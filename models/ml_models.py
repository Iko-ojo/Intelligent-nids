"""
Machine Learning model definitions and training functions.

This module provides implementations of various traditional ML models
for network anomaly detection including Random Forest, XGBoost, SVM, etc.
"""
from typing import Dict, Any
from scipy.optimize import linear_sum_assignment
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans, AgglomerativeClustering
from xgboost import XGBClassifier
import numpy as np


class KMeansClassifier:
    """
    Wrapper around KMeans that exposes a classifier interface (fit(X, y), predict(X))
    for use in supervised pipelines.

    It first fits KMeans on X, then maps each cluster ID to a class label using
    statistics from y. The mapping is **anomaly-aware**:

    - By default it uses majority vote within each cluster.
    - Clusters that are small or contain a high fraction of "attack" samples
      (labels != 0) are biased towards an attack label to improve recall/F1 for
      attacks while still supporting multi-class labels.
    """

    def __init__(
        self,
        n_clusters: int = 2,
        random_state: int = 42,
        n_init: int = 10,
        anomaly_fraction_threshold: float = 0.4,
        small_cluster_fraction_threshold: float = 0.02,
        single_benign_cluster: bool = False,
        use_hungarian_mapping: bool = False,
        **kwargs: Any
    ):
        self.n_clusters = n_clusters
        self.random_state = random_state
        # Thresholds for anomaly-aware cluster -> class mapping.
        # We assume label 0 is "benign" and any non-zero label is some form of attack
        # (consistent with the data loader's use of 0 for benign and 1+ for attacks).
        self.anomaly_fraction_threshold = anomaly_fraction_threshold
        self.small_cluster_fraction_threshold = small_cluster_fraction_threshold
        # If True, only one cluster is assigned to benign (0); others are forced to an attack label.
        # Pop from kwargs so it's never passed to KMeans (handles old reloads where it wasn't in signature)
        self.single_benign_cluster = kwargs.pop("single_benign_cluster", single_benign_cluster)
        # If True, use optimal 1:1 cluster-to-class assignment (Hungarian) when n_clusters >= n_classes.
        self.use_hungarian_mapping = kwargs.pop("use_hungarian_mapping", use_hungarian_mapping)
        self.kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=random_state,
            n_init=n_init,
            **kwargs
        )
        self.cluster_to_class_: np.ndarray = np.array([])

    def fit(self, X: np.ndarray, y: np.ndarray) -> "KMeansClassifier":
        self.kmeans.fit(X)
        cluster_labels = self.kmeans.labels_
        n_samples = len(y)

        # Map each cluster to a class label with anomaly-aware heuristics.
        # - Start from majority vote within the cluster.
        # - If the cluster is small or has a high fraction of non-zero labels,
        #   prefer an attack label (non-zero) to improve anomaly recall.
        benign_label = 0
        self.cluster_to_class_ = np.zeros(self.n_clusters, dtype=np.intp)

        for c in range(self.n_clusters):
            mask = cluster_labels == c
            if not mask.any():
                # Empty cluster: default to benign
                self.cluster_to_class_[c] = benign_label
                continue

            y_cluster = y[mask]
            values, counts = np.unique(y_cluster, return_counts=True)
            cluster_size = counts.sum()

            # Baseline: majority label in this cluster
            majority_idx = int(counts.argmax())
            majority_label = int(values[majority_idx])

            # Benign vs attack statistics
            if benign_label in values:
                benign_count = int(counts[values == benign_label].sum())
            else:
                benign_count = 0
            attack_count = int(cluster_size - benign_count)

            attack_fraction = attack_count / float(cluster_size)
            cluster_fraction = cluster_size / float(n_samples) if n_samples > 0 else 0.0

            assigned_label = majority_label

            # If there are any attack samples in this cluster, decide whether we
            # should bias this cluster towards an attack label instead of benign.
            if attack_count > 0:
                # Pick the attack label with the highest count in this cluster
                attack_mask = values != benign_label
                attack_values = values[attack_mask]
                attack_counts = counts[attack_mask]
                top_attack_label = int(attack_values[int(attack_counts.argmax())])

                # Override majority-benign clusters when:
                # - the fraction of attacks in the cluster is high enough, OR
                # - the cluster is very small (outlier-like) but contains attacks.
                if majority_label == benign_label and (
                    attack_fraction >= self.anomaly_fraction_threshold
                    or cluster_fraction <= self.small_cluster_fraction_threshold
                ):
                    assigned_label = top_attack_label

            self.cluster_to_class_[c] = assigned_label

        # Optionally enforce only one cluster mapped to benign (0).
        if self.single_benign_cluster:
            benign_label = 0
            benign_counts = np.zeros(self.n_clusters, dtype=np.intp)
            for c in range(self.n_clusters):
                mask = cluster_labels == c
                if mask.any():
                    benign_counts[c] = (y[mask] == benign_label).sum()
            chosen_benign = int(np.argmax(benign_counts))
            for c in range(self.n_clusters):
                if c != chosen_benign and self.cluster_to_class_[c] == benign_label:
                    mask = cluster_labels == c
                    if mask.any():
                        y_c = y[mask]
                        attack_mask = y_c != benign_label
                        if attack_mask.any():
                            values, counts = np.unique(
                                y_c[attack_mask], return_counts=True
                            )
                            self.cluster_to_class_[c] = int(
                                values[int(counts.argmax())]
                            )
                        else:
                            self.cluster_to_class_[c] = 1  # fallback to first attack class
                    else:
                        self.cluster_to_class_[c] = 1

        # Optionally replace with optimal 1:1 cluster-to-class assignment.
        if self.use_hungarian_mapping:
            unique_labels = np.unique(y)
            n_classes = len(unique_labels)
            if self.n_clusters >= n_classes:
                # Count matrix: C[c, j] = samples in cluster c with label unique_labels[j]
                C = np.zeros((self.n_clusters, n_classes), dtype=np.float64)
                for c in range(self.n_clusters):
                    mask = cluster_labels == c
                    if mask.any():
                        for j, lab in enumerate(unique_labels):
                            C[c, j] = (y[mask] == lab).sum()
                # Maximize total agreement: minimize -C
                row_ind, col_ind = linear_sum_assignment(-C)
                # Assign: cluster row_ind[i] -> unique_labels[col_ind[i]]
                new_mapping = np.copy(self.cluster_to_class_)
                assigned_rows = set(row_ind)
                for i in range(len(row_ind)):
                    new_mapping[row_ind[i]] = int(unique_labels[col_ind[i]])
                # Unassigned clusters (when n_clusters > n_classes) keep previous mapping
                for c in range(self.n_clusters):
                    if c not in assigned_rows:
                        mask = cluster_labels == c
                        if mask.any():
                            values, counts = np.unique(y[mask], return_counts=True)
                            new_mapping[c] = int(values[int(counts.argmax())])
                self.cluster_to_class_ = new_mapping

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        cluster_labels = self.kmeans.predict(X)
        return self.cluster_to_class_[cluster_labels]


def _map_clusters_to_classes(
    cluster_labels: np.ndarray,
    y: np.ndarray,
    n_clusters: int,
    anomaly_fraction_threshold: float,
    small_cluster_fraction_threshold: float,
    benign_label: int = 0,
) -> np.ndarray:
    """
    Map cluster IDs to class labels using anomaly-aware heuristics.
    Shared logic between KMeansClassifier and HierarchicalClusteringClassifier.
    """
    n_samples = len(y)
    cluster_to_class = np.zeros(n_clusters, dtype=np.intp)

    for c in range(n_clusters):
        mask = cluster_labels == c
        if not mask.any():
            cluster_to_class[c] = benign_label
            continue

        y_cluster = y[mask]
        values, counts = np.unique(y_cluster, return_counts=True)
        cluster_size = counts.sum()

        majority_idx = int(counts.argmax())
        majority_label = int(values[majority_idx])

        if benign_label in values:
            benign_count = int(counts[values == benign_label].sum())
        else:
            benign_count = 0
        attack_count = int(cluster_size - benign_count)
        attack_fraction = attack_count / float(cluster_size)
        cluster_fraction = cluster_size / float(n_samples) if n_samples > 0 else 0.0

        assigned_label = majority_label

        if attack_count > 0:
            attack_mask = values != benign_label
            attack_values = values[attack_mask]
            attack_counts = counts[attack_mask]
            top_attack_label = int(attack_values[int(attack_counts.argmax())])

            if majority_label == benign_label and (
                attack_fraction >= anomaly_fraction_threshold
                or cluster_fraction <= small_cluster_fraction_threshold
            ):
                assigned_label = top_attack_label

        cluster_to_class[c] = assigned_label

    return cluster_to_class


class HierarchicalClusteringClassifier:
    """
    Wrapper around AgglomerativeClustering that exposes a classifier interface
    (fit(X, y), predict(X)) for use in supervised pipelines.

    Uses the same anomaly-aware cluster-to-class mapping as KMeansClassifier.
    For prediction on new data, assigns each point to the nearest cluster
    centroid (computed from training data).
    """

    def __init__(
        self,
        n_clusters: int = 2,
        linkage: str = "ward",
        anomaly_fraction_threshold: float = 0.4,
        small_cluster_fraction_threshold: float = 0.02,
        max_fit_samples=None,
        random_state: int = 42,
        **kwargs: Any
    ):
        self.n_clusters = n_clusters
        self.linkage = linkage
        self.anomaly_fraction_threshold = anomaly_fraction_threshold
        self.small_cluster_fraction_threshold = small_cluster_fraction_threshold
        # To keep hierarchical clustering tractable on large datasets, we can
        # subsample in fit() using max_fit_samples.
        self.max_fit_samples = max_fit_samples
        self.random_state = random_state
        self.hierarchical = AgglomerativeClustering(
            n_clusters=n_clusters,
            linkage=linkage,
            **kwargs
        )
        self.cluster_to_class_: np.ndarray = np.array([])
        self.cluster_centroids_: np.ndarray = np.array([])

    def fit(self, X: np.ndarray, y: np.ndarray) -> "HierarchicalClusteringClassifier":
        # Optionally subsample for fitting to avoid O(n^2) scaling on very large datasets.
        if self.max_fit_samples is not None and X.shape[0] > self.max_fit_samples:
            rng = np.random.RandomState(self.random_state)
            indices = rng.choice(X.shape[0], size=self.max_fit_samples, replace=False)
            X_fit = X[indices]
            y_fit = y[indices]
        else:
            X_fit = X
            y_fit = y

        self.hierarchical.fit(X_fit)
        cluster_labels = self.hierarchical.labels_
        benign_label = 0

        self.cluster_to_class_ = _map_clusters_to_classes(
            cluster_labels,
            y_fit,
            self.n_clusters,
            self.anomaly_fraction_threshold,
            self.small_cluster_fraction_threshold,
            benign_label,
        )

        # Compute cluster centroids for prediction on new data
        self.cluster_centroids_ = np.zeros((self.n_clusters, X.shape[1]), dtype=np.float64)
        for c in range(self.n_clusters):
            mask = cluster_labels == c
            if mask.any():
                self.cluster_centroids_[c] = X_fit[mask].mean(axis=0)
            else:
                # Fallback to global mean over full training data
                self.cluster_centroids_[c] = X.mean(axis=0)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        # Assign each point to nearest centroid (vectorized via cdist)
        from scipy.spatial.distance import cdist
        dists = cdist(X, self.cluster_centroids_, metric="euclidean")
        cluster_labels = np.argmin(dists, axis=1).astype(np.intp)
        return self.cluster_to_class_[cluster_labels]


def _distances_to_centroids(X: np.ndarray, centroids: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Return distance of each point to its assigned centroid."""
    distances = np.zeros(len(X), dtype=np.float64)
    for i in range(len(X)):
        c = int(labels[i])
        distances[i] = np.linalg.norm(X[i] - centroids[c])
    return distances


class HybridKMeansXGBoost:
    """
    Hybrid model: KMeans flags anomalies (atypical clusters + high-distance samples),
    XGBoost classifies those flagged samples for final prediction.
    Non-flagged samples use KMeans cluster-to-class mapping.
    """

    def __init__(
        self,
        n_clusters: int = 10,
        distance_percentile: float = 0.9,
        small_cluster_fraction: float = 0.02,
        random_state: int = 42,
        kmeans_kwargs: Dict[str, Any] = None,
        xgb_kwargs: Dict[str, Any] = None,
    ):
        self.n_clusters = n_clusters
        self.distance_percentile = distance_percentile
        self.small_cluster_fraction = small_cluster_fraction
        self.random_state = random_state
        self.kmeans_kwargs = kmeans_kwargs or {}
        self.xgb_kwargs = xgb_kwargs or {}
        self.kmeans_clf_: KMeansClassifier = None
        self.xgb_: XGBClassifier = None
        self.anomaly_clusters_: np.ndarray = np.array([])  # bool mask: cluster c is anomaly
        self.distance_threshold_: float = 0.0
        self.small_clusters_: np.ndarray = np.array([])  # bool: cluster c is small

    def fit(self, X: np.ndarray, y: np.ndarray) -> "HybridKMeansXGBoost":
        self.kmeans_clf_ = KMeansClassifier(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            **self.kmeans_kwargs
        )
        self.kmeans_clf_.fit(X, y)
        n_classes = len(np.unique(y))
        xgb_params = dict(self.xgb_kwargs)
        # For binary (2 classes), do NOT set num_class - XGBoost uses binary:logistic.
        # Setting num_class=2 forces multi:softprob, causing "Invalid shape of labels"
        # (preds size = n_samples * n_classes vs labels = n_samples).
        if n_classes > 2:
            xgb_params.setdefault("num_class", n_classes)
        xgb_params.setdefault("use_label_encoder", False)
        self.xgb_ = XGBClassifier(
            random_state=self.random_state,
            n_jobs=-1,
            **xgb_params
        )
        self.xgb_.fit(X, y)

        cluster_labels = self.kmeans_clf_.kmeans.labels_
        centroids = self.kmeans_clf_.kmeans.cluster_centers_
        n_samples = len(y)

        # Anomaly clusters = clusters mapped to non-benign (label != 0)
        self.anomaly_clusters_ = (self.kmeans_clf_.cluster_to_class_ != 0)

        # Small clusters = cluster size < small_cluster_fraction of data
        cluster_sizes = np.array([(cluster_labels == c).sum() for c in range(self.n_clusters)])
        threshold_size = max(1, int(n_samples * self.small_cluster_fraction))
        self.small_clusters_ = (cluster_sizes < threshold_size)

        # Distance threshold: percentile of within-cluster distances on training set
        distances = _distances_to_centroids(X, centroids, cluster_labels)
        self.distance_threshold_ = float(np.percentile(distances, self.distance_percentile * 100))

        return self

    def _get_flagged_mask(self, X: np.ndarray) -> np.ndarray:
        cluster_labels = self.kmeans_clf_.kmeans.predict(X)
        centroids = self.kmeans_clf_.kmeans.cluster_centers_
        distances = _distances_to_centroids(X, centroids, cluster_labels)

        in_anomaly_cluster = self.anomaly_clusters_[cluster_labels]
        in_small_cluster = self.small_clusters_[cluster_labels]
        high_distance = distances > self.distance_threshold_
        flagged = in_anomaly_cluster | in_small_cluster | high_distance
        return flagged, cluster_labels

    def predict(self, X: np.ndarray) -> np.ndarray:
        flagged, cluster_labels = self._get_flagged_mask(X)
        y_pred = np.zeros(len(X), dtype=np.intp)
        y_pred[~flagged] = self.kmeans_clf_.cluster_to_class_[cluster_labels[~flagged]]
        if flagged.any():
            y_pred[flagged] = self.xgb_.predict(X[flagged])
        return y_pred

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities for X. Uses XGBoost proba for flagged, one-hot for KMeans."""
        flagged, cluster_labels = self._get_flagged_mask(X)
        n_classes = self.xgb_.n_classes_
        proba = np.zeros((len(X), n_classes), dtype=np.float64)
        proba[~flagged] = 0.0
        for i in np.where(~flagged)[0]:
            lab = self.kmeans_clf_.cluster_to_class_[cluster_labels[i]]
            proba[i, lab] = 1.0
        if flagged.any():
            proba[flagged] = self.xgb_.predict_proba(X[flagged])
        return proba


def create_ml_models(
    random_state: int = 42,
    n_jobs: int = -1,
    xgboost_params: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Create instances of all machine learning models.
    
    Args:
        random_state: Random state for reproducibility
        n_jobs: Number of parallel jobs (-1 for all cores)
        xgboost_params: Additional parameters for XGBoost
        
    Returns:
        Dictionary mapping model names to model instances
    """
    if xgboost_params is None:
        xgboost_params = {}
    
    models = {
        'Random Forest': RandomForestClassifier(
            n_estimators=100,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=n_jobs,
            verbose=0
        ),
        # 'Decision Tree': DecisionTreeClassifier(
        #     max_depth=20,
        #     min_samples_split=5,
        #     min_samples_leaf=2,
        #     random_state=random_state
        # ),
        'XGBoost': XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=random_state,
            n_jobs=n_jobs,
            **xgboost_params
        ),
        # 'SVM': SVC(
        #     kernel='rbf',
        #     C=1.0,
        #     gamma='scale',
        #     probability=True,
        #     random_state=random_state,
        #     verbose=False
        # ),
        # 'Logistic Regression': LogisticRegression(
        #     max_iter=1000,
        #     random_state=random_state,
        #     n_jobs=n_jobs,
        #     verbose=0
        # ),
        # 'KNN': KNeighborsClassifier(
        #     n_neighbors=5,
        #     n_jobs=n_jobs
        # ),
        'KMeans': KMeansClassifier(
            n_clusters=5,
            random_state=random_state,
            n_init=50,
            max_iter=500
        )
    }

    return models


def train_ml_models(
    models: Dict[str, Any],
    X_train: np.ndarray,
    y_train: np.ndarray,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Train all machine learning models.
    
    Args:
        models: Dictionary of model name -> model instance
        X_train: Training feature matrix
        y_train: Training labels
        verbose: Whether to print training progress
        
    Returns:
        Dictionary of trained models
    """
    trained_models = {}
    
    for name, model in models.items():
        if verbose:
            print(f"Training {name}...", end=" ", flush=True)
        
        try:
            model.fit(X_train, y_train)
            trained_models[name] = model
            
            if verbose:
                print("✓ Completed")
        except Exception as e:
            if verbose:
                print(f"✗ Failed: {str(e)}")
            # Continue with other models even if one fails
            continue
    
    if verbose:
        print(f"\nSuccessfully trained {len(trained_models)}/{len(models)} ML models\n")
    
    return trained_models

