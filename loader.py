"""
Data loading and preprocessing functions for network anomaly detection.

This module handles loading CSV data, preprocessing features, encoding
categorical variables, and preparing data for model training.
"""
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple, Dict, Optional, Any, Callable
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from collections import Counter
import warnings

warnings.filterwarnings("ignore")


def _standardize_raw_label(value: Any) -> str:
    """
    Helper to normalise a raw label value into a clean, uppercase string.
    """
    return str(value).strip().upper()


def map_label_to_four_classes(value: Any) -> int:
    """
    Map the original UNSW‑NB15 style textual label into four canonical classes:
        0 -> BENIGN
        1 -> Web Attacks*
        2 -> DDOS*
        3 -> PortScan

    Notes
    -----
    - This helper is kept for backward compatibility with older merged CSV
      files that contain human‑readable string labels.
    - The new CIC‑UNSW dataset you are using provides a separate `Label.csv`
      file with *numeric* labels. For that case we **do not** apply this
      mapping and instead treat the integers as the ground‑truth classes.
    """
    label = _standardize_raw_label(value)

    # BENIGN traffic
    if "BENIGN" in label:
        return 0

    # Web application attacks (e.g. "Web Attack - Brute Force",
    # "Web Attack - XSS", "Web Attack - SQL Injection", etc.)
    if "WEB ATTACK" in label or "WEB_ATTACK" in label or "WEB-ATTACK" in label:
        return 1

    # DDoS style flooding attacks (e.g. "DDoS Hulk", "DDoS GoldenEye")
    if "DDOS" in label or "DDoS" in label:
        return 2

    # PortScan attacks
    if "PORTSCAN" in label or "PORT SCAN" in label:
        return 3

    # Fallback: treat any other non-benign label as Web Attack.
    return 1


def load_data(
    train_path: Optional[Path] = None,
    test_path: Optional[Path] = None,
    dataset_path: Optional[Path] = None,
    features_path: Optional[Path] = None,
    labels_path: Optional[Path] = None,
    label_column: str = "Label",
    test_size: float = 0.3,
    random_state: int = 42,
    output_dir: Optional[Path] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load training and test data from CSV files.
    
    Supports three usage patterns:
    1) Separate train and test CSV files
       (if ``train_path`` and ``test_path`` are provided)
    2) Single dataset CSV file that will be split
       (if ``dataset_path`` is provided)
    3) Separate feature and label CSV files (e.g. CIC‑UNSW)
       (if ``features_path`` and ``labels_path`` are provided)
    
    Args:
        train_path: Path to training data CSV file (optional if dataset_path provided)
        test_path: Path to test data CSV file (optional if dataset_path provided)
        dataset_path: Path to single dataset CSV file to split (optional if train_path/test_path provided)
        features_path: Path to CSV file containing only features (no label column)
        labels_path: Path to CSV file containing labels (e.g. single ``Label`` column)
        label_column: Name of the label column
        test_size: Proportion of dataset to include in test split (default: 0.3)
        random_state: Random state for train_test_split (default: 42)
        
    Returns:
        Tuple of (train_df, test_df) DataFrames
        
    Raises:
        FileNotFoundError: If data files don't exist
        ValueError: If data files are empty or malformed, or if the
                    combination of paths is invalid.
    """
    def _plot_four_class_distribution(y_raw: pd.Series, split_name: str) -> None:
        """
        Plot class distribution for the four canonical classes:
        - 0: BENIGN
        - 1: Web Attacks*
        - 2: DDOS*
        - 3: PortScan
        """
        if output_dir is None:
            return

        output_dir.mkdir(parents=True, exist_ok=True)

        # For legacy string labels we map into the four canonical classes.
        # When the labels are already numeric (e.g. CIC‑UNSW) we only plot
        # a 4‑class view if the values are exactly a subset of {0,1,2,3}.
        try:
            import pandas.api.types as ptypes  # type: ignore
        except Exception:
            ptypes = None

        if ptypes is not None and (
            ptypes.is_integer_dtype(y_raw) or ptypes.is_float_dtype(y_raw)
        ):
            unique_vals = set(pd.Series(y_raw).dropna().unique().tolist())
            if unique_vals.issubset({0, 1, 2, 3}):
                mapped = y_raw.astype(int)
            else:
                # Labels are numeric but not in the 0‑3 range; skip this
                # specific 4‑class plot to avoid misleading visuals.
                print(
                    f"Skipping 4‑class distribution plot for {split_name} "
                    f"(numeric labels not limited to {{0,1,2,3}})."
                )
                return
        else:
            mapped = y_raw.apply(map_label_to_four_classes)
        counts = mapped.value_counts().reindex([0, 1, 2, 3], fill_value=0)

        plt.figure(figsize=(8, 5))
        # Plot in a fixed, explicit order so labels can't drift
        x_labels = ["BENIGN (0)", "Web Attacks (1)", "DDOS (2)", "PortScan (3)"]
        y_vals = [int(counts.loc[0]), int(counts.loc[1]), int(counts.loc[2]), int(counts.loc[3])]
        ax = sns.barplot(x=x_labels, y=y_vals)
        ax.set_xlabel("Class")
        ax.set_ylabel("Count")
        ax.set_title(f"{split_name} Set Class Distribution (4-class)")
        plt.xticks(rotation=15)
        plt.tight_layout()

        save_path = output_dir / f"class_distribution_{split_name.lower()}_4class.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved {split_name.lower()} 4-class distribution plot to {save_path}")

    def _plot_train_test_split_overview(train_count: int, test_count: int) -> None:
        """
        Plot a single horizontal bar showing 100% split into Train/Test.
        """
        if output_dir is None:
            return

        output_dir.mkdir(parents=True, exist_ok=True)

        total = train_count + test_count
        if total <= 0:
            return

        train_pct = (train_count / total) * 100
        test_pct = (test_count / total) * 100

        fig, ax = plt.subplots(figsize=(10, 2.2))
        ax.barh(["Dataset split"], [train_pct], color="#4C78A8", label=f"Training ({train_pct:.1f}%)")
        ax.barh(["Dataset split"], [test_pct], left=[train_pct], color="#F58518", label=f"Test ({test_pct:.1f}%)")

        # Add text labels centered in each segment (only if segment wide enough)
        if train_pct >= 8:
            ax.text(train_pct / 2, 0, f"Train\n{train_pct:.1f}%\n(n={train_count})",
                    va="center", ha="center", color="white", fontsize=10, fontweight="bold")
        if test_pct >= 8:
            ax.text(train_pct + (test_pct / 2), 0, f"Test\n{test_pct:.1f}%\n(n={test_count})",
                    va="center", ha="center", color="white", fontsize=10, fontweight="bold")

        ax.set_xlim(0, 100)
        ax.set_xlabel("Percentage of full dataset (%)")
        ax.set_yticks([])
        ax.set_title("Train/Test Split Overview (100%)")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=2, frameon=False)
        plt.tight_layout()

        save_path = output_dir / "dataset_split_train_test_100pct.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved train/test split overview plot to {save_path}")

    # Determine which loading method to use
    if features_path is not None and labels_path is not None:
        # ------------------------------------------------------------------
        # 3) Separate features and labels CSVs (e.g. CIC‑UNSW Data/Label)
        # ------------------------------------------------------------------
        if not features_path.exists():
            raise FileNotFoundError(f"Features file not found: {features_path}")
        if not labels_path.exists():
            raise FileNotFoundError(f"Labels file not found: {labels_path}")

        print(f"Loading features from {features_path}...")
        X_df = pd.read_csv(features_path, low_memory=False)
        X_df.columns = X_df.columns.str.strip()
        print(f"Features shape: {X_df.shape}")

        print(f"Loading labels from {labels_path}...")
        y_df = pd.read_csv(labels_path, low_memory=False)
        y_df.columns = y_df.columns.str.strip()
        print(f"Labels shape: {y_df.shape}")

        if X_df.shape[0] != y_df.shape[0]:
            raise ValueError(
                f"Features and labels have different number of rows: "
                f"{X_df.shape[0]} vs {y_df.shape[0]}"
            )

        # Determine which column holds the labels
        if label_column in y_df.columns:
            y_series = y_df[label_column]
        else:
            if y_df.shape[1] != 1:
                raise ValueError(
                    f"Labels CSV must contain a '{label_column}' column or a "
                    f"single unnamed label column. Available columns: "
                    f"{list(y_df.columns)}"
                )
            y_series = y_df.iloc[:, 0]

        # Merge into a single DataFrame for the rest of the pipeline
        df = X_df.copy()
        df[label_column] = y_series.values

        print(f"Merged dataset shape (features + label): {df.shape}")

        # ------------------------------------------------------------------
        # Rebalance benign vs attack classes (CIC‑UNSW numeric labels)
        # ------------------------------------------------------------------
        # The raw dataset is heavily skewed (~80% benign, ~20% attacks).
        # To obtain an overall distribution of ~70% benign / 30% attacks
        # (where "attacks" are all classes != 0), we downsample the benign
        # class before performing the train/test split.
        benign_label = 0
        benign_mask = df[label_column] == benign_label
        n_benign = int(benign_mask.sum())
        n_total = len(df)
        n_attack = n_total - n_benign

        print(
            f"Original overall class counts -> "
            f"benign (0): {n_benign}, attacks (1+): {n_attack}"
        )

        if n_attack > 0 and n_benign > 0:
            target_benign_ratio = 0.7
            target_attack_ratio = 0.3
            # For a target 70/30 split we want:
            #   B' / (B' + A') = 0.7  =>  B' = (0.7 / 0.3) * A'
            target_benign_count = int((target_benign_ratio / target_attack_ratio) * n_attack)

            # We only downsample benign; never oversample beyond available data.
            target_benign_count = min(target_benign_count, n_benign)

            if target_benign_count < n_benign:
                print(
                    f"\nRebalancing dataset to approximately "
                    f"{int(target_benign_ratio * 100)}/{int(target_attack_ratio * 100)} "
                    "Benign/Attack by downsampling benign class..."
                )
                df_benign_sampled = df[benign_mask].sample(
                    n=target_benign_count, random_state=random_state
                )
                df_attacks = df[~benign_mask]

                df = (
                    pd.concat([df_benign_sampled, df_attacks], axis=0)
                    .sample(frac=1.0, random_state=random_state)
                    .reset_index(drop=True)
                )

                print(f"Rebalanced dataset shape: {df.shape}")

                # Show new distribution
                new_counts = df[label_column].value_counts().sort_index()
                new_total = len(df)
                print("New overall class distribution:")
                for cls, cnt in new_counts.items():
                    pct = (cnt / new_total) * 100
                    print(f"  {cls}: {cnt} ({pct:.2f}%)")

        # Separate features and target (after any rebalancing)
        X = df.drop([label_column], axis=1)
        y = df[label_column]

        # Split into train and test (stratified so both keep ~70/30)
        print(
            f"\nSplitting dataset into train/test with "
            f"test_size={test_size}, random_state={random_state}..."
        )
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=y,
        )

        # Create train and test DataFrames
        train_df = X_train.copy()
        train_df[label_column] = y_train.values

        test_df = X_test.copy()
        test_df[label_column] = y_test.values

        print(f"Training data shape: {train_df.shape}")
        print(f"Test data shape: {test_df.shape}")

        # Display unique counts for y_train
        unique, counts = np.unique(y_train, return_counts=True)
        class_counts = dict(zip(unique, counts))
        print(f"\nTraining set class distribution:")
        for class_name, count in class_counts.items():
            percentage = (count / len(y_train)) * 100
            print(f"  {class_name}: {count} ({percentage:.2f}%)")

        # Display unique counts for y_test
        unique_test, counts_test = np.unique(y_test, return_counts=True)
        class_counts_test = dict(zip(unique_test, counts_test))
        print(f"\nTest set class distribution:")
        for class_name, count in class_counts_test.items():
            percentage = (count / len(y_test)) * 100
            print(f"  {class_name}: {count} ({percentage:.2f}%)")

        # Four-class plots are only generated when labels are compatible
        _plot_four_class_distribution(y_train, "Training")
        _plot_four_class_distribution(y_test, "Test")
        _plot_train_test_split_overview(
            train_count=len(train_df), test_count=len(test_df)
        )

        return train_df, test_df

    if dataset_path is not None:
        # Load from single dataset and split
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
        
        print(f"Loading dataset from {dataset_path}...")
        df = pd.read_csv(dataset_path, low_memory=False)
        # Strip whitespace from column names
        df.columns = df.columns.str.strip()
        print(f"Dataset shape: {df.shape}")
        
        # Validate label column exists
        if label_column not in df.columns:
            raise ValueError(f"Label column '{label_column}' not found in dataset. Available columns: {list(df.columns[-5:])}")
        
        # Separate features and target
        X = df.drop([label_column], axis=1)
        y = df[label_column]
        
        # Split into train and test
        print(f"\nSplitting dataset into train/test with test_size={test_size}, random_state={random_state}...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Create train and test DataFrames
        train_df = X_train.copy()
        train_df[label_column] = y_train.values
        
        test_df = X_test.copy()
        test_df[label_column] = y_test.values
        
        print(f"Training data shape: {train_df.shape}")
        print(f"Test data shape: {test_df.shape}")
        
        # Display unique counts for y_train
        unique, counts = np.unique(y_train, return_counts=True)
        class_counts = dict(zip(unique, counts))
        print(f"\nTraining set class distribution:")
        for class_name, count in class_counts.items():
            percentage = (count / len(y_train)) * 100
            print(f"  {class_name}: {count} ({percentage:.2f}%)")
        
        # Display unique counts for y_test
        unique_test, counts_test = np.unique(y_test, return_counts=True)
        class_counts_test = dict(zip(unique_test, counts_test))
        print(f"\nTest set class distribution:")
        for class_name, count in class_counts_test.items():
            percentage = (count / len(y_test)) * 100
            print(f"  {class_name}: {count} ({percentage:.2f}%)")

        # Four-class labels for plotting (0: BENIGN, 1: Web Attacks, 2: DDOS, 3: PortScan)
        _plot_four_class_distribution(y_train, "Training")
        _plot_four_class_distribution(y_test, "Test")
        _plot_train_test_split_overview(train_count=len(train_df), test_count=len(test_df))
        
        return train_df, test_df
    
    elif train_path is not None and test_path is not None:
        # Load from separate train and test files (backward compatibility)
        if not train_path.exists():
            raise FileNotFoundError(f"Training data file not found: {train_path}")
        if not test_path.exists():
            raise FileNotFoundError(f"Test data file not found: {test_path}")
        
        print(f"Loading training data from {train_path}...")
        train_df = pd.read_csv(train_path, low_memory=False)
        # Strip whitespace from column names
        train_df.columns = train_df.columns.str.strip()
        print(f"Training data shape: {train_df.shape}")
        
        print(f"Loading test data from {test_path}...")
        test_df = pd.read_csv(test_path, low_memory=False)
        # Strip whitespace from column names
        test_df.columns = test_df.columns.str.strip()
        print(f"Test data shape: {test_df.shape}")
        
        # Validate label column exists
        if label_column not in train_df.columns:
            raise ValueError(f"Label column '{label_column}' not found in training data. Available columns: {list(train_df.columns[-5:])}")
        if label_column not in test_df.columns:
            raise ValueError(f"Label column '{label_column}' not found in test data. Available columns: {list(test_df.columns[-5:])}")
        
        # Display unique counts for training set
        y_train = train_df[label_column]
        unique, counts = np.unique(y_train, return_counts=True)
        class_counts = dict(zip(unique, counts))
        print(f"\nTraining set class distribution:")
        for class_name, count in class_counts.items():
            percentage = (count / len(y_train)) * 100
            print(f"  {class_name}: {count} ({percentage:.2f}%)")
        
        # Display unique counts for test set
        y_test = test_df[label_column]
        unique_test, counts_test = np.unique(y_test, return_counts=True)
        class_counts_test = dict(zip(unique_test, counts_test))
        print(f"\nTest set class distribution:")
        for class_name, count in class_counts_test.items():
            percentage = (count / len(y_test)) * 100
            print(f"  {class_name}: {count} ({percentage:.2f}%)")

        # Four-class labels for plotting (0: BENIGN, 1: Web Attacks, 2: DDOS, 3: PortScan)
        _plot_four_class_distribution(y_train, "Training")
        _plot_four_class_distribution(y_test, "Test")
        _plot_train_test_split_overview(train_count=len(train_df), test_count=len(test_df))
        
        return train_df, test_df
    
    else:
        raise ValueError("Either dataset_path must be provided, or both train_path and test_path must be provided")


def convert_to_binary_classification(
    df: pd.DataFrame,
    label_column: str,
    class_mapping: Optional[Dict[str, str]] = None
) -> pd.DataFrame:
    """
    Legacy helper kept for backward compatibility.

    NOTE: The current pipeline uses explicit 4‑class labels
    (0 = BENIGN, 1 = Web Attacks, 2 = DDOS, 3 = PortScan) and does not rely
    on this function any more.
    """
    df = df.copy()

    # Preserve previous behaviour in case older code imports this.
    if class_mapping is None:
        df[label_column] = df[label_column].apply(
            lambda x: "normal" if _standardize_raw_label(x) == "BENIGN" else "attack"
        )
    else:
        df[label_column] = df[label_column].map(class_mapping)
        df[label_column] = df[label_column].fillna("attack")

    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing values in the dataset.
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with missing values handled
    """
    df = df.copy()
    
    # Fill numeric columns with median
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isna().any():
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)
    
    # Fill categorical columns with mode
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if df[col].isna().any():
            mode_val = df[col].mode()[0] if not df[col].mode().empty else "unknown"
            df[col].fillna(mode_val, inplace=True)
    
    return df


def preprocess_data(
    df: pd.DataFrame,
    label_column: str,
    fit_encoders: bool = True,
    label_encoders: Optional[Dict[str, LabelEncoder]] = None,
    scaler: Optional[StandardScaler] = None,
    class_mapping: Optional[Dict[str, str]] = None,
    handle_missing: bool = True,
    apply_four_class_mapping: bool = True,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, LabelEncoder], StandardScaler]:
    """
    Preprocess the dataset: handle missing values, encode labels, and scale features.
    
    Args:
        df: DataFrame to preprocess
        label_column: Name of the label column
        fit_encoders: Whether to fit encoders (True for training, False for test)
        label_encoders: Dictionary of label encoders (None if fit_encoders=True)
        scaler: StandardScaler object (None if fit_encoders=True)
        class_mapping: Mapping for binary classification
        handle_missing: Whether to handle missing values
        
    Returns:
        Tuple of (X, y, label_encoders, scaler)
        - X: Scaled feature matrix
        - y: Encoded target labels
        - label_encoders: Dictionary of fitted label encoders
        - scaler: Fitted StandardScaler
    """
    df = df.copy()
    
    # Handle missing values
    if handle_missing:
        df = handle_missing_values(df)
    
    # Optionally map labels to the canonical four‑class integer representation
    # used by the original UNSW‑NB15 experiments:
    #   0 -> BENIGN
    #   1 -> Web Attacks*
    #   2 -> DDOS*
    #   3 -> PortScan
    #
    # For the new CIC‑UNSW dataset where labels are already numeric in a
    # dedicated `Label.csv`, we **disable** this mapping so that we preserve
    # the native class IDs.
    if apply_four_class_mapping:
        df[label_column] = df[label_column].apply(map_label_to_four_classes)

    # Initialize encoders if needed (used only for feature columns)
    if fit_encoders:
        label_encoders = {}
        scaler = StandardScaler()
    else:
        # Ensure we have feature names stored
        if scaler is not None and not hasattr(scaler, 'feature_names_in_'):
            # Try to get from label_encoders if stored there
            if label_encoders is not None and '_feature_names' in label_encoders:
                pass  # Will use stored names
    
    # Separate features and target
    X = df.drop([label_column], axis=1)
    # y is already an integer array in {0, 1, 2, 3}
    y = df[label_column].to_numpy()
    
    # Ensure consistent column order and handle missing columns
    if not fit_encoders and label_encoders is not None:
        # For test data, align columns with training data
        # This handles cases where test set has different columns
        pass  # We'll handle this after conversion
    
    # Convert all columns to numeric (handle any remaining non-numeric columns)
    for col in X.columns:
        if X[col].dtype == 'object':
            # Try to convert to numeric, if fails, encode it
            try:
                X[col] = pd.to_numeric(X[col], errors='coerce')
            except:
                # If conversion fails, use label encoding
                if fit_encoders:
                    if col not in label_encoders:
                        label_encoders[col] = LabelEncoder()
                    X[col] = label_encoders[col].fit_transform(X[col].astype(str))
                else:
                    if col in label_encoders:
                        # Handle unseen categories
                        X[col] = X[col].astype(str)
                        known_cats = set(label_encoders[col].classes_)
                        X[col] = X[col].apply(
                            lambda x: label_encoders[col].transform([list(known_cats)[0]])[0]
                            if x not in known_cats
                            else label_encoders[col].transform([x])[0]
                        )
                    else:
                        # If encoder doesn't exist, fill with 0
                        X[col] = 0
    
    # Fill any remaining NaN values with 0
    X = X.fillna(0)
    
    # Replace infinity and very large values
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(0)
    
    # Clip very large values to prevent overflow
    # Use a reasonable maximum value (e.g., 1e10)
    max_val = 1e10
    X = X.clip(lower=-max_val, upper=max_val)
    
    # Scale features
    if fit_encoders:
        X_transformed = scaler.fit_transform(X)
        # Store feature names for later alignment in label_encoders dict
        if label_encoders is not None:
            label_encoders['_feature_names'] = X.columns.tolist()
    else:
        # For test data, align columns with training data
        if label_encoders is not None and '_feature_names' in label_encoders:
            expected_cols = label_encoders['_feature_names']
            
            # Add missing columns with zeros
            for col in expected_cols:
                if col not in X.columns:
                    X[col] = 0
            
            # Remove extra columns and reorder to match training
            X = X[expected_cols]
        
        X_transformed = scaler.transform(X)
    
    return X_transformed, y, label_encoders, scaler


def explore_data(
    df: pd.DataFrame,
    X: np.ndarray,
    y: np.ndarray,
    output_dir: Optional[Path] = None
) -> None:
    """
    Perform basic data exploration and visualization.
    
    Args:
        df: Original DataFrame
        X: Feature matrix
        y: Target vector
        output_dir: Directory to save plots (optional)
    """
    print("\n" + "="*80)
    print("DATA EXPLORATION")
    print("="*80)
    
    print(f"\nDataset shape: {df.shape}")
    print(f"Feature matrix shape: {X.shape}")
    print(f"Target vector shape: {y.shape}")
    
    print(f"\nData types:")
    print(df.dtypes.value_counts())
    
    print(f"\nMissing values:")
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print(missing[missing > 0])
    else:
        print("No missing values found.")
    
    print(f"\nClass distribution:")
    class_counts = Counter(y)
    for class_name, count in class_counts.items():
        percentage = (count / len(y)) * 100
        print(f"  Class {class_name}: {count} ({percentage:.2f}%)")
    
    # Plot class distribution
    plt.figure(figsize=(10, 6))
    sns.countplot(x=y)
    plt.title('Class Distribution', fontsize=14, fontweight='bold')
    plt.xlabel('Class', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.tight_layout()
    
    if output_dir:
        plt.savefig(output_dir / "class_distribution.png", dpi=300, bbox_inches='tight')
        print(f"\nSaved class distribution plot to {output_dir / 'class_distribution.png'}")
    plt.close()
    
    # Correlation matrix for numeric features
    numeric_df = df.select_dtypes(include=[np.number])
    if len(numeric_df.columns) > 1:
        print(f"\nComputing correlation matrix for {len(numeric_df.columns)} numeric features...")
        
        # Sample if too many features for visualization
        if len(numeric_df.columns) > 50:
            print("Too many features for full correlation matrix. Sampling top 30 by variance...")
            variances = numeric_df.var().sort_values(ascending=False)
            top_features = variances.head(30).index.tolist()
            numeric_df = numeric_df[top_features]
        
        corr_matrix = numeric_df.corr()
        
        plt.figure(figsize=(16, 12))
        sns.heatmap(
            corr_matrix,
            annot=False,
            fmt=".2f",
            cmap="coolwarm",
            center=0,
            square=True,
            linewidths=0.5,
            cbar_kws={"shrink": 0.8}
        )
        plt.title('Feature Correlation Matrix', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if output_dir:
            plt.savefig(output_dir / "correlation_matrix.png", dpi=300, bbox_inches='tight')
            print(f"Saved correlation matrix to {output_dir / 'correlation_matrix.png'}")
        plt.close()
    
    print("\n" + "="*80)

