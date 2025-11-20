"""
Data Loading Module

This module handles loading raw data and splitting it into train/test sets.
"""

import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split


def load_data(data_path: str = "data/UCI_Credit_Card.csv") -> pd.DataFrame:
    """
    Load the credit card dataset from CSV file.
    
    Args:
        data_path: Path to the CSV file
        
    Returns:
        DataFrame containing the raw data
    """
    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found at {data_path}")
    
    data = pd.read_csv(data_path)
    return data


def split_data(
    data: pd.DataFrame,
    target_col: str = "default.payment.next.month",
    test_size: float = 0.2,
    random_state: int = 42,
    stratify: bool = True
) -> tuple:
    """
    Split data into train and test sets.
    
    Args:
        data: Full dataset
        target_col: Name of the target column
        test_size: Proportion of data to use for testing
        random_state: Random seed for reproducibility
        stratify: Whether to use stratified splitting (recommended for imbalanced data)
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    X = data.drop(columns=[target_col, "ID"] if "ID" in data.columns else [target_col])
    y = data[target_col]
    
    stratify_param = y if stratify else None
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_param
    )
    
    return X_train, X_test, y_train, y_test

