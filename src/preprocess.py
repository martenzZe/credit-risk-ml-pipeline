"""
Preprocessing Module

This module handles all feature engineering and preprocessing steps,
including temporal features based on EDA findings.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib


def create_temporal_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Create temporal features from payment history columns.
    
    Based on EDA findings, these features are highly predictive:
    - Bill amount trends and volatility
    - Payment amount trends and volatility
    - Payment-to-bill ratios
    - Delinquency patterns and streaks
    - Limit balance
    
    Args:
        data: DataFrame with BILL_AMT, PAY_AMT, and PAY_ columns, and LIMIT_BAL
        
    Returns:
        DataFrame with temporal features added
    """
    data = data.copy()
    
    # Extract column groups
    limit_bal_col = ['LIMIT_BAL']
    bill_cols = [f'BILL_AMT{i}' for i in range(1, 7)]
    pay_amt_cols = [f'PAY_AMT{i}' for i in range(1, 7)]
    pay_status_cols = ['PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']
    
    bills = data[bill_cols]
    pays = data[pay_amt_cols]
    pay_status = data[pay_status_cols]
    limit_balance = data[limit_bal_col]
    
    # 1. Trend statistics (slopes and volatility)
    bill_deltas = bills.diff(axis=1).fillna(0)
    pay_deltas = pays.diff(axis=1).fillna(0)
    limit_bal_deltas = limit_balance.diff(axis=1).fillna(0)
    
    data['bill_slope'] = bill_deltas.mean(axis=1)
    data['bill_volatility'] = bills.std(axis=1)
    data['pay_slope'] = pay_deltas.mean(axis=1)
    data['pay_volatility'] = pays.std(axis=1)
    data['limit_bal_slope'] = limit_bal_deltas.mean(axis=1)
    data['limit_bal_volatility'] = limit_balance.std(axis=1)
    data['limit_bal_growth'] = limit_balance.diff(axis=1).fillna(0).mean(axis=1)
    
    # 2. Payment-to-bill ratios
    pay_to_bill = (pays / bills.replace(0, np.nan)).clip(0, 5).fillna(0)
    data['pay_to_bill_mean'] = pay_to_bill.mean(axis=1)
    data['pay_to_bill_min'] = pay_to_bill.min(axis=1)
    data['pay_to_bill_last'] = pay_to_bill.iloc[:, -1]

    # 2b. Ratio of bill and payment features to limit balance
    # These features show how much of their limit people use or repay
    for col in bill_cols:
        data[f'{col}_to_limit'] = data[col] / data['LIMIT_BAL']
    for col in pay_amt_cols:
        data[f'{col}_to_limit'] = data[col] / data['LIMIT_BAL']

    data['bill_mean_to_limit'] = data[[f'{col}_to_limit' for col in bill_cols]].mean(axis=1)
    data['pay_mean_to_limit'] = data[[f'{col}_to_limit' for col in pay_amt_cols]].mean(axis=1)
    
    # 3. Delinquency flags and counts
    late_flags = (pay_status > 0).astype(int)
    severe_flags = (pay_status >= 2).astype(int)
    
    data['late_months'] = late_flags.sum(axis=1)
    data['severe_late_months'] = severe_flags.sum(axis=1)
    data['recent_late'] = late_flags.iloc[:, -1]
    
    # 4. Delinquency streaks
    def longest_streak(row):
        """Calculate the longest consecutive streak of late payments."""
        streak = best = 0
        for val in row:
            if val:
                streak += 1
                best = max(best, streak)
            else:
                streak = 0
        return best
    
    data['late_streak'] = late_flags.apply(longest_streak, axis=1)
    data['severe_streak'] = severe_flags.apply(longest_streak, axis=1)
    
    # 5. Growth features (early vs late periods)
    first_half = slice(0, 3)
    second_half = slice(3, 6)
    
    data['bill_mean_early'] = bills.iloc[:, first_half].mean(axis=1)
    data['bill_mean_late'] = bills.iloc[:, second_half].mean(axis=1)
    data['bill_growth'] = data['bill_mean_late'] - data['bill_mean_early']
    
    data['pay_mean_early'] = pays.iloc[:, first_half].mean(axis=1)
    data['pay_mean_late'] = pays.iloc[:, second_half].mean(axis=1)
    data['pay_growth'] = data['pay_mean_late'] - data['pay_mean_early']
    
    return data


def clean_categorical_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Clean categorical features based on EDA findings.
    
    EDUCATION and MARRIAGE have unusual values (0, 5, 6) with low counts.
    We'll collapse them into an "other" category.
    
    Args:
        data: DataFrame with categorical columns
        
    Returns:
        DataFrame with cleaned categorical features
    """
    data = data.copy()
    
    # Collapse EDUCATION: 0, 5, 6 -> 4 (other)
    if 'EDUCATION' in data.columns:
        data['EDUCATION'] = data['EDUCATION'].replace([0, 5, 6], 4)
    
    # Collapse MARRIAGE: 0 -> 3 (other)
    if 'MARRIAGE' in data.columns:
        data['MARRIAGE'] = data['MARRIAGE'].replace(0, 3)
    
    return data


def select_features(
    data: pd.DataFrame,
    include_raw_features: bool = False
) -> pd.DataFrame:
    """
    Select features for modeling.

    Based on EDA:
    - Use temporal features (highly predictive)
    - Include LIMIT_BAL
    - Include AGE
    - Optionally include raw BILL/PAY features (but they're highly correlated)

    Args:
        data: DataFrame with all features
        include_raw_features: Whether to include raw monthly BILL/PAY features

    Returns:
        DataFrame with selected features only
    """
    temporal_features = [
        'bill_slope', 'bill_volatility', 'pay_slope', 'pay_volatility',
        'pay_to_bill_mean', 'pay_to_bill_min', 'pay_to_bill_last',
        'late_months', 'severe_late_months', 'recent_late',
        'late_streak', 'severe_streak',
        'bill_growth', 'pay_growth'
    ]

    # Ensure LIMIT_BAL is included as a core feature
    base_features = ['LIMIT_BAL', 'AGE', 'SEX', 'EDUCATION', 'MARRIAGE']
    
    selected_features = base_features + temporal_features

    # Optionally include raw payment features
    if include_raw_features:
        raw_pay_features = ['PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']
        selected_features.extend(raw_pay_features)

    # Only select features that exist in the data (including LIMIT_BAL if present)
    available_features = [f for f in selected_features if f in data.columns]

    # Check that LIMIT_BAL is present in available_features
    if 'LIMIT_BAL' not in available_features:
        raise KeyError("LIMIT_BAL must be present in the dataset for modeling.")

    return data[available_features]


class Preprocessor:
    """
    Preprocessor class that encapsulates all preprocessing steps
    and can be saved/loaded for inference.
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.is_fitted = False
        
    def fit_transform(self, X: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Apply all preprocessing steps.
        
        Args:
            X: Input features
            fit: Whether to fit the scaler (True for training, False for inference)
            
        Returns:
            Preprocessed features
        """
        X_processed = X.copy()
        
        # Clean categorical features
        X_processed = clean_categorical_features(X_processed)
        
        # Scale features
        if fit:
            X_scaled = self.scaler.fit_transform(X_processed)
            self.is_fitted = True
        else:
            if not self.is_fitted:
                raise ValueError("Preprocessor must be fitted before transform")
            X_scaled = self.scaler.transform(X_processed)
        
        X_scaled_df = pd.DataFrame(
            X_scaled,
            columns=X_processed.columns,
            index=X_processed.index
        )
        
        return X_scaled_df
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform new data using fitted preprocessor."""
        return self.fit_transform(X, fit=False)
    
    def save(self, filepath: str):
        """Save preprocessor to disk."""
        joblib.dump(self, filepath)
    
    @staticmethod
    def load(filepath: str):
        """Load preprocessor from disk."""
        return joblib.load(filepath)

