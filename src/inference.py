"""
Inference Module

This module handles making predictions on new data.
"""

import joblib
import pandas as pd
import numpy as np
from pathlib import Path

from .preprocess import Preprocessor, create_temporal_features, select_features


class CreditRiskPredictor:
    """
    Predictor class for making credit risk predictions.
    
    Loads saved model and preprocessor to make predictions on new data.
    """
    
    def __init__(self, model_path: str, preprocessor_path: str):
        """
        Initialize predictor with saved model and preprocessor.
        
        Args:
            model_path: Path to saved model file
            preprocessor_path: Path to saved preprocessor file
        """
        self.model = joblib.load(model_path)
        self.preprocessor = Preprocessor.load(preprocessor_path)
    
    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """
        Predict class labels (0 or 1) for new data.
        
        Args:
            data: DataFrame with features (should include temporal features)
            
        Returns:
            Array of predicted class labels
        """
        # Create temporal features if not present
        if 'bill_slope' not in data.columns:
            data = create_temporal_features(data)
        
        # Select features
        X_selected = select_features(data, include_raw_features=False)
        
        # Preprocess
        X_processed = self.preprocessor.transform(X_selected)
        
        # Predict
        return self.model.predict(X_processed)
    
    def predict_proba(self, data: pd.DataFrame) -> np.ndarray:
        """
        Predict class probabilities for new data.
        
        Args:
            data: DataFrame with features (should include temporal features)
            
        Returns:
            Array of predicted probabilities [prob_class_0, prob_class_1]
        """
        # Create temporal features if not present
        if 'bill_slope' not in data.columns:
            data = create_temporal_features(data)
        
        # Select features
        X_selected = select_features(data, include_raw_features=False)
        
        # Preprocess
        X_processed = self.preprocessor.transform(X_selected)
        
        # Predict probabilities
        return self.model.predict_proba(X_processed)
    
    def predict_risk_score(self, data: pd.DataFrame) -> pd.Series:
        """
        Predict risk scores (probability of default) for new data.
        
        Args:
            data: DataFrame with features
            
        Returns:
            Series of risk scores (probability of default)
        """
        proba = self.predict_proba(data)
        return pd.Series(proba[:, 1], index=data.index, name='risk_score')


def load_predictor(model_dir: str = "models", model_type: str = "logistic"):
    """
    Convenience function to load a predictor.
    
    Args:
        model_dir: Directory containing saved model artifacts
        model_type: Type of model to load
        
    Returns:
        CreditRiskPredictor instance
    """
    model_dir = Path(model_dir)
    model_path = model_dir / f"{model_type}_model.joblib"
    preprocessor_path = model_dir / "preprocessor.joblib"
    
    return CreditRiskPredictor(model_path, preprocessor_path)

