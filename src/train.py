"""
Training Module

This module handles model training and saving.
"""

import joblib
import pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score, classification_report

from .data_loader import load_data, split_data
from .preprocess import Preprocessor, create_temporal_features, select_features
from .model import get_model


def train_pipeline(
    data_path: str = "data/UCI_Credit_Card.csv",
    model_type: str = 'logistic',
    target_col: str = "default.payment.next.month",
    test_size: float = 0.2,
    random_state: int = 42,
    output_dir: str = "models",
    model_params: dict = None
) -> dict:
    """
    Complete training pipeline: load data, preprocess, train model, and save artifacts.
    
    Args:
        data_path: Path to data CSV file
        model_type: Type of model to train ('logistic', 'rf', 'gbm')
        target_col: Name of target column
        test_size: Proportion of data for testing
        random_state: Random seed
        output_dir: Directory to save model artifacts
        model_params: Additional parameters for the model
        
    Returns:
        Dictionary containing model, preprocessor, and metrics
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # 1. Load data
    print("Loading data...")
    data = load_data(data_path)
    
    # 2. Create temporal features
    print("Creating temporal features...")
    data = create_temporal_features(data)
    
    # 3. Split data
    print("Splitting data...")
    X_train, X_test, y_train, y_test = split_data(
        data, target_col=target_col, test_size=test_size, random_state=random_state
    )
    
    # 4. Select features
    print("Selecting features...")
    X_train_selected = select_features(X_train, include_raw_features=False)
    X_test_selected = select_features(X_test, include_raw_features=False)
    
    # 5. Preprocess
    print("Preprocessing features...")
    preprocessor = Preprocessor()
    X_train_processed = preprocessor.fit_transform(X_train_selected, fit=True)
    X_test_processed = preprocessor.transform(X_test_selected)
    
    # 6. Train model
    print(f"Training {model_type} model...")
    model_params = model_params or {}
    model = get_model(model_type, **model_params)
    model.fit(X_train_processed, y_train)
    
    # 7. Evaluate
    print("Evaluating model...")
    y_train_pred_proba = model.predict_proba(X_train_processed)[:, 1]
    y_test_pred_proba = model.predict_proba(X_test_processed)[:, 1]
    
    train_auc = roc_auc_score(y_train, y_train_pred_proba)
    test_auc = roc_auc_score(y_test, y_test_pred_proba)
    
    print(f"\nTrain AUC: {train_auc:.4f}")
    print(f"Test AUC: {test_auc:.4f}")
    
    # 8. Save artifacts
    print(f"Saving artifacts to {output_dir}...")
    model_path = output_path / f"{model_type}_model.joblib"
    preprocessor_path = output_path / "preprocessor.joblib"
    
    joblib.dump(model, model_path)
    preprocessor.save(preprocessor_path)
    
    print(f"Model saved to {model_path}")
    print(f"Preprocessor saved to {preprocessor_path}")
    
    return {
        'model': model,
        'preprocessor': preprocessor,
        'train_auc': train_auc,
        'test_auc': test_auc,
        'model_path': model_path,
        'preprocessor_path': preprocessor_path
    }

