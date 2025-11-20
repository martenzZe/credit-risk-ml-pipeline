"""
Model Definition Module

This module defines the machine learning models for credit risk prediction.
"""

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV


def create_logistic_model(class_weight: str = 'balanced', **kwargs):
    """
    Create a logistic regression model.
    
    Args:
        class_weight: How to handle class imbalance ('balanced' recommended)
        **kwargs: Additional parameters for LogisticRegression
        
    Returns:
        LogisticRegression model
    """
    default_params = {
        'max_iter': 1000,
        'random_state': 42,
        'class_weight': class_weight
    }
    default_params.update(kwargs)
    return LogisticRegression(**default_params)


def create_random_forest_model(class_weight: str = 'balanced', **kwargs):
    """
    Create a random forest model.
    
    Args:
        class_weight: How to handle class imbalance ('balanced' recommended)
        **kwargs: Additional parameters for RandomForestClassifier
        
    Returns:
        RandomForestClassifier model
    """
    default_params = {
        'n_estimators': 100,
        'max_depth': 10,
        'random_state': 42,
        'class_weight': class_weight,
        'n_jobs': -1
    }
    default_params.update(kwargs)
    return RandomForestClassifier(**default_params)


def create_gradient_boosting_model(**kwargs):
    """
    Create a gradient boosting model.
    
    Args:
        **kwargs: Additional parameters for GradientBoostingClassifier
        
    Returns:
        GradientBoostingClassifier model
    """
    default_params = {
        'n_estimators': 100,
        'max_depth': 5,
        'random_state': 42,
        'learning_rate': 0.1
    }
    default_params.update(kwargs)
    return GradientBoostingClassifier(**default_params)


def get_model(model_type: str = 'logistic', **kwargs):
    """
    Factory function to create a model by type.
    
    Args:
        model_type: Type of model ('logistic', 'rf', 'gbm')
        **kwargs: Model-specific parameters
        
    Returns:
        Model instance
    """
    models = {
        'logistic': create_logistic_model,
        'rf': create_random_forest_model,
        'random_forest': create_random_forest_model,
        'gbm': create_gradient_boosting_model,
        'gradient_boosting': create_gradient_boosting_model
    }
    
    if model_type.lower() not in models:
        raise ValueError(f"Unknown model type: {model_type}. Choose from {list(models.keys())}")
    
    return models[model_type.lower()](**kwargs)


def tune_model(model, param_grid: dict, X_train, y_train, cv: int = 5, scoring: str = 'roc_auc'):
    """
    Perform grid search for hyperparameter tuning.
    
    Args:
        model: Base model to tune
        param_grid: Dictionary of parameters to search
        X_train: Training features
        y_train: Training target
        cv: Number of cross-validation folds
        scoring: Scoring metric
        
    Returns:
        Best model from grid search
    """
    grid_search = GridSearchCV(
        model,
        param_grid,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        verbose=1
    )
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_

