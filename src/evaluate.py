"""
Evaluation Module

This module handles model evaluation including metrics and visualizations.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, roc_curve, 
    classification_report, confusion_matrix,
    precision_recall_curve, average_precision_score
)
from pathlib import Path


def evaluate_model(model, X_test, y_test, output_dir: str = "outputs"):
    """
    Evaluate model and generate metrics and plots.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test targets
        output_dir: Directory to save evaluation outputs
        
    Returns:
        Dictionary of evaluation metrics
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Metrics
    auc = roc_auc_score(y_test, y_pred_proba)
    ap = average_precision_score(y_test, y_pred_proba)
    
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print(f"\nROC AUC: {auc:.4f}")
    print(f"Average Precision: {ap:.4f}")
    
    # Generate plots
    plot_roc_curve(y_test, y_pred_proba, output_path / "roc_curve.png")
    plot_precision_recall_curve(y_test, y_pred_proba, output_path / "pr_curve.png")
    plot_confusion_matrix(y_test, y_pred, output_path / "confusion_matrix.png")
    
    return {
        'roc_auc': auc,
        'average_precision': ap,
        'classification_report': classification_report(y_test, y_pred, output_dict=True)
    }


def plot_roc_curve(y_true, y_pred_proba, save_path: Path = None):
    """Plot ROC curve."""
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    auc = roc_auc_score(y_true, y_pred_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"ROC curve saved to {save_path}")
    plt.close()


def plot_precision_recall_curve(y_true, y_pred_proba, save_path: Path = None):
    """Plot Precision-Recall curve."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
    ap = average_precision_score(y_true, y_pred_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label=f'PR Curve (AP = {ap:.3f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Precision-Recall curve saved to {save_path}")
    plt.close()


def plot_confusion_matrix(y_true, y_pred, save_path: Path = None):
    """Plot confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    plt.close()


def plot_feature_importance(model, feature_names, top_n: int = 20, save_path: Path = None):
    """
    Plot feature importance for tree-based models.
    
    Args:
        model: Trained model with feature_importances_ attribute
        feature_names: List of feature names
        top_n: Number of top features to show
        save_path: Path to save the plot
    """
    if not hasattr(model, 'feature_importances_'):
        print("Model does not support feature importance")
        return
    
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]
    
    plt.figure(figsize=(10, 8))
    plt.barh(range(top_n), importances[indices])
    plt.yticks(range(top_n), [feature_names[i] for i in indices])
    plt.xlabel('Importance')
    plt.title(f'Top {top_n} Feature Importances')
    plt.gca().invert_yaxis()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Feature importance plot saved to {save_path}")
    plt.close()

