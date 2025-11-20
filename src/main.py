"""
Main Pipeline Orchestrator

This script orchestrates the entire ML pipeline:
1. Load and preprocess data
2. Train model
3. Evaluate model
4. Make predictions

Usage:
    python -m src.main --mode train
    python -m src.main --mode evaluate
    python -m src.main --mode predict
"""

import argparse
import sys
from pathlib import Path

from .data_loader import load_data, split_data
from .preprocess import Preprocessor, create_temporal_features, select_features
from .train import train_pipeline
from .evaluate import evaluate_model, plot_feature_importance
from .inference import load_predictor


def train_mode(args):
    """Train a new model."""
    print("=" * 60)
    print("TRAINING MODE")
    print("=" * 60)
    
    results = train_pipeline(
        data_path=args.data_path,
        model_type=args.model_type,
        test_size=args.test_size,
        random_state=args.random_state,
        output_dir=args.output_dir,
        model_params=args.model_params
    )
    
    print("\nTraining completed successfully!")
    print(f"Train AUC: {results['train_auc']:.4f}")
    print(f"Test AUC: {results['test_auc']:.4f}")


def evaluate_mode(args):
    """Evaluate a trained model."""
    print("=" * 60)
    print("EVALUATION MODE")
    print("=" * 60)
    
    # Load data
    data = load_data(args.data_path)
    data = create_temporal_features(data)
    X_train, X_test, y_train, y_test = split_data(
        data, test_size=args.test_size, random_state=args.random_state
    )
    
    # Load model and preprocessor
    import joblib
    model_path = Path(args.model_dir) / f"{args.model_type}_model.joblib"
    preprocessor_path = Path(args.model_dir) / "preprocessor.joblib"
    
    model = joblib.load(model_path)
    preprocessor = Preprocessor.load(preprocessor_path)
    
    # Preprocess test data
    X_test_selected = select_features(X_test, include_raw_features=False)
    X_test_processed = preprocessor.transform(X_test_selected)
    
    # Evaluate
    metrics = evaluate_model(model, X_test_processed, y_test, output_dir=args.output_dir)
    
    # Feature importance for tree-based models
    if hasattr(model, 'feature_importances_'):
        feature_names = X_test_processed.columns
        plot_feature_importance(
            model, feature_names, 
            save_path=Path(args.output_dir) / "feature_importance.png"
        )


def predict_mode(args):
    """Make predictions on new data."""
    import pandas as pd
    
    print("=" * 60)
    print("PREDICTION MODE")
    print("=" * 60)
    
    # Load predictor
    predictor = load_predictor(args.model_dir, args.model_type)
    
    # Load data to predict
    from .data_loader import load_data
    data = load_data(args.data_path)
    data = create_temporal_features(data)
    
    # Make predictions
    risk_scores = predictor.predict_risk_score(data)
    
    # Save predictions
    output_path = Path(args.output_dir)
    output_path.mkdir(exist_ok=True)
    
    predictions_df = pd.DataFrame({
        'ID': data['ID'] if 'ID' in data.columns else data.index,
        'risk_score': risk_scores,
        'prediction': (risk_scores > args.threshold).astype(int)
    })
    
    output_file = output_path / "predictions.csv"
    predictions_df.to_csv(output_file, index=False)
    print(f"\nPredictions saved to {output_file}")
    print(f"\nSample predictions:")
    print(predictions_df.head())


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Credit Risk ML Pipeline")
    parser.add_argument(
        '--mode', 
        choices=['train', 'evaluate', 'predict'], 
        required=True,
        help='Pipeline mode'
    )
    parser.add_argument(
        '--data_path',
        default='data/UCI_Credit_Card.csv',
        help='Path to data CSV file'
    )
    parser.add_argument(
        '--model_type',
        choices=['logistic', 'rf', 'gbm'],
        default='logistic',
        help='Type of model to train/use'
    )
    parser.add_argument(
        '--model_dir',
        default='models',
        help='Directory containing saved model artifacts'
    )
    parser.add_argument(
        '--output_dir',
        default='outputs',
        help='Directory for outputs (predictions, plots, etc.)'
    )
    parser.add_argument(
        '--test_size',
        type=float,
        default=0.2,
        help='Proportion of data for testing'
    )
    parser.add_argument(
        '--random_state',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.5,
        help='Threshold for binary predictions (predict mode)'
    )
    
    args = parser.parse_args()
    
    # Route to appropriate mode
    if args.mode == 'train':
        train_mode(args)
    elif args.mode == 'evaluate':
        evaluate_mode(args)
    elif args.mode == 'predict':
        predict_mode(args)
    else:
        print(f"Unknown mode: {args.mode}")
        sys.exit(1)


if __name__ == "__main__":
    main()

