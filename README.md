# Credit Risk ML Pipeline

An end-to-end machine learning pipeline for credit risk prediction using the UCI Credit Card default dataset.

## Project Structure

```
credit-risk-ml-pipeline/
├── src/                      # Main package
│   ├── __init__.py          # Package initialization
│   ├── data_loader.py       # Data loading and splitting
│   ├── preprocess.py        # Feature engineering and preprocessing
│   ├── model.py             # Model definitions
│   ├── train.py             # Training pipeline
│   ├── evaluate.py          # Model evaluation and visualization
│   ├── inference.py         # Prediction functions
│   └── main.py              # Pipeline orchestrator
├── notebooks/               # Jupyter notebooks for EDA
│   └── 01_data.ipynb       # Exploratory data analysis
├── data/                    # Data directory
│   └── UCI_Credit_Card.csv
├── models/                  # Saved model artifacts (created during training)
├── outputs/                 # Evaluation outputs and predictions
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## Installation

1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

The pipeline is orchestrated through `main.py` with three modes: `train`, `evaluate`, and `predict`.

### 1. Train a Model

Train a new model and save artifacts:

```bash
python -m src.main --mode train --model_type logistic
```

Available model types:
- `logistic`: Logistic Regression (fast, good baseline)
- `rf`: Random Forest (more robust)
- `gbm`: Gradient Boosting (often best performance)

Options:
- `--data_path`: Path to data CSV (default: `data/UCI_Credit_Card.csv`)
- `--model_type`: Type of model to train (default: `logistic`)
- `--output_dir`: Directory to save models (default: `models`)
- `--test_size`: Proportion for testing (default: `0.2`)
- `--random_state`: Random seed (default: `42`)

### 2. Evaluate a Model

Evaluate a trained model on test data:

```bash
python -m src.main --mode evaluate --model_type logistic
```

This generates:
- Classification report
- ROC curve plot (`outputs/roc_curve.png`)
- Precision-Recall curve plot (`outputs/pr_curve.png`)
- Confusion matrix plot (`outputs/confusion_matrix.png`)
- Feature importance plot (`outputs/feature_importance.png`) - for tree-based models

### 3. Make Predictions

Generate predictions on new data:

```bash
python -m src.main --mode predict --model_type logistic --data_path data/UCI_Credit_Card.csv
```

Predictions are saved to `outputs/predictions.csv` with:
- `ID`: Customer ID
- `risk_score`: Probability of default (0-1)
- `prediction`: Binary prediction (0/1) based on threshold

Options:
- `--threshold`: Probability threshold for binary predictions (default: `0.5`)

## Key Features

### Temporal Feature Engineering

Based on EDA findings, the pipeline creates highly predictive temporal features:

1. **Trend Statistics**:
   - Bill amount slope and volatility
   - Payment amount slope and volatility

2. **Payment Adequacy**:
   - Payment-to-bill ratios (mean, min, last)

3. **Delinquency Patterns**:
   - Total late months
   - Severe late months
   - Recent late flag
   - Longest late streak
   - Longest severe streak

4. **Growth Features**:
   - Early vs late period comparisons
   - Bill and payment growth

### Preprocessing

- Categorical feature cleaning (EDUCATION, MARRIAGE)
- Feature selection (temporal features + base features)
- Standard scaling for all features
- Handles class imbalance with stratified splitting and class weights

## Using as a Package

You can also import and use the modules directly:

```python
from src.data_loader import load_data, split_data
from src.preprocess import create_temporal_features, Preprocessor
from src.model import get_model
from src.inference import CreditRiskPredictor

# Load data
data = load_data("data/UCI_Credit_Card.csv")
data = create_temporal_features(data)

# Split
X_train, X_test, y_train, y_test = split_data(data)

# Preprocess
preprocessor = Preprocessor()
X_train_processed = preprocessor.fit_transform(X_train)

# Train
model = get_model('logistic')
model.fit(X_train_processed, y_train)

# Predict
predictor = CreditRiskPredictor("models/logistic_model.joblib", "models/preprocessor.joblib")
risk_scores = predictor.predict_risk_score(new_data)
```

## Model Performance

Baseline logistic regression with temporal features achieves:
- **ROC AUC: ~0.73** (as validated in EDA)

The temporal features show strong predictive power, especially:
- Late payment counts and streaks (highest importance)
- Recent delinquency status
- Payment-to-bill ratios

## Next Steps

- Experiment with hyperparameter tuning using `model.tune_model()`
- Try ensemble methods
- Deploy model using the API module (`api/`)
- Add monitoring and model versioning
