"""
Model training and evaluation script for Project 2:
Floral Extract Yield & Efficacy Predictor.

Reads `data/tabular/floral_extract_data.csv`, trains a regression model
(`RandomForestRegressor` by default, or `xgboost` if requested/installed),
computes MAE, RMSE, and R^2, and saves `saved_models/yield_regressor.joblib`.
"""

import argparse
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

FEATURE_COLUMNS: list[str] = [
    "extraction_temp_c",
    "solvent_concentration_pct",
    "steeping_time_mins",
    "raw_flower_mass_g",
]
TARGET_COLUMN: str = "yield_pct"


def get_project_root() -> Path:
    """Resolve repository root from <repo>/project2_regressor/regression_train.py."""
    return Path(__file__).resolve().parent.parent


def load_dataset(data_path: Path | str) -> pd.DataFrame:
    """Load and validate the tabular extraction dataset."""
    csv_path = Path(data_path)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at '{csv_path}'. "
            "Run `project2_regressor/generate_data.py` first."
        )

    df = pd.read_csv(csv_path)
    required_cols = FEATURE_COLUMNS + [TARGET_COLUMN]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in dataset: {missing_cols}")

    # Drop any rows with missing target values; impute numeric feature NaNs with column median
    df = df.dropna(subset=[TARGET_COLUMN]).copy()
    for col in FEATURE_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())

    df[TARGET_COLUMN] = pd.to_numeric(df[TARGET_COLUMN], errors="coerce")
    df = df.dropna(subset=[TARGET_COLUMN])
    return df


def build_model(model_type: str = "rf", random_state: int = 42):
    """Instantiate the selected regression model."""
    model_key = model_type.lower().strip()
    if model_key in ("rf", "random_forest", "randomforest"):
        return RandomForestRegressor(
            n_estimators=200,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=random_state,
            n_jobs=-1,
        )
    if model_key in ("xgb", "xgboost"):
        try:
            from xgboost import XGBRegressor
        except ImportError as exc:
            raise ImportError(
                "XGBoost is not installed. Install `xgboost` or use `--model rf`."
            ) from exc

        return XGBRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=random_state,
            n_jobs=-1,
        )

    raise ValueError(f"Unsupported model type '{model_type}'. Choose 'rf' or 'xgb'.")


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute MAE, RMSE, and R^2 metrics."""
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))
    return {"mae": mae, "rmse": rmse, "r2": r2}


def train_and_evaluate(
    data_path: Path | str | None = None,
    model_output_path: Path | str | None = None,
    model_type: str = "rf",
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict[str, float]:
    """Train the regressor, log metrics, and serialize the model artifact."""
    project_root = get_project_root()
    resolved_data = (
        Path(data_path)
        if data_path is not None
        else project_root / "data" / "tabular" / "floral_extract_data.csv"
    )
    resolved_output = (
        Path(model_output_path)
        if model_output_path is not None
        else project_root / "saved_models" / "yield_regressor.joblib"
    )

    df = load_dataset(resolved_data)
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    model = build_model(model_type=model_type, random_state=random_state)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_pred_clipped = np.clip(np.asarray(y_pred, dtype=float), 0.0, 100.0)
    metrics = evaluate_predictions(y_test.to_numpy(dtype=float), y_pred_clipped)

    resolved_output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, resolved_output)

    print("--- Regression Model Evaluation ---")
    print(f"Model Type                 : {model_type}")
    print(f"Training Samples           : {len(X_train)} | Test Samples: {len(X_test)}")
    print(f"Mean Absolute Error (MAE)  : {metrics['mae']:.2f}%")
    print(f"Root Mean Squared Error    : {metrics['rmse']:.2f}%")
    print(f"R-squared (R^2) Score      : {metrics['r2']:.4f}")
    print(f"Saved model artifact to    : {resolved_output}")

    if hasattr(model, "feature_importances_"):
        importances = pd.Series(model.feature_importances_, index=FEATURE_COLUMNS).sort_values(
            ascending=False
        )
        print("\nFeature Importances:")
        for feat, score in importances.items():
            print(f"  {feat:<28}: {score:.4f}")

    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train floral extract yield regressor.")
    parser.add_argument(
        "--model",
        type=str,
        default="rf",
        choices=["rf", "xgb"],
        help="Regression algorithm to train ('rf' for RandomForestRegressor, 'xgb' for XGBRegressor).",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of dataset reserved for testing (default: 0.2).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_and_evaluate(
        model_type=args.model,
        test_size=args.test_size,
        random_state=args.seed,
    )


if __name__ == "__main__":
    main()