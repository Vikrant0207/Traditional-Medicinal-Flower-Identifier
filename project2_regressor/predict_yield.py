"""
Inference script for Project 2: Floral Extract Yield & Efficacy Predictor.

Loads the trained regression model from `saved_models/yield_regressor.joblib`
and predicts `yield_pct` for new laboratory extraction parameters.
"""

from pathlib import Path
from typing import Any
import joblib
import numpy as np
import pandas as pd

FEATURE_COLUMNS: list[str] = [
    "extraction_temp_c",
    "solvent_concentration_pct",
    "steeping_time_mins",
    "raw_flower_mass_g",
]


def get_default_model_path() -> Path:
    """Resolve default path to `saved_models/yield_regressor.joblib`."""
    project_root = Path(__file__).resolve().parent.parent
    return project_root / "saved_models" / "yield_regressor.joblib"


def load_yield_model(model_path: Path | str | None = None) -> Any:
    """Load and return the trained scikit-learn / XGBoost regressor."""
    resolved_path = Path(model_path) if model_path is not None else get_default_model_path()
    if not resolved_path.exists():
        raise FileNotFoundError(
            f"Trained model not found at '{resolved_path}'. "
            "Run `project2_regressor/regression_train.py` first."
        )
    return joblib.load(resolved_path)


def _validate_and_prepare_features(data: pd.DataFrame) -> pd.DataFrame:
    """Ensure required feature columns exist, are numeric, and are ordered properly."""
    missing_cols = [col for col in FEATURE_COLUMNS if col not in data.columns]
    if missing_cols:
        raise ValueError(f"Missing required feature columns: {missing_cols}")

    prepared = data[FEATURE_COLUMNS].copy()
    for col in FEATURE_COLUMNS:
        prepared[col] = pd.to_numeric(prepared[col], errors="raise")

    if prepared.isna().any().any():
        raise ValueError("Input features contain NaN/missing values.")

    return prepared


def predict_yield_batch(
    experiments: pd.DataFrame | list[dict[str, float]],
    model: Any | None = None,
    model_path: Path | str | None = None,
) -> pd.DataFrame:
    """
    Predict `yield_pct` (clipped to [0.0, 100.0]) for a DataFrame or list of dicts.

    Returns a copy of the input data with a `predicted_yield_pct` column appended.
    """
    if isinstance(experiments, list):
        if not experiments:
            raise ValueError("`experiments` list cannot be empty.")
        input_df = pd.DataFrame(experiments)
    elif isinstance(experiments, pd.DataFrame):
        if experiments.empty:
            raise ValueError("`experiments` DataFrame cannot be empty.")
        input_df = experiments.copy()
    else:
        raise TypeError("`experiments` must be a pandas DataFrame or a list of dicts.")

    features_df = _validate_and_prepare_features(input_df)
    regressor = model if model is not None else load_yield_model(model_path)

    raw_predictions = regressor.predict(features_df)
    clipped_predictions = np.clip(np.asarray(raw_predictions, dtype=float), 0.0, 100.0)

    result_df = input_df.copy()
    result_df["predicted_yield_pct"] = np.round(clipped_predictions, 2)
    return result_df


def predict_yield(
    extraction_temp_c: float,
    solvent_concentration_pct: float,
    steeping_time_mins: float,
    raw_flower_mass_g: float,
    model: Any | None = None,
    model_path: Path | str | None = None,
) -> float:
    """Predict extract yield percentage (`yield_pct`) for a single lab run."""
    single_row = [
        {
            "extraction_temp_c": extraction_temp_c,
            "solvent_concentration_pct": solvent_concentration_pct,
            "steeping_time_mins": steeping_time_mins,
            "raw_flower_mass_g": raw_flower_mass_g,
        }
    ]
    result_df = predict_yield_batch(single_row, model=model, model_path=model_path)
    return float(result_df["predicted_yield_pct"].iloc[0])


def main() -> None:
    """Run a sample prediction from the command line."""
    sample_params = {
        "extraction_temp_c": 72.5,
        "solvent_concentration_pct": 85.0,
        "steeping_time_mins": 120.0,
        "raw_flower_mass_g": 50.0,
    }
    predicted = predict_yield(**sample_params)
    print("--- Floral Extract Yield Prediction ---")
    for key, val in sample_params.items():
        print(f"  {key}: {val}")
    print(f"Predicted Compound Yield: {predicted:.2f}%")


if __name__ == "__main__":
    main()