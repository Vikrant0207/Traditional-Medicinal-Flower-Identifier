"""
Generate synthetic laboratory extraction dataset for Project 2:
Floral Extract Yield & Efficacy Predictor.

Saves output to: data/tabular/floral_extract_data.csv
"""

from pathlib import Path
import numpy as np
import pandas as pd


def generate_floral_extract_data(
    num_samples: int = 1000,
    seed: int = 42,
    output_path: Path | str | None = None,
) -> pd.DataFrame:
    """Generate synthetic floral extraction dataset and optionally save to CSV."""
    rng = np.random.default_rng(seed)

    # Lab extraction features
    extraction_temp_c = rng.uniform(40.0, 90.0, num_samples)          # °C
    solvent_concentration_pct = rng.uniform(50.0, 95.0, num_samples)  # % (e.g., ethanol/water)
    steeping_time_mins = rng.uniform(30.0, 240.0, num_samples)        # minutes
    raw_flower_mass_g = rng.uniform(10.0, 100.0, num_samples)         # grams

    # Non-linear yield relationship: optimal temperature around 65°C with thermal degradation above it
    base_yield = (
        (extraction_temp_c * 0.40)
        + (solvent_concentration_pct * 0.20)
        + (steeping_time_mins * 0.05)
        - ((extraction_temp_c - 65.0) ** 2) * 0.01
    )

    # Slight diminishing/saturation effect at very high mass or long steeping times
    mass_factor = 1.0 - 0.0005 * np.maximum(raw_flower_mass_g - 50.0, 0.0)
    noise = rng.normal(0.0, 2.0, num_samples)

    yield_pct = np.clip((base_yield * mass_factor) + noise, 0.0, 100.0)

    df = pd.DataFrame(
        {
            "extraction_temp_c": np.round(extraction_temp_c, 2),
            "solvent_concentration_pct": np.round(solvent_concentration_pct, 2),
            "steeping_time_mins": np.round(steeping_time_mins, 1),
            "raw_flower_mass_g": np.round(raw_flower_mass_g, 2),
            "yield_pct": np.round(yield_pct, 2),
        }
    )

    if output_path is not None:
        out_file = Path(output_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_file, index=False)

    return df


def main() -> None:
    # Resolve repository root from <repo>/project2_regressor/generate_data.py
    project_root = Path(__file__).resolve().parent.parent
    default_output = project_root / "data" / "tabular" / "floral_extract_data.csv"

    df = generate_floral_extract_data(
        num_samples=1000,
        seed=42,
        output_path=default_output,
    )

    print(f"Generated {len(df)} rows and saved dataset to: {default_output}")
    print("\nDataset preview:")
    print(df.head())
    print("\nSummary statistics:")
    print(df.describe().round(2))


if __name__ == "__main__":
    main()