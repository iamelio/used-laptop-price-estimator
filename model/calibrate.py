from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "api"))

from app.predictors.model import ModelPredictor  # noqa: E402
from app.schemas import LaptopSpecs  # noqa: E402

LISTINGS = ROOT / "nigeria_listings.csv"
ARTIFACTS = ROOT / "artifacts"
SOURCE = "jiji.ng used and refurbished laptop listings, November 2026"


def load_listings() -> pd.DataFrame:
    return pd.read_csv(LISTINGS)


def listing_specs(row, brands: list[str]) -> LaptopSpecs:
    # Listings rarely state condition or battery health, so each is priced as a typical used laptop.
    return LaptopSpecs(
        brand=row.brand if row.brand in brands else "Other",
        model_name=row.model_name,
        laptop_type=row.laptop_type,
        year=int(row.year),
        cpu=row.cpu,
        ram_gb=int(row.ram_gb),
        storage_gb=int(row.storage_gb),
        storage_type=row.storage_type,
        gpu=None if pd.isna(row.gpu) else row.gpu,
        screen_inches=None,
        issues=["scratches"],
        battery_runtime="unknown",
    )


def uncalibrated_prices(listings: pd.DataFrame, artifacts: Path = ARTIFACTS) -> np.ndarray:
    predictor = ModelPredictor(artifacts, calibrated=False)
    brands = predictor.options().brands
    return np.array([predictor.appraise(listing_specs(row, brands)).estimate for row in listings.itertuples()], dtype=float)


def calibrate(artifacts: Path = ARTIFACTS) -> dict:
    listings = load_listings()
    actual = listings["price_ngn"].to_numpy(dtype=float)
    predicted = uncalibrated_prices(listings, artifacts)
    ratio = actual / predicted

    # Leave-one-out: each listing is scored with a factor fitted on the other listings only.
    loo_factor = np.array([np.median(np.delete(ratio, i)) for i in range(len(ratio))])
    calibrated = predicted * loo_factor
    low, high = np.quantile(actual / calibrated, [0.1, 0.9])
    market = {
        "source": SOURCE,
        "listings": int(len(listings)),
        "factor": float(np.median(ratio)),
        "range_ratio": [float(low), float(high)],
        "mae": int(round(np.mean(np.abs(actual - calibrated)))),
        "median_error": float(np.median(np.abs(actual - calibrated) / actual)),
        "r2": round(float(1 - np.sum((actual - calibrated) ** 2) / np.sum((actual - actual.mean()) ** 2)), 3),
        "uncalibrated_median_error": float(np.median(np.abs(actual - predicted) / actual)),
    }
    meta_path = artifacts / "meta.json"
    meta = json.loads(meta_path.read_text())
    meta["market"] = market
    meta_path.write_text(json.dumps(meta, indent=2))
    return market


if __name__ == "__main__":
    result = calibrate()
    print(json.dumps(result, indent=2))
