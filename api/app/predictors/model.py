from __future__ import annotations

import json
import math
import re
from datetime import datetime
from pathlib import Path

import lightgbm as lgb
import numpy as np

from app.predictors.base import Predictor, RawContribution, build_appraisal
from app.rules import CONDITION_RULE, grade_from_issues
from app.schemas import Appraisal, FormOptions, LaptopSpecs, ModelInfo

ARTIFACTS_DIR = Path(__file__).resolve().parents[3] / "model" / "artifacts"

# The model prices a refurbished laptop, which sellers grade as like new, so like new is the 1.0 point.
GRADES = {"like_new": (1.00, "Like new"), "good": (0.95, "Good"), "fair": (0.85, "Fair"), "worn": (0.70, "Worn")}
RUNTIMES = {
    "all_day": (1.00, "Battery: all day"),
    "half_day": (0.97, "Battery: half a day"),
    "one_to_three_hours": (0.90, "Battery: 1 to 3 hours"),
    "needs_charger": (0.80, "Battery: needs charger"),
    "unknown": (0.95, "Battery not checked"),
}


def _storage_label(gb: int) -> str:
    return f"{gb // 1024} TB" if gb >= 1024 and gb % 1024 == 0 else f"{gb} GB"


class ModelPredictor(Predictor):
    def __init__(self, artifacts_dir: Path = ARTIFACTS_DIR, calibrated: bool = True) -> None:
        meta_path = artifacts_dir / "meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"No trained model in {artifacts_dir}. Run: api/.venv/bin/python model/train.py")
        self.meta = json.loads(meta_path.read_text())
        self.boosters = [lgb.Booster(model_file=str(artifacts_dir / name)) for name in self.meta["models"]]
        self.features: list[str] = self.meta["features"]
        self.max_age: int = self.meta["max_age"]
        self.extra_age_multiplier: float = self.meta["extra_age_multiplier"]
        self.cpu_aliases: dict[str, str] = self.meta.get("cpu_aliases", {})
        self.series_patterns = [
            (name, re.compile(rf"(?<![a-z0-9]){re.escape(name)}(?![a-z])", re.I))
            for name in sorted(self.meta["series"][:-1], key=len, reverse=True)
        ]
        self.market: dict | None = self.meta.get("market") if calibrated else None
        scores = self.market or self.meta
        self.info = ModelInfo(
            kind="trained",
            name="LightGBM price model, calibrated to Nigerian listings" if self.market else "LightGBM price model",
            version=self.meta["version"],
            mae=scores["mae"],
            r2=scores["r2"],
            trained_on_rows=self.meta["trained_on_rows"],
        )
        self._options = FormOptions(
            brands=self.meta["brands"],
            laptop_types=self.meta["laptop_types"],
            cpus=self.meta.get("cpu_options", self.meta["cpus"]),
            ram_gb=self.meta["ram_gb"],
            storage_gb=self.meta["storage_gb"],
            screen_inches=self.meta["screen_inches"],
            year_min=self.meta["year_min"],
            year_max=datetime.now().year,
            condition_rule=CONDITION_RULE,
            model=self.info,
        )

    def options(self) -> FormOptions:
        return self._options

    def _series(self, model_name: str | None) -> str | None:
        if not model_name or not model_name.strip():
            return None
        return next((name for name, pattern in self.series_patterns if pattern.search(model_name)), "Other")

    def _row(self, specs: LaptopSpecs, age: int, series: str | None) -> np.ndarray:
        def index(value: str, choices: list[str], fallback: str | None = None) -> float:
            if value in choices:
                return float(choices.index(value))
            return float(choices.index(fallback)) if fallback in choices else math.nan

        values = {
            "brand": index(specs.brand, self.meta["brands"], "Other"),
            "laptop_type": index(specs.laptop_type, self.meta["laptop_types"]),
            "cpu": index(self.cpu_aliases.get(specs.cpu, specs.cpu), self.meta["cpus"], "Other"),
            "series": math.nan if series is None else index(series, self.meta["series"]),
            "cpu_tier": float(self.meta["cpu_tier"].get(specs.cpu, math.nan)),
            "ram_gb": float(specs.ram_gb),
            "storage_gb": float(specs.storage_gb),
            "storage_type": 1.0 if specs.storage_type == "hdd" else 0.0,
            "gpu": math.nan if specs.gpu is None else float(specs.gpu == "dedicated"),
            "screen_inches": math.nan if specs.screen_inches is None else float(specs.screen_inches),
            "age": float(age),
            "refurbished": 1.0,  # price it as a second-hand machine
        }
        return np.array([[values[name] for name in self.features]])

    def _label(self, field: str, specs: LaptopSpecs, age: int, series: str | None) -> str:
        if field == "year":
            years = f"Age, {age} year{'s' if age != 1 else ''}"
            return f"{years} ({age - self.max_age} estimated)" if age > self.max_age else years
        if field == "model_name":
            if series is None:
                return "Model not given"
            return f"{series} series" if series != "Other" else f"{specs.model_name.strip()} model"
        if field == "cpu":
            return f"{specs.cpu} (estimated)" if specs.cpu in self.cpu_aliases else specs.cpu
        labels = {
            "brand": f"{specs.brand} brand",
            "laptop_type": f"{specs.laptop_type} build",
            "ram_gb": f"{specs.ram_gb} GB RAM",
            "storage_gb": f"{_storage_label(specs.storage_gb)} storage",
            "storage_type": "Hard drive or eMMC, not SSD" if specs.storage_type == "hdd" else "SSD storage",
            "gpu": {"dedicated": "Dedicated graphics", "integrated": "Integrated graphics"}.get(
                specs.gpu or "", "Graphics not known"
            ),
            "screen_inches": f'{specs.screen_inches:g}" screen' if specs.screen_inches is not None
            else "Screen size not known",
        }
        return labels[field]

    def appraise(self, specs: LaptopSpecs) -> Appraisal:
        age = max(0, datetime.now().year - specs.year)
        series = self._series(specs.model_name)
        row = self._row(specs, min(age, self.max_age), series)
        # TreeSHAP, computed by LightGBM itself: one value per column, plus the expected value last.
        phi = np.mean([booster.predict(row, pred_contrib=True)[0] for booster in self.boosters], axis=0)

        field_of = {"age": "year", "series": "model_name", "cpu_tier": "cpu"}
        shap_by_field: dict[str, float] = {}
        for column, value in zip(self.features, phi[:-1]):
            field = field_of.get(column, column)
            shap_by_field[field] = shap_by_field.get(field, 0.0) + float(value)

        # The model works in log(naira), and "refurbished" is the same for every request, so it
        # becomes part of the starting point: the average price of a second-hand laptop.
        base_log = phi[-1] + shap_by_field.pop("refurbished", 0.0)
        base_value = math.exp(base_log)
        total_phi = sum(shap_by_field.values())
        model_estimate = math.exp(base_log + total_phi)

        amounts: dict[str, float] = {}
        for field, value in shap_by_field.items():
            if abs(total_phi) > 1e-9:
                amounts[field] = (model_estimate - base_value) * value / total_phi
            else:
                amounts[field] = base_value * value
        price = base_value + sum(amounts.values())

        # Past the oldest age in the data the trees go flat, so carry on the yearly depreciation they learned.
        if (extra := age - self.max_age) > 0:
            older = price * (self.extra_age_multiplier**extra - 1)
            amounts["year"] = amounts.get("year", 0.0) + older
            price += older

        # The training prices come from a European shop, so scale the whole receipt to the Nigerian used market.
        if self.market:
            factor = self.market["factor"]
            base_value *= factor
            amounts = {field: amount * factor for field, amount in amounts.items()}
            price *= factor

        contributions: list[RawContribution] = [
            (field, self._label(field, specs, age, series), amount, "model") for field, amount in amounts.items()
        ]
        grade_multiplier, grade_name = GRADES[grade_from_issues(specs.issues)]
        adjustments = [("issues", f"{grade_name} condition", grade_multiplier)]
        if specs.battery_health_pct is not None:
            adjustments.append((
                "battery_health_pct",
                f"Battery health {specs.battery_health_pct}%",
                1 - max(0, 85 - specs.battery_health_pct) * 0.006,
            ))
        else:
            multiplier, label = RUNTIMES[specs.battery_runtime]
            adjustments.append(("battery_runtime", label, multiplier))
        for feature, label, multiplier in adjustments:
            contributions.append((feature, label, price * (multiplier - 1), "adjustment"))
            price *= multiplier

        low_ratio, high_ratio = (self.market or self.meta)["range_ratio"]
        return build_appraisal(
            specs=specs,
            base_value=base_value,
            contributions=contributions,
            low=price * low_ratio,
            high=price * high_ratio,
            model=self.info,
        )
