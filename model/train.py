"""Train the Tally price model.

Run from the repo root:

    api/.venv/bin/python model/train.py

Writes model/artifacts/model_0.txt ... model_4.txt (LightGBM, one per seed) and
model/artifacts/meta.json, which ModelPredictor in api/app/predictors/model.py loads at
startup. Hyperparameters come from model/best_lgb.json, which model/tune.py writes.

Dataset: 2,160 laptop listings (new and refurbished) scraped from a Spanish retailer
in 2023, published on Kaggle by Juan Merino and mirrored for ML Zoomcamp. Prices are
in euros. The dataset has no release-year or laptop-type column, so both are derived
from the listing title: year from the CPU generation, type from the product line.
"""

from __future__ import annotations

import json
import re
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA_URL = "https://raw.githubusercontent.com/alexeygrigorev/datasets/master/laptops.csv"
DATA_PATH = ROOT / "data" / "laptops.csv"
ARTIFACTS = ROOT / "artifacts"
TUNED = ROOT / "best_lgb.json"  # tracked in git, so a fresh clone trains the tuned model

# The listings were collected in 2023, so a laptop's age in the data is 2023 - year.
DATA_YEAR = 2023
# Euro to naira. Update this when the rate moves; retraining takes seconds.
EUR_TO_NGN = 1_700
SEED = 42
# Oldest year the form offers, same as the mock. Years past the data's oldest are priced by
# carrying on the learned depreciation (extra_age_multiplier), so they still get a fair price.
YEAR_MIN = 2012

BRANDS = ["Acer", "Apple", "Asus", "Dell", "Gigabyte", "HP", "Lenovo", "LG", "Microsoft",
          "MSI", "Razer", "Samsung", "Toshiba", "Other"]
LAPTOP_TYPES = ["Notebook", "Ultrabook", "Gaming", "2-in-1", "Workstation"]
CPUS = ["Intel Celeron / Pentium", "AMD Athlon / budget", "Intel Core i3", "Intel Core i5",
        "Intel Core i7", "Intel Core i9", "AMD Ryzen 3", "AMD Ryzen 5", "AMD Ryzen 7", "AMD Ryzen 9",
        "Apple M1", "Apple M1 Pro", "Apple M2", "Apple M2 Pro", "Other"]
# Product lines with at least 8 listings, read from the seller's free-text model_name.
# Generic words ("Go", "Thin", "Edge") are left out because they match too much free text.
SERIES = ["15S", "IdeaPad", "ROG", "ThinkPad", "VivoBook", "MacBook Pro", "TUF", "V15", "Chromebook",
          "MacBook Air", "ExpertBook", "ZenBook", "ThinkBook", "Latitude", "Creator", "Modern",
          "Surface Laptop", "Aspire", "Prestige", "EliteBook", "Blade", "250", "Stealth", "Raider", "Victus",
          "Omen", "Vostro", "Surface Pro", "ProBook", "Gram", "Katana", "Legion", "Pavilion", "Extensa",
          "Revolt", "Aorus", "Pulse", "Yoga", "ZBook", "Akoya", "255", "F515", "Nitro", "Aero", "Galaxy Book",
          "Flex Advance", "Summit", "Predator", "Leopard", "ConceptD", "TravelMate", "Vector",
          "Satellite Pro", "Ioxbook", "Nubian", "Other"]
# Rough performance tier per CPU family, so the trees can share what they learn across vendors.
CPU_TIER = {"Intel Celeron / Pentium": 0, "AMD Athlon / budget": 0, "Intel Core i3": 1, "AMD Ryzen 3": 1,
            "Intel Core i5": 2, "AMD Ryzen 5": 2, "Apple M1": 2.5, "Intel Core i7": 3, "AMD Ryzen 7": 3,
            "Apple M2": 3, "Apple M1 Pro": 3.5, "Intel Core i9": 4, "AMD Ryzen 9": 4, "Apple M2 Pro": 4}
CATEGORY_VALUES = {"brand": BRANDS, "laptop_type": LAPTOP_TYPES, "cpu": CPUS, "series": SERIES}

# Model columns, in order. Each is a LaptopSpecs field except `age` (shown as the `year` line),
# `series` (read from `model_name`), `cpu_tier` (read from `cpu`, shown on the cpu line) and `refurbished` (always 1 when pricing, folded into the
# starting point).
FEATURES = ["brand", "laptop_type", "cpu", "series", "cpu_tier", "ram_gb", "storage_gb", "storage_type", "gpu",
            "screen_inches", "age", "refurbished"]
CATEGORICAL = ["brand", "laptop_type", "cpu", "series"]
MONOTONE = {"ram_gb": 1, "storage_gb": 1, "age": -1, "cpu_tier": 1}
# Columns a seller can leave blank. Training adds copies of rows with each one blanked.
OPTIONAL = ["gpu", "screen_inches", "series"]


# ---------------------------------------------------------------- feature derivation

def brand_of(raw: str) -> str:
    raw = "Toshiba" if "Toshiba" in raw else raw
    return raw if raw in BRANDS else "Other"


def cpu_of(raw: str) -> str:
    raw = raw.replace("Evo ", "")
    if raw in CPUS:
        return raw
    if raw in ("Intel Celeron", "Intel Pentium"):
        return "Intel Celeron / Pentium"
    if raw in ("AMD Athlon", "AMD 3020e", "AMD 3015e", "AMD 3015Ce"):
        return "AMD Athlon / budget"
    return "Other"


_SERIES_PATTERNS = [(s, re.compile(rf"(?<![a-z0-9]){re.escape(s)}(?![a-z])", re.I))
                    for s in sorted(SERIES[:-1], key=len, reverse=True)]


def series_of(text: str | None) -> str | None:
    """Product line named in free text, "Other" when none matches, None when there's no text."""
    if text is None or not str(text).strip():
        return None
    for name, pattern in _SERIES_PATTERNS:
        if pattern.search(str(text)):
            return name
    return "Other"


INTEL_GEN_YEAR = {6: 2016, 7: 2017, 8: 2018, 9: 2019, 10: 2020, 11: 2021, 12: 2022, 13: 2023, 14: 2024}
RYZEN_SERIES_YEAR = {2: 2018, 3: 2019, 4: 2020, 5: 2021, 6: 2022, 7: 2023}
BUDGET_CHIP_YEAR = {"N3350": 2016, "N4200": 2016, "N4000": 2018, "N4020": 2020, "N4120": 2020,
                    "N5030": 2020, "N4500": 2021, "N5100": 2021, "N6000": 2021, "N100": 2023,
                    "N200": 2023, "3050U": 2020, "3020e": 2020, "3015e": 2020, "3015Ce": 2020}
APPLE_YEAR = {"Apple M1": 2020, "Apple M1 Pro": 2021, "Apple M2": 2022, "Apple M2 Pro": 2023}


def year_of(title: str, cpu_raw: str) -> float:
    """Release year, from the title's own year if it has one, else from the CPU generation."""
    if m := re.search(r"\b(201[5-9]|202[0-4])\b", title):
        return int(m.group(1))
    if cpu_raw in APPLE_YEAR:
        return APPLE_YEAR[cpu_raw]
    if m := re.search(r"i\d[- ](\d{4,5})", title, re.I):
        digits = m.group(1)
        gen = int(digits[:2]) if len(digits) == 5 or digits[0] == "1" else int(digits[0])
        return INTEL_GEN_YEAR.get(gen, np.nan)
    if m := re.search(r"Ryzen \d (?:PRO )?(\d)\d{3}", title, re.I):
        return RYZEN_SERIES_YEAR.get(int(m.group(1)), np.nan)
    for chip, year in BUDGET_CHIP_YEAR.items():
        if re.search(rf"\b{chip}\b", title):
            return year
    return np.nan


GAMING = re.compile(r"\b(ROG|TUF|Legion|Omen|Victus|Nitro|Predator|Katana|Raider|Stealth|Blade|Aorus|"
                    r"Pulse|Titan|Vector|Leopard|Revolt|Sword|Cyborg|Bravo|Alpha|Crosshair|Delta|Strix|"
                    r"Gaming|GF63|Thin GF|G5|G15|G16|Alienware|Deep Gaming)\b", re.I)
WORKSTATION = re.compile(r"\b(ZBook|Precision|ThinkPad P\d+\w*|ConceptD|Creator|ProArt|StudioBook|Aero)\b"
                         r"|RTX A\d|\bT ?\d{3,4}\b|Quadro", re.I)
TWO_IN_ONE = re.compile(r"\b(Surface Pro|Surface Go|x360|Flip|Yoga(?! Slim)|Flex|360|Detachable|"
                        r"Convertible|2 en 1|2-in-1|Duet|Spin|Book\d? Fold)\b", re.I)
ULTRABOOK = re.compile(r"\b(MacBook|ZenBook|XPS|Swift|Gram|Spectre|Envy|Surface Laptop|Prestige|Summit|"
                       r"Galaxy Book\d? Pro|X1 Carbon|Yoga Slim|Evo|Modern 14|Inspiron 14 Plus|"
                       r"ExpertBook B9|Elite Dragonfly|Swift Go)\b", re.I)


def laptop_type_of(title: str, gpu: object, cpu_raw: str) -> str:
    if WORKSTATION.search(title) or (isinstance(gpu, str) and re.match(r"(RTX A|T |Radeon Pro)", gpu)
                                     and "MacBook" not in title):
        return "Workstation"
    if GAMING.search(title) or (isinstance(gpu, str) and re.match(r"(RTX [234]0[6-9]0|RTX 30[6-8]0)", gpu)):
        return "Gaming"
    if TWO_IN_ONE.search(title):
        return "2-in-1"
    if ULTRABOOK.search(title) or "Evo" in cpu_raw:
        return "Ultrabook"
    return "Notebook"


def load_dataset() -> pd.DataFrame:
    if not DATA_PATH.exists():
        DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(DATA_URL, DATA_PATH)
    raw = pd.read_csv(DATA_PATH)

    df = pd.DataFrame({
        "title": raw["Laptop"],
        "brand": raw["Brand"].map(brand_of),
        "laptop_type": [laptop_type_of(t, g, c) for t, g, c in zip(raw["Laptop"], raw["GPU"], raw["CPU"])],
        "cpu": raw["CPU"].map(cpu_of),
        "series": raw["Model"].map(series_of),
        "ram_gb": raw["RAM"].astype(float),
        "storage_gb": raw["Storage"].replace({1000: 1024, 2000: 2048, 3000: 3072, 4000: 4096, 500: 512,
                                              240: 256}).astype(float),
        # No hard drives in this data, so eMMC (also slow, cheap, non-SSD) stands in for "hdd".
        "storage_type": (raw["Storage type"] == "eMMC").astype(float),
        "gpu": raw["GPU"].notna().astype(float),
        "screen_inches": raw["Screen"],
        "year": [year_of(t, c) for t, c in zip(raw["Laptop"], raw["CPU"])],
        "refurbished": (raw["Status"] == "Refurbished").astype(float),
        "price_ngn": raw["Final Price"] * EUR_TO_NGN,
    })
    df = df[df["year"].notna() & (df["storage_gb"] >= 32)].reset_index(drop=True)
    df["age"] = (DATA_YEAR - df["year"]).clip(lower=0)
    df["cpu_tier"] = df["cpu"].map(CPU_TIER).astype(float)
    return df


def spec_groups(df: pd.DataFrame) -> np.ndarray:
    """Listings with identical model inputs (usually colour variants) share a group id."""
    return pd.util.hash_pandas_object(df[FEATURES], index=False).factorize()[0]


# ---------------------------------------------------------------- LightGBM

def encode(df: pd.DataFrame, features: list[str] = FEATURES) -> np.ndarray:
    cols = []
    for name in features:
        if name in CATEGORY_VALUES:
            lookup = {v: i for i, v in enumerate(CATEGORY_VALUES[name])}
            cols.append(df[name].map(lookup))
        else:
            cols.append(df[name])
    return np.column_stack([np.asarray(c, dtype=float) for c in cols])


def with_unknowns(df: pd.DataFrame, rng: np.random.Generator, features: list[str] = FEATURES,
                  frac: float = 0.2) -> pd.DataFrame:
    """Copies of some rows with an optional answer blanked, so the model learns what "not sure" is worth."""
    copies = [df]
    for col in OPTIONAL:
        if col in features:
            copies.append(df.sample(frac=frac, random_state=int(rng.integers(1e9))).assign(**{col: np.nan}))
    return pd.concat(copies, ignore_index=True)


PARAMS = {
    "objective": "regression",
    "learning_rate": 0.03,
    "num_leaves": 31,
    "min_data_in_leaf": 8,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.9,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "cat_smooth": 10,
    "monotone_constraints_method": "advanced",
    "verbose": -1,
    "seed": SEED,
    "num_threads": 1,
}


def fit(train: pd.DataFrame, rounds: int, valid: pd.DataFrame | None = None,
        features: list[str] = FEATURES, params: dict | None = None, weight: np.ndarray | None = None) -> lgb.Booster:
    params = {**(params or PARAMS), "monotone_constraints": [MONOTONE.get(f, 0) for f in features]}
    cat_idx = [i for i, f in enumerate(features) if f in CATEGORICAL]
    dtrain = lgb.Dataset(encode(train, features), np.log(train["price_ngn"]), feature_name=features,
                         categorical_feature=cat_idx, weight=weight, free_raw_data=False)
    sets, callbacks = [], []
    if valid is not None:
        sets = [lgb.Dataset(encode(valid, features), np.log(valid["price_ngn"]), reference=dtrain)]
        callbacks = [lgb.early_stopping(200, verbose=False)]
    return lgb.train(params, dtrain, num_boost_round=rounds, valid_sets=sets, callbacks=callbacks)


# ---------------------------------------------------------------- final training

def holdout_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """15% test set, grouped so colour variants of one laptop never straddle the split."""
    from sklearn.model_selection import GroupShuffleSplit
    gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=SEED)
    dev_idx, test_idx = next(gss.split(df, groups=spec_groups(df)))
    return df.iloc[dev_idx].reset_index(drop=True), df.iloc[test_idx].reset_index(drop=True)


def tuned() -> tuple[dict, int, int]:
    """Hyperparameters, boosting rounds and seed count from model/tune.py, or the defaults."""
    if not TUNED.exists():
        return PARAMS, 600, 1
    best = json.loads(TUNED.read_text())
    return {**PARAMS, **best["params"]}, int(best["rounds"]), int(best.get("seeds", 1))


def fit_seeds(train: pd.DataFrame, params: dict, rounds: int, seeds: int) -> list[lgb.Booster]:
    """Several boosters that differ only in their random seed. Their average is steadier than any one."""
    boosters = []
    for k in range(seeds):
        rng = np.random.default_rng(SEED + k)
        boosters.append(fit(with_unknowns(train, rng), rounds, params={**params, "seed": SEED + k}))
    return boosters


def predict(boosters: list[lgb.Booster], df: pd.DataFrame) -> np.ndarray:
    X = encode(df)
    return np.exp(np.mean([b.predict(X) for b in boosters], axis=0))


def main() -> None:
    from sklearn.metrics import mean_absolute_error, r2_score
    from sklearn.model_selection import GroupKFold

    df = load_dataset()
    dev, test = holdout_split(df)
    params, rounds, seeds = tuned()
    print(f"{len(df)} rows ({len(dev)} dev, {len(test)} test); {rounds} rounds x {seeds} seeds")

    # Out-of-fold predictions on dev set the fair range: real price / predicted price, 10th to 90th percentile.
    oof = np.zeros(len(dev))
    for tr, va in GroupKFold(5, shuffle=True, random_state=SEED).split(dev, groups=spec_groups(dev)):
        oof[va] = predict(fit_seeds(dev.iloc[tr], params, rounds, seeds), dev.iloc[va])
    low_q, high_q = np.quantile(dev["price_ngn"] / oof, [0.1, 0.9])

    # Held-out test score, from a model that never saw the test rows.
    pred = predict(fit_seeds(dev, params, rounds, seeds), test)
    y = test["price_ngn"].to_numpy()
    mae, r2 = mean_absolute_error(y, pred), r2_score(y, pred)
    mape = float(np.median(np.abs(y - pred) / y))
    coverage = float(np.mean((y >= pred * low_q) & (y <= pred * high_q)))
    print(f"test: MAE ₦{mae:,.0f}, R² {r2:.3f}, median error {mape:.1%}, "
          f"80% range {low_q:.3f}–{high_q:.3f} covers {coverage:.0%}")

    # Final model: every row, slightly more rounds since it has ~18% more data than each CV fold.
    final = fit_seeds(df, params, int(rounds * 1.15), seeds)
    # The data stops at age MAX_AGE, and past it the trees go flat. Measure the average yearly
    # depreciation the model learned across 0..max_age, so pricing can carry that trend on.
    max_age = int(df["age"].max())
    as_used = df.assign(refurbished=1.0)
    gap = np.log(predict(final, as_used.assign(age=float(max_age)))) - np.log(predict(final, as_used.assign(age=0.0)))
    extra_age_multiplier = float(np.exp(gap.mean() / max_age))
    print(f"learned depreciation over ages 0-{max_age}: {extra_age_multiplier - 1:+.1%} a year")

    for old in ARTIFACTS.glob("model*.txt"):
        old.unlink()
    for k, booster in enumerate(final):
        booster.save_model(str(ARTIFACTS / f"model_{k}.txt"))
    meta = {
        "version": datetime.now(timezone.utc).strftime("%Y.%m.%d"),
        "models": [f"model_{k}.txt" for k in range(len(final))],
        "features": FEATURES,
        "brands": BRANDS,
        "laptop_types": LAPTOP_TYPES,
        "cpus": CPUS,
        "cpu_tier": CPU_TIER,
        "series": SERIES,
        "ram_gb": sorted(int(v) for v in df["ram_gb"].value_counts().loc[lambda s: s >= 10].index),
        "storage_gb": sorted(int(v) for v in df["storage_gb"].value_counts().loc[lambda s: s >= 10].index),
        "screen_inches": [11.6, 13.3, 14.0, 15.6, 16.0, 17.3],
        "year_min": min(YEAR_MIN, int(df["year"].min())),
        "data_year": DATA_YEAR,
        "max_age": max_age,
        "extra_age_multiplier": extra_age_multiplier,
        "eur_to_ngn": EUR_TO_NGN,
        "range_ratio": [float(low_q), float(high_q)],
        "mae": int(round(mae)),
        "r2": round(float(r2), 3),
        "trained_on_rows": int(len(df)),
    }
    (ARTIFACTS / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"saved {len(final)} model file(s) and meta.json to {ARTIFACTS}")


if __name__ == "__main__":
    main()
