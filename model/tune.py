"""Experiments for the Tally price model: features, hyperparameters, model families.

    api/.venv/bin/python model/tune.py baseline
    api/.venv/bin/python model/tune.py lgb  [trials]
    api/.venv/bin/python model/tune.py xgb  [trials]
    api/.venv/bin/python model/tune.py ensemble

Findings (Nov 2026): the series feature (from model_name) and cpu_tier help; weighting
duplicate listings and adding the 2018 Indian-retail dataset hurt; CatBoost breaks under
monotone constraints; blending XGBoost into 5 seed-averaged LightGBMs gains only ~0.0004
RMSLE, so the API ships LightGBM alone.

Every score comes from grouped cross-validation: listings with identical specs (colour
variants of one laptop) always land in the same fold, so the model is never scored on a
near-copy of something it trained on. A final test split (15%, also grouped) is set aside
and never used for tuning.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train as T  # noqa: E402

RESULTS = Path(__file__).resolve().parent / "artifacts" / "tuning"  # Optuna history, git-ignored


def split():
    return T.holdout_split(T.load_dataset())


def cv_scores(dev: pd.DataFrame, fit_predict, features: list[str], folds: int = 5, repeats: int = 2):
    """fit_predict(train_df, valid_df, features) -> predicted naira for valid_df."""
    groups = T.spec_groups(dev)
    maes, r2s, rmsles = [], [], []
    for r in range(repeats):
        for tr, va in GroupKFold(folds, shuffle=True, random_state=T.SEED + r).split(dev, groups=groups):
            pred = fit_predict(dev.iloc[tr], dev.iloc[va], features)
            y = dev.iloc[va]["price_ngn"].to_numpy()
            maes.append(mean_absolute_error(y, pred))
            r2s.append(r2_score(y, pred))
            rmsles.append(float(np.sqrt(np.mean((np.log(y) - np.log(pred)) ** 2))))
    return float(np.mean(maes)), float(np.mean(r2s)), float(np.mean(rmsles))


def report(name, scores, extra=""):
    mae, r2, rmsle = scores
    print(f"{name:<40} MAE ₦{mae:>9,.0f}  R² {r2:.4f}  RMSLE {rmsle:.4f} {extra}", flush=True)


# ---------------------------------------------------------------- baseline / features

def baseline(dev):
    from concurrent.futures import ThreadPoolExecutor
    def make(frac=0.25, dedup=None):
        def fp(tr, va, feats):
            tr = T.with_unknowns(tr, np.random.default_rng(0), feats, frac)
            w = None
            if dedup is not None:
                sizes = pd.Series(T.spec_groups(tr)).map(pd.Series(T.spec_groups(tr)).value_counts()).to_numpy()
                w = sizes ** -dedup
            booster = T.fit(tr, 3000, va, feats, T.PARAMS, w)
            return np.exp(booster.predict(T.encode(va, feats), num_iteration=booster.best_iteration))
        return fp
    base = [f for f in T.FEATURES if f not in ("series", "cpu_tier")]
    tier = T.FEATURES
    runs = {
        "original": (make(), base),
        "+ series": (make(), [f for f in T.FEATURES if f != "cpu_tier"]),
        "+ series + cpu_tier": (make(), tier),
        "+ series, blank 10%": (make(0.10), T.FEATURES),
        "+ series, blank 40%": (make(0.40), T.FEATURES),
        "+ series, weight 1/sqrt(dupes)": (make(dedup=0.5), T.FEATURES),
        "+ series, weight 1/dupes": (make(dedup=1.0), T.FEATURES),
    }
    with ThreadPoolExecutor(8) as ex:
        futures = {k: ex.submit(cv_scores, dev, fp, feats) for k, (fp, feats) in runs.items()}
        for k, f in futures.items():
            report(k, f.result())


# ---------------------------------------------------------------- LightGBM

def lgb_params(trial):
    return {
        **T.PARAMS,
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 4, 96, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 2, 40, log=True),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
        "bagging_freq": 1,
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-4, 10, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-4, 30, log=True),
        "min_gain_to_split": trial.suggest_float("min_gain_to_split", 0, 0.05),
        "cat_smooth": trial.suggest_float("cat_smooth", 1, 50, log=True),
        "cat_l2": trial.suggest_float("cat_l2", 1, 50, log=True),
        "min_data_per_group": trial.suggest_int("min_data_per_group", 2, 50, log=True),
        "max_cat_threshold": trial.suggest_int("max_cat_threshold", 8, 64, log=True),
        "objective": trial.suggest_categorical("objective", ["regression", "huber", "fair"]),
    }


def lgb_fit_predict(params):
    rounds = []
    def fp(tr, va, feats):
        booster = T.fit(T.with_unknowns(tr, np.random.default_rng(0), feats), 5000, va, feats, params)
        rounds.append(booster.best_iteration)
        return np.exp(booster.predict(T.encode(va, feats), num_iteration=booster.best_iteration))
    return fp, rounds


# ---------------------------------------------------------------- XGBoost

def xgb_fit(tr, feats, params, rounds, va=None):
    import xgboost as xgb
    tr = T.with_unknowns(tr, np.random.default_rng(0), feats)
    def dm(df):
        X = df[feats].copy()
        for c in T.CATEGORICAL:
            if c in X:
                X[c] = pd.Categorical(X[c], categories=T.CATEGORY_VALUES[c])
        return xgb.DMatrix(X, np.log(df["price_ngn"]), enable_categorical=True)
    mono = "(" + ",".join(str(T.MONOTONE.get(f, 0)) for f in feats) + ")"
    p = {"objective": "reg:squarederror", "tree_method": "hist", "monotone_constraints": mono, "seed": T.SEED,
         "max_cat_to_onehot": 1, "nthread": 1, **params}
    evals = [(dm(va), "va")] if va is not None else []
    return xgb.train(p, dm(tr), rounds, evals=evals, early_stopping_rounds=200 if va is not None else None,
                     verbose_eval=False), dm


def xgb_params(trial):
    return {
        "eta": trial.suggest_float("eta", 0.01, 0.15, log=True),
        "max_depth": trial.suggest_int("max_depth", 2, 10),
        "min_child_weight": trial.suggest_float("min_child_weight", 0.5, 20, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "lambda": trial.suggest_float("lambda", 1e-3, 30, log=True),
        "alpha": trial.suggest_float("alpha", 1e-4, 5, log=True),
        "gamma": trial.suggest_float("gamma", 0, 0.05),
        "max_cat_threshold": trial.suggest_int("max_cat_threshold", 4, 64, log=True),
    }


def xgb_fit_predict(params):
    rounds = []
    def fp(tr, va, feats):
        booster, dm = xgb_fit(tr, feats, params, 5000, va)
        rounds.append(booster.best_iteration + 1)
        return np.exp(booster.predict(dm(va), iteration_range=(0, booster.best_iteration + 1)))
    return fp, rounds


# ---------------------------------------------------------------- search driver

def search(kind, dev, trials):
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    make_params, make_fp = {"lgb": (lgb_params, lgb_fit_predict),
                            "xgb": (xgb_params, xgb_fit_predict)}[kind]
    RESULTS.mkdir(parents=True, exist_ok=True)
    storage = f"sqlite:///{RESULTS / 'optuna.db'}"
    study = optuna.create_study(study_name=kind, direction="minimize", storage=storage, load_if_exists=True,
                                sampler=optuna.samplers.TPESampler(seed=T.SEED, multivariate=True))
    started = time.time()

    def objective(trial):
        params = make_params(trial)
        fp, rounds = make_fp(params)
        mae, r2, rmsle = cv_scores(dev, fp, T.FEATURES, folds=5, repeats=1)
        trial.set_user_attr("mae", mae)
        trial.set_user_attr("r2", r2)
        trial.set_user_attr("rounds", int(np.median(rounds)))
        return rmsle

    def log(study, trial):
        if trial.number % 10 == 0 or study.best_trial.number == trial.number:
            b = study.best_trial
            print(f"[{kind} {trial.number:>4} {time.time() - started:>5.0f}s] best RMSLE {b.value:.4f} "
                  f"MAE ₦{b.user_attrs['mae']:,.0f} R² {b.user_attrs['r2']:.4f}", flush=True)

    study.optimize(objective, n_trials=trials, callbacks=[log], n_jobs=8)
    best = study.best_trial
    # Re-score the winner with more repeats so the families are compared on equal footing.
    fp, rounds = make_fp(make_params(optuna.trial.FixedTrial(best.params)))
    scores = cv_scores(dev, fp, T.FEATURES, folds=5, repeats=3)
    report(f"{kind} best (5x3 grouped CV)", scores, f"rounds≈{int(np.median(rounds))}")
    out = {"params": best.params, "rounds": int(np.median(rounds)), "mae": scores[0], "r2": scores[1],
           "rmsle": scores[2]}
    if kind == "lgb":
        out["seeds"] = 5  # train.py averages this many boosters; see ensemble()
    (T.ROOT / f"best_{kind}.json").write_text(json.dumps(out, indent=2) + "\n")


def ensemble(dev):
    """Out-of-fold predictions from each tuned family, then the best blend of them in log space."""
    from concurrent.futures import ThreadPoolExecutor

    def best(kind):
        return json.loads((T.ROOT / f"best_{kind}.json").read_text())

    lgb_best, xgb_best = best("lgb"), best("xgb")
    lgb_p = {**T.PARAMS, **lgb_best["params"]}

    def lgb_seeds(n):
        def f(tr, va):
            return np.mean([T.fit(T.with_unknowns(tr, np.random.default_rng(k)), lgb_best["rounds"],
                                  params={**lgb_p, "seed": T.SEED + k}).predict(T.encode(va)) for k in range(n)], 0)
        return f

    def xgb(tr, va):
        booster, dm = xgb_fit(tr, T.FEATURES, xgb_best["params"], xgb_best["rounds"])
        return booster.predict(dm(va))

    families = {"lgb": lgb_seeds(1), "lgb x5 seeds": lgb_seeds(5), "xgb": xgb}
    splits = [(r, tr, va) for r in range(3) for tr, va in
              GroupKFold(5, shuffle=True, random_state=T.SEED + r).split(dev, groups=T.spec_groups(dev))]

    def oof(f):
        out = np.zeros((3, len(dev)))
        for r, tr, va in splits:
            out[r, va] = f(dev.iloc[tr], dev.iloc[va])
        return out

    with ThreadPoolExecutor(8) as ex:
        preds = dict(zip(families, ex.map(oof, families.values())))
    y = np.log(dev["price_ngn"].to_numpy())

    def score(p):
        rmsle = np.mean([np.sqrt(np.mean((y - p[r]) ** 2)) for r in range(3)])
        mae = np.mean([mean_absolute_error(np.exp(y), np.exp(p[r])) for r in range(3)])
        r2 = np.mean([r2_score(np.exp(y), np.exp(p[r])) for r in range(3)])
        return mae, r2, rmsle

    for name, p in preds.items():
        report(name, score(p))
    for a in np.arange(0, 1.01, 0.1):
        report(f"blend lgb x5 {a:.1f} / xgb {1 - a:.1f}", score(a * preds["lgb x5 seeds"] + (1 - a) * preds["xgb"]))


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "baseline"
    dev, test = split()
    print(f"dev {len(dev)} rows, test {len(test)} rows (held out)")
    if what == "baseline":
        baseline(dev)
    elif what == "ensemble":
        ensemble(dev)
    else:
        search(what, dev, int(sys.argv[2]) if len(sys.argv) > 2 else 100)
