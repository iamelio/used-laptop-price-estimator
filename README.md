# Used laptop price estimator

Prices a used laptop in naira and explains the price line by line, printed as a receipt. The working name is **Tally**. This is a group project for our ML class.

**Live site: https://tally.madebyelio.com**

To walk through the data and the model step by step, open the notebook in Google Colab:

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/iamelio/used-laptop-price-estimator/blob/main/notebooks/laptop_price_model.ipynb)

## How it fits together

```
site (web/)  ──POST /api/appraise──▶  FastAPI  ──▶  LightGBM model  ──▶  Appraisal  ──▶  receipt
                                                    trained on European prices,
                                                    calibrated to Nigerian listings
```

The site and the model only meet at the API contract. [`docs/CONTRACT.md`](docs/CONTRACT.md) explains every field and the rules responses follow. One container serves both the site and the API.

## What's where

| Path | What it is |
|---|---|
| `web/index.html` | The live site: answers on the left, a receipt that reprints as answers change on the right |
| `design/appraisal-slip.html` | The design prototype the site grew from, with a built-in sample pricer |
| `api/app/schemas.py` | The contract: request and response models |
| `api/app/rules.py` | Shared condition grading rule |
| `api/app/predictors/model.py` | The trained model's predictor: loads it, prices a laptop and explains the price |
| `api/app/predictors/mock.py` | Hand-tuned stand-in, used when `PREDICTOR` isn't set to `model` |
| `api/tests/` | Contract tests that every predictor must pass |
| `contract/` | Generated OpenAPI spec and example request, options and response |
| `model/train.py` | Downloads the dataset, trains the model into `model/artifacts/`, then calibrates it |
| `model/calibrate.py` | Scales prices to the Nigerian market using `model/nigeria_listings.csv` |
| `model/nigeria_listings.csv` | 80 used-laptop listings from Jiji, November 2026, with how each year was worked out |
| `model/artifacts/` | The trained model the API loads. It's committed, so the model works straight after cloning. |
| `model/tune.py` | Experiments behind the model: features, hyperparameter search, model families |
| `notebooks/laptop_price_model.ipynb` | Colab notebook and project report: data, cleaning, features, a cross-validated comparison of six models, calibration, how one price becomes the receipt, limitations, and likely defence questions |
| `Dockerfile` | Builds the site and API into one container |

## Who does what

- **Design, frontend and architecture:** the design lead.
- **Dataset, training and `ModelPredictor`:** Kosi.

## Run it locally

```bash
cd api
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
PREDICTOR=model .venv/bin/uvicorn app.main:app --reload
```

The site is then at http://localhost:8000 and the API docs at http://localhost:8000/docs. Without `PREDICTOR=model`, the API uses the mock pricer.

```bash
.venv/bin/python -m pytest
PREDICTOR=model .venv/bin/python -m pytest
```

Retrain and recalibrate after changing the data or the model, from the repo root. It takes about a minute and downloads the dataset on its first run.

```bash
api/.venv/bin/pip install -r model/requirements.txt
api/.venv/bin/python model/train.py
```

## Deploy

The repo builds with the `Dockerfile` on any Docker host. The live site runs on Coolify: a Dockerfile app from this repo, port 8000, health check `/api/health`.

## The model

A LightGBM model trained on 2,061 of 2,160 laptop listings from a 2023 European retailer (the other 99 have no year that can be worked out). The data has no year or laptop-type column, so `train.py` works them out from the listing title and CPU. On European laptops it never saw in training:

| Measure | Result |
|---|---|
| Median error | 13.6% |
| R² | 0.82 |
| Real price inside the fair range | 80% |

Each receipt line comes from the model's SHAP values. Condition and battery are fixed adjustments, marked `source="adjustment"`, since the data has no columns for them. Ages past the 7 years the data covers carry on the depreciation the model learned, shown on the same Age line, for example "Age, 13 years (6 estimated)".

### Calibrated to Nigerian prices

European shop prices run about 2.5 times Nigerian used prices. `model/calibrate.py` prices 80 current Jiji listings with the uncalibrated model, then scales every price by the median ratio of real to predicted price (0.40). The fair range is re-measured on the same listings. Scored leave-one-out, so each listing is scored with a factor fitted on the other 79:

| Measure on Nigerian listings | Before | After |
|---|---|---|
| Median error | 153% | 19.0% |
| Average error | ₦650,666 | ₦119,966 |

### Experiments

`model/tune.py` holds the experiments behind these choices: `baseline`, `lgb`, `xgb` and `ensemble` (for example `api/.venv/bin/python model/tune.py lgb 100`). What they found:

- **Helped:** reading the product line from `model_name`, and a CPU performance tier.
- **Hurt:** down-weighting duplicate listings, and adding a 2018 Indian retail dataset.
- **Ruled out:** CatBoost breaks under monotonic constraints, and blending in XGBoost gains almost nothing (about 0.0004 RMSLE), so the API uses LightGBM alone.

### Known limits

- **Small calibration sample:** 80 asking prices from one marketplace in one month, mostly business laptops listed in Lagos. Asking prices run above sale prices.
- **Older laptops:** few training listings are more than 5 years old.
- **Newest chips:** Apple M3 and M4 aren't in the data, so they're priced like an M2 of the same age and marked "(estimated)".

## Status

- **Done:** API contract, trained and calibrated model, contract tests, live site, Colab notebook.
- **Next:** a bigger Nigerian sample with real sale prices, and learning condition and battery from it instead of fixed adjustments.
