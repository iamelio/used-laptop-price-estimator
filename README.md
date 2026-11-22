# Used laptop price estimator

Prices a used laptop in naira and explains the price line by line, printed as a receipt. The working name is **Tally**. This is a group project for our ML class.

To see the design, open [`design/appraisal-slip.html`](design/appraisal-slip.html) in a browser. It's a working prototype, with the slip updating live as you change answers.

To walk through the data and the model step by step, open the notebook in Google Colab:

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/iamelio/used-laptop-price-estimator/blob/main/notebooks/laptop_price_model.ipynb)

## How it fits together

```
form (web)  ──POST /api/appraise──▶  FastAPI  ──▶  predictor  ──▶  Appraisal  ──▶  slip
                                              mock now, trained model later
```

The web app and the model only meet at the API contract, so both can be built at the same time. [`docs/CONTRACT.md`](docs/CONTRACT.md) explains every field, the rules responses follow, and the steps for building the model.

## What's where

| Path | What it is |
|---|---|
| `design/appraisal-slip.html` | Interactive UI prototype |
| `api/app/schemas.py` | The contract: request and response models |
| `api/app/rules.py` | Shared condition grading rule |
| `api/app/predictors/mock.py` | Hand-tuned stand-in for the model |
| `api/app/predictors/model.py` | The trained model's predictor: loads it, prices a laptop and explains the price |
| `api/tests/` | Contract tests that every predictor must pass |
| `contract/` | Generated OpenAPI spec and example request, options and response |
| `model/train.py` | Downloads the dataset and trains the model into `model/artifacts/` |
| `model/artifacts/` | The trained model the API loads. It's committed, so the model works straight after cloning. |
| `model/tune.py` | Experiments behind the model: features, hyperparameter search, model families |
| `model/best_lgb.json` | The tuned settings `train.py` uses |
| `notebooks/laptop_price_model.ipynb` | Colab notebook and project report: the data, cleaning, features, a cross-validated comparison of six models, the chosen model, how one price becomes the receipt, limitations, and likely defence questions |

## Who does what

- **Design, frontend and architecture:** the design lead.
- **Dataset, training and `ModelPredictor`:** Kosi.

## Run the API

```bash
cd api
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/uvicorn app.main:app --reload
```

Interactive docs are then at http://localhost:8000/docs.

```bash
.venv/bin/python -m pytest
```

`PREDICTOR=model` switches from the mock to the trained model in `model/artifacts/`:

```bash
cd api
PREDICTOR=model .venv/bin/python -m pytest
PREDICTOR=model .venv/bin/uvicorn app.main:app --reload
```

Retrain after changing the data or the model, from the repo root. It takes under a minute and downloads the dataset on its first run.

```bash
api/.venv/bin/pip install -r model/requirements.txt
api/.venv/bin/python model/train.py
```

## The model

A LightGBM model trained on 2,061 of 2,160 laptop listings from a 2023 European retailer (the other 99 have no year that can be worked out), converted to naira at €1 = ₦1,700. The data has no year or laptop-type column, so `train.py` works them out from the listing title and CPU. On laptops it never saw in training:

| Measure | Result |
|---|---|
| Median error | 13.6% |
| Mean absolute error | ₦422,000 |
| R² | 0.82 |
| Real price inside the "likely between" range | 80% |

Each slip line comes from the model's SHAP values. Condition and battery are fixed adjustments, marked `source="adjustment"`, since the data has no columns for them. Ages past the 7 years the data covers carry on the depreciation the model learned, shown on the same Age line, for example "Age, 13 years (6 estimated)".

`model/tune.py` holds the experiments behind these choices: `baseline`, `lgb`, `xgb` and `ensemble` (for example `api/.venv/bin/python model/tune.py lgb 100`). What they found:

- **Helped:** reading the product line from `model_name`, and a CPU performance tier.
- **Hurt:** down-weighting duplicate listings, and adding a 2018 Indian retail dataset.
- **Ruled out:** CatBoost breaks under monotonic constraints, and blending in XGBoost gains almost nothing (about 0.0004 RMSLE), so the API uses LightGBM alone.

**Known limits:**

- **Price level:** prices come from a European shop, not the Nigerian used market, so estimates run high.
- **Older laptops:** few listings are more than 5 years old.
- **Newest chips:** Apple M3 and M4 aren't in the data, so they're priced like an M2 of the same age and marked "(estimated)".

## Status

- **Done:** API contract, mock pricer, contract tests, UI prototype, dataset and trained model.
- **Next:** React frontend built from the prototype, hosting, and calibrating prices to the Nigerian used market.
