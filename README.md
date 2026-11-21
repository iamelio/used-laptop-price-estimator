# Used laptop price estimator

Prices a used laptop in naira and explains the price line by line, printed as a receipt. The working name is **Tally**. This is a group project for our ML class.

To see the design, open [`design/appraisal-slip.html`](design/appraisal-slip.html) in a browser. It's a working prototype, with the slip updating live as you change answers.

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
| `model/train.py` | Downloads the dataset and trains the model. Saved model files go in `model/artifacts/`, which git ignores. |
| `model/tune.py` | Experiments behind the model: features, hyperparameter search, model families |
| `model/best_lgb.json` | The tuned settings `train.py` uses |

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

`PREDICTOR=model` switches from the mock to the trained model. The model files aren't in git, so train it once first, from the repo root. It takes about 10 seconds and downloads the dataset on its first run.

```bash
api/.venv/bin/pip install -r model/requirements.txt
api/.venv/bin/python model/train.py
```

```bash
cd api
PREDICTOR=model .venv/bin/python -m pytest
PREDICTOR=model .venv/bin/uvicorn app.main:app --reload
```

## The model

A LightGBM model trained on 2,160 laptop listings from a 2023 European retailer, converted to naira at €1 = ₦1,700. The data has no year or laptop-type column, so `train.py` works them out from the listing title and CPU. On laptops it never saw in training:

| Measure | Result |
|---|---|
| Median error | 13.6% |
| Mean absolute error | ₦422,000 |
| R² | 0.82 |
| Real price inside the "likely between" range | 80% |

Each slip line comes from the model's SHAP values. Condition, battery and any age past the 7 years the data covers are fixed adjustments, marked `source="adjustment"`, since the data has no columns for them.

**Known limits:** prices come from a European shop, not the Nigerian used market, so estimates run high. Few listings are more than 5 years old.

## Status

- **Done:** API contract, mock pricer, contract tests, UI prototype, dataset and trained model.
- **Next:** React frontend built from the prototype, and hosting. The host has to run `model/train.py` at build time.
