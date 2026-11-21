# Tally API contract

The web app and the model service only talk through the three endpoints below.
As long as both sides keep to this contract, the frontend and the model can be
built at the same time without waiting on each other.

- **Source of truth:** [`api/app/schemas.py`](../api/app/schemas.py). This page explains it.
- **Examples:** [`contract/`](../contract), generated from the code. The frontend can build against them before the model exists.
- **Ownership:** the web app and design sit with the design lead. Everything under `api/app/predictors/model.py` and `model/` is Kosi's.

## Run it

```bash
cd api
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/uvicorn app.main:app --reload      # http://localhost:8000/docs
.venv/bin/python -m pytest                   # contract tests
```

`PREDICTOR=mock` (the default) uses the hand-tuned mock. `PREDICTOR=model` uses the real model in `model.py`.

## Endpoints

| Method | Path | Returns |
|---|---|---|
| GET | `/api/health` | `{"status": "ok", "model": ModelInfo}`. The page calls it on load to wake a sleeping server. |
| GET | `/api/options` | `FormOptions`: everything the form needs to render. |
| POST | `/api/appraise` | Takes `LaptopSpecs`, returns `Appraisal`, which is everything the slip prints. |

Bad requests get a `422`. Unknown fields are rejected, so a typo fails loudly instead of being ignored.

## Request: `LaptopSpecs`

| Field | Type | Notes |
|---|---|---|
| `brand` | string | One of `options.brands`. |
| `model_name` | string or null | Free text such as "XPS 13". The trained model reads the product line from it (ROG, ThinkPad, MacBook Air and so on), so filling it in sharpens the price. Null is fine. |
| `laptop_type` | string | One of `options.laptop_types`. |
| `year` | int | Between `options.year_min` and `options.year_max`. |
| `cpu` | string | One of `options.cpus`. |
| `ram_gb` | int | |
| `storage_gb` | int | |
| `storage_type` | `"ssd"` or `"hdd"` | |
| `gpu` | `"integrated"`, `"dedicated"` or null | Null means the seller isn't sure. |
| `screen_inches` | number or null | Null means the seller isn't sure. |
| `issues` | list of issue ids | Every condition item the seller ticked, with no repeats. An empty list means "None of these". |
| `battery_runtime` | `"all_day"`, `"half_day"`, `"one_to_three_hours"`, `"needs_charger"` or `"unknown"` | How long a full charge lasts. |
| `battery_health_pct` | int 1 to 100, or null | From the exact-health helper. Takes priority over `battery_runtime` when present. |

Issue ids: `scratches`, `dent_or_crack`, `screen_damage`, `worn_keys`, `something_broken`, `loose_hinge`, `repaired`.

## Response: `Appraisal`

| Field | Slip part | Notes |
|---|---|---|
| `id`, `created_at` | Header | |
| `condition_grade` | Item block | `like_new`, `good`, `fair` or `worn`, from the shared rule below. |
| `base_value` | Starting point | The model's average prediction over the training data (SHAP's expected value). |
| `contributions[]` | Lines | `feature`, `label`, `amount` (signed naira), and `source` (`"model"` or `"adjustment"`). |
| `estimate` | Estimate | |
| `range` | Fair range | `low`, `high`, and `confidence` (0.8 means 8 in 10 real prices fall inside). |
| `model` | Accuracy line | `kind` (`mock` or `trained`), `name`, `version`, `mae` in naira, optional `r2` and `trained_on_rows`. |

### Rules every response follows

The schema enforces the first two, and the tests check the rest.

1. **The slip adds up exactly:** `base_value + sum(amounts) == estimate`, to the naira.
2. **The estimate sits inside the range.**
3. **`condition_grade` comes from the shared rule,** never a separate guess.
4. **Each line's `feature` is a `LaptopSpecs` field name,** such as `ram_gb` or `battery_runtime`. The frontend uses it to link a slip line to its form field.
5. **No zero-amount lines.**
6. **Unknown answers don't print as "None".**
7. **When `battery_health_pct` is given, there's no `battery_runtime` line.**
8. **Sanity checks:** a newer laptop, more RAM, or less damage never makes the price go down.

`build_appraisal()` in [`predictors/base.py`](../api/app/predictors/base.py) takes care of rules 1, 2, 3 and 5. It rounds everything to ₦1,000, puts the rounding leftover on the largest line, drops zero lines and works out the grade. Always build responses with it.

## Condition rule

[`api/app/rules.py`](../api/app/rules.py) holds the rule. `/api/options` serves it as `condition_rule`, so the form's live grade and the server's grade can't drift apart.

| Issue | Points |
|---|---|
| `scratches`, `worn_keys`, `repaired` | 1 |
| `dent_or_crack`, `screen_damage`, `loose_hinge` | 2 |
| `something_broken` | 3 |

| Total points | Grade |
|---|---|
| 0 | like new |
| 1 to 2 | good |
| 3 to 4 | fair |
| 5 or more | worn |

## The slip updates live

The page calls `/api/appraise` about 250 ms after every answer change, so each call needs to be fast. Aim for under 200 ms on the server.

- **Explain with `shap.TreeExplainer`.** It takes milliseconds per laptop. `KernelExplainer` takes seconds, which is far too slow here.
- **Load the model once at startup.** `ModelPredictor.__init__` runs once, so don't reload the model on each request.
- **Free hosting tiers often sleep when idle,** and the first request can then take most of a minute. The page pings `/api/health` on load to wake the server early.

## Building the real model (Kosi)

1. **Pick the dataset.** List its brands, types and CPUs, and return them from `ModelPredictor.options()` together with the shared `CONDITION_RULE`.
2. **Train on the request fields.**
   - **Missing values:** `gpu` and `screen_inches` can be null. LightGBM and scikit-learn's `HistGradientBoostingRegressor` handle missing values natively.
   - **Sanity rules:** monotonic constraints on `year`, `ram_gb` and `storage_gb` keep rule 8 true.
3. **Explain each prediction** with `shap.TreeExplainer`. Add up the SHAP values of every encoded column that came from the same field, so all `brand_*` columns become one `brand` line.
4. **Price in naira.** If the model predicts `log(price)`, its SHAP values are in log space and won't add up in naira. Convert them:
   - `base_value = exp(φ₀)` and `estimate = exp(φ₀ + Σφ)`
   - Give each field its share of the difference: `amount_i = (estimate − base_value) × φᵢ / Σφ`. Signs are preserved, because `exp` only ever goes up.
   - If `Σφ` is close to 0, give each field `base_value × φᵢ` instead.
5. **Fixed adjustments.** If the dataset has no battery or condition column, apply them as fixed adjustments after the model, and pass `source="adjustment"` so the slip can say so.
6. **Fair range.** The simplest option: take the 10th and 90th percentiles of `actual / predicted` on the validation set, and multiply the estimate by each. Quantile models (LightGBM `objective="quantile"`) also work.
7. **Accuracy.** Put the test-set MAE, R² and row count in `ModelInfo(kind="trained", ...)`.
8. **Check it.** Run `PREDICTOR=model .venv/bin/python -m pytest`. It's done when every test passes.

## Changing the contract

1. Edit `api/app/schemas.py`, and `rules.py` if needed.
2. Update `contract/example-request.json` if the request changed.
3. Run `.venv/bin/python -m app.export_contract` to regenerate `openapi.json` and the example options and response.
4. Run `.venv/bin/python -m pytest`. A stale `openapi.json` fails the tests on purpose.
5. Tell the other person, and update the frontend's types in the same pull request.
