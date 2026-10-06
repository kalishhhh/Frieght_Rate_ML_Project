# Freight Rate Prediction Challenge

Predicts `posted_rate` for 12,000 validation loads (Nov–Dec 2025) and a fixed-lane December 2025 chart
(Lexington → Fort Wayne, 360 mi, Dry Van, 32,000 lb).

**Approach in one paragraph:** a linear model on `log(rate)` (distance, equipment, weight, market index,
quote signal, interactions, time trend) plus a LightGBM model on its residuals. ~1.4% of training labels are
corrupted (rate ×/÷ 2–5), so they are detected and excluded from *training*. Validation is time-ordered
(rolling origin, never training on the future). Full write-up: `Freight_Rate_Report.docx`.

## Results (time-ordered validation)

| | MAE | MAPE | Median APE | MAPE on intact labels |
|---|---|---|---|---|
| Naive $/mile baseline | $220.7 | 10.5% | 6.9% | 8.3% |
| **Final model** (5-fold rolling origin) | **$99.5** | **4.1%** | **1.5%** | **1.9%** |

RMSE is ≈ $630 for every model because the corrupted labels dominate squared error; see the report.

## Repository layout

```
data/train_test.csv                      training data (Jan–Oct 2025)
data/validation.csv                      12,000 loads to predict (Nov–Dec 2025)
data/validation_predictions_template.csv template (load_id order)
data/december_chart_inputs.csv           filled in by train.py (predicted_rate)
features.py                              cleaning + feature engineering
train.py                                 validation, final fit, predictions, figures
score.py                                 provided scorer
validation_predictions.csv               submission file (output of train.py)
outputs/                                 metrics (CSV/JSON), figures, training log
Freight_Rate_Report.docx                 report
```

## Run it

```bash
python -m venv .venv && source .venv/bin/activate      # optional
python -m pip install -r requirements.txt
python train.py                                         # ~80 s; seeds are fixed
python score.py --predictions validation_predictions.csv \
                --december-predictions data/december_chart_inputs.csv
```

`train.py` runs the rolling-origin validation and the unseen-city stress test, trains the final model on all
48,000 rows, writes `validation_predictions.csv` and fills `predicted_rate` in
`data/december_chart_inputs.csv`, and saves figures/metrics to `outputs/`. `score.py` then validates both files and
creates `scorer_results/candidate_december.png`.

## Method details

- **Cleaning** (`features.py`): negative weights → absolute value; missing weight kept as NaN + flag;
  missing `market_index` filled with that day's median across all rows (features only, no labels).
- **Features:** distance (+log, polynomial, short-haul hinge), haversine distance and circuity,
  coordinates (instead of city names, so unseen cities work), equipment, weight, market index, quote signal,
  interactions, linear time trend. No calendar features: December is unseen and they only added noise.
- **Validation:** expanding window, origins 1 May … 1 Sep, each tested on the following two months; primary
  hold-out is train Jan–Aug → test Sep–Oct. All fitting steps happen inside each fold.
- **December chart inputs:** the chart file has no market/quote data, so `market_index` = that day's mean in
  `validation.csv`, `quote_signal` = validation median, coordinates from the training data.

## Assumptions and limitations

- December never appears in training, so real holiday seasonality cannot be learned.
- The trend term is frozen at the last training date (no extrapolation); drift could bias predictions by ~1–2%.
- Corrupted labels in the validation set are unpredictable and will inflate RMSE/MAPE for any model.
