"""Freight rate prediction: validation, final training, predictions and report figures.

Usage:  python train.py
Outputs: validation_predictions.csv, data/december_chart_inputs.csv (filled in), outputs/*
"""
from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from features import TARGET, add_trend_features, build_features, clean

OUT = Path("outputs"); OUT.mkdir(exist_ok=True)
TRIM = 0.30      # |log residual| above this => corrupted label (clean gap: none between 0.2 and 0.5)
SEED = 0
GBM = dict(n_estimators=1000, learning_rate=0.02, num_leaves=15, min_child_samples=60, subsample=0.8,
           subsample_freq=1, colsample_bytree=0.8, reg_lambda=5.0, verbose=-1, random_state=SEED, n_jobs=4)


# ----------------------------------------------------------------------------- model
class FreightModel:
    """log(rate) = Ridge(linear stage, with trend + interactions)  +  LightGBM(residual correction).

    * Rows whose label is corrupted (|resid| > TRIM under a first-pass fit) are dropped from training.
    * The trend term is frozen at the last training date when predicting later dates (no extrapolation).
    """

    def __init__(self, use_gbm=True, use_trend=True):
        self.use_gbm, self.use_trend = use_gbm, use_trend

    def _lin(self):
        return make_pipeline(StandardScaler(), Ridge(alpha=1.0))

    def _design(self, d):
        base = build_features(d)
        lin = add_trend_features(base, d)
        if not self.use_trend:
            lin = lin.drop(columns="t")
        return base, lin

    def fit(self, d):
        y = np.log(d[TARGET].values)
        base, lin = self._design(d)
        self.med_ = lin.median()
        lin = lin.fillna(self.med_)
        first = self._lin().fit(lin, y)
        self.keep_ = np.abs(y - first.predict(lin)) < TRIM
        self.lin_ = self._lin().fit(lin[self.keep_], y[self.keep_])
        self.t_max_ = lin["t"].max() if self.use_trend else None
        if self.use_gbm:
            resid = y - self.lin_.predict(lin)
            self.gbm_ = lgb.LGBMRegressor(**GBM).fit(base[self.keep_], resid[self.keep_])
        return self

    def predict(self, d):
        base, lin = self._design(d)
        if self.use_trend:
            lin["t"] = np.minimum(lin["t"], self.t_max_)
        out = self.lin_.predict(lin.fillna(self.med_))
        if self.use_gbm:
            out = out + self.gbm_.predict(base)
        return np.exp(out)


class RatePerMileBaseline:
    """Naive benchmark: median $/mile per equipment type x distance."""
    def fit(self, d):
        self.rpm_ = (d[TARGET] / d.distance).groupby(d.equipment).median(); return self
    def predict(self, d):
        return (d.distance * d.equipment.map(self.rpm_)).values


class GbmOnly:
    """Gradient boosting alone (trimmed labels, no linear stage / trend)."""
    def fit(self, d):
        y = np.log(d[TARGET].values)
        helper = FreightModel(use_gbm=False).fit(d)
        self.g_ = lgb.LGBMRegressor(**GBM).fit(build_features(d)[helper.keep_], y[helper.keep_]); return self
    def predict(self, d):
        return np.exp(self.g_.predict(build_features(d)))


def metrics(y, p, clean_mask=None):
    e = p - y
    m = dict(MAE=np.abs(e).mean(), RMSE=np.sqrt((e**2).mean()), MAPE=100 * np.mean(np.abs(e) / y),
             MedAPE=100 * np.median(np.abs(e) / y), Bias=e.mean())
    if clean_mask is not None:
        m["MAPE_clean"] = 100 * np.mean((np.abs(e) / y)[clean_mask])
        m["MAE_clean"] = np.abs(e)[clean_mask].mean()
    return m


# ----------------------------------------------------------------------------- data
raw = pd.read_csv("data/train_test.csv", parse_dates=["date"])
val_raw = pd.read_csv("data/validation.csv", parse_dates=["date"])
df = clean(raw, reference_dates=val_raw)      # features only are shared; no target leakage
val = clean(val_raw, reference_dates=raw)
print(f"train/test rows: {len(df):,}   validation rows: {len(val):,}")

# ----------------------------------------------------------------------------- 1) rolling-origin validation
ORIGINS = ["2025-05-01", "2025-06-01", "2025-07-01", "2025-08-01", "2025-09-01"]
MODELS = {"Naive $/mile baseline": RatePerMileBaseline, "LightGBM only": GbmOnly,
          "Ridge (no trend)": lambda: FreightModel(use_gbm=False, use_trend=False),
          "Ridge + trend": lambda: FreightModel(use_gbm=False),
          "Ridge + trend + LightGBM (final)": FreightModel}
rows = []
holdout_pred = None
for o in ORIGINS:
    end = pd.Timestamp(o) + pd.DateOffset(months=2)
    tr, te = df[df.date < o], df[(df.date >= o) & (df.date < end)]
    y = te[TARGET].values
    ref = FreightModel(use_gbm=False).fit(tr).predict(te)
    cm = np.abs(np.log(y) - np.log(ref)) < TRIM          # rows whose label is not corrupted
    for name, make in MODELS.items():
        p = np.asarray(make().fit(tr).predict(te))
        rows.append(dict(origin=o, test=f"{o[:7]} to {(end - pd.Timedelta(days=1)).strftime('%Y-%m')}",
                         model=name, n_train=len(tr), n_test=len(te), **metrics(y, p, cm)))
        if o == ORIGINS[-1] and name.endswith("(final)"):
            holdout_pred = te.assign(pred=p, label_ok=cm)
cv = pd.DataFrame(rows); cv.to_csv(OUT / "cv_rolling_results.csv", index=False)
summary = cv.groupby("model")[["MAE", "RMSE", "MAPE", "MedAPE", "MAPE_clean", "MAE_clean"]].mean().round(2)
summary = summary.loc[list(MODELS)]
summary.to_csv(OUT / "cv_summary.csv")
last = cv[cv.origin == ORIGINS[-1]].set_index("model").loc[list(MODELS), ["MAE", "RMSE", "MAPE", "MedAPE", "MAPE_clean"]].round(2)
last.to_csv(OUT / "holdout_sep_oct.csv")
print("\nRolling-origin CV (mean of 5 folds, 2-month test windows):\n", summary)
print("\nPrimary hold-out (train Jan-Aug -> test Sep-Oct):\n", last)

# ----------------------------------------------------------------------------- 2) unseen-city stress test
rng = np.random.RandomState(1)
cities = np.array(sorted(set(df.pickup) | set(df.delivery)))
st = []
for rep in range(5):
    hold = set(rng.choice(cities, 8, replace=False))
    tr = df[(df.date < "2025-09-01") & ~df.pickup.isin(hold) & ~df.delivery.isin(hold)]
    te = df[(df.date >= "2025-09-01") & (df.pickup.isin(hold) | df.delivery.isin(hold))]
    y = te[TARGET].values
    ref = FreightModel(use_gbm=False).fit(tr).predict(te); cm = np.abs(np.log(y) - np.log(ref)) < TRIM
    for name, make in (("Ridge + trend", lambda: FreightModel(use_gbm=False)), ("Final model", FreightModel)):
        st.append(dict(rep=rep, model=name, n_test=len(te), **metrics(y, make().fit(tr).predict(te), cm)))
stress = pd.DataFrame(st).groupby("model")[["MAE", "MAPE", "MedAPE", "MAPE_clean"]].mean().round(2)
stress.to_csv(OUT / "unseen_city_stress_test.csv")
print("\nUnseen-city stress test (8 cities hidden from training, 5 random draws):\n", stress)

# ----------------------------------------------------------------------------- 3) final model + predictions
final = FreightModel().fit(df)
print(f"\nFinal fit on {len(df):,} rows; dropped {int((~final.keep_).sum())} corrupted-label rows "
      f"({100 * (~final.keep_).mean():.2f}%)")
val["predicted_rate"] = final.predict(val).round(2)
template = pd.read_csv("data/validation_predictions_template.csv")
sub = template[["load_id"]].merge(val[["load_id", "predicted_rate"]], on="load_id", how="left")
assert sub.predicted_rate.notna().all() and (sub.predicted_rate > 0).all() and len(sub) == 12_000
sub.to_csv("validation_predictions.csv", index=False)

# December chart: fixed lane; only the date changes. Features not supplied in the chart file are
# looked up as follows: lane coordinates from training data; market_index = that day's mean
# market_index in validation.csv (a market-level signal known at prediction time);
# quote_signal = validation median (a per-load noise-like signal with no date information).
dec = pd.read_csv("data/december_chart_inputs.csv")
coords = {**{r.pickup: (r.pickup_lat, r.pickup_lon) for r in raw.itertuples()},
          **{r.delivery: (r.delivery_lat, r.delivery_lon) for r in raw.itertuples()}}
d = dec.copy(); d["date"] = pd.to_datetime(d["date"])
d["pickup_lat"], d["pickup_lon"] = zip(*d.pickup.map(coords)); d["delivery_lat"], d["delivery_lon"] = zip(*d.delivery.map(coords))
mi = val[val.date >= "2025-12-01"].groupby("date").market_index.mean()
d["market_index"] = d.date.map(mi); d["quote_signal"] = val.quote_signal.median()
d["weight_missing"] = 0; d["market_index_missing"] = 0
assert d.market_index.notna().all()
dec["predicted_rate"] = final.predict(d).round(2)
dec.to_csv("data/december_chart_inputs.csv", index=False)
print("\nDecember predictions: min %.2f  mean %.2f  max %.2f" % (dec.predicted_rate.min(), dec.predicted_rate.mean(), dec.predicted_rate.max()))
lane = df[(df.pickup == "Lexington") & (df.delivery == "Fort Wayne") & (df.distance.between(340, 380)) & (df.equipment == "Dry Van") & (np.abs(np.log(df[TARGET]) - np.log(final.predict(df))) < TRIM)]
print("Sanity - same lane/equipment in training (Sep-Oct, clean): mean $%.0f (n=%d)" % (lane[lane.date >= '2025-09-01'][TARGET].mean(), (lane.date >= '2025-09-01').sum()))

# sanity: distribution of validation predictions vs training labels, drift check
rpm_tr = (df.loc[final.keep_, TARGET] / df.loc[final.keep_, "distance"]).median()
rpm_va = (val.predicted_rate / val.distance).median()
print("median $/mile  train(clean labels): %.3f   validation predictions: %.3f" % (rpm_tr, rpm_va))

# ----------------------------------------------------------------------------- 4) figures for the report
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "font.size": 10})
C = {"main": "#064A56", "alt": "#E07A5F", "grey": "#9DAFB3"}

# (a) label corruption: residual histogram
lin_only = FreightModel(use_gbm=False).fit(df)
_, L = lin_only._design(df)
resid = np.log(df[TARGET].values) - lin_only.lin_.predict(L.fillna(lin_only.med_))
fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.6))
ax[0].hist(resid, bins=200, color=C["main"]); ax[0].set_yscale("log")
ax[0].axvline(TRIM, color=C["alt"], ls="--"); ax[0].axvline(-TRIM, color=C["alt"], ls="--")
ax[0].set_title("Residuals of a first-pass fit (log scale)", loc="left", fontweight="bold")
ax[0].set_xlabel("log(actual) - log(predicted)"); ax[0].set_ylabel("rows (log axis)")
ax[1].hist(resid[np.abs(resid) < TRIM], bins=80, color=C["main"])
ax[1].set_title("Zoom: the other 98.6%% of rows (sd = %.3f)" % resid[np.abs(resid) < TRIM].std(), loc="left", fontweight="bold")
ax[1].set_xlabel("log(actual) - log(predicted)")
fig.tight_layout(); fig.savefig(OUT / "fig_label_corruption.png", dpi=170); plt.close(fig)

# (b) rolling CV comparison
fig, ax = plt.subplots(figsize=(8.6, 3.8))
order = list(MODELS); w = 0.38; x = np.arange(len(order))
ax.bar(x - w/2, summary.loc[order, "MAPE"], w, label="MAPE, all rows", color=C["grey"])
ax.bar(x + w/2, summary.loc[order, "MAPE_clean"], w, label="MAPE, rows with intact labels", color=C["main"])
ax.set_xticks(x); ax.set_xticklabels([o.replace(" (final)", "\n(final)").replace(" + ", "\n+ ") for o in order], fontsize=8)
ax.set_ylabel("MAPE (%)"); ax.legend(frameon=False)
ax.set_title("Rolling-origin validation (5 folds, 2-month test windows)", loc="left", fontweight="bold")
fig.tight_layout(); fig.savefig(OUT / "fig_cv_comparison.png", dpi=170); plt.close(fig)

# (c) hold-out predicted vs actual + error by month
h = holdout_pred
fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.9))
g = h[h.label_ok]
ax[0].scatter(g[TARGET], g.pred, s=3, alpha=0.25, color=C["main"]); lim = [0, g[TARGET].max()]
ax[0].plot(lim, lim, color=C["alt"], lw=1); ax[0].set_xlabel("actual rate ($)"); ax[0].set_ylabel("predicted rate ($)")
ax[0].set_title("Sep-Oct hold-out (intact labels)", loc="left", fontweight="bold")
b = h[~h.label_ok]
ax[1].scatter(h[TARGET], h.pred, s=3, alpha=0.15, color=C["grey"]); ax[1].scatter(b[TARGET], b.pred, s=10, color=C["alt"], label="corrupted labels (%d)" % len(b))
ax[1].plot(lim, lim, color="k", lw=0.6); ax[1].set_xscale("log"); ax[1].set_yscale("log"); ax[1].legend(frameon=False)
ax[1].set_xlabel("actual rate ($, log)"); ax[1].set_ylabel("predicted ($, log)")
ax[1].set_title("Same hold-out incl. corrupted labels", loc="left", fontweight="bold")
fig.tight_layout(); fig.savefig(OUT / "fig_holdout_scatter.png", dpi=170); plt.close(fig)

json.dump({"rolling_cv_mean": summary.to_dict("index"), "holdout_sep_oct": last.to_dict("index"),
           "unseen_city_stress": stress.to_dict("index"), "rows_dropped_as_corrupt": int((~final.keep_).sum()),
           "n_train": len(df), "n_val": len(val)}, open(OUT / "metrics.json", "w"), indent=2)
print("\nDone.")
