"""
Probabilistic demand forecast for each device, 1-6 months ahead.

Model
  One global LightGBM model per quantile, direct multi-horizon (horizon is a
  feature). Target and features are expressed relative to each product's
  12-month level (log space), so the trees learn shape and growth rather than
  absolute scale and can follow a growing product.

Inputs (all known at the forecast origin)
  recent demand, same month last year, calendar month, horizon, product, and
  the weighted sales pipeline for the target month.

Calibration
  Quantile GBMs trained on small data give intervals that are too narrow.
  A scale factor on the spread around P50 is fitted on a separate set of
  earlier origins so the P10-P90 band covers ~80% (conformal-style), then
  checked on the later backtest origins it never saw.

Validation
  Rolling-origin backtest against seasonal naive and a 3-month moving average.
"""
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
H = 6
QUANTILES = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
I10, I50, I90 = 1, 3, 5
N_BACKTEST, N_CALIB = 8, 8
PARAMS = dict(n_estimators=300, learning_rate=0.03, num_leaves=7, min_child_samples=15,
              subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0, verbose=-1)

demand = pd.read_csv(DATA / "demand_history.csv")
pipe = pd.read_csv(DATA / "pipeline_snapshots.csv")
products = pd.read_csv(DATA / "products.csv")
pids = list(products.product_id)
months = sorted(demand.month.unique())
T = len(months)
Y = demand.pivot(index="month", columns="product_id", values="units").loc[months, pids].values.astype(float)
PIPE = pipe.set_index(["origin", "horizon", "product_id"]).weighted_pipeline_units
FIRST_ORIGIN = 12


def level(i, p):
    return np.log1p(Y[i - 11:i + 1, p].mean())


def features(i, h, p):
    y, base = Y[:, p], level(i, p)
    return {
        "product": p, "horizon": h,
        "target_month": int(str(pd.Period(months[i], "M") + h)[5:]),
        "lag0": np.log1p(y[i]) - base, "lag1": np.log1p(y[i - 1]) - base, "lag2": np.log1p(y[i - 2]) - base,
        "mean3": np.log1p(y[i - 2:i + 1].mean()) - base, "mean6": np.log1p(y[i - 5:i + 1].mean()) - base,
        "same_month_ly": np.log1p(y[i + h - 12]) - base,
        "trend_yoy": base - np.log1p(y[max(0, i - 23):i - 11].mean()),
        "pipeline": np.log1p(PIPE[(months[i], h, pids[p])]) - base,
    }


def rows_for(origins, last_observed):
    """Training rows whose target month index <= last_observed."""
    X, y = [], []
    for i in origins:
        for h in range(1, H + 1):
            if i + h > last_observed:
                continue
            for p in range(len(pids)):
                X.append(features(i, h, p))
                y.append(np.log1p(Y[i + h, p]) - level(i, p))
    return pd.DataFrame(X), np.array(y)


def predict_origin(o):
    """Train on everything observable at origin o; return log-space quantiles [H*P, Q] and meta."""
    Xtr, ytr = rows_for(range(FIRST_ORIGIN, o + 1), o)
    meta = [(h, p) for h in range(1, H + 1) for p in range(len(pids))]
    Xp = pd.DataFrame([features(o, h, p) for h, p in meta])
    base = np.array([level(o, p) for h, p in meta])
    preds, importance, models = [], None, []
    for q in QUANTILES:
        m = lgb.LGBMRegressor(objective="quantile", alpha=q, **PARAMS)
        m.fit(Xtr, ytr, categorical_feature=["product"])
        preds.append(m.predict(Xp) + base)
        models.append(m)
        if q == 0.5:
            importance = dict(zip(Xtr.columns, m.booster_.feature_importance("gain")))
    predict_origin.last_models, predict_origin.last_X = models, Xp
    return np.sort(np.column_stack(preds), axis=1), meta, importance


def widen(logq, s):
    mid = logq[:, [I50]]
    return mid + (logq - mid) * s


last_bt = T - 1 - H
bt_origins = list(range(last_bt - N_BACKTEST + 1, last_bt + 1))
cal_origins = list(range(bt_origins[0] - N_CALIB, bt_origins[0]))


def collect(origins):
    out = []
    for o in origins:
        logq, meta, _ = predict_origin(o)
        for k, (h, p) in enumerate(meta):
            out.append({"origin": months[o], "o": o, "horizon": h, "product_id": pids[p], "p": p,
                        "actual": Y[o + h, p], "logq": logq[k],
                        "seasonal_naive": Y[o + h - 12, p], "moving_avg_3": Y[o - 2:o + 1, p].mean()})
    return out


def coverage(recs, s):
    hits = [np.log1p(r["actual"]) >= widen(r["logq"][None], s)[0, I10] and
            np.log1p(r["actual"]) <= widen(r["logq"][None], s)[0, I90] for r in recs]
    return float(np.mean(hits))


def fit_scale(recs, target=0.80):
    for s in np.arange(1.0, 4.01, 0.05):
        if coverage(recs, s) >= target:
            return round(float(s), 2)
    return 4.0


cal = collect(cal_origins)
scale = fit_scale(cal)
bt = collect(bt_origins)

rows = []
for r in bt:
    q = np.expm1(widen(r["logq"][None], scale)[0])
    raw = np.expm1(r["logq"])
    rows.append({"origin": r["origin"], "horizon": r["horizon"], "product_id": r["product_id"], "actual": r["actual"],
                 "p10": q[I10], "p50": q[I50], "p90": q[I90], "raw_p10": raw[I10], "raw_p90": raw[I90],
                 "seasonal_naive": r["seasonal_naive"], "moving_avg_3": r["moving_avg_3"]})
bt_df = pd.DataFrame(rows)


def wape(a, f):
    return float(np.abs(a - f).sum() / a.sum())


def summarize(d):
    return {
        "wape_model": wape(d.actual, d.p50),
        "wape_seasonal_naive": wape(d.actual, d.seasonal_naive),
        "wape_moving_avg": wape(d.actual, d.moving_avg_3),
        "bias_model": float((d.p50 - d.actual).sum() / d.actual.sum()),
        "coverage_p10_p90": float(((d.actual >= d.p10) & (d.actual <= d.p90)).mean()),
        "coverage_raw": float(((d.actual >= d.raw_p10) & (d.actual <= d.raw_p90)).mean()),
        "n": int(len(d)),
    }


summary = {
    "backtest_origins": [months[o] for o in bt_origins],
    "calibration_origins": [months[o] for o in cal_origins],
    "interval_scale": scale,
    "overall": summarize(bt_df),
    "by_product": {pid: summarize(bt_df[bt_df.product_id == pid]) for pid in pids},
    "by_horizon": {int(h): summarize(bt_df[bt_df.horizon == h]) for h in range(1, H + 1)},
}

# ---------------------------------------------------------- final forecast
final_scale = fit_scale(cal + bt)          # use all held-out origins for the production scale
logq, meta, importance = predict_origin(T - 1)
q = np.maximum(np.expm1(widen(logq, final_scale)), 0)
fut = [str(pd.Period(months[-1], "M") + h) for h in range(1, H + 1)]
fc = pd.DataFrame([{"month": fut[h - 1], "horizon": h, "product_id": pids[p],
                    **{f"q{int(qq * 100):02d}": float(q[k, j]) for j, qq in enumerate(QUANTILES)}}
                   for k, (h, p) in enumerate(meta)])
tot = sum(importance.values())
summary["final_interval_scale"] = final_scale
summary["feature_importance"] = {k: round(v / tot, 4) for k, v in sorted(importance.items(), key=lambda x: -x[1])}

fc.to_csv(DATA / "forecast.csv", index=False)

# ------------------------------------------------ save production models
MODELS = ROOT / "models"
MODELS.mkdir(exist_ok=True)
for qq, m in zip(QUANTILES, predict_origin.last_models):
    m.booster_.save_model(str(MODELS / f"quantile_q{int(qq * 100):02d}.txt"))


def compact(tree):
    """Flatten one LightGBM tree into arrays a browser can walk.
    Child index >= 0 is an internal node, < 0 is leaf ~index."""
    feat, thr, cats, left, right, leaves = [], [], [], [], [], []

    def walk(n):
        if "leaf_value" in n:
            leaves.append(round(n["leaf_value"], 7))
            return ~(len(leaves) - 1)
        i = len(feat)
        feat.append(n["split_feature"]); left.append(0); right.append(0)
        if n["decision_type"] == "==":
            thr.append(None); cats.append([int(c) for c in str(n["threshold"]).split("||")])
        else:
            thr.append(round(float(n["threshold"]), 7)); cats.append(None)
        left[i] = walk(n["left_child"]); right[i] = walk(n["right_child"])
        return i

    root = walk(tree["tree_structure"])
    return {"root": root, "f": feat, "t": thr, "c": cats, "l": left, "r": right, "v": leaves}


def walk_predict(trees, x):
    total = 0.0
    for t in trees:
        node = t["root"]
        while node >= 0:
            f = x[t["f"][node]]
            go_left = (int(f) in t["c"][node]) if t["c"][node] is not None else (f <= t["t"][node])
            node = t["l"][node] if go_left else t["r"][node]
        total += t["v"][~node]
    return total


export = {"features": list(predict_origin.last_X.columns), "quantiles": QUANTILES, "scale": final_scale, "models": []}
X_check = predict_origin.last_X.values
max_err = 0.0
for qq, m in zip(QUANTILES, predict_origin.last_models):
    dump = m.booster_.dump_model()
    trees = [compact(t) for t in dump["tree_info"]]
    export["models"].append(trees)
    ref = m.predict(predict_origin.last_X)
    mine = np.array([walk_predict(trees, row) for row in X_check])
    max_err = max(max_err, float(np.abs(ref - mine).max()))
(MODELS / "forecast_model.json").write_text(json.dumps(export, separators=(",", ":")))
print(f"exported {sum(len(t) for t in export['models'])} trees; max |tree walk - LightGBM| = {max_err:.2e}")
bt_df.to_csv(DATA / "backtest_detail.csv", index=False)
(DATA / "backtest_summary.json").write_text(json.dumps(summary, indent=2))

o = summary["overall"]
print(f"calibration scale {scale} (final {final_scale})")
print(f"Backtest {o['n']} forecasts: WAPE model {o['wape_model']:.3f} | seasonal naive {o['wape_seasonal_naive']:.3f} | MA3 {o['wape_moving_avg']:.3f}")
print(f"bias {o['bias_model']:+.3f}  coverage raw {o['coverage_raw']:.2f} -> calibrated {o['coverage_p10_p90']:.2f}")
for pid, s in summary["by_product"].items():
    print(f"  {pid:9s} model {s['wape_model']:.3f} snaive {s['wape_seasonal_naive']:.3f} ma3 {s['wape_moving_avg']:.3f} bias {s['bias_model']:+.2f} cov {s['coverage_p10_p90']:.2f}")
print("horizon WAPE:", {h: round(s['wape_model'], 3) for h, s in summary["by_horizon"].items()})
print("importance:", list(summary["feature_importance"].items())[:6])
print(fc.pivot(index="month", columns="product_id", values="q50").round(0))
