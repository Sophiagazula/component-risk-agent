"""
Production forecast run: backtest, calibrate, train the final models, export.

Model
  One global LightGBM model per quantile, direct multi-horizon (horizon is a
  feature). Target and features are relative to each device's 12-month level
  in log space, so the trees learn shape and growth rather than scale.

Calibration
  Quantile GBMs trained on small data are overconfident. A spread factor
  around P50 is fitted on 8 earlier origins so P10-P90 covers ~80%, then
  checked on the 8 later backtest origins it never saw.

Outputs
  data/forecast.csv, data/backtest_summary.json, data/backtest_detail.csv,
  models/quantile_qXX.txt, models/forecast_model.json (browser trees),
  models/parity_fixture.json (used by the JS parity test).
"""
import json

import numpy as np
import pandas as pd

from model import (DATA, ROOT, H, QUANTILES, I10, I50, I90, ForecastData,
                   widen, wape, compact, walk_predict)

fd = ForecastData.from_csv()
pids, months, Y, T = fd.pids, fd.months, fd.Y, fd.T
bt_origins, cal_origins = fd.origins()


def collect(origins):
    out = []
    for o in origins:
        logq, meta, *_ = fd.predict_origin(o)
        for k, (h, p) in enumerate(meta):
            out.append({"origin": months[o], "o": o, "horizon": h, "product_id": pids[p], "p": p,
                        "actual": Y[o + h, p], "logq": logq[k],
                        "seasonal_naive": Y[o + h - 12, p], "moving_avg_3": Y[o - 2:o + 1, p].mean()})
    return out


def coverage(recs, s):
    hits = []
    for r in recs:
        q, a = widen(r["logq"][None], s)[0], np.log1p(r["actual"])
        hits.append(q[I10] <= a <= q[I90])
    return float(np.mean(hits))


def fit_scale(recs, target=0.80):
    for s in np.arange(1.0, 4.01, 0.05):
        if coverage(recs, s) >= target:
            return round(float(s), 2)
    return 4.0


def reliability(recs, s):
    """Share of actuals at or below each predicted quantile (ideal: equals the quantile)."""
    out = {}
    for j, q in enumerate(QUANTILES):
        out[str(q)] = float(np.mean([np.log1p(r["actual"]) <= widen(r["logq"][None], s)[0, j] for r in recs]))
    return out


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
    "reliability": {"raw": reliability(bt, 1.0), "calibrated": reliability(bt, scale)},
}

# ---------------------------------------------------------- final forecast
final_scale = fit_scale(cal + bt)          # all held-out origins set the production scale
logq, meta, importance, models, Xp = fd.predict_origin(T - 1)
q = np.maximum(np.expm1(widen(logq, final_scale)), 0)
fut = [str(pd.Period(months[-1], "M") + h) for h in range(1, H + 1)]
fc = pd.DataFrame([{"month": fut[h - 1], "horizon": h, "product_id": pids[p],
                    **{f"q{int(qq * 100):02d}": float(q[k, j]) for j, qq in enumerate(QUANTILES)}}
                   for k, (h, p) in enumerate(meta)])
tot = sum(importance.values())
summary["final_interval_scale"] = final_scale
summary["feature_importance"] = {k: round(v / tot, 4) for k, v in sorted(importance.items(), key=lambda x: -x[1])}

# ------------------------------------------------ save and export models
MODELS = ROOT / "models"
MODELS.mkdir(exist_ok=True)
export = {"features": list(Xp.columns), "quantiles": QUANTILES, "scale": final_scale, "models": []}
fixture = {"features": list(Xp.columns), "X": Xp.values.tolist(), "expected": []}
max_err = 0.0
for qq, m in zip(QUANTILES, models):
    m.booster_.save_model(str(MODELS / f"quantile_q{int(qq * 100):02d}.txt"))
    trees = [compact(t) for t in m.booster_.dump_model()["tree_info"]]
    export["models"].append(trees)
    ref = m.predict(Xp)
    fixture["expected"].append(ref.tolist())
    mine = np.array([walk_predict(trees, row) for row in Xp.values])
    max_err = max(max_err, float(np.abs(ref - mine).max()))
(MODELS / "forecast_model.json").write_text(json.dumps(export, separators=(",", ":")))
(MODELS / "parity_fixture.json").write_text(json.dumps(fixture))

fc.to_csv(DATA / "forecast.csv", index=False)
bt_df.to_csv(DATA / "backtest_detail.csv", index=False)
(DATA / "backtest_summary.json").write_text(json.dumps(summary, indent=2))

o = summary["overall"]
print(f"exported {sum(len(t) for t in export['models'])} trees; max |tree walk - LightGBM| = {max_err:.2e}")
print(f"calibration scale {scale} (final {final_scale})")
print(f"Backtest {o['n']} forecasts: WAPE model {o['wape_model']:.3f} | seasonal naive {o['wape_seasonal_naive']:.3f} | MA3 {o['wape_moving_avg']:.3f}")
print(f"bias {o['bias_model']:+.3f}  coverage raw {o['coverage_raw']:.2f} -> calibrated {o['coverage_p10_p90']:.2f}")
print("reliability (calibrated):", {k: round(v, 2) for k, v in summary["reliability"]["calibrated"].items()})
print("horizon WAPE:", {h: round(s['wape_model'], 3) for h, s in summary["by_horizon"].items()})
