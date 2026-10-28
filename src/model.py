"""
Shared forecasting code: features, training, calibration and model export.
Used by forecast.py (production run), evaluate.py (ablations) and the tests.
"""
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
H = 6                                            # months ahead
QUANTILES = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
I10, I50, I90 = 1, 3, 5
FIRST_ORIGIN = 12                                # need a year of history for the features
PARAMS = dict(n_estimators=300, learning_rate=0.03, num_leaves=7, min_child_samples=15,
              subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0, verbose=-1)


class ForecastData:
    """Demand history and pipeline snapshots, plus feature building.

    Every feature for origin i (the last observed month) uses only demand up to
    month i and the pipeline snapshot taken at month i.
    """

    def __init__(self, demand, pipe, products):
        self.pids = list(products.product_id)
        self.months = sorted(demand.month.unique())
        self.T = len(self.months)
        self.Y = (demand.pivot(index="month", columns="product_id", values="units")
                  .loc[self.months, self.pids].values.astype(float))
        self.PIPE = pipe.set_index(["origin", "horizon", "product_id"]).weighted_pipeline_units

    @classmethod
    def from_csv(cls, data_dir=DATA):
        return cls(pd.read_csv(data_dir / "demand_history.csv"),
                   pd.read_csv(data_dir / "pipeline_snapshots.csv"),
                   pd.read_csv(data_dir / "products.csv"))

    def level(self, i, p):
        return np.log1p(self.Y[i - 11:i + 1, p].mean())

    def features(self, i, h, p):
        y, base = self.Y[:, p], self.level(i, p)
        return {
            "product": p, "horizon": h,
            "target_month": int(str(pd.Period(self.months[i], "M") + h)[5:]),
            "lag0": np.log1p(y[i]) - base, "lag1": np.log1p(y[i - 1]) - base, "lag2": np.log1p(y[i - 2]) - base,
            "mean3": np.log1p(y[i - 2:i + 1].mean()) - base, "mean6": np.log1p(y[i - 5:i + 1].mean()) - base,
            "same_month_ly": np.log1p(y[i + h - 12]) - base,
            "trend_yoy": base - np.log1p(y[max(0, i - 23):i - 11].mean()),
            "pipeline": np.log1p(self.PIPE[(self.months[i], h, self.pids[p])]) - base,
        }

    def rows_for(self, origins, last_observed):
        """Training rows from these origins whose target month is already observed."""
        X, y, targets = [], [], []
        for i in origins:
            for h in range(1, H + 1):
                if i + h > last_observed:
                    continue
                for p in range(len(self.pids)):
                    X.append(self.features(i, h, p))
                    y.append(np.log1p(self.Y[i + h, p]) - self.level(i, p))
                    targets.append(i + h)
        self.last_targets = targets
        return pd.DataFrame(X), np.array(y)

    def predict_origin(self, o, quantiles=QUANTILES, drop=()):
        """Train on everything observable at origin o, predict months o+1..o+H.
        Returns log-space quantiles [H*P, Q], (h, p) meta, P50 gain importance, fitted models, X."""
        Xtr, ytr = self.rows_for(range(FIRST_ORIGIN, o + 1), o)
        meta = [(h, p) for h in range(1, H + 1) for p in range(len(self.pids))]
        Xp = pd.DataFrame([self.features(o, h, p) for h, p in meta])
        cols = [c for c in Xtr.columns if c not in drop]
        cat = ["product"] if "product" in cols else []
        base = np.array([self.level(o, p) for h, p in meta])
        preds, importance, models = [], None, []
        for q in quantiles:
            m = lgb.LGBMRegressor(objective="quantile", alpha=q, **PARAMS)
            m.fit(Xtr[cols], ytr, categorical_feature=cat)
            preds.append(m.predict(Xp[cols]) + base)
            models.append(m)
            if q == 0.5:
                importance = dict(zip(cols, m.booster_.feature_importance("gain")))
        return np.sort(np.column_stack(preds), axis=1), meta, importance, models, Xp[cols]

    def origins(self, n_backtest=8, n_calib=8):
        last_bt = self.T - 1 - H
        bt = list(range(last_bt - n_backtest + 1, last_bt + 1))
        cal = list(range(bt[0] - n_calib, bt[0]))
        return bt, cal


def widen(logq, s):
    """Scale the spread of log-space quantiles around the median."""
    mid = logq[:, [I50]]
    return mid + (logq - mid) * s


def wape(actual, forecast):
    actual, forecast = np.asarray(actual, float), np.asarray(forecast, float)
    return float(np.abs(actual - forecast).sum() / actual.sum())


# ------------------------------------------------------------- export
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
    """Same tree walk as app/engine.js, in Python."""
    total = 0.0
    for t in trees:
        node = t["root"]
        while node >= 0:
            f = x[t["f"][node]]
            go_left = (int(f) in t["c"][node]) if t["c"][node] is not None else (f <= t["t"][node])
            node = t["l"][node] if go_left else t["r"][node]
        total += t["v"][~node]
    return total
