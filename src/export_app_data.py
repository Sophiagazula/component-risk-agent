"""
Sets the current component position (on hand + open purchase orders) and
exports everything the web app needs to app/data.js.

Inventory is set relative to forecast component requirements so the scenario
is realistic: most parts are covered, a few are tight.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA, APP = ROOT / "data", ROOT / "app"
APP.mkdir(exist_ok=True)

products = pd.read_csv(DATA / "products.csv")
suppliers = pd.read_csv(DATA / "suppliers.csv")
components = pd.read_csv(DATA / "components.csv")
bom = pd.read_csv(DATA / "bom.csv")
demand = pd.read_csv(DATA / "demand_history.csv")
fc = pd.read_csv(DATA / "forecast.csv")
bt = json.loads((DATA / "backtest_summary.json").read_text())

fmonths = sorted(fc.month.unique())
# average monthly P50 requirement per component
req = fc.merge(bom, on="product_id").assign(r=lambda d: d.q50 * d.qty_per).groupby("component_id").r.sum() / len(fmonths)

# months of cover on hand, and open POs as (month index 0..5, months of cover)
POSITION = {
    "LTE-MOD": (2.0, [(1, 1.4), (2, 1.4), (3, 1.4), (4, 1.4)]),
    "SOC-AP": (2.2, [(1, 1.8), (3, 1.8), (5, 1.6)]),
    "MCU-LP": (3.0, [(2, 2.5), (4, 2.0)]),
    "CAM-FWD": (1.6, [(1, 1.6), (3, 1.6), (5, 1.4)]),
    "CAM-IR": (1.9, [(2, 2.4), (4, 2.4)]),
    "GNSS": (2.6, [(2, 2.6), (4, 2.6)]),
    "BLE": (3.0, [(2, 2.6), (4, 2.4)]),
    "ENC-GW": (1.6, [(i, 1.1) for i in range(6)]),
    "ENC-CAM": (1.6, [(i, 1.1) for i in range(6)]),
    "BATT": (2.1, [(2, 2.0), (4, 2.0)]),
    "SNS-ARR": (2.0, [(2, 2.4), (4, 2.4)]),
    "FLASH": (2.4, [(1, 2.5), (3, 2.5)]),
}


def round_to(x, step=500):
    return int(max(step, round(x / step) * step))


comp_out, pos = [], []
po_id = 4100
for _, c in components.iterrows():
    on_hand_cover, pos_cover = POSITION[c.component_id]
    r = req[c.component_id]
    for m, cov in pos_cover:
        po_id += 1
        pos.append({"po": f"PO-{po_id}", "component_id": c.component_id, "supplier_id": c.supplier_id,
                    "month_index": m, "qty": round_to(r * cov)})
    comp_out.append({
        "id": c.component_id, "name": c["name"], "supplier": c.supplier_id, "unitCost": c.unit_cost,
        "leadMonths": int(c.lead_months), "moq": int(c.moq), "onHand": round_to(r * on_hand_cover),
        "alt": None if pd.isna(c.alt_supplier_id) else {"supplier": c.alt_supplier_id, "premium": c.alt_premium,
                                                         "leadMonths": int(c.alt_lead_months)},
        "expeditePremium": c.expedite_premium,
    })

hist = demand.copy()
hist_months = sorted(hist.month.unique())[-24:]
history = {pid: [int(hist[(hist.product_id == pid) & (hist.month == m)].units.iloc[0]) for m in hist_months]
           for pid in products.product_id}
forecast = {pid: [[round(float(r[q]), 1) for q in ["q05", "q10", "q25", "q50", "q75", "q90", "q95"]]
                  for _, r in fc[fc.product_id == pid].sort_values("horizon").iterrows()]
            for pid in products.product_id}

app = {
    "asOf": "2026-10",
    "historyMonths": hist_months,
    "forecastMonths": fmonths,
    "quantiles": [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95],
    "products": [{"id": r.product_id, "name": r.name, "revenue": int(r.revenue_per_unit), "margin": int(r.margin_per_unit)}
                 for r in products.itertuples()],
    "suppliers": [{"id": r.supplier_id, "name": r.name, "region": r.region, "onTime": r.on_time_rate}
                  for r in suppliers.itertuples()],
    "components": comp_out,
    "bom": [{"product": r.product_id, "component": r.component_id, "qty": int(r.qty_per)} for r in bom.itertuples()],
    "openPOs": pos,
    "history": history,
    "forecast": forecast,
    "backtest": bt,
}
(APP / "data.js").write_text("window.APP_DATA = " + json.dumps(app, separators=(",", ":")) + ";\n")
print("components:", len(comp_out), "open POs:", len(pos), "bytes:", (APP / "data.js").stat().st_size)
print(pd.Series({c["id"]: c["onHand"] for c in comp_out}))
