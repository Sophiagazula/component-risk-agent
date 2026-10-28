"""
Synthetic connected-hardware supply chain.

Creates 48 months of device demand driven by trend, seasonality and large
fleet deals, a sales-pipeline signal that sees those deals coming, a bill of
materials with shared components, suppliers with lead-time variability, and
current component inventory and open purchase orders.

All data is synthetic. Product, component and supplier names are generic.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)
OUT = Path(__file__).resolve().parents[1] / "data"
OUT.mkdir(exist_ok=True)

N_MONTHS = 48                       # Nov 2022 .. Oct 2026
months = pd.period_range("2022-11", periods=N_MONTHS, freq="M")

# ---------------------------------------------------------------- products
# revenue = hardware + first-year subscription; margin = gross margin per unit
PRODUCTS = [
    # id,   name,              base, growth/yr, revenue, margin, deal_rate, deal_size, season
    ("GW-PRO", "Vehicle Gateway Pro", 5200, 0.22, 1150, 520, 0.35, 2600, "q4"),
    ("GW-LITE", "Vehicle Gateway Lite", 2600, 0.15, 700, 280, 0.25, 1400, "q4"),
    ("CAM-RD", "Road-Facing Dashcam", 3400, 0.34, 1600, 760, 0.40, 2200, "q4"),
    ("CAM-DUAL", "Dual-Facing Dashcam", 1900, 0.48, 2200, 1080, 0.35, 1500, "q4"),
    ("TAG-AST", "Asset Tag", 900, 0.55, 120, 55, 0.22, 6000, "spring"),
    ("SNS-ENV", "Environmental Sensor", 160, 0.30, 350, 150, 0.15, 900, "spring"),
]

SEASON = {
    # multiplicative index by calendar month (1..12)
    "q4": {1: .88, 2: .90, 3: 1.0, 4: .97, 5: .98, 6: 1.02, 7: .95, 8: .96, 9: 1.05, 10: 1.06, 11: 1.08, 12: 1.15},
    "spring": {1: .70, 2: .78, 3: 1.10, 4: 1.30, 5: 1.30, 6: 1.15, 7: 1.0, 8: .95, 9: .95, 10: .90, 11: .90, 12: .97},
}

# pipeline visibility: share of a future month's deal volume already in the
# weighted pipeline h months ahead
VISIBILITY = {1: .85, 2: .70, 3: .55, 4: .40, 5: .28, 6: .18}
H = 6
TOTAL = N_MONTHS + H                # simulate 6 extra months for pipeline only

demand_rows, pipe_rows = [], []
for pid, name, base, g, rev, mar, deal_rate, deal_size, season in PRODUCTS:
    t = np.arange(TOTAL)
    level = base * (1 + g) ** (t / 12)
    seas = np.array([SEASON[season][(m.month)] for m in pd.period_range("2022-11", periods=TOTAL, freq="M")])
    # large fleet deals: Poisson count, lognormal size, scaled with growth
    n_deals = RNG.poisson(deal_rate, TOTAL)
    deals = np.array([RNG.lognormal(np.log(deal_size), 0.45, k).sum() for k in n_deals]) * (1 + g) ** (t / 12)
    run_rate = level * seas * RNG.lognormal(0, 0.10, TOTAL)
    demand = np.maximum(0, np.round(run_rate + deals))
    if pid == "SNS-ENV":            # intermittent run-rate months
        demand = np.where(RNG.random(TOTAL) < 0.25, np.round(deals), demand)

    for i in range(N_MONTHS):
        demand_rows.append({"month": str(months[i]), "product_id": pid, "units": int(demand[i]),
                            "deal_units": int(round(deals[i]))})
    # pipeline snapshot at each origin i for months i+1..i+6
    for i in range(N_MONTHS):
        for h in range(1, H + 1):
            target = i + h
            seen = deals[target] * VISIBILITY[h] * RNG.lognormal(0, 0.25)
            noise = RNG.lognormal(np.log(deal_size * 0.15), 0.5)   # deals that won't close
            pipe_rows.append({"origin": str(pd.Period("2022-11", "M") + i), "horizon": h,
                              "target_month": str(pd.Period("2022-11", "M") + target),
                              "product_id": pid, "weighted_pipeline_units": int(round(seen + noise))})

demand_df = pd.DataFrame(demand_rows)
pipe_df = pd.DataFrame(pipe_rows)
products_df = pd.DataFrame(PRODUCTS, columns=["product_id", "name", "base", "growth", "revenue_per_unit",
                                              "margin_per_unit", "deal_rate", "deal_size", "season"])

# ------------------------------------------------------------- suppliers
SUPPLIERS = [
    # id,  name,                    region,     on_time_rate
    ("S1", "Cellular modem supplier", "Taiwan", 0.72),
    ("S2", "Silicon distributor", "Singapore", 0.85),
    ("S3", "Microcontroller supplier", "Malaysia", 0.90),
    ("S4", "Camera module supplier", "China", 0.78),
    ("S5", "GNSS supplier", "Korea", 0.92),
    ("S6", "Enclosure molder (JDM)", "Mexico", 0.95),
    ("S7", "Battery & sensor supplier", "Japan", 0.88),
]
suppliers_df = pd.DataFrame(SUPPLIERS, columns=["supplier_id", "name", "region", "on_time_rate"])

# ------------------------------------------------------------ components
# lead time in months; alt = qualified second source (None if single-source)
COMPONENTS = [
    # id,     name,                     supplier, unit_cost, lead_mo, moq,   alt_supplier, alt_premium, alt_lead_mo, expedite_premium
    ("LTE-MOD", "LTE Cat-4 modem module", "S1", 38.0, 4, 5000, None, None, None, 0.40),
    ("SOC-AP", "Application processor SoC", "S2", 24.0, 5, 5000, "S3", 0.18, 3, 0.35),
    ("MCU-LP", "Low-power microcontroller", "S3", 3.5, 3, 10000, "S2", 0.12, 2, 0.30),
    ("CAM-FWD", "Forward camera module", "S4", 18.0, 3, 3000, None, None, None, 0.45),
    ("CAM-IR", "Driver-facing IR camera", "S4", 22.0, 3, 2000, None, None, None, 0.45),
    ("GNSS", "GNSS receiver module", "S5", 7.0, 2, 5000, "S2", 0.10, 2, 0.25),
    ("BLE", "Bluetooth LE radio", "S3", 1.8, 2, 20000, "S5", 0.08, 2, 0.25),
    ("ENC-GW", "Gateway enclosure", "S6", 4.0, 1, 5000, None, None, None, 0.20),
    ("ENC-CAM", "Camera enclosure", "S6", 6.0, 1, 3000, None, None, None, 0.20),
    ("BATT", "Battery cell", "S7", 2.5, 2, 20000, "S5", 0.15, 2, 0.30),
    ("SNS-ARR", "Temp/humidity sensor array", "S7", 9.0, 3, 1000, None, None, None, 0.35),
    ("FLASH", "eMMC flash memory", "S2", 5.0, 4, 10000, "S3", 0.10, 3, 0.35),
]
components_df = pd.DataFrame(COMPONENTS, columns=["component_id", "name", "supplier_id", "unit_cost", "lead_months",
                                                  "moq", "alt_supplier_id", "alt_premium", "alt_lead_months",
                                                  "expedite_premium"])

# ----------------------------------------------------------------- BOM
BOM = {
    "GW-PRO": {"LTE-MOD": 1, "SOC-AP": 1, "GNSS": 1, "BLE": 1, "ENC-GW": 1, "FLASH": 1},
    "GW-LITE": {"LTE-MOD": 1, "MCU-LP": 1, "GNSS": 1, "ENC-GW": 1},
    "CAM-RD": {"LTE-MOD": 1, "SOC-AP": 1, "CAM-FWD": 1, "GNSS": 1, "ENC-CAM": 1, "FLASH": 1},
    "CAM-DUAL": {"LTE-MOD": 1, "SOC-AP": 1, "CAM-FWD": 1, "CAM-IR": 1, "GNSS": 1, "ENC-CAM": 1, "FLASH": 2},
    "TAG-AST": {"MCU-LP": 1, "BLE": 1, "BATT": 1},
    "SNS-ENV": {"MCU-LP": 1, "BLE": 1, "BATT": 2, "SNS-ARR": 1},
}
PART_NUMBERS = {"LTE-MOD": "412-00118", "SOC-AP": "412-00231", "MCU-LP": "412-00307", "CAM-FWD": "520-00044",
                "CAM-IR": "520-00051", "GNSS": "412-00402", "BLE": "412-00415", "ENC-GW": "610-00012",
                "ENC-CAM": "610-00021", "BATT": "705-00009", "SNS-ARR": "530-00063", "FLASH": "412-00520"}
components_df["part_number"] = components_df.component_id.map(PART_NUMBERS)

bom_df = pd.DataFrame([{"product_id": p, "component_id": c, "qty_per": q} for p, d in BOM.items() for c, q in d.items()])

demand_df.to_csv(OUT / "demand_history.csv", index=False)
pipe_df.to_csv(OUT / "pipeline_snapshots.csv", index=False)
products_df.to_csv(OUT / "products.csv", index=False)
suppliers_df.to_csv(OUT / "suppliers.csv", index=False)
components_df.to_csv(OUT / "components.csv", index=False)
bom_df.to_csv(OUT / "bom.csv", index=False)
print(f"demand rows {len(demand_df)}, pipeline rows {len(pipe_df)}")
print(demand_df.groupby("product_id").units.agg(["mean", "min", "max"]).round(0))
