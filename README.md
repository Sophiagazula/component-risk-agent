# Component Risk Agent

**From probabilistic demand forecasts to component shortage risk, cost-based buy decisions and a tiered-autonomy agent, for a connected-hardware supply chain.**

**Live demo:** https://sophiagazula.github.io/component-risk-agent/

> All data is synthetic. Product, part and supplier names are generic.

## Why this exists

In hardware, supply chains rarely break at the finished-goods level. They break when one long-lead component (a modem, a camera module, a battery cell) blocks thousands of devices. This prototype follows that problem end to end:

| Step | What it does | Method |
|---|---|---|
| 1. Predict | Forecast each device 1–6 months ahead as a range (P5–P95) | Global LightGBM quantile models, direct multi-horizon, sales pipeline as a leading signal, conformal-style interval calibration |
| 2. Explode | Turn device demand into part demand | Bill of materials with shared parts |
| 3. Simulate | Probability each part runs short before the horizon ends | 3,000 Monte Carlo runs: correlated demand + supplier lateness |
| 4. Decide | How much to buy, and when to pull orders in | Newsvendor critical ratio per part (cost of short vs. cost of holding); pull-ins when the problem is timing, not quantity |
| 5. Stress | What a supplier delay breaks, and the cheapest fix | Monthly build simulation; options scored by margin protected minus cost |
| 6. Act | Who approves what | Tiered autonomy: auto for free and reversible actions, planner up to a spend limit, director above it or for strategic calls; audit log |

## Results (synthetic backtest)

- 288 forecasts over 8 rolling origins, 1–6 months ahead
- **WAPE 20%** vs. 43% for same-month-last-year and 45% for a 3-month moving average
- Bias −7%
- Raw quantile intervals covered 68% of actuals; after calibration on earlier held-out origins, **P10–P90 covers 87%** of backtest actuals (target 80%)
- Known weakness: the intermittent environmental sensor, where the model loses to seasonal naive. In production I would use a Croston-style method or forecast at a higher level.

## Run it

```bash
pip install -r requirements.txt
python src/simulate_supply_chain.py   # synthetic demand, pipeline, BOM, suppliers
python src/forecast.py                # quantile models + backtest + calibration
python src/export_app_data.py         # inventory position + app/data.js
python src/build_site.py              # docs/ for GitHub Pages
```

Open `docs/index.html` in a browser, or serve it: `python -m http.server -d docs`.

## Repo layout

```
src/simulate_supply_chain.py   synthetic supply chain
src/forecast.py                LightGBM quantile forecast, backtest, calibration
src/export_app_data.py         inventory position and app data export
src/build_site.py              builds docs/ for GitHub Pages
app/index.html                 the web app (forecast, risk, shock, agent)
data/                          generated CSVs and backtest summary
docs/                          published site
```

## What would change with real data

- ERP orders and shipments, BOMs from PLM, supplier commit dates from planning tools
- CRM pipeline and customer rollout schedules as the leading demand signal; installed-base telemetry for replacement demand
- Supplier lateness learned per supplier and lane from receipt history
- Validate against numbers Finance already trusts before any recommendation reaches a planner
- An LLM drafts the agent briefing from the same numbers; decisions stay with the model + policy + human approval

Built in a day with Python and plain JavaScript, using Claude Code.
