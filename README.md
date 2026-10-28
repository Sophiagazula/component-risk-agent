# Component Risk Agent

![Train, gate and deploy](https://github.com/Sophiagazula/component-risk-agent/actions/workflows/train-and-deploy.yml/badge.svg)

Which parts will run short, what is the cheapest fix, and who should approve it. A prototype for a connected-hardware supply chain: probabilistic device forecasts, part-level shortage risk, cost-based buy and expedite decisions, supplier-delay scenarios, and an agent with tiered approvals.

**Live demo:** https://sophiagazula.github.io/component-risk-agent/
**Design notes:** [DESIGN.md](DESIGN.md) · **Evaluation:** [reports/evaluation.md](reports/evaluation.md)

All data is synthetic. Product, part and supplier names are generic.

## How it works

1. **Forecast.** LightGBM quantile models (P5–P95), one per quantile, trained across all devices, 1–6 months ahead. The weighted sales pipeline is the main leading signal. Ranges are calibrated on held-out months.
2. **Explode.** Device demand goes through the bill of materials to part demand. Shared parts link devices.
3. **Simulate.** 3,000 futures with correlated demand and supplier lateness give each part's chance of running short.
4. **Decide.** Newsvendor critical ratio per part sets the buy. When the problem is timing rather than quantity, open POs are pulled in instead.
5. **Stress test.** Delay a supplier, rerun clear-to-build month by month, score mitigations in margin.
6. **Act.** Free, reversible actions run automatically; spend goes to a planner or director; everything is logged.

The trained trees are exported and evaluated in the browser, so adding a what-if deal re-forecasts demand and updates everything downstream.

## Results

Rolling backtest, 288 forecasts (8 origins × 6 horizons × 6 devices):

| | WAPE |
|---|---|
| This model | 20.3% |
| Same month last year | 42.5% |
| 3-month moving average | 45.4% |

- Bias −7%. P10–P90 coverage 68% raw, 87% after calibration.
- Removing the sales pipeline raises WAPE to 35.5%. Calendar month adds nothing.
- Weak spot: the intermittent environmental sensor, where the model loses to seasonal naive.

## Pipeline

`.github/workflows/train-and-deploy.yml` runs on every push, on demand and monthly:

build data → train + backtest → tests → feature ablation → quality gate → export → deploy

The gate releases a model only if it beats the best naive baseline by 10%+, absolute bias is under 15% and P10–P90 coverage is 70–95%. A failed step stops the run and the previous version stays live. Each run posts the gate and evaluation tables to its summary page and keeps the models and model card as artifacts.

## Tests

```bash
pytest -q tests                       # leakage, data integrity, exported trees vs LightGBM, forecast sanity
node --test tests/engine.test.mjs     # browser engine: parity with LightGBM, quantiles, critical ratio, MOQ
```

The leakage test scrambles every month after the forecast origin and checks no feature changes.

## Run locally

```bash
pip install -r requirements.txt -r requirements-dev.txt
python src/simulate_supply_chain.py
python src/forecast.py
pytest -q tests && node --test tests/engine.test.mjs
python src/evaluate.py
python src/quality_gate.py
python src/export_app_data.py
python src/build_site.py
python -m http.server -d docs
```

## Layout

```
src/model.py                   features, training, calibration, tree export (shared)
src/simulate_supply_chain.py   synthetic demand, pipeline, BOM, suppliers
src/forecast.py                backtest, calibration, final models, export
src/evaluate.py                feature ablation and evaluation report
src/quality_gate.py            release checks and model card
src/export_app_data.py         inventory position and app data
src/build_site.py              static site for GitHub Pages
app/engine.js                  pure calculations used by the app and tests
app/index.html                 the app
tests/                         pytest and node tests
models/                        saved models, browser model file, model card
reports/evaluation.md          ablation, error by horizon, calibration
DESIGN.md                      decisions, alternatives, limitations
```
