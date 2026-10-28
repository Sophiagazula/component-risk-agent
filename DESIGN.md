# Design notes: Component Risk Agent

Author: Sophia Gazula · October 2026 · Status: prototype on synthetic data

## Problem

Hardware supply chains rarely fail at the finished-goods level. They fail when one long-lead part blocks thousands of devices: a single-source modem, a camera module, a $2.50 battery cell. Planners see the device forecast, but the decisions they actually make are about parts: what to buy, what to expedite, which customer ships first, and who signs off.

The question this prototype answers: **which parts will run short, what is the cheapest fix, and who should approve it?**

## Scope

In scope: probabilistic device demand, part-level shortage risk, buy and expedite decisions, supplier-delay scenarios, and an approval workflow for the resulting actions.

Not in scope: finished-goods inventory, multi-site allocation, supplier capacity limits, a purchasing budget constraint, and any connection to real systems. All data is synthetic.

## Decisions

### 1. Forecast devices, decide on parts

Demand is forecast per device (6 series) and pushed through the bill of materials, rather than forecasting each part directly. Part demand is derived demand: forecasting it directly throws away the information that the modem moves with four devices at once, and it breaks every time the BOM changes.

### 2. Quantile gradient boosting, one global model per quantile

**Chosen:** LightGBM with quantile loss, one model per quantile (P5 to P95), trained across all devices, with months-ahead as a feature (direct multi-horizon).

**Considered:**
- *Per-device ETS/ARIMA.* Strong baseline for smooth series, but it can't use the sales pipeline, and six short series give each model very little data.
- *Prophet.* Easy seasonality, but no natural way to use a leading signal like weighted pipeline, and uncertainty intervals are not calibrated to the loss we care about.
- *DeepAR / Temporal Fusion Transformer.* Right family at scale (thousands of SKUs), but overkill for 6 series × 48 months and much harder to explain to planners.
- *Point forecast plus normal errors.* Simple, but B2B fleet demand is lumpy and skewed. The upper tail is what drives buys, and a symmetric assumption gets it wrong.

**Why:** tabular data, few series, strong exogenous signals. Boosted trees use the pipeline directly, train in seconds, and quantile loss gives the range without a distribution assumption.

### 3. Features relative to each device's level

Target and features are in log space relative to the device's 12-month average. One model then serves a 400-unit sensor and a 12,000-unit dashcam, and the trees can follow a growing product. A first version used absolute levels: it under-forecast growing devices by 18% because trees cannot extrapolate. The relative version cut bias to −7%.

### 4. The sales pipeline earns its place

Feature ablation on the same backtest (`reports/evaluation.md`):

| Variant | WAPE |
|---|---|
| All features | 20.4% |
| Without sales pipeline | 35.5% |
| Without trend | 22.2% |
| Without same month last year | 21.2% |
| Without calendar month | 20.4% |
| History only | 37.5% |

The pipeline is the signal. Calendar month adds nothing here and is a candidate to drop; I kept it because real hardware demand has fiscal-year effects the synthetic data doesn't fully capture.

### 5. Calibrate the ranges, then check them

Raw quantile models were overconfident: P10–P90 covered 68% of actuals instead of 80%. I fit one spread factor around the median on 8 earlier origins and checked it on the 8 later origins it never saw: coverage 87%.

Per-quantile reliability shows the fix is uneven. The upper tail is now good (P90 → 89%, P95 → 94%), which is the side that drives buy decisions. The lower tail is now too wide (P10 → 2%). **Next step:** fit separate factors for each tail (conformalized quantile regression). I left it for now because it changes the downstream numbers and the upper tail is the one that matters for shortages.

### 6. Simulate shortages instead of using a safety-stock formula

The textbook formula assumes independent normal demand and ignores shared parts and late deliveries. The prototype simulates 3,000 futures: demand drawn from the forecast quantiles with correlation across months (ρ = 0.6, forecast errors persist), and each open PO arriving on time or 1–2 months late according to the supplier's on-time rate. Output is the probability each part runs short and the month it most often first does.

### 7. Newsvendor buy rule per part

**Chosen:** buy up to the critical-ratio percentile of 6-month need, CR = Cu / (Cu + Co).
- Cu: share of short orders lost × device margin per part + share expedited × expedite premium.
- Co: unit cost × (holding rate × months + obsolescence).

**Considered:** a joint optimization (MILP) across all parts with a budget and supplier capacity. It's the right next step, but the per-part rule ties buying to costs the business already understands, and it's transparent enough for a planner to challenge. The three cost inputs are sliders because they are business decisions, not model outputs.

### 8. Timing is not quantity

The first version recommended "buy nothing" for a part with an 84% chance of running short. Enough was on order; it just landed too late. A quantity rule can't see that. The plan now:
1. Sets the buy quantity with the newsvendor rule.
2. If risk is still 40%+, pulls open POs one month earlier, 25% of a PO at a time, until risk is under 15%.

The 40% / 15% thresholds are judgment calls: expediting costs a 20–45% premium, so I don't pay it for watch-list parts. In production I'd set them with planners, per part class.

### 9. Supplier shocks are scored in margin, with late cost made explicit

The scenario reruns a month-by-month build plan with backlog. Options (do nothing, reallocate to highest-margin devices, expedite, alternate source, combinations) are scored as margin protected minus cost, where a late shipment costs a set share of its margin. That share is an explicit input: at 0% the free option (reallocate) wins; at 15% expedite plus reallocate wins. Making that trade-off visible is the point.

### 10. Tiered autonomy, not full autonomy

The agent executes only free, reversible actions (flag, alert, notify account teams). Spend goes to a planner up to a limit; anything above it, or strategic (qualifying a second source), goes to a director. Every action is logged. This is the same pattern I used for late-shipment exceptions in production, and published.

The briefing is generated from templates so every number traces back to a calculation. An LLM would only draft the wording; it never makes a decision with cost attached.

### 11. Inference in the browser, training in CI

The trained trees are exported and evaluated in the browser, which makes the what-if ("add a 12,000-unit deal") instant and the demo free to host. A test checks the browser engine against LightGBM's own predictions on every run (max difference under 1e-5).

Trade-off: no backend means no live data, no access control and no audit storage. In production the decision engine would be a service and the models would sit in a registry.

### 12. A pipeline that can refuse to ship

Every push and every month: build data → train → tests → ablation → quality gate → export → deploy. The gate releases a model only if it beats the best naive baseline by 10%+, absolute bias is under 15%, and P10–P90 coverage is 70–95%. The first CI run failed on a missing dependency; deploy was skipped and the live site was untouched, which is the behavior I want.

## Evaluation summary

288 forecasts (8 rolling origins × 6 horizons × 6 devices): WAPE 20% vs 43% for same month last year and 45% for a 3-month average; bias −7%. Full tables in `reports/evaluation.md`.

## Known limitations

- Synthetic data. Structure is realistic (growth, seasonality, lumpy fleet deals, shared parts, single-source suppliers, late deliveries); the numbers are not.
- The environmental sensor is intermittent and the model loses to seasonal naive (61% vs 57% WAPE). It should get a Croston/TSB-style model or be forecast at a higher level.
- Lower-tail calibration is too wide (see 5).
- No budget or capacity constraints in the buy plan; no multi-site allocation.
- Monthly buckets. Real planning runs weekly near-term.

## What I'd do with real data

1. **Weeks 1–4:** connect orders, shipments, BOMs and PO commit dates; reproduce numbers Finance already trusts before any recommendation reaches a planner.
2. **Weeks 4–8:** ship the forecast and part-risk view for the single-source, long-lead parts only. Track overrides with reason codes.
3. **After that:** weekly buckets, supplier lateness learned per supplier and lane, a budget-constrained buy optimization, and the agent's low-risk actions switched on once planners trust the recommendations.

## Questions I'd ask the team

- Which decision hurts most today: what to buy, what to expedite, or who ships first?
- How are supplier commit dates captured, and how often do they slip?
- What does a late shipment actually cost: delayed subscription revenue, penalties, churn risk?
- Who approves spend today, and at what thresholds?

## How it was built

Python (pandas, LightGBM) for data and training, plain JavaScript for the app, GitHub Actions for CI. I used Claude Code to write most of the implementation; I set the approach, the decision rules and the thresholds, and reviewed and tested the output. Issues I caught along the way: the absolute-level model's growth bias (3), the "buy nothing at 84% risk" timing gap (8), unrealistic starting inventory, and a missing dependency caught by CI (12).
