"""
Release gate: a newly trained model is only published if it beats the
baselines and its intervals are calibrated. Writes a model card either way
and exits non-zero on failure, which stops the deployment in CI.
"""
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
s = json.loads((ROOT / "data" / "backtest_summary.json").read_text())
o = s["overall"]

best_baseline = min(o["wape_seasonal_naive"], o["wape_moving_avg"])
checks = [
    ("Beats best baseline by 10%+ (WAPE)", o["wape_model"] <= 0.9 * best_baseline,
     f"{o['wape_model']:.3f} vs {best_baseline:.3f}"),
    ("Absolute bias under 15%", abs(o["bias_model"]) < 0.15, f"{o['bias_model']:+.3f}"),
    ("P10-P90 coverage between 70% and 95%", 0.70 <= o["coverage_p10_p90"] <= 0.95, f"{o['coverage_p10_p90']:.2f}"),
]
passed = all(ok for _, ok, _ in checks)

card = {
    "model": "LightGBM quantile regression, global across devices, direct 1-6 month horizon",
    "quantiles": [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95],
    "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    "git_commit": os.environ.get("GITHUB_SHA", "local"),
    "backtest": {"forecasts": o["n"], "origins": s["backtest_origins"], "wape": o["wape_model"],
                 "wape_seasonal_naive": o["wape_seasonal_naive"], "wape_moving_avg": o["wape_moving_avg"],
                 "bias": o["bias_model"], "coverage_raw": o["coverage_raw"], "coverage_calibrated": o["coverage_p10_p90"]},
    "interval_scale": s["final_interval_scale"],
    "gate": [{"check": n, "passed": bool(ok), "value": v} for n, ok, v in checks],
    "released": passed,
}
(ROOT / "models").mkdir(exist_ok=True)
(ROOT / "models" / "model_card.json").write_text(json.dumps(card, indent=2))

for n, ok, v in checks:
    print(f"{'PASS' if ok else 'FAIL'}  {n}: {v}")
if not passed:
    print("Quality gate failed: model not released.")
    sys.exit(1)
print("Quality gate passed: model released.")
