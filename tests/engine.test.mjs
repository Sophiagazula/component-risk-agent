// Tests for app/engine.js, the calculations the web app runs. Run: node --test tests/engine.test.mjs
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { createRequire } from "node:module";

const require = createRequire(import.meta.url);
const Engine = require("../app/engine.js");
const root = new URL("../", import.meta.url);
const QK = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95];

test("browser tree walk matches LightGBM predictions", () => {
  const model = JSON.parse(readFileSync(new URL("models/forecast_model.json", root)));
  const fx = JSON.parse(readFileSync(new URL("models/parity_fixture.json", root)));
  let worst = 0;
  model.models.forEach((trees, j) => fx.X.forEach((x, i) => {
    worst = Math.max(worst, Math.abs(Engine.walk(trees, x) - fx.expected[j][i]));
  }));
  assert.ok(worst < 1e-5, `max difference ${worst}`);
});

test("predicted quantiles are ordered and non-negative", () => {
  const model = JSON.parse(readFileSync(new URL("models/forecast_model.json", root)));
  const fx = JSON.parse(readFileSync(new URL("models/parity_fixture.json", root)));
  for (const x of fx.X) {
    const q = Engine.predictQuantiles(model, x, 8.5);
    q.forEach((v, i) => { assert.ok(v >= 0); if (i) assert.ok(v >= q[i - 1]); });
  }
});

test("quantile interpolation hits the knots and is monotonic", () => {
  const qs = [10, 20, 40, 60, 80, 100, 120];
  QK.forEach((u, i) => assert.ok(Math.abs(Engine.qAt(qs, u, QK) - qs[i]) < 1e-9));
  let prev = -Infinity;
  for (let u = 0.001; u < 1; u += 0.01) { const v = Engine.qAt(qs, u, QK); assert.ok(v >= prev); assert.ok(v >= 0); prev = v; }
});

test("critical ratio responds to the business inputs the right way", () => {
  const base = { pLost: 0.15, marginPerUnit: 400, unitCost: 2.5, expeditePremium: 0.3, holdRate: 0.25, months: 6, obsolescence: 0.1 };
  const cr = x => Engine.criticalRatio({ ...base, ...x }).cr;
  assert.ok(cr({}) > 0 && cr({}) < 1);
  assert.ok(cr({ pLost: 0.4 }) > cr({}), "more lost orders -> buy deeper");
  assert.ok(cr({ obsolescence: 0.3 }) < cr({}), "more obsolescence -> buy less");
  assert.ok(cr({ unitCost: 50 }) < cr({}), "expensive part -> tighter buffer");
});

test("orders round up to the minimum order quantity", () => {
  assert.equal(Engine.roundUpToMoq(0, 5000), 0);
  assert.equal(Engine.roundUpToMoq(1, 5000), 5000);
  assert.equal(Engine.roundUpToMoq(10001, 5000), 15000);
});

test("simulation random numbers are reproducible", () => {
  const a = Engine.mulberry32(42), b = Engine.mulberry32(42);
  for (let i = 0; i < 100; i++) assert.equal(a(), b());
});

test("normal CDF is accurate", () => {
  assert.ok(Math.abs(Engine.Phi(0) - 0.5) < 1e-6);
  assert.ok(Math.abs(Engine.Phi(1.2816) - 0.9) < 1e-4);
  assert.ok(Math.abs(Engine.Phi(-1.6449) - 0.05) < 1e-4);
});
