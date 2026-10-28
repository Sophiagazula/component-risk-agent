/*
 * Pure calculation functions shared by the app and the tests.
 * No DOM access here, so `node --test tests/` can exercise them directly.
 */
(function (root) {
  "use strict";

  // Interpolate a value at probability u from a set of quantile knots (qs at probabilities QK).
  function qAt(qs, u, QK) {
    u = Math.min(0.995, Math.max(0.005, u));
    const n = QK.length - 1;
    if (u <= QK[0]) return Math.max(0, qs[0] - (QK[0] - u) * (qs[1] - qs[0]) / (QK[1] - QK[0]));
    if (u >= QK[n]) return qs[n] + (u - QK[n]) * (qs[n] - qs[n - 1]) / (QK[n] - QK[n - 1]);
    for (let i = 1; i <= n; i++) if (u <= QK[i]) return qs[i - 1] + (u - QK[i - 1]) * (qs[i] - qs[i - 1]) / (QK[i] - QK[i - 1]);
    return qs[n];
  }

  // Seeded random numbers, so every simulation run uses the same draws.
  function mulberry32(a) {
    return () => {
      a |= 0; a = a + 0x6D2B79F5 | 0;
      let t = Math.imul(a ^ a >>> 15, 1 | a);
      t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
      return ((t ^ t >>> 14) >>> 0) / 4294967296;
    };
  }

  // Standard normal CDF (Abramowitz-Stegun 26.2.17).
  function Phi(x) {
    const t = 1 / (1 + 0.2316419 * Math.abs(x)), d = 0.3989423 * Math.exp(-x * x / 2);
    const p = d * t * (0.3193815 + t * (-0.3565638 + t * (1.781478 + t * (-1.821256 + t * 1.330274))));
    return x > 0 ? 1 - p : p;
  }

  // Sum of one LightGBM ensemble for feature vector x (trees exported by src/model.py:compact).
  function walk(trees, x) {
    let s = 0;
    for (const t of trees) {
      let n = t.root;
      while (n >= 0) { const f = x[t.f[n]]; n = (t.c[n] ? t.c[n].includes(f) : f <= t.t[n]) ? t.l[n] : t.r[n]; }
      s += t.v[~n];
    }
    return s;
  }

  // All quantiles for one device-month: sort, widen around the median by the calibration scale, back to units.
  function predictQuantiles(model, x, base) {
    const lq = model.models.map(trees => walk(trees, x) + base).sort((a, b) => a - b);
    const mid = lq[Math.floor(lq.length / 2)];
    return lq.map(v => Math.max(0, Math.expm1(mid + (v - mid) * model.scale)));
  }

  // Newsvendor: cost of being short per unit (cu), cost of holding per unit (co), critical ratio.
  function criticalRatio({ pLost, marginPerUnit, unitCost, expeditePremium, holdRate, months, obsolescence }) {
    const cu = pLost * marginPerUnit + (1 - pLost) * unitCost * expeditePremium;
    const co = unitCost * (holdRate * months / 12 + obsolescence);
    return { cu, co, cr: cu / (cu + co) };
  }

  // Value at quantile q of an already-sorted sample.
  function sortedQuantile(arr, q) { return arr[Math.min(arr.length - 1, Math.floor(q * arr.length))]; }

  // Round an order up to a multiple of the supplier's minimum order quantity.
  function roundUpToMoq(q, moq) { return q > 0 ? Math.ceil(q / moq) * moq : 0; }

  const Engine = { qAt, mulberry32, Phi, walk, predictQuantiles, criticalRatio, sortedQuantile, roundUpToMoq };
  root.Engine = Engine;
  if (typeof module === "object" && module.exports) module.exports = Engine;
})(typeof window !== "undefined" ? window : globalThis);
