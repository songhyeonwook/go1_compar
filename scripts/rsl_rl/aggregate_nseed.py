#!/usr/bin/env python3
"""Aggregate biomech metrics across n seeds -> mean +/- std (paper §4.11).

For each seed's biomech npz, compute the per-policy scalars:
  - GRF reduction % (mean over the four injured limbs)
  - SI eq.7 % (mean)
  - direction-of-change agreement % (literature-grounded signs, from biomech_analyze)
then report mean +/- std (and 95% bootstrap CI) across seeds.

Usage: python3 aggregate_nseed.py seed42=biomech/ns_42.npz seed43=biomech/ns_43.npz ...
"""
import sys
import numpy as np
from biomech_analyze import analyze, REL, LEFT_LEGS


def si_eq7(xh, xa):
    return (xh - xa) / (0.5 * (xh + xa) + 1e-9) * 100.0


def per_seed(npz):
    agg = analyze(npz)
    if 0 not in agg:
        return None
    N = agg[0]
    reds, sis, hits, tot = [], [], 0, 0
    for cond in [1, 2, 3, 4]:
        if cond not in agg:
            continue
        a = agg[cond]; leg = cond - 1; r = REL[leg]
        reds.append((1 - a["peak"][leg] / max(N["peak"][leg], 1e-6)) * 100)
        sis.append(si_eq7(N["peak"][leg], a["peak"][leg]))
        side = +1.0 if leg in LEFT_LEGS else -1.0
        checks = [
            (a["peak"][leg] - N["peak"][leg], -1),                                   # affected GRF down
            (a["duty"][leg] - N["duty"][leg], +1),                                   # stance prolonged
            (a["impulse"][leg] - N["impulse"][leg], -1),                             # affected impulse down
            (a["impulse"][r["contra"]] - N["impulse"][r["contra"]], +1),             # contra up
            (a["impulse"][r["ipsi"]] - N["impulse"][r["ipsi"]], +1),                 # ipsi up
            ((a["base_y"] - N["base_y"]) * (-side), +1),                             # CoM to intact
        ]
        hits += sum(1 for v, e in checks if np.sign(v) == e)
        tot += len(checks)
    return dict(grf=float(np.mean(reds)), si=float(np.mean(sis)),
                doc=100.0 * hits / tot if tot else float("nan"))


def ci95(x):
    x = np.asarray(x, float)
    if len(x) < 2:
        return (float("nan"), float("nan"))
    bs = [np.mean(np.random.choice(x, len(x), replace=True)) for _ in range(2000)]
    return (float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)))


def main():
    items = []
    for arg in sys.argv[1:]:
        name, path = arg.split("=", 1)
        items.append((name, path))
    rows = {}
    print(f"{'seed':10} {'GRFred%':>8} {'SI(eq7)%':>9} {'direction%':>11}")
    for name, path in items:
        m = per_seed(path)
        if m is None:
            print(f"{name:10}  (no Normal baseline — skip)"); continue
        rows[name] = m
        print(f"{name:10} {m['grf']:8.1f} {m['si']:9.1f} {m['doc']:11.1f}")
    if len(rows) >= 2:
        print("-" * 42)
        for key, lab in [("grf", "GRF reduction %"), ("si", "SI (eq.7) %"), ("doc", "direction-of-change %")]:
            vals = [r[key] for r in rows.values()]
            lo, hi = ci95(vals)
            print(f"  {lab:24}: {np.mean(vals):.1f} ± {np.std(vals, ddof=1):.1f}  "
                  f"(n={len(vals)}, 95% CI [{lo:.1f}, {hi:.1f}])")


if __name__ == "__main__":
    main()
