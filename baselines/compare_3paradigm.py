#!/usr/bin/env python3
"""3-paradigm comparison table (antalgic vs fault-tolerant vs symmetry).

For each paradigm's biomech npz, compute the per-policy scalars (GRF reduction %,
SI eq.7 %, direction-of-change agreement %) by reusing this repo's
scripts/rsl_rl/aggregate_nseed.per_seed(), and print the comparison table.

Expected (paper §2.3/§2.5): antalgic direction-of-change > 80%,
fault-tolerant < 40%, symmetry-encouraging < 50%.

Usage:
  python3 compare_3paradigm.py antalgic=<npz> faulttol=<npz> symmetry=<npz>
"""
import os
import sys

# reuse the antalgic metric code from this repo (portable, relative to this file)
HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.abspath(os.path.join(HERE, "..", "scripts", "rsl_rl"))
sys.path.insert(0, SCRIPTS)
from aggregate_nseed import per_seed  # noqa: E402


def main():
    items = []
    for arg in sys.argv[1:]:
        name, path = arg.split("=", 1)
        if not os.path.isabs(path):
            cand = os.path.join(SCRIPTS, path)
            path = cand if os.path.exists(cand) else path
        items.append((name, path))

    print("=" * 62)
    print("3-PARADIGM COMPARISON  (injured-animal biomechanics match)")
    print("=" * 62)
    print(f"{'paradigm':16} {'GRFred%':>8} {'SI(eq7)%':>9} {'direction%':>11}  expect")
    exp = {"antalgic": ">80", "faulttol": "<40", "symmetry": "<50"}
    for name, path in items:
        if not os.path.exists(path):
            print(f"{name:16}  (npz not found: {path})"); continue
        m = per_seed(path)
        if m is None:
            print(f"{name:16}  (no Normal baseline)"); continue
        e = exp.get(name.split("_")[0], "")
        print(f"{name:16} {m['grf']:8.1f} {m['si']:9.1f} {m['doc']:11.1f}  {e}")
    print("-" * 62)
    print("direction-of-change = sign match vs weight-bearing-lameness literature")
    print("(Weishaupt 2006; Fischer 2013; dog forelimb PMC3530583).")


if __name__ == "__main__":
    main()
