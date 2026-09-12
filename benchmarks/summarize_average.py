#!/usr/bin/env python3
"""Summarize all recorded blocks; never drop outliers. Ratios are paired by round."""
import argparse
import csv
import random
import statistics as st
from collections import defaultdict

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("csv")
args = parser.parse_args()
with open(args.csv, encoding="utf-8-sig", newline="") as f:
    rows = list(csv.DictReader(f))
groups = defaultdict(dict)
for r in rows:
    name, round_id = r["backend"], int(r["round"])
    if round_id in groups[name]:
        raise ValueError("duplicate round/backend")
    groups[name][round_id] = float(r["per_call_ms"])
def quantile(values, p):
    values = sorted(values)
    x = (len(values) - 1) * p
    i = int(x)
    return values[i] + (values[min(i + 1, len(values) - 1)] - values[i]) * (x - i)

print("backend,blocks,median_ms,p10_ms,p90_ms,MAD_percent")
for name, blocks in groups.items():
    values = list(blocks.values())
    median = st.median(values)
    mad = st.median(abs(x - median) for x in values)
    print(f"{name},{len(values)},{median:.6f},{quantile(values,.1):.6f},{quantile(values,.9):.6f},{100*mad/median:.2f}")
baseline = groups.get("upstream-avx2")
if baseline:
    print("backend,paired_median_ratio_to_upstream,bootstrap_low,bootstrap_high")
    rng = random.Random(9721)
    for name, blocks in groups.items():
        if name == "upstream-avx2":
            continue
        if blocks.keys() != baseline.keys():
            raise ValueError("incomplete paired rounds")
        ratios = [blocks[i] / baseline[i] for i in sorted(blocks)]
        # Exploratory percentile bootstrap, not proof that OS noise is independent.
        boot = [st.median(rng.choices(ratios, k=len(ratios))) for _ in range(5000)]
        print(f"{name},{st.median(ratios):.4f},{quantile(boot,.025):.4f},{quantile(boot,.975):.4f}")
