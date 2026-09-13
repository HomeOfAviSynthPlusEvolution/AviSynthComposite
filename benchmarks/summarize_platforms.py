#!/usr/bin/env python3
"""Refresh the generated part of a cross-platform Markdown report, preserving notes."""
import argparse
import csv
import statistics
from pathlib import Path

KEY = ("workload", "bits", "step", "mask", "width", "height", "opacity", "layout")
BEGIN = "<!-- BEGIN PLATFORM MATRIX -->"
END = "<!-- END PLATFORM MATRIX -->"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--platform", action="append", required=True, metavar="LABEL=CSV")
    parser.add_argument("--missing", action="append", default=[])
    parser.add_argument("--compare", action="append", default=[], metavar="NUMERATOR/DENOMINATOR",
                        help="Add a direct comparison, for example FreeBSD-EPYC/Linux-EPYC")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    platforms = []
    for item in args.platform:
        label, filename = item.split("=", 1)
        with Path(filename).open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        data = {}
        for row in rows:
            if row["target"] == "0":
                continue
            key = tuple(row[k] for k in KEY)
            if key in data:
                raise SystemExit(f"{label}: multiple backends for one case; use the native-only collection")
            data[key] = row
        platforms.append((label, data))
    reference_label, reference = platforms[0]
    by_label = dict(platforms)
    comparisons = [(label, reference_label) for label, _ in platforms[1:]]
    for value in args.compare:
        pair = value.split("/")
        if len(pair) != 2 or any(label not in by_label for label in pair) or pair[0] == pair[1]:
            parser.error("comparison requires two distinct existing platform labels")
        if tuple(pair) not in comparisons:
            comparisons.append(tuple(pair))
    medians = {}
    for numerator, denominator in comparisons:
        data, base = by_label[numerator], by_label[denominator]
        ratios = [float(data[k]["median_ms"]) / float(base[k]["median_ms"])
                  for k in base.keys() & data.keys()]
        medians[numerator, denominator] = statistics.median(ratios) if ratios else None
    text = [BEGIN, "", "计时为毫秒，每项5次计时的中位数；空缺不按0计算。各平台采用自身默认后端。",
            "不同硬件的绝对差距不直接视为缺陷；候选异常需结合整体差距、共同后端和指令检查。", "",
            "| 平台 | 已测同条件项目 | native target | 相对首列平台的耗时比中位数 |",
            "|---|---:|---|---:|"]
    for index, (label, data) in enumerate(platforms):
        targets = ", ".join(sorted({r["target"] for r in data.values()}))
        median = 1.0 if index == 0 else medians[label, reference_label]
        ratio = f"{median:.3f}" if median is not None else "—"
        text.append(f"| {label} | {len(data)} | {targets} | {ratio} |")
    for label in args.missing:
        text.append(f"| {label} | — | — | — |")
    keys = sorted(set().union(*(d.keys() for _, d in platforms)))
    for workload in sorted({k[0] for k in keys}):
        text.extend(["", f"### {workload}", ""])
        columns = ["类型 / 布局 / 尺寸", "mask", "opacity"] + [p[0] + " ms" for p in platforms] + args.missing
        columns += [f"{numerator}/{denominator}" for numerator, denominator in comparisons]
        text.extend(["| " + " | ".join(columns) + " |", "|" + "|".join("---" for _ in columns) + "|"])
        for key in (k for k in keys if k[0] == workload):
            _, bits, step, mask, width, height, opacity, layout = key
            datatype = "F32" if bits == "32" else "U" + bits
            fields = [f"{datatype} / step{step} {layout} / {width}×{height}", mask, opacity]
            fields += [f"{float(data[key]['median_ms']):.4f}" if key in data else "—" for _, data in platforms]
            fields += ["—"] * len(args.missing)
            for numerator, denominator in comparisons:
                data, base = by_label[numerator], by_label[denominator]
                if key not in base or key not in data:
                    fields.append("—")
                    continue
                a, b = float(base[key]["median_ms"]), float(data[key]["median_ms"])
                ratio = b / a
                normalized = ratio / medians[numerator, denominator]
                candidate = abs(a-b) >= .05 and (normalized >= 1.5 or normalized <= 1/1.5)
                value = f"{ratio:.2f}×"
                fields.append(f"**{value}**" if candidate else value)
            text.append("| " + " | ".join(fields) + " |")
    text.extend(["", "粗体只标记候选：相对该平台整体耗时比再偏离至少1.5倍，且绝对差至少0.05ms；不是已确认缺陷。", "", END])
    generated = "\n".join(text)
    dest = Path(args.output)
    old = dest.read_text(encoding="utf-8") if dest.exists() else "# 跨硬件软件平台性能对照\n\n"
    if BEGIN in old and END in old:
        old = old[:old.index(BEGIN)] + generated + old[old.index(END)+len(END):]
    else:
        old = old.rstrip() + "\n\n" + generated + "\n"
    dest.write_text(old, encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
