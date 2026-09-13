#!/usr/bin/env python3
"""Collect a small, resumable cross-platform sample using the production benchmark."""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import platform
import subprocess
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bench", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--workload", action="append", help="Restrict a diagnostic follow-up")
    parser.add_argument("--all-targets", action="store_true")
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    metadata = {"label": args.label, "revision": args.revision,
                "system": platform.platform(), "machine": platform.machine(),
                "benchmark": str(Path(args.bench).resolve()), "trials": 5,
                "all_targets": args.all_targets, "workloads": sorted(args.workload or []),
                "benchmark_sha256": hashlib.sha256(Path(args.bench).read_bytes()).hexdigest(),
                "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    workloads = ("mix", "overlay_mul", "sample420", "affine", "clamp",
                 "continuous_product", "continuous_add", "luma", "copy")
    if args.workload and any(w not in workloads for w in args.workload):
        parser.error("unknown workload")
    manifest = output / "metadata.json"
    if manifest.exists():
        previous = json.loads(manifest.read_text(encoding="utf-8"))
        identity = ("revision", "all_targets", "workloads", "benchmark_sha256")
        if any(previous.get(k) != metadata[k] for k in identity):
            parser.error("output belongs to a different build or collection; choose a new output directory")
    manifest.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    statuses = []
    combined = []
    for workload in workloads:
        if args.workload and workload not in args.workload:
            continue
        for bits in (8, 16, 32):
            for width, height, step in ((1920, 1080, 1), (641, 360, 4)):
                for opacity in ((".17", ".625") if workload in ("mix", "overlay_mul") else (".17",)):
                    key = f"{workload}-{bits}-{width}x{height}-s{step}-o{opacity}"
                    dest = output / (key + ".csv")
                    err = output / (key + ".err")
                    command = [args.bench, "--workload", workload, "--bits", str(bits),
                               "--width", str(width), "--height", str(height), "--step", str(step),
                               "--opacity", opacity, "--trials", "5"]
                    if args.all_targets:
                        command.append("--all-targets")
                    # Each completed job is durable. A lost Spot host needs no cleanup/retry.
                    done = output / (key + ".done")
                    if not (done.exists() and dest.exists()):
                        started = time.monotonic()
                        try:
                            result = subprocess.run(command, capture_output=True, text=True, timeout=180)
                            dest.write_text(result.stdout, encoding="utf-8")
                            err.write_text(result.stderr, encoding="utf-8")
                            status = {"case": key, "returncode": result.returncode,
                                      "elapsed_s": time.monotonic() - started}
                            if result.returncode == 0:
                                done.write_text("completed\n", encoding="utf-8")
                        except subprocess.TimeoutExpired:
                            status = {"case": key, "returncode": "timeout"}
                    else:
                        status = {"case": key, "returncode": 0, "cached": True}
                    statuses.append(status)
                    if status["returncode"] == 0:
                        combined.extend(csv.DictReader(io.StringIO(dest.read_text(encoding="utf-8"))))
                    (output / "status.json").write_text(json.dumps(statuses, indent=2) + "\n", encoding="utf-8")
                    print(json.dumps(status), flush=True)
    if combined:
        with (output / "matrix.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(combined[0]))
            writer.writeheader()
            writer.writerows(combined)
    return int(any(s["returncode"] != 0 for s in statuses))


if __name__ == "__main__":
    raise SystemExit(main())
