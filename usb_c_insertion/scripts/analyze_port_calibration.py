#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
from typing import Dict, List, Tuple


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze USB-C port calibration samples.")
    parser.add_argument(
        "path",
        nargs="?",
        default="/tmp/usb_c_insertion_port_calibration_samples.jsonl",
        help="JSONL sample file written by port_calibration_gui.py",
    )
    args = parser.parse_args()

    samples = _load_samples(args.path)
    if not samples:
        print("No samples found in %s" % args.path)
        return

    print("Samples: %d" % len(samples))
    _print_stats("Base error actual_port - estimated_port", _vectors(samples, "error_base"))
    _print_stats("Tool error actual_port - estimated_port", _vectors(samples, "error_tool"))


def _load_samples(path: str) -> List[Dict]:
    samples = []
    with open(path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                samples.append(json.loads(line))
            except ValueError as exc:
                raise ValueError("Invalid JSON in %s:%d: %s" % (path, line_number, exc))
    return samples


def _vectors(samples: List[Dict], key: str) -> List[Tuple[float, float, float]]:
    result = []
    for sample in samples:
        value = sample.get(key, {})
        result.append((float(value.get("x", 0.0)), float(value.get("y", 0.0)), float(value.get("z", 0.0))))
    return result


def _print_stats(label: str, vectors: List[Tuple[float, float, float]]) -> None:
    mean = tuple(sum(vector[index] for vector in vectors) / float(len(vectors)) for index in range(3))
    std = tuple(
        math.sqrt(sum((vector[index] - mean[index]) ** 2 for vector in vectors) / float(len(vectors)))
        for index in range(3)
    )
    norms = [math.sqrt(sum(component * component for component in vector)) for vector in vectors]
    mean_norm = sum(norms) / float(len(norms))
    max_norm = max(norms)
    print("")
    print(label)
    print("  mean_m:  x=% .6f y=% .6f z=% .6f" % mean)
    print("  mean_mm: x=% .3f y=% .3f z=% .3f" % tuple(component * 1000.0 for component in mean))
    print("  std_mm:  x=% .3f y=% .3f z=% .3f" % tuple(component * 1000.0 for component in std))
    print("  norm_mm: mean=% .3f max=% .3f" % (mean_norm * 1000.0, max_norm * 1000.0))


if __name__ == "__main__":
    main()
