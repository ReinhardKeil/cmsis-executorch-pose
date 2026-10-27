#!/usr/bin/env python3
# Copyright 2026 Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Compare pose SDS output with a reference using separate value tolerances."""

import argparse
import math
import struct
from pathlib import Path


JOINT_COUNT = 17
VALUES_PER_JOINT = 3
RECORD_BYTES = JOINT_COUNT * VALUES_PER_JOINT * 4


def read_records(path: Path) -> dict[int, tuple[float, ...]]:
    records = {}
    with path.open("rb") as stream:
        while header := stream.read(8):
            if len(header) != 8:
                raise ValueError(f"{path}: truncated SDS record header")
            timeslot, size = struct.unpack("<II", header)
            payload = stream.read(size)
            if len(payload) != size:
                raise ValueError(f"{path}: truncated payload at timeslot {timeslot}")
            if size != RECORD_BYTES:
                raise ValueError(
                    f"{path}: expected {RECORD_BYTES} bytes at timeslot {timeslot}, got {size}"
                )
            if timeslot in records:
                raise ValueError(f"{path}: duplicate timeslot {timeslot}")
            records[timeslot] = struct.unpack(f"<{JOINT_COUNT * VALUES_PER_JOINT}f", payload)
    if not records:
        raise ValueError(f"{path}: no pose records")
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("actual", type=Path)
    parser.add_argument("reference", type=Path)
    parser.add_argument("--coordinate-atol", type=float, default=0.02)
    parser.add_argument("--confidence-atol", type=float, default=0.05)
    args = parser.parse_args()

    actual = read_records(args.actual)
    reference = read_records(args.reference)
    if actual.keys() != reference.keys():
        raise SystemExit(
            f"timeslots differ: actual={sorted(actual)} reference={sorted(reference)}"
        )

    max_coordinate_error = 0.0
    max_confidence_error = 0.0
    failures = []
    for timeslot in sorted(reference):
        for index, (value, expected) in enumerate(zip(actual[timeslot], reference[timeslot])):
            if not math.isfinite(value):
                failures.append(f"timeslot {timeslot}, value {index}: non-finite {value}")
                continue
            is_confidence = index % VALUES_PER_JOINT == 2
            error = abs(value - expected)
            tolerance = args.confidence_atol if is_confidence else args.coordinate_atol
            if is_confidence:
                max_confidence_error = max(max_confidence_error, error)
            else:
                max_coordinate_error = max(max_coordinate_error, error)
            if error > tolerance and len(failures) < 10:
                kind = "confidence" if is_confidence else "coordinate"
                failures.append(
                    f"timeslot {timeslot}, joint {index // VALUES_PER_JOINT} {kind}: "
                    f"actual={value:.6f}, reference={expected:.6f}, error={error:.6f}"
                )

    if failures:
        raise SystemExit("pose reference comparison failed:\n  " + "\n  ".join(failures))
    print(
        f"PASS: {len(actual)} frames; max coordinate error={max_coordinate_error:.6f} "
        f"(limit {args.coordinate_atol:.6f}); max confidence error={max_confidence_error:.6f} "
        f"(limit {args.confidence_atol:.6f})"
    )


if __name__ == "__main__":
    main()
