#!/usr/bin/env python3
"""Independently recompute and validate all 7,820 E3 certificates.

This validator does not import the main E3 program.  It enumerates subsets in
independent bit-mask batches and applies the unspecialized Bott--Kunneth rule
directly to every twist.  The main program instead constructs its subset table
recursively and uses the CICY-specialized Bott conditions.  Agreement therefore
compares two independent implementations of the same Koszul bounds.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


HERE = Path(__file__).resolve().parent
PACKAGE_INPUT = HERE.parent.parent / "data/cicy_criteria_input_7820.jsonl"
DEFAULT_INPUT = PACKAGE_INPUT
DEFAULT_OUTPUT_DIRECTORY = HERE.parent.parent / "reproduction_output"
DEFAULT_CERTIFICATES = DEFAULT_OUTPUT_DIRECTORY / "e3_certificates_7820.jsonl"
DEFAULT_SUMMARY = DEFAULT_OUTPUT_DIRECTORY / "e3_summary_7820.json"
DEFAULT_VALIDATION = DEFAULT_OUTPUT_DIRECTORY / "e3_independent_validation_7820.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def independent_rows(
    conf: list[list[int]], batch_size: int = 8192
) -> list[dict[str, Any]]:
    dimensions = [sum(row) - 1 for row in conf]
    columns = np.asarray(list(zip(*conf)), dtype=np.int16)
    column_count, row_count = columns.shape
    subset_count = 1 << column_count
    bit_positions = np.arange(column_count, dtype=np.uint64)
    U = np.zeros(row_count, dtype=np.int64)
    V = np.zeros(row_count, dtype=np.int64)

    for start in range(0, subset_count, batch_size):
        stop = min(start + batch_size, subset_count)
        masks = np.arange(start, stop, dtype=np.uint64)
        bits = ((masks[:, None] >> bit_positions[None, :]) & 1).astype(np.int16)
        sizes = bits.sum(axis=1, dtype=np.int16)
        degree_sums = bits @ columns

        for target in range(row_count):
            valid = np.ones(stop - start, dtype=bool)
            cohomological_degree = np.zeros(stop - start, dtype=np.int16)
            term_dimension = np.ones(stop - start, dtype=np.int64)
            for ambient_row, n in enumerate(dimensions):
                twists = -degree_sums[:, ambient_row]
                if ambient_row == target:
                    twists = twists + 1
                degree_zero = twists >= 0
                degree_top = twists <= -n - 1
                local_valid = degree_zero | degree_top
                valid &= local_valid
                cohomological_degree += degree_top.astype(np.int16) * n

                local_dimension = np.zeros(stop - start, dtype=np.int64)
                for k in np.unique(twists[local_valid]):
                    k_int = int(k)
                    if k_int >= 0:
                        dimension = math.comb(n + k_int, n)
                    else:
                        dimension = math.comb(-k_int - 1, n)
                    local_dimension[twists == k] = dimension
                term_dimension *= local_dimension

            u_mask = valid & (cohomological_degree == sizes)
            v_mask = valid & (sizes >= 1) & (cohomological_degree == sizes - 1)
            U[target] += term_dimension[u_mask].sum(dtype=np.int64)
            V[target] += term_dimension[v_mask].sum(dtype=np.int64)

    rows: list[dict[str, Any]] = []
    for target in range(row_count):
        ambient_h0 = dimensions[target] + 1
        U_i = int(U[target])
        V_i = int(V[target])
        rows.append(
            {
                "row_1based": target + 1,
                "ambient_dimension": dimensions[target],
                "ambient_h0_dimension": ambient_h0,
                "U_i": U_i,
                "V_i": V_i,
                "restriction_isomorphism_certified": (
                    U_i == ambient_h0 and V_i == 0
                ),
            }
        )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--certificates", type=Path, default=DEFAULT_CERTIFICATES)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--output", type=Path, default=DEFAULT_VALIDATION)
    parser.add_argument("--expected-count", type=int, default=7820)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = args.input.resolve()
    certificates_path = args.certificates.resolve()
    summary_path = args.summary.resolve()
    output_path = args.output.resolve()

    dataset = read_jsonl(input_path)
    certificates = read_jsonl(certificates_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if len(dataset) != args.expected_count or len(certificates) != args.expected_count:
        raise ValueError("input or certificate count differs from expected count")
    dataset_by_num = {int(row["Num"]): row for row in dataset}
    certificate_by_num = {int(row["Num"]): row for row in certificates}
    if len(dataset_by_num) != args.expected_count or len(certificate_by_num) != args.expected_count:
        raise ValueError("duplicate CICY numbers in input or certificates")
    if set(dataset_by_num) != set(certificate_by_num):
        raise ValueError("input and certificate CICY-number sets differ")
    if summary["certificates"]["sha256"] != sha256_file(certificates_path):
        raise ValueError("certificate hash differs from main summary")
    if summary["input"]["sha256"] != sha256_file(input_path):
        raise ValueError("input hash differs from main summary")

    mismatch_count = 0
    independently_passed = 0
    relation_counts: Counter[str] = Counter()
    row_count = 0
    first_mismatches: list[dict[str, Any]] = []

    for position, num in enumerate(sorted(dataset_by_num), start=1):
        record = dataset_by_num[num]
        saved = certificate_by_num[num]
        recomputed = independent_rows(record["working_conf"])
        row_count += len(recomputed)
        relation = record.get("presentation_relation")
        if relation is None:  # backward-compatible input reader
            relation = record.get("topology_audit", {}).get("fixed_X_relation")
        relation_counts[str(relation)] += 1
        expected_pass = all(
            row["restriction_isomorphism_certified"] for row in recomputed
        )
        independently_passed += int(expected_pass)
        reasons: list[str] = []
        if saved["row_certificates"] != recomputed:
            reasons.append("row_certificates")
        if bool(saved["E3_certified"]) != expected_pass:
            reasons.append("E3_certified")
        if saved.get("canonical_hash_working") != record.get("canonical_hash_working"):
            reasons.append("canonical_hash_working")
        if reasons:
            mismatch_count += 1
            if len(first_mismatches) < 20:
                first_mismatches.append({"Num": num, "reasons": reasons})
        if position % 100 == 0 or position == len(dataset_by_num):
            print(
                f"validated {position}/{len(dataset_by_num)} configurations; "
                f"mismatches={mismatch_count}",
                flush=True,
            )

    validation = {
        "audit_id": "independent_E3_validation_all_7820_20260920",
        "status": "pass" if mismatch_count == 0 else "fail",
        "implementation_independence": (
            "The main program constructs a recursive subset table and uses the "
            "CICY-specialized Bott conditions; this validator enumerates subsets "
            "in independent bit-mask batches and applies the unspecialized "
            "Bott--Kunneth formula directly."
        ),
        "configuration_count": len(dataset),
        "row_test_count": row_count,
        "independently_E3_certified_count": independently_passed,
        "mismatch_count": mismatch_count,
        "first_mismatches": first_mismatches,
        "working_relation_counts": dict(sorted(relation_counts.items())),
        "input_sha256": sha256_file(input_path),
        "certificates_sha256": sha256_file(certificates_path),
        "main_summary_sha256": sha256_file(summary_path),
        "validator_sha256": sha256_file(Path(__file__).resolve()),
        "special_defining_polynomial_used": False,
    }
    output_path.write_text(
        json.dumps(validation, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(validation, ensure_ascii=False, indent=2, sort_keys=True))
    return 1 if mismatch_count else 0


if __name__ == "__main__":
    raise SystemExit(main())
