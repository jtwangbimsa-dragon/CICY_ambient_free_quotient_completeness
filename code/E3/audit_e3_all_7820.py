#!/usr/bin/env python3
"""Compute coefficient-independent E3 certificates for all 7,820 CICY frames.

For every working configuration and every ambient factor i, the program
computes the two Koszul--Bott bounds

    U_i = sum_{p, |S|=p} h^p(A, O_A(e_i-q_S)),
    V_i = sum_{p>=1, |S|=p} h^(p-1)(A, O_A(e_i-q_S)).

The restriction map H^0(A,O_A(e_i)) -> H^0(X,O_X(e_i)) is certified to be
an isomorphism when U_i=n_i+1 and V_i=0.  All quantities depend only on the
working configuration matrix; no defining-polynomial coefficients are read.

The output has one JSON record per CICY and contains the complete row-by-row
evidence.  A failed row is written as unresolved and makes the process return
a nonzero exit status.
"""

from __future__ import annotations

import argparse
import hashlib
import json
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


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def validate_conf(conf: Any) -> list[list[int]]:
    if not isinstance(conf, list) or not conf or not isinstance(conf[0], list) or not conf[0]:
        raise ValueError("working_conf must be a nonempty matrix")
    width = len(conf[0])
    if any(not isinstance(row, list) or len(row) != width for row in conf):
        raise ValueError("working_conf must be rectangular")
    if any(
        not isinstance(value, int) or isinstance(value, bool) or value < 0
        for row in conf
        for value in row
    ):
        raise ValueError("working_conf entries must be nonnegative integers")
    return conf


def subset_tables(columns: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Enumerate all subsets exactly, returning sizes and multidegree sums."""
    column_count, row_count = columns.shape
    subset_count = 1 << column_count
    cardinalities = np.empty(subset_count, dtype=np.int16)
    degree_sums = np.empty((subset_count, row_count), dtype=np.int16)
    cardinalities[0] = 0
    degree_sums[0, :] = 0
    populated = 1
    for column in columns:
        next_populated = 2 * populated
        cardinalities[populated:next_populated] = cardinalities[:populated] + 1
        degree_sums[populated:next_populated, :] = degree_sums[:populated, :] + column
        populated = next_populated
    return cardinalities, degree_sums


def audit_configuration(record: dict[str, Any]) -> dict[str, Any]:
    num = int(record["Num"])
    conf = validate_conf(record["working_conf"])
    dimensions = [sum(row) - 1 for row in conf]
    if any(n < 1 for n in dimensions):
        raise ValueError(f"Num {num}: invalid ambient dimensions {dimensions}")
    if record.get("working_ambient_dimensions") != dimensions:
        raise ValueError(f"Num {num}: stored working dimensions do not match working_conf")

    columns = np.asarray(list(zip(*conf)), dtype=np.int16)
    for row, n in enumerate(dimensions):
        if int(columns[:, row].sum()) != n + 1:
            raise ValueError(f"Num {num}: Calabi--Yau row-sum condition fails at row {row+1}")

    cardinalities, degree_sums = subset_tables(columns)
    subset_count = len(cardinalities)
    row_certificates: list[dict[str, Any]] = []

    for target_row, n_target in enumerate(dimensions):
        valid = np.ones(subset_count, dtype=bool)
        cohomological_degrees = np.zeros(subset_count, dtype=np.int16)

        # This is the Bott rule specialized to a CICY row sum n_r+1.
        # On a non-target row O(-s) contributes only for s=0 or s=n_r+1.
        # On the target row O(1-s) contributes only for s=0 or s=1.
        for ambient_row, n in enumerate(dimensions):
            sums = degree_sums[:, ambient_row]
            if ambient_row == target_row:
                valid &= sums <= 1
            else:
                top = sums == n + 1
                valid &= (sums == 0) | top
                cohomological_degrees += top.astype(np.int16) * n

        target_sums = degree_sums[:, target_row]
        term_dimensions = np.where(target_sums == 0, n_target + 1, 1)
        u_mask = valid & (cohomological_degrees == cardinalities)
        v_mask = valid & (cardinalities >= 1) & (
            cohomological_degrees == cardinalities - 1
        )
        U_i = int(term_dimensions[u_mask].sum(dtype=np.int64))
        V_i = int(term_dimensions[v_mask].sum(dtype=np.int64))
        passed = U_i == n_target + 1 and V_i == 0
        row_certificates.append(
            {
                "row_1based": target_row + 1,
                "ambient_dimension": n_target,
                "ambient_h0_dimension": n_target + 1,
                "U_i": U_i,
                "V_i": V_i,
                "restriction_isomorphism_certified": passed,
            }
        )

    passed = all(row["restriction_isomorphism_certified"] for row in row_certificates)
    relation = record.get("presentation_relation")
    if relation is None:  # backward-compatible input reader
        relation = record.get("topology_audit", {}).get("fixed_X_relation")
    return {
        "Num": num,
        "canonical_hash_working": record.get("canonical_hash_working"),
        "source": record.get("source"),
        "working_relation": relation,
        "graph_embedded": relation == "scheme_isomorphic_graph_embedding",
        "ambient_factor_count": len(dimensions),
        "defining_equation_count": len(conf[0]),
        "subset_count": subset_count,
        "row_certificates": row_certificates,
        "E3_certified": passed,
        "status": "pass_E3" if passed else "unresolved_E3",
        "special_defining_polynomial_used": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--certificates", type=Path, default=DEFAULT_CERTIFICATES)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--expected-count", type=int, default=7820)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = args.input.resolve()
    certificates_path = args.certificates.resolve()
    summary_path = args.summary.resolve()
    records = read_jsonl(input_path)
    nums = [int(record["Num"]) for record in records]
    if len(records) != args.expected_count:
        raise ValueError(
            f"expected {args.expected_count} input records, found {len(records)}"
        )
    if len(set(nums)) != len(nums):
        raise ValueError("input contains duplicate CICY numbers")

    certificates_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = certificates_path.with_suffix(certificates_path.suffix + ".tmp")
    status_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    relation_counts: Counter[str] = Counter()
    row_test_count = 0

    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        for position, record in enumerate(records, start=1):
            certificate = audit_configuration(record)
            stream.write(canonical_json(certificate) + "\n")
            status_counts[certificate["status"]] += 1
            source_counts[str(certificate["source"])] += 1
            relation_counts[str(certificate["working_relation"])] += 1
            row_test_count += len(certificate["row_certificates"])
            if position % 100 == 0 or position == len(records):
                print(
                    f"processed {position}/{len(records)} configurations; "
                    f"unresolved={status_counts['unresolved_E3']}",
                    flush=True,
                )
    temporary.replace(certificates_path)

    unresolved = status_counts["unresolved_E3"]
    summary = {
        "audit_id": "coefficient_independent_E3_all_7820_20260920",
        "status": "pass" if unresolved == 0 else "unresolved",
        "input": {
            "path": str(input_path),
            "sha256": sha256_file(input_path),
            "record_count": len(records),
        },
        "program": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__).resolve()),
        },
        "certificates": {
            "path": str(certificates_path),
            "sha256": sha256_file(certificates_path),
            "record_count": len(records),
        },
        "configuration_count": len(records),
        "row_test_count": row_test_count,
        "E3_certified_count": status_counts["pass_E3"],
        "unresolved_count": unresolved,
        "source_counts": dict(sorted(source_counts.items())),
        "working_relation_counts": dict(sorted(relation_counts.items())),
        "coefficient_independent": True,
        "special_defining_polynomial_used": False,
        "formulae": {
            "U_i": "sum_{p=0}^K sum_{|S|=p} h^p(A,O_A(e_i-q_S))",
            "V_i": "sum_{p=1}^K sum_{|S|=p} h^(p-1)(A,O_A(e_i-q_S))",
            "pass_condition": "U_i=n_i+1 and V_i=0 for every ambient row i",
        },
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 1 if unresolved else 0


if __name__ == "__main__":
    raise SystemExit(main())
