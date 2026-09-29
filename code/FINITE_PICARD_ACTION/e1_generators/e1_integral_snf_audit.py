#!/usr/bin/env python3
"""Exact integral-lattice audit for the favourable CICY E1 gate.

The official favourable calculation is a calculation over complex cohomology.
It proves that the restricted ambient hyperplane classes have full rank in
``Pic(X) tensor Q``; it does not, by itself, prove that their integral span is
saturated.  This module supplies a sufficient saturation certificate.

Let ``L = sum_i Z H_i`` and let ``C_alpha`` be *known integral homology
classes* in ``H_2(X,Z)``.  The columns of the pairing matrix are

    P[i, alpha] = <H_i, C_alpha>.

If the Smith invariant factors of P are all one, the restrictions of the
known homology classes generate ``L^*``.  Since restriction
``H_2(X,Z)/tors -> L^*`` factors through those columns, this forces L to be a
primitive full-rank sublattice of ``Pic(X)/tors``.

Only the following mathematically realised integral H_2 classes are used:

* ``PD(H_j cup H_k)`` for restricted integral hyperplane classes H_j,H_k;
* ``PD(c_2(TX))``;
* a degree-e_i stable-map class when the freshly recomputed projection audit
  gives a non-zero degree-one Gromov--Witten/top-Chern witness.

These columns are sufficient witnesses, not a claim to list all curve or
homology classes.  Picard torsion is a separate gate.  This audit deliberately
does not turn the standard simply-connected-CICY assertion into an implicit
machine certificate: records with Smith factors all one are reported as
``pass_snf_unit_invariants`` and remain conditional on an external
``Pic(X) torsion-free`` theorem/certificate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from sympy import Matrix, ZZ
from sympy.matrices.normalforms import smith_normal_form

try:
    from .e1_lattice_close import c2_from_conf
except ImportError:  # direct execution from this directory
    from e1_lattice_close import c2_from_conf


PACKAGE_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = PACKAGE_ROOT.parent
DEFAULT_DATASET = WORKSPACE_ROOT / "intrinsic_e1e3_fixed_20260821" / "cicy_e1e3_fixed.jsonl"
DEFAULT_E2 = WORKSPACE_ROOT / "intrinsic_e1e3_fixed_20260821" / "e2_projection_records.jsonl"
DEFAULT_OUTPUT_DIR = PACKAGE_ROOT / "e1_integral_audit"
DEFAULT_RECORDS_NAME = "e1_integral_snf_records.jsonl"
DEFAULT_SUMMARY_NAME = "e1_integral_snf_summary.json"
SCHEMA_VERSION = 1
AUDIT_ID = "cicy_e1_integral_snf_20260821"


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ambient_dimensions(conf: list[list[int]]) -> list[int]:
    if not conf or not conf[0]:
        raise ValueError("configuration must be a non-empty matrix")
    width = len(conf[0])
    if any(len(row) != width for row in conf):
        raise ValueError("configuration matrix is not rectangular")
    if any(not isinstance(q, int) or isinstance(q, bool) or q < 0 for row in conf for q in row):
        raise ValueError("configuration entries must be non-negative integers")
    dimensions = [sum(row) - 1 for row in conf]
    if any(n < 1 for n in dimensions):
        raise ValueError(f"invalid ambient dimensions {dimensions}")
    return dimensions


def normal_product(conf: list[list[int]]) -> dict[tuple[int, ...], int]:
    """Expand prod_a(sum_i q[i,a] H_i), truncated in the ambient Chow ring."""

    dimensions = ambient_dimensions(conf)
    row_count = len(conf)
    polynomial: dict[tuple[int, ...], int] = {(0,) * row_count: 1}
    for column in zip(*conf):
        updated: dict[tuple[int, ...], int] = {}
        for exponent, coefficient in polynomial.items():
            for row_index, degree in enumerate(column):
                if degree == 0 or exponent[row_index] >= dimensions[row_index]:
                    continue
                next_exponent = list(exponent)
                next_exponent[row_index] += 1
                key = tuple(next_exponent)
                updated[key] = updated.get(key, 0) + coefficient * degree
        polynomial = updated
    return polynomial


def triple_intersection_curve_columns(conf: list[list[int]]) -> list[list[int]]:
    """Return degree vectors of the integral classes PD(H_j cup H_k)."""

    dimensions = ambient_dimensions(conf)
    row_count = len(conf)
    top = tuple(dimensions)
    defining_class = normal_product(conf)
    columns: list[list[int]] = []
    for left in range(row_count):
        for right in range(left, row_count):
            degree_vector: list[int] = []
            for divisor in range(row_count):
                exponent = [0] * row_count
                exponent[divisor] += 1
                exponent[left] += 1
                exponent[right] += 1
                complement = tuple(top[i] - exponent[i] for i in range(row_count))
                value = 0 if min(complement) < 0 else defining_class.get(complement, 0)
                degree_vector.append(value)
            columns.append(degree_vector)
    return columns


def degree_one_witness(row: dict[str, Any]) -> tuple[int, str] | None:
    """Return the non-zero deformation-invariant degree-one witness, if present."""

    if not row.get("uniform_vertical_curve_witness", False):
        return None
    if row.get("kind") == "type1_square_linear_system":
        value = row.get("degree_one_GW")
        field = "degree_one_GW"
    elif row.get("kind") == "type2_quadratic_linear_system":
        value = row.get("degree_one_GW_line_class_Ci")
        field = "degree_one_GW_line_class_Ci"
    else:
        return None
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        return None
    return value, field


def known_integral_h2_columns(
    dataset_record: dict[str, Any], e2_record: dict[str, Any]
) -> tuple[list[list[int]], dict[str, Any]]:
    """Build the sufficient-witness matrix and provenance metadata."""

    conf = dataset_record["original_selected_conf"]
    row_count = len(conf)
    columns = triple_intersection_curve_columns(conf)
    divisor_intersection_count = len(columns)

    c2 = c2_from_conf(conf)
    supplied_c2 = dataset_record.get("official_fields", {}).get("C2_selected")
    if supplied_c2 is not None and [int(value) for value in supplied_c2] != c2:
        raise ValueError(f"Num {dataset_record['Num']}: supplied c2 disagrees with configuration")
    columns.append(list(c2))

    degree_one_rows: list[dict[str, Any]] = []
    for row in e2_record["original_rows"]:
        witness = degree_one_witness(row)
        if witness is None:
            continue
        value, field = witness
        row_index = row["row_1based"] - 1
        if not 0 <= row_index < row_count:
            raise ValueError(f"Num {dataset_record['Num']}: invalid E2 row index")
        degree_vector = [0] * row_count
        degree_vector[row_index] = 1
        columns.append(degree_vector)
        degree_one_rows.append(
            {
                "row_1based": row_index + 1,
                "witness_field": field,
                "witness_value": value,
                "class_degree_vector": degree_vector,
            }
        )

    provenance = {
        "PD_Hj_cup_Hk_column_count": divisor_intersection_count,
        "PD_c2_column_count": 1,
        "PD_c2_computed_from_configuration": True,
        "degree_one_stable_map_column_count": len(degree_one_rows),
        "degree_one_stable_map_rows": degree_one_rows,
        "scope_note": (
            "Every column is the pairing vector of a specified integral H_2 class. "
            "The list is only a sufficient witness set and is not asserted to generate all H_2."
        ),
    }
    return columns, provenance


def smith_invariant_factors(columns: list[list[int]], row_count: int) -> tuple[list[int], int]:
    if any(len(column) != row_count for column in columns):
        raise ValueError("pairing column height mismatch")
    matrix = Matrix(row_count, len(columns), lambda i, j: columns[j][i])
    diagonal = smith_normal_form(matrix, domain=ZZ)
    invariants = [
        abs(int(diagonal[i, i]))
        for i in range(min(diagonal.rows, diagonal.cols))
        if diagonal[i, i] != 0
    ]
    return invariants, len(invariants)


def audit_record(dataset_record: dict[str, Any], e2_record: dict[str, Any]) -> dict[str, Any]:
    if dataset_record["Num"] != e2_record["Num"]:
        raise ValueError("dataset/E2 Num mismatch")
    if dataset_record["source"] != e2_record["source"]:
        raise ValueError(f"Num {dataset_record['Num']}: dataset/E2 source mismatch")

    row_count = len(dataset_record["original_selected_conf"])
    h11 = dataset_record["official_fields"]["H11"]
    if h11 != row_count:
        raise ValueError(f"Num {dataset_record['Num']}: favourable rank mismatch H11={h11}, rows={row_count}")

    columns, provenance = known_integral_h2_columns(dataset_record, e2_record)
    invariants, rank = smith_invariant_factors(columns, row_count)
    primitive = rank == row_count and all(value == 1 for value in invariants)
    index_upper_bound = math.prod(invariants) if rank == row_count else None

    return {
        "audit_id": AUDIT_ID,
        "schema_version": SCHEMA_VERSION,
        "Num": dataset_record["Num"],
        "source": dataset_record["source"],
        "canonical_hash_original": dataset_record["canonical_hash_original"],
        "ambient_hyperplane_rank": row_count,
        "official_H11": h11,
        "rational_E1_status": "pass_official_favourable_full_rank",
        "known_integral_h2_columns": provenance,
        "pairing_matrix_shape": [row_count, len(columns)],
        "smith_rank": rank,
        "smith_invariant_factors": invariants,
        "known_pairing_lattice_index_in_Zm": index_upper_bound,
        "primitivity_mod_torsion_status": (
            "pass_snf_unit_invariants" if primitive else "unverified_known_columns_not_saturated"
        ),
        "torsion_gate": {
            "status": "not_certified_by_this_audit",
            "required_statement": "Pic(X) has no torsion (for example from a cited simply-connectedness theorem)",
            "reason": (
                "Complex Hodge/coherent-cohomology favourability does not detect finite torsion in Pic(X)."
            ),
        },
        "integral_E1_status": (
            "conditional_pass_requires_torsion_free_gate"
            if primitive
            else "unverified_primitivity_and_torsion"
        ),
    }


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"{path}:{line_number}: {error}") from error


def run_audit(dataset_path: Path, e2_path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    e2_by_num = {record["Num"]: record for record in read_jsonl(e2_path)}
    records: list[dict[str, Any]] = []
    for dataset_record in read_jsonl(dataset_path):
        num = dataset_record["Num"]
        if num not in e2_by_num:
            raise ValueError(f"Num {num}: missing E2 record")
        records.append(audit_record(dataset_record, e2_by_num[num]))
    if len(records) != len(e2_by_num):
        raise ValueError("dataset and E2 record populations differ")

    status_counts = Counter(record["primitivity_mod_torsion_status"] for record in records)
    source_status_counts: dict[str, dict[str, int]] = {}
    for source in sorted({record["source"] for record in records}):
        source_status_counts[source] = dict(
            sorted(
                Counter(
                    record["primitivity_mod_torsion_status"]
                    for record in records
                    if record["source"] == source
                ).items()
            )
        )

    summary = {
        "audit_id": AUDIT_ID,
        "schema_version": SCHEMA_VERSION,
        "record_count": len(records),
        "input_files": {
            "e1_integral_snf_audit.py": {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256_file(Path(__file__).resolve()),
            },
            "cicy_e1e3_fixed.jsonl": {
                "path": str(dataset_path),
                "sha256": sha256_file(dataset_path),
            },
            "e2_projection_records.jsonl": {
                "path": str(e2_path),
                "sha256": sha256_file(e2_path),
            },
        },
        "theorem": {
            "rational_scope": (
                "Official favourability supplies full rank only after tensoring Pic with Q/C."
            ),
            "snf_certificate": (
                "If specified integral H_2 pairing columns have Smith rank h11 and every invariant "
                "factor equals one, the ambient hyperplane lattice is primitive in Pic(X)/tors."
            ),
            "integral_curve_column_sources": [
                "PD(H_j cup H_k)",
                "PD(c_2(TX))",
                "degree-e_i stable-map class from a non-zero exact deformation-invariant witness",
            ],
            "non_exhaustiveness": (
                "These are sufficient witness columns only; non-unit SNF is unverified, not a proof "
                "that the ambient lattice has non-trivial index."
            ),
        },
        "primitivity_mod_torsion_status_counts": dict(sorted(status_counts.items())),
        "source_status_counts": source_status_counts,
        "torsion_gate": {
            "status": "not_certified_by_this_audit_for_any_record",
            "effect": (
                "A pass_snf_unit_invariants record becomes a full integral Picard isomorphism only "
                "after an external Pic(X)-torsion-free certificate is attached."
            ),
        },
    }
    return records, summary


def write_outputs(records: list[dict[str, Any]], summary: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    records_path = output_dir / DEFAULT_RECORDS_NAME
    summary_path = output_dir / DEFAULT_SUMMARY_NAME
    with records_path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(canonical_json(record) + "\n")
    with summary_path.open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, ensure_ascii=False, sort_keys=True, indent=2)
        stream.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--e2-records", type=Path, default=DEFAULT_E2)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    records, summary = run_audit(args.dataset.resolve(), args.e2_records.resolve())
    write_outputs(records, summary, args.output_dir.resolve())
    counts = summary["primitivity_mod_torsion_status_counts"]
    print(canonical_json({"record_count": len(records), "counts": counts, "output_dir": str(args.output_dir.resolve())}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
