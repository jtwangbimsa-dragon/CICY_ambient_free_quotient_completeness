#!/usr/bin/env python3
"""Certify E1 and E2 simultaneously from configuration data for 7,820 CICYs.

For each ambient factor i, the program computes the degree-e_i vertical-line
top Chern number

    B_i = integral_{Gr(2,n_i+1) x base} c_top(F_i)

by exact coefficient extraction.  If B_i is nonzero, every defining tuple in
the selected complete-intersection family contains an effective curve C_i of
ambient degree e_i.  Hence J_j.C_i=delta_ij.

If this happens in every coordinate direction, the same curves prove both:

* E1: rational favourability plus integral divisor--curve pairing makes the
  ambient Picard lattice saturated; the standard torsion-free theorem for the
  smooth non-product CICYs in this dataset then gives an integral basis.
* E2: a nef divisor D=sum a_j J_j has a_i=D.C_i>=0 for every i, while every
  J_i is ambient-nef.  Thus Nef(X) is exactly the nonnegative ambient orthant.

A zero B_i is deliberately reported only as NOT_CERTIFIED_BY_THIS_TEST.  It is
not treated as a proof that E1 or E2 fails for a special member.  No defining
polynomial is selected or read.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


HERE = Path(__file__).resolve().parent
PACKAGE_INPUT = HERE.parent.parent / "data/cicy_criteria_input_7820.jsonl"
DEFAULT_INPUT = PACKAGE_INPUT
DEFAULT_OUTPUT_DIRECTORY = HERE.parent.parent / "reproduction_output"
DEFAULT_CERTIFICATES = DEFAULT_OUTPUT_DIRECTORY / "e1_e2_certificates_7820.jsonl"
DEFAULT_SUMMARY = DEFAULT_OUTPUT_DIRECTORY / "e1_e2_summary_7820.json"

Poly = dict[tuple[int, ...], int]


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


def validate_conf(conf: Any, *, num: int, label: str) -> list[list[int]]:
    if not isinstance(conf, list) or not conf or not isinstance(conf[0], list) or not conf[0]:
        raise ValueError(f"Num {num}: {label} must be a nonempty matrix")
    width = len(conf[0])
    if any(not isinstance(row, list) or len(row) != width for row in conf):
        raise ValueError(f"Num {num}: {label} must be rectangular")
    if any(
        not isinstance(value, int) or isinstance(value, bool) or value < 0
        for row in conf
        for value in row
    ):
        raise ValueError(f"Num {num}: {label} entries must be nonnegative integers")
    dimensions = [sum(row) - 1 for row in conf]
    if any(n < 1 for n in dimensions):
        raise ValueError(f"Num {num}: {label} has invalid ambient dimensions")
    if sum(dimensions) - width != 3:
        raise ValueError(f"Num {num}: {label} is not a threefold configuration")
    return conf


def add_term(poly: Poly, exponent: tuple[int, ...], coefficient: int) -> None:
    if not coefficient:
        return
    value = poly.get(exponent, 0) + coefficient
    if value:
        poly[exponent] = value
    elif exponent in poly:
        del poly[exponent]


def multiply_by_linear(
    poly: Poly,
    weights: tuple[int, ...],
    target: tuple[int, ...],
) -> Poly:
    result: Poly = {}
    for exponent, coefficient in poly.items():
        for index, weight in enumerate(weights):
            if not weight or exponent[index] >= target[index]:
                continue
            new_exponent = list(exponent)
            new_exponent[index] += 1
            add_term(result, tuple(new_exponent), coefficient * weight)
    return result


def product_of_linears(
    factors: Iterable[tuple[int, ...]], target: tuple[int, ...]
) -> Poly:
    result: Poly = {(0,) * len(target): 1}
    for factor in factors:
        result = multiply_by_linear(result, factor, target)
        if not result:
            break
    return result


def vertical_line_number(conf: list[list[int]], row_index: int) -> int:
    """Return B_i by Grassmannian coefficient extraction over the integers."""
    dimensions = [sum(row) - 1 for row in conf]
    n = dimensions[row_index]
    base_rows = [row for row in range(len(conf)) if row != row_index]
    base_target = tuple(dimensions[row] for row in base_rows)
    columns = list(zip(*conf))
    base_linears = [tuple(column[row] for row in base_rows) for column in columns]
    target = (n, n - 1) + base_target

    factors: list[tuple[int, ...]] = []
    for degree, base_linear in zip(conf[row_index], base_linears):
        for r in range(degree + 1):
            factors.append((r, degree - r) + base_linear)
    polynomial = product_of_linears(factors, target)

    first = (n - 1, n - 1) + base_target
    first_value = polynomial.get(first, 0)
    if n == 1:
        second_value = 0
    else:
        second = (n, n - 2) + base_target
        second_value = polynomial.get(second, 0)
    return first_value - second_value


def audit_record(record: dict[str, Any]) -> dict[str, Any]:
    num = int(record["Num"])
    selected = validate_conf(record["original_selected_conf"], num=num, label="selected conf")
    working = validate_conf(record["working_conf"], num=num, label="working conf")
    if len(selected) != len(working):
        raise ValueError(f"Num {num}: graph embedding changed the divisor-frame rank")

    h11 = int(record["official_fields"]["H11"])
    hypotheses = record.get("hypotheses", {})
    rational_declared = hypotheses.get("rational_favourability_certified")
    if rational_declared is None:  # backward-compatible input reader
        rational_declared = (
            record.get("E1_status", {}).get("original")
            == "pass_official_favourable_representative"
        )
    nonproduct_declared = hypotheses.get("standard_nonproduct_cicy")
    if nonproduct_declared is None:  # backward-compatible input reader
        nonproduct_declared = (
            record.get("official_fields", {}).get("IsProduct_original") is False
        )
    rational_gate = rational_declared is True and h11 == len(selected)
    nonproduct_gate = nonproduct_declared is True

    selected_numbers = [vertical_line_number(selected, row) for row in range(len(selected))]
    if any(number < 0 for number in selected_numbers):
        raise ValueError(f"Num {num}: a vertical-line top Chern number is negative")

    graph_embedded = bool(record.get("obstruction_mappings"))
    graph_cross_check: dict[str, Any]
    if graph_embedded:
        working_numbers = [vertical_line_number(working, row) for row in range(len(working))]
        if working_numbers != selected_numbers:
            raise ValueError(f"Num {num}: selected/working vertical-line numbers differ")
        graph_cross_check = {
            "status": "pass",
            "working_vertical_line_numbers": working_numbers,
        }
    else:
        if working != selected:
            raise ValueError(f"Num {num}: untransformed record changed configuration")
        graph_cross_check = {"status": "identical_presentation"}

    official_gwi_check: dict[str, Any]
    if record.get("source") == "Conf" and record["official_fields"].get("GWI_original") is not None:
        official = record["official_fields"].get("GWI_original")
        if official != selected_numbers:
            raise ValueError(f"Num {num}: computed line numbers disagree with official GWI")
        official_gwi_check = {"status": "pass", "checked_row_count": len(selected)}
    else:
        official_gwi_check = {
            "status": "not_supplied_or_not_applicable",
            "checked_row_count": 0,
        }

    row_witnesses = [
        {
            "row_1based": row + 1,
            "vertical_line_top_chern_B_i": number,
            "nonzero_effective_curve_witness": number != 0,
            "curve_pairings": "J_j.C_i=delta_ij" if number != 0 else None,
        }
        for row, number in enumerate(selected_numbers)
    ]
    zero_rows = [
        witness["row_1based"]
        for witness in row_witnesses
        if not witness["nonzero_effective_curve_witness"]
    ]
    all_directions_witnessed = not zero_rows

    # The torsion statement is a theorem-level gate, not a numerical inference.
    torsion_gate = {
        "status": (
            "pass_standard_smooth_nonproduct_CICY_theorem"
            if nonproduct_gate
            else "UNKNOWN_standard_nonproduct_hypothesis_not_supplied"
        ),
        "statement": (
            "The smooth non-product CICY is simply connected; UCT makes H^2(X,Z) "
            "torsion-free, and H^1(O_X)=0 embeds Pic(X) into H^2(X,Z)."
        ),
    }
    E1_certified = rational_gate and nonproduct_gate and all_directions_witnessed
    E2_certified = E1_certified and all_directions_witnessed
    both = E1_certified and E2_certified

    return {
        "Num": num,
        "source": record.get("source"),
        "canonical_hash_selected": record.get("canonical_hash_original"),
        "canonical_hash_working": record.get("canonical_hash_working"),
        "graph_embedded": graph_embedded,
        "ambient_factor_count": len(selected),
        "rational_favourability_gate": {
            "status": "pass" if rational_gate else "UNKNOWN",
            "h11": h11,
            "ambient_divisor_rank": len(selected),
        },
        "picard_torsion_gate": torsion_gate,
        "row_witnesses": row_witnesses,
        "zero_witness_rows_1based": zero_rows,
        "all_coordinate_directions_witnessed": all_directions_witnessed,
        "graph_cross_check": graph_cross_check,
        "official_GWI_cross_check": official_gwi_check,
        "E1_certified_by_this_test": E1_certified,
        "E2_certified_by_this_test": E2_certified,
        "E1_and_E2_certified": both,
        "status": (
            "PASS_E1_AND_E2"
            if both
            else "NOT_CERTIFIED_BY_VERTICAL_LINE_TEST"
        ),
        "zero_witness_interpretation": (
            None
            if both
            else (
                "The direct criterion is unresolved.  A zero B_i is inconclusive, "
                "and a missing favourability or non-product hypothesis is not inferred "
                "from the matrix."
            )
        ),
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
    if len(records) != args.expected_count or len(set(nums)) != args.expected_count:
        raise ValueError("input is not the expected unique 7,820-record dataset")

    certificates_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = certificates_path.with_suffix(certificates_path.suffix + ".tmp")
    status_counts: Counter[str] = Counter()
    source_status_counts: Counter[str] = Counter()
    relation_counts: Counter[str] = Counter()
    zero_witness_count = 0
    row_test_count = 0
    official_gwi_row_checks = 0

    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        for position, record in enumerate(records, start=1):
            certificate = audit_record(record)
            stream.write(canonical_json(certificate) + "\n")
            status_counts[certificate["status"]] += 1
            source_status_counts[
                f"{certificate['source']}:{certificate['status']}"
            ] += 1
            relation_counts[
                "graph_embedding" if certificate["graph_embedded"] else "identity"
            ] += 1
            zero_witness_count += len(certificate["zero_witness_rows_1based"])
            row_test_count += len(certificate["row_witnesses"])
            official_gwi_row_checks += certificate["official_GWI_cross_check"][
                "checked_row_count"
            ]
            if position % 100 == 0 or position == len(records):
                print(
                    f"processed {position}/{len(records)} configurations; "
                    f"positive={status_counts['PASS_E1_AND_E2']}",
                    flush=True,
                )
    temporary.replace(certificates_path)

    positive_count = status_counts["PASS_E1_AND_E2"]
    unclassified_count = status_counts["NOT_CERTIFIED_BY_VERTICAL_LINE_TEST"]
    summary = {
        "audit_id": "direct_E1_E2_vertical_lines_all_7820_20260920",
        "status": "pass",
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
        "E1_and_E2_certified_count": positive_count,
        "not_certified_by_this_test_count": unclassified_count,
        "zero_vertical_line_witness_row_count": zero_witness_count,
        "source_status_counts": dict(sorted(source_status_counts.items())),
        "presentation_relation_counts": dict(sorted(relation_counts.items())),
        "official_GWI_row_cross_check_count": official_gwi_row_checks,
        "official_GWI_mismatch_count": 0,
        "graph_cross_check_mismatch_count": 0,
        "special_defining_polynomial_used": False,
        "claim_scope": {
            "positive": (
                "PASS_E1_AND_E2 is a sufficient configuration-only certificate "
                "for the fixed smooth CICY and, before graph transport, uniformly "
                "over the selected smooth complete-intersection family."
            ),
            "nonpositive": (
                "NOT_CERTIFIED_BY_VERTICAL_LINE_TEST is unresolved, not a proof "
                "that E1 or E2 fails for the fixed special member."
            ),
        },
        "proof": {
            "curve_existence": (
                "B_i!=0 is the top Chern number of the compact vertical-line "
                "bundle and forces an effective degree-e_i curve C_i."
            ),
            "E1": (
                "J_j.C_i=delta_ij makes every coefficient of an integral divisor "
                "integral; rational favourability and the torsion-free theorem "
                "therefore give an integral Picard basis."
            ),
            "E2": (
                "For nef D=sum a_j J_j, a_i=D.C_i>=0; conversely each J_i is "
                "ambient-nef, so Nef(X) is exactly the nonnegative orthant."
            ),
        },
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
