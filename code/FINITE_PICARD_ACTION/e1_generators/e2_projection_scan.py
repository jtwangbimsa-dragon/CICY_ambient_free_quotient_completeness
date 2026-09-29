#!/usr/bin/env python3
"""Exact projection and vertical-curve calculations used by integral E1.

The public E1 entry point passes each supplied configuration record directly
to ``scan_record``.  The resulting curve data are then fed immediately into
the Smith-normal-form and finite-index calculations; no saved E2 result is
used as proof input.

No defining polynomial is sampled.  All arithmetic is exact integer
intersection theory in the Chow ring of a product of projective spaces.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Optional


ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "cicy_e1e3_fixed.jsonl"
RECORDS_OUTPUT = ROOT / "e2_projection_records.jsonl"
AUDIT_OUTPUT = ROOT / "e2_projection_audit.json"

Poly = dict[tuple[int, ...], int]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def padd(left: Poly, right: Poly, scale: int = 1) -> Poly:
    result = dict(left)
    for monomial, coefficient in right.items():
        value = result.get(monomial, 0) + scale * coefficient
        if value:
            result[monomial] = value
        elif monomial in result:
            del result[monomial]
    return result


def pscale(poly: Poly, scale: int) -> Poly:
    if not scale:
        return {}
    return {
        monomial: scale * coefficient
        for monomial, coefficient in poly.items()
        if coefficient
    }


def pmul(left: Poly, right: Poly, target: tuple[int, ...]) -> Poly:
    if not left or not right:
        return {}
    result: Poly = {}
    for left_monomial, left_coefficient in left.items():
        for right_monomial, right_coefficient in right.items():
            monomial = tuple(
                left_exponent + right_exponent
                for left_exponent, right_exponent in zip(
                    left_monomial, right_monomial
                )
            )
            if all(
                exponent <= limit
                for exponent, limit in zip(monomial, target)
            ):
                result[monomial] = (
                    result.get(monomial, 0)
                    + left_coefficient * right_coefficient
                )
    return {
        monomial: coefficient
        for monomial, coefficient in result.items()
        if coefficient
    }


def linear_poly(weights: tuple[int, ...]) -> Poly:
    zero = (0,) * len(weights)
    result: Poly = {}
    for index, coefficient in enumerate(weights):
        if coefficient:
            monomial = list(zero)
            monomial[index] = 1
            result[tuple(monomial)] = coefficient
    return result


def product_of_linears(
    forms: list[tuple[int, ...]], target: tuple[int, ...]
) -> Poly:
    result: Poly = {(0,) * len(target): 1}
    for weights in forms:
        result = pmul(result, linear_poly(weights), target)
        if not result:
            break
    return result


def chern_classes(
    forms: list[tuple[int, ...]],
    target: tuple[int, ...],
    max_degree: int = 3,
) -> list[Poly]:
    zero = (0,) * len(target)
    classes: list[Poly] = [{zero: 1}] + [
        {} for _ in range(max_degree)
    ]
    for weights in forms:
        line = linear_poly(weights)
        for degree in range(max_degree, 0, -1):
            classes[degree] = padd(
                classes[degree],
                pmul(classes[degree - 1], line, target),
            )
    return classes


def degree_one_gw(
    n: int,
    base_target: tuple[int, ...],
    row_degrees: list[int],
    base_linears: list[tuple[int, ...]],
) -> int:
    """Compute the degree-e_i genus-zero invariant on G(2,n+1) x B.

    For Chern roots x,y of S*, Grassmannian integration is coefficient
    extraction [x^n y^(n-1)] (x-y)(...).
    """
    target = (n, n - 1) + base_target
    factors: list[tuple[int, ...]] = []
    for degree, base_linear in zip(row_degrees, base_linears):
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


def row_base_data(conf: list[list[int]], row_index: int):
    dimensions = [sum(row) - 1 for row in conf]
    base_rows = [
        index for index in range(len(conf)) if index != row_index
    ]
    base_target = tuple(dimensions[index] for index in base_rows)
    columns = list(zip(*conf))
    base_linears = [
        tuple(column[index] for index in base_rows)
        for column in columns
    ]
    return dimensions[row_index], base_target, base_linears


def scan_row(conf: list[list[int]], row_index: int) -> dict:
    n, base_target, base_linears = row_base_data(conf, row_index)
    row_degrees = conf[row_index]
    positive_columns = [
        index for index, degree in enumerate(row_degrees) if degree > 0
    ]
    zero_columns = [
        index for index, degree in enumerate(row_degrees) if degree == 0
    ]
    support_size = len(positive_columns)
    common = {
        "row_1based": row_index + 1,
        "ambient_dimension": n,
        "positive_degree_column_count": support_size,
        "row_degrees": row_degrees,
    }

    if support_size < n:
        return common | {
            "kind": "forced_positive_dimensional_fiber",
            "uniform_vertical_curve_witness": True,
            "generic_projection_finite": False,
            "fixed_X_polynomial_test_needed": False,
        }

    z_class = product_of_linears(
        [base_linears[index] for index in zero_columns],
        base_target,
    )

    if support_size == n + 1 and all(
        row_degrees[index] == 1 for index in positive_columns
    ):
        bundle_forms = [
            base_linears[index] for index in positive_columns
        ]
        _, c1, c2, c3 = chern_classes(bundle_forms, base_target)
        porteous = padd(
            pmul(c2, c2, base_target),
            pmul(c1, c3, base_target),
            scale=-1,
        )
        number = pmul(z_class, porteous, base_target).get(
            base_target, 0
        )
        if number < 0:
            raise AssertionError(
                f"row {row_index + 1}: negative Type-1 degeneracy degree"
            )
        return common | {
            "kind": "type1_square_linear_system",
            "porteous_corank2_degree": number,
            "degree_one_GW": number,
            "uniform_vertical_curve_witness": number > 0,
            "generic_projection_finite": number == 0,
            "fixed_X_polynomial_test_needed": number == 0,
            "fixed_X_test": (
                None
                if number > 0
                else "On the base complete intersection cut by q_i=0 columns, "
                "form the (n+1)x(n+1) coefficient matrix of the q_i=1 "
                "equations.  After multigraded irrelevant-ideal saturation, "
                "test whether its n-by-n minors have a common zero."
            ),
        }

    twos = [
        index for index in positive_columns if row_degrees[index] == 2
    ]
    ones = [
        index for index in positive_columns if row_degrees[index] == 1
    ]
    if (
        support_size == n
        and len(twos) == 1
        and len(ones) == n - 1
    ):
        bundle_forms = [base_linears[index] for index in ones]
        _, c1, c2, c3 = chern_classes(bundle_forms, base_target)
        c1_squared = pmul(c1, c1, base_target)
        delta_1_3 = padd(
            pmul(c1_squared, c1, base_target),
            pscale(pmul(c1, c2, base_target), -2),
        )
        delta_1_3 = padd(delta_1_3, c3)
        conic_number = pmul(z_class, delta_1_3, base_target).get(
            base_target, 0
        )
        line_number = degree_one_gw(
            n, base_target, row_degrees, base_linears
        )
        if conic_number < 0 or line_number < 0:
            raise AssertionError(
                f"row {row_index + 1}: negative Type-2 witness"
            )
        passed = conic_number > 0 or line_number > 0
        return common | {
            "kind": "type2_quadratic_linear_system",
            "porteous_linear_corank_degree_conic_class_2Ci": conic_number,
            "degree_one_GW_line_class_Ci": line_number,
            "uniform_vertical_curve_witness": passed,
            "generic_projection_finite": not passed,
            "fixed_X_polynomial_test_needed": not passed,
        }

    raise AssertionError(
        f"unexpected CY row type at row {row_index + 1}: "
        f"n={n}, degrees={row_degrees}"
    )


def scan_record(record: dict) -> dict:
    original_rows = [
        scan_row(record["original_selected_conf"], row_index)
        for row_index in range(len(record["original_selected_conf"]))
    ]
    if record["obstruction_mappings"]:
        working_rows = [
            scan_row(record["working_conf"], row_index)
            for row_index in range(len(record["working_conf"]))
        ]
    else:
        working_rows = original_rows

    def degree_one_value(row: dict) -> Optional[int]:
        if row["kind"] == "type1_square_linear_system":
            return row["degree_one_GW"]
        if row["kind"] == "type2_quadratic_linear_system":
            return row["degree_one_GW_line_class_Ci"]
        return None

    if len(original_rows) != len(working_rows):
        raise AssertionError(f"Num {record['Num']}: graph repair changed row count")
    for original_row, working_row in zip(original_rows, working_rows):
        if (
            original_row["uniform_vertical_curve_witness"]
            != working_row["uniform_vertical_curve_witness"]
            or original_row["generic_projection_finite"]
            != working_row["generic_projection_finite"]
            or degree_one_value(original_row) != degree_one_value(working_row)
        ):
            raise AssertionError(
                f"Num {record['Num']} row {original_row['row_1based']}: "
                "original/working graph E2 witness mismatch"
            )

    zero_rows = [
        row["row_1based"]
        for row in original_rows
        if row["fixed_X_polynomial_test_needed"]
    ]
    all_witnessed = not zero_rows
    result = {
        "dataset_id": record["dataset_id"],
        "schema_version": 1,
        "Num": record["Num"],
        "source": record["source"],
        "canonical_hash_working": record["canonical_hash_working"],
        "working_graph_transformed": bool(record["obstruction_mappings"]),
        "authoritative_generic_family_presentation": "original_selected_conf",
        "row_count": len(original_rows),
        "original_rows": original_rows,
        "working_graph_cross_check": (
            "identical_configuration"
            if not record["obstruction_mappings"]
            else "pass_same_projection_finiteness_and_degree_one_witnesses"
        ),
        "uniform_E2_pass": all_witnessed,
        "generic_E2_pass": all_witnessed,
        "generic_E2_status": (
            "pass_all_coordinate_Mori_rays_witnessed"
            if all_witnessed
            else "fail_at_least_one_coordinate_projection_is_generically_finite"
        ),
        "fixed_X_status": (
            "pass_without_defining_polynomials"
            if all_witnessed
            else "undetermined_until_listed_projection_tests_use_defining_polynomials"
        ),
        "fixed_X_projection_tests_needed": len(zero_rows),
        "fixed_X_projection_test_rows_1based": zero_rows,
    }
    if record["obstruction_mappings"]:
        result["working_rows_cross_check"] = working_rows
    return result


def build_audit(input_records: list[dict], results: list[dict]) -> dict:
    input_by_num = {record["Num"]: record for record in input_records}
    all_rows = [
        row for result in results for row in result["original_rows"]
    ]
    working_rows = [
        row
        for result in results
        for row in result.get(
            "working_rows_cross_check", result["original_rows"]
        )
    ]
    row_kinds = Counter(row["kind"] for row in all_rows)
    zero_rows = [
        row for row in all_rows if row["fixed_X_polynomial_test_needed"]
    ]
    zero_distribution = Counter(
        result["fixed_X_projection_tests_needed"] for result in results
    )

    regression_checked = 0
    regression_mismatches: list[dict] = []
    for result in results:
        source_record = input_by_num[result["Num"]]
        if result["source"] != "Conf":
            continue
        if source_record["official_fields"]["KahlerPos_original"] is not True:
            raise AssertionError(
                f"Num {result['Num']}: Conf source is not KahlerPos=True"
            )
        official_gwi = source_record["official_fields"]["GWI_original"]
        if len(official_gwi) != len(result["original_rows"]):
            raise AssertionError(
                f"Num {result['Num']}: official GWI length mismatch"
            )
        for row, expected in zip(result["original_rows"], official_gwi):
            if row["kind"] == "forced_positive_dimensional_fiber":
                continue
            actual = (
                row["degree_one_GW"]
                if row["kind"] == "type1_square_linear_system"
                else row["degree_one_GW_line_class_Ci"]
            )
            regression_checked += 1
            if actual != expected:
                regression_mismatches.append(
                    {
                        "Num": result["Num"],
                        "row_1based": row["row_1based"],
                        "computed": actual,
                        "official": expected,
                    }
                )

    if regression_mismatches:
        raise AssertionError(
            f"official GWI regression mismatches: {regression_mismatches[:5]}"
        )

    conf_results = [result for result in results if result["source"] == "Conf"]
    favour_results = [
        result for result in results if result["source"] == "FavourConf"
    ]
    if not all(result["uniform_E2_pass"] for result in conf_results):
        raise AssertionError("not all 4,874 official KahlerPos cases passed")
    if any(result["generic_E2_pass"] for result in favour_results):
        raise AssertionError("expected every one of the 2,946 FavourConf cases to have a fake wall")

    type1_values = [
        row["porteous_corank2_degree"]
        for row in all_rows
        if row["kind"] == "type1_square_linear_system"
    ]
    type2_conics = [
        row["porteous_linear_corank_degree_conic_class_2Ci"]
        for row in all_rows
        if row["kind"] == "type2_quadratic_linear_system"
    ]
    type2_lines = [
        row["degree_one_GW_line_class_Ci"]
        for row in all_rows
        if row["kind"] == "type2_quadratic_linear_system"
    ]

    return {
        "dataset_id": "intrinsic_e1e3_fixed_20260821",
        "schema_version": 1,
        "audit_kind": "E2 projection-finiteness and vertical-curve audit",
        "input_file": INPUT.name,
        "input_sha256": sha256_file(INPUT),
        "authoritative_generic_family_presentation": "original_selected_conf",
        "working_conf_role": "cross-check only; graph-transformed working_conf must be used with its specialized graph equations for the fixed X and must not be interpreted as a generic working_conf member",
        "record_count": len(results),
        "source_counts": dict(Counter(result["source"] for result in results)),
        "criterion": {
            "statement": "With E1, the ambient H_i form a divisor basis and their positive orthant P lies in Nef(X). E2, Nef(X)=P, holds iff for every i the projection omitting factor i contracts a curve; equivalently L_i=sum_{j!=i} H_j is non-ample.",
            "fixed_X_consequence": "A nonzero row witness proves the required vertical curve without defining polynomials. A zero Type-1 Porteous degree proves finiteness only for general coefficients; a fixed special X must be tested with its coefficient matrix.",
        },
        "quantifiers": {
            "uniform_pass": "forced positive-dimensional fibers or nonzero exact GW/Porteous witnesses force a vertical curve throughout the smooth family",
            "generic_fail": "zero Type-1 degeneracy degree makes the corank-two locus empty for a general coefficient matrix",
            "fixed_special_member": "zero generic degree does not exclude an excess degeneracy locus for special defining polynomials",
        },
        "formulae": {
            "type1": "On the fourfold Z_i cut by q_ia=0 columns, F_i=direct_sum_{q_ia=1} O(l_a), and N_i=integral_Z_i(c2(F_i)^2-c1(F_i)c3(F_i)).",
            "type2_conic": "On the threefold Z_i, for F_i=direct_sum over the n_i-1 linear columns, A_i=integral_Z_i(c1(F_i)^3-2c1(F_i)c2(F_i)+c3(F_i)). A_i counts rank-drop fibers containing a conic of class 2C_i.",
            "degree_one_GW": "B_i is the exact top-Chern integral on G(2,n_i+1) times the base: product over columns a and r=0..q_ia of (l_a+r*x+(q_ia-r)*y), followed by Grassmannian coefficient extraction [x^n y^(n-1)](x-y).",
        },
        "row_count": len(all_rows),
        "authoritative_original_row_kind_counts": dict(row_kinds),
        "working_graph_cross_check_row_kind_counts": dict(
            Counter(row["kind"] for row in working_rows)
        ),
        "uniformly_witnessed_row_count": sum(
            row["uniform_vertical_curve_witness"] for row in all_rows
        ),
        "zero_type1_row_count": len(zero_rows),
        "zero_row_kind_counts": dict(Counter(row["kind"] for row in zero_rows)),
        "type1_degree_range": [min(type1_values), max(type1_values)],
        "type2_conic_degree_range": [min(type2_conics), max(type2_conics)],
        "type2_line_GW_range": [min(type2_lines), max(type2_lines)],
        "uniform_E2_pass_configuration_count": sum(
            result["uniform_E2_pass"] for result in results
        ),
        "generic_E2_fail_configuration_count": sum(
            not result["generic_E2_pass"] for result in results
        ),
        "conf_4874_uniform_E2_pass_count": sum(
            result["uniform_E2_pass"] for result in conf_results
        ),
        "favourconf_2946_generic_E2_fail_count": sum(
            not result["generic_E2_pass"] for result in favour_results
        ),
        "favourconf_2946_fixed_X_undetermined_count": sum(
            result["fixed_X_status"].startswith("undetermined")
            for result in favour_results
        ),
        "fixed_X_projection_test_count": sum(
            result["fixed_X_projection_tests_needed"] for result in results
        ),
        "fixed_X_projection_test_count_distribution": {
            str(key): value for key, value in sorted(zero_distribution.items())
        },
        "official_4874_GWI_regression": {
            "nonforced_rows_checked": regression_checked,
            "mismatch_count": len(regression_mismatches),
            "note": "The authoritative regression uses original_selected_conf. A separate exact comparison verifies that every graph-transformed working_conf has the same projection-finiteness result and degree-e_i witness row by row.",
        },
        "working_graph_cross_check": {
            "transformed_configuration_count": sum(
                result["working_graph_transformed"] for result in results
            ),
            "row_mismatch_count": 0,
        },
        "example_Num_7800": next(
            result for result in results if result["Num"] == 7800
        ),
    }


def main() -> None:
    with INPUT.open(encoding="utf-8") as stream:
        input_records = [json.loads(line) for line in stream if line.strip()]
    results = [scan_record(record) for record in input_records]
    audit = build_audit(input_records, results)

    with RECORDS_OUTPUT.open("w", encoding="utf-8") as stream:
        for result in results:
            stream.write(json.dumps(result, separators=(",", ":")))
            stream.write("\n")
    AUDIT_OUTPUT.write_text(
        json.dumps(audit, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    print(
        "PASS: 7820 E2 projection records; "
        f"{audit['conf_4874_uniform_E2_pass_count']} uniform passes; "
        f"{audit['favourconf_2946_generic_E2_fail_count']} generic failures; "
        f"{audit['zero_type1_row_count']} fixed-X projection tests; "
        f"{audit['official_4874_GWI_regression']['nonforced_rows_checked']} "
        "official GWI values matched"
    )


if __name__ == "__main__":
    main()
