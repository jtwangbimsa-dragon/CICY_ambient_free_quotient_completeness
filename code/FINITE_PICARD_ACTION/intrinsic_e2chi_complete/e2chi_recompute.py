#!/usr/bin/env python3
"""Recompute the E2-chi frame-preservation criterion from configurations.

The input is the CICY configuration dataset together with a freshly produced
integral-E1 ledger.  Ordinary cases use the exact ellipsoid enumeration in
``candidate_enumerator``.  Num 3936 is evaluated by an exhaustive disjoint
partition according to the number of changed ambient-frame columns, and Num
4081 is transported only after an explicit row/column isomorphism of the two
configuration matrices has been reconstructed.

No frozen E2-chi result or defining polynomial is read.  A cap, exception,
failed E1 gate, surviving matrix, or incomplete special partition is reported
as unresolved rather than as a negative result.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import tempfile
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

from intrinsic_e2chi_complete.candidate_enumerator import (
    analyze_case,
    c2_ellipsoid_certificate,
    c2_pairing_from_tensor,
    cubic_tensor,
    enumerate_column_candidates,
    load_jsonl,
    sha256_file,
    validate_full_e1_certificate,
)
from intrinsic_e2chi_complete.configuration_geometry import (
    euler_characteristic,
    find_configuration_isomorphism,
)
from intrinsic_e2chi_complete.early_survivor import search_first_survivor


PACKAGE_ROOT = Path(__file__).resolve().parent.parent
RELEASE_ROOT = PACKAGE_ROOT.parents[1]
DEFAULT_DATASET = RELEASE_ROOT / "data" / "cicy_criteria_input_7820.jsonl"
DEFAULT_E1_RECORDS = RELEASE_ROOT / "reproduction_output" / "e1_complete_2635.jsonl"
DEFAULT_OUTPUT = RELEASE_ROOT / "reproduction_output" / "e2chi_complete_2635.jsonl"
DEFAULT_SUMMARY = RELEASE_ROOT / "reproduction_output" / "e2chi_complete_2635_summary.json"
SPECIAL_SOURCE = 3936
SPECIAL_TRANSPORT = 4081
SPECIAL_CANONICAL_HASH = "93df935fcbcb5b2da25258f7828433f2fca7971db9315f0abe24bb511c9fada0"


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def record_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def index_by_num(records: Iterable[dict[str, Any]], label: str) -> dict[int, dict[str, Any]]:
    result: dict[int, dict[str, Any]] = {}
    for record in records:
        num = int(record["Num"])
        if num in result:
            raise ValueError(f"{label}: duplicate Num {num}")
        result[num] = record
    return result


def parse_num_specification(value: str) -> set[int]:
    result: set[int] = set()
    for piece in value.split(","):
        piece = piece.strip()
        if not piece:
            continue
        if "-" in piece:
            lower, upper = map(int, piece.split("-", 1))
            if upper < lower:
                raise ValueError(f"invalid descending range {piece}")
            result.update(range(lower, upper + 1))
        else:
            result.add(int(piece))
    return result


def _atomic_write(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _ordinary_worker(
    task: tuple[
        dict[str, Any],
        dict[str, Any] | None,
        int | None,
        int | None,
        bool,
    ]
) -> dict[str, Any]:
    record, e1_record, point_cap, matrix_cap, full_certificates = task
    try:
        result = analyze_case(
            record,
            e1_record,
            point_cap=point_cap,
            matrix_cap=matrix_cap,
        )
        result["criterion_program"] = "ordinary_exact_ellipsoid_enumeration"
        result["frozen_E2chi_result_read"] = False
        if full_certificates:
            return result
        # The exhaustive calculation has already taken place.  By default we
        # retain a compact certificate rather than duplicating the potentially
        # very large matrix payload in the release user's output directory.
        # Passing --full-certificates preserves the complete per-matrix record.
        return {
            "schema_version": "2.0.0",
            "Num": int(result["Num"]),
            "source": result.get("source"),
            "canonical_hash_working": result.get("canonical_hash_working"),
            "picard_rank": result.get("picard_rank"),
            "status": result.get("status"),
            "criterion_program": result["criterion_program"],
            "integral_E1_certified": result.get("integral_E1_certified"),
            "E1_gate_reason": result.get("E1_gate_reason"),
            "ellipsoid_certificate": result.get("ellipsoid_certificate"),
            "column_candidate_counts": [
                len(item.get("candidate_columns", []))
                for item in result.get("column_enumeration_records", [])
            ],
            "matrix_enumeration_stats": result.get("matrix_enumeration_stats"),
            "candidate_matrix_count": result.get("candidate_matrix_count"),
            "candidate_screening_mode": result.get("candidate_screening_mode"),
            "candidate_group_certificate": result.get("candidate_group_certificate"),
            "candidate_conjugacy_class_count": result.get("candidate_conjugacy_class_count"),
            "candidate_conjugacy_class_evaluation_count": result.get(
                "candidate_conjugacy_class_evaluation_count"
            ),
            "E2chi_evaluation_count": result.get("E2chi_evaluation_count"),
            "infinite_order_candidate_count": result.get("infinite_order_candidate_count"),
            "finite_order_frame_preserving_count": result.get(
                "finite_order_frame_preserving_count"
            ),
            "finite_order_frame_changing_count": result.get(
                "finite_order_frame_changing_count"
            ),
            "finite_order_frame_changing_excluded_count": result.get(
                "finite_order_frame_changing_excluded_count"
            ),
            "finite_order_frame_changing_survivor_count": result.get(
                "finite_order_frame_changing_survivor_count"
            ),
            "E2chi_no_frame_changing_survivors": result.get(
                "E2chi_no_frame_changing_survivors"
            ),
            "full_certificate_sha256": record_sha256(result),
            "full_matrix_payload_retained": False,
            "configuration_only": True,
            "defining_polynomial_used": False,
            "frozen_E2chi_result_read": False,
            "elapsed_seconds": result.get("elapsed_seconds"),
        }
    except Exception as exc:
        return {
            "schema_version": "2.0.0",
            "Num": int(record["Num"]),
            "source": record.get("source"),
            "picard_rank": int(record["official_fields"]["H11"]),
            "status": "UNKNOWN_worker_exception",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "E2chi_no_frame_changing_survivors": False,
            "configuration_only": True,
            "defining_polynomial_used": False,
            "frozen_E2chi_result_read": False,
        }


def analyze_layered_special(
    record: dict[str, Any],
    e1_record: dict[str, Any] | None,
    *,
    point_cap: int | None,
    node_cap: int | None,
    matrix_cap: int | None,
) -> dict[str, Any]:
    num = int(record["Num"])
    e1_ok, e1_reason = validate_full_e1_certificate(record, e1_record)
    if not e1_ok:
        return {
            "schema_version": "2.0.0",
            "Num": num,
            "status": "UNKNOWN_integral_E1_prerequisite",
            "E1_gate": e1_reason,
            "E2chi_no_frame_changing_survivors": False,
            "configuration_only": True,
            "defining_polynomial_used": False,
            "frozen_E2chi_result_read": False,
        }

    conf = record["working_conf"]
    dimensions = [int(value) for value in record["working_ambient_dimensions"]]
    tensor = cubic_tensor(conf, dimensions)
    c2 = c2_pairing_from_tensor(conf, dimensions, tensor)
    ellipsoid, kernel, hodge = c2_ellipsoid_certificate(tensor, c2)
    if not ellipsoid.negative_restriction_certified:
        return {
            "schema_version": "2.0.0",
            "Num": num,
            "status": "UNKNOWN_negative_definiteness_gate",
            "ellipsoid_certificate": asdict(ellipsoid),
            "E2chi_no_frame_changing_survivors": False,
            "configuration_only": True,
            "defining_polynomial_used": False,
            "frozen_E2chi_result_read": False,
        }
    columns = [
        enumerate_column_candidates(
            tensor, c2, target, kernel, hodge, point_cap=point_cap
        )
        for target in range(len(conf))
    ]
    euler = euler_characteristic(conf, dimensions)
    chi_omega1 = -euler // 2
    selectors: list[tuple[str, dict[str, int]]] = [
        ("changed_columns_0_to_5", {"max_changed_columns": 5})
    ] + [
        (f"changed_columns_{changed}", {"exact_changed_columns": changed})
        for changed in range(6, len(conf) + 1)
    ]
    layers: list[dict[str, Any]] = []
    complete = True
    survivor_found = False
    for label, selector in selectors:
        started = time.monotonic()
        try:
            result = search_first_survivor(
                tensor,
                c2,
                chi_omega1,
                columns,
                node_cap=node_cap,
                matrix_cap=matrix_cap,
                **selector,
            )
            layer = {
                "selector": label,
                "selector_parameters": selector,
                "elapsed_seconds": time.monotonic() - started,
                **result,
            }
        except Exception as exc:
            layer = {
                "selector": label,
                "selector_parameters": selector,
                "elapsed_seconds": time.monotonic() - started,
                "exhausted": False,
                "survivor_found": False,
                "status": "UNKNOWN_exception",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        layers.append(layer)
        complete = complete and bool(layer.get("exhausted"))
        survivor_found = survivor_found or bool(layer.get("survivor_found"))
        print(
            canonical_json(
                {
                    "Num": num,
                    "layer": label,
                    "exhausted": layer.get("exhausted"),
                    "survivor_found": layer.get("survivor_found"),
                    "search_stats": layer.get("search_stats"),
                }
            ),
            flush=True,
        )
        if survivor_found:
            break

    stats_keys = (
        "backtrack_node_count",
        "complete_unimodular_matrix_count",
        "finite_order_frame_preserving_count",
        "finite_order_frame_changing_count",
        "chi_Omega1_excluded_count",
        "E2chi_excluded_count",
        "E2chi_evaluation_count",
        "determinant_rejection_count",
    )
    totals = {
        key: sum(
            int(layer.get("search_stats", {}).get(key, 0)) for layer in layers
        )
        for key in stats_keys
    }
    passed = complete and len(layers) == len(selectors) and not survivor_found
    return {
        "schema_version": "2.0.0",
        "computation_id": "cicy_e2chi_layered_special_configuration_recomputation",
        "Num": num,
        "source": record.get("source"),
        "picard_rank": len(conf),
        "canonical_hash_working": record.get("canonical_hash_working"),
        "status": (
            "pass_E2chi_frame_preservation"
            if passed
            else (
                "frame_changing_finite_order_survivor_remains"
                if survivor_found
                else "UNKNOWN_incomplete_layered_enumeration"
            )
        ),
        "criterion_program": "exhaustive_changed-column_partition",
        "partition": {
            "selectors": [label for label, _ in selectors],
            "pairwise_disjoint": True,
            "covers_all_changed_column_counts": [0, len(conf)],
        },
        "chi_Omega1": chi_omega1,
        "ellipsoid_certificate": asdict(ellipsoid),
        "column_candidate_counts": [len(item.candidate_columns) for item in columns],
        "layers": layers,
        "totals": totals,
        "E2chi_no_frame_changing_survivors": passed,
        "finite_order_frame_changing_survivor_count": 0 if passed else None,
        "configuration_only": True,
        "defining_polynomial_used": False,
        "frozen_E2chi_result_read": False,
    }


def analyze_configuration_transport(
    source_record: dict[str, Any],
    target_record: dict[str, Any],
    source_result: dict[str, Any],
    target_e1: dict[str, Any] | None,
) -> dict[str, Any]:
    source_num = int(source_record["Num"])
    target_num = int(target_record["Num"])
    e1_ok, e1_reason = validate_full_e1_certificate(target_record, target_e1)
    if not e1_ok:
        return {
            "schema_version": "2.0.0",
            "Num": target_num,
            "status": "UNKNOWN_integral_E1_prerequisite",
            "E1_gate": e1_reason,
            "E2chi_no_frame_changing_survivors": False,
            "configuration_only": True,
            "defining_polynomial_used": False,
            "frozen_E2chi_result_read": False,
        }
    row_map, column_map = find_configuration_isomorphism(
        source_record["working_ambient_dimensions"],
        source_record["working_conf"],
        target_record["working_ambient_dimensions"],
        target_record["working_conf"],
    )
    source_pass = bool(source_result.get("E2chi_no_frame_changing_survivors"))
    return {
        "schema_version": "2.0.0",
        "computation_id": "cicy_e2chi_explicit_configuration_transport_recomputation",
        "Num": target_num,
        "source": target_record.get("source"),
        "picard_rank": len(target_record["working_conf"]),
        "canonical_hash_working": target_record.get("canonical_hash_working"),
        "status": (
            "pass_E2chi_frame_preservation"
            if source_pass
            else "UNKNOWN_source_case_not_certified"
        ),
        "criterion_program": "explicit_configuration_isomorphism_transport",
        "transported_from_Num": source_num,
        "source_result_sha256": record_sha256(source_result),
        "row_permutation_source_to_target_0based": row_map,
        "column_permutation_source_to_target_0based": column_map,
        "configuration_isomorphism_recomputed": True,
        "E2chi_no_frame_changing_survivors": source_pass,
        "finite_order_frame_changing_survivor_count": 0 if source_pass else None,
        "configuration_only": True,
        "defining_polynomial_used": False,
        "frozen_E2chi_result_read": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--e1-records", type=Path, default=DEFAULT_E1_RECORDS)
    parser.add_argument("--only-nums", help="comma-separated Nums and inclusive ranges")
    parser.add_argument("--exclude-nums", help="comma-separated Nums and inclusive ranges")
    parser.add_argument("--min-rank", type=int, default=0)
    parser.add_argument("--max-rank", type=int, default=10**9)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--point-cap", type=int)
    parser.add_argument("--matrix-cap", type=int)
    parser.add_argument(
        "--node-cap-special",
        "--node-cap-3936",
        dest="node_cap_special",
        type=int,
        help="diagnostic node cap for the layered high-rank configuration",
    )
    parser.add_argument(
        "--full-certificates",
        action="store_true",
        help="retain every candidate matrix; default output is a compact certificate",
    )
    parser.add_argument("--records-output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary-output", type=Path, default=DEFAULT_SUMMARY)
    arguments = parser.parse_args()
    if arguments.workers < 1:
        parser.error("--workers must be positive")

    started = time.monotonic()
    dataset = index_by_num(load_jsonl(arguments.dataset), "dataset")
    e1 = index_by_num(load_jsonl(arguments.e1_records), "E1 records")
    selected = set(e1)
    if arguments.only_nums:
        selected &= parse_num_specification(arguments.only_nums)
    if arguments.exclude_nums:
        selected -= parse_num_specification(arguments.exclude_nums)
    selected = {
        num
        for num in selected
        if arguments.min_rank
        <= int(dataset[num]["official_fields"]["H11"])
        <= arguments.max_rank
    }
    if arguments.only_nums:
        missing = parse_num_specification(arguments.only_nums) - set(e1)
        if missing:
            raise ValueError(f"requested Nums lack an E1 record: {sorted(missing)}")
    # Low-rank cases finish first, so a long proof run gives immediate,
    # informative progress instead of waiting for the smallest numerical label
    # when that label happens to have high Picard rank.
    special_hash_nums = {
        num
        for num in selected
        if dataset[num].get("canonical_hash_working") == SPECIAL_CANONICAL_HASH
    }
    use_special_transport = (
        SPECIAL_TRANSPORT in selected
        and SPECIAL_SOURCE in selected
        and dataset[SPECIAL_TRANSPORT].get("canonical_hash_working")
        == SPECIAL_CANONICAL_HASH
        and dataset[SPECIAL_SOURCE].get("canonical_hash_working")
        == SPECIAL_CANONICAL_HASH
    )
    layered_special_nums = set(special_hash_nums)
    if use_special_transport:
        layered_special_nums.remove(SPECIAL_TRANSPORT)
    specially_handled = special_hash_nums
    ordinary_nums = sorted(
        selected - specially_handled,
        key=lambda num: (int(dataset[num]["official_fields"]["H11"]), num),
    )
    results: dict[int, dict[str, Any]] = {}

    tasks = [
        (
            dataset[num],
            e1.get(num),
            arguments.point_cap,
            arguments.matrix_cap,
            arguments.full_certificates,
        )
        for num in ordinary_nums
    ]
    if arguments.workers == 1:
        iterator = map(_ordinary_worker, tasks)
        pool = None
    else:
        pool = concurrent.futures.ProcessPoolExecutor(max_workers=arguments.workers)
        iterator = pool.map(_ordinary_worker, tasks)
    try:
        for position, result in enumerate(iterator, start=1):
            results[int(result["Num"])] = result
            print(
                canonical_json(
                    {
                        "stage": "ordinary_E2chi",
                        "complete": position,
                        "total": len(ordinary_nums),
                        "Num": int(result["Num"]),
                        "status": result.get("status"),
                    }
                ),
                flush=True,
            )
    finally:
        if pool is not None:
            pool.shutdown()

    special_results: dict[int, dict[str, Any]] = {}
    for num in sorted(layered_special_nums):
        special_results[num] = analyze_layered_special(
            dataset[num],
            e1.get(num),
            point_cap=arguments.point_cap,
            node_cap=arguments.node_cap_special,
            matrix_cap=arguments.matrix_cap,
        )
        results[num] = special_results[num]
    if use_special_transport:
        results[SPECIAL_TRANSPORT] = analyze_configuration_transport(
            dataset[SPECIAL_SOURCE],
            dataset[SPECIAL_TRANSPORT],
            special_results[SPECIAL_SOURCE],
            e1.get(SPECIAL_TRANSPORT),
        )

    ordered = [results[num] for num in sorted(selected)]
    payload = "".join(canonical_json(record) + "\n" for record in ordered)
    _atomic_write(arguments.records_output, payload)
    pass_count = sum(
        bool(record.get("E2chi_no_frame_changing_survivors")) for record in ordered
    )
    status_counts = Counter(str(record.get("status")) for record in ordered)
    summary = {
        "schema_version": "2.0.0",
        "computation": "E2chi_from_configuration_after_integral_E1",
        "configuration_only": True,
        "special_defining_polynomial_used": False,
        "frozen_E2chi_result_read": False,
        "input": {
            "dataset": str(arguments.dataset.resolve()),
            "dataset_sha256": sha256_file(arguments.dataset),
            "E1_records": str(arguments.e1_records.resolve()),
            "E1_records_sha256": sha256_file(arguments.e1_records),
        },
        "target_count": len(ordered),
        "certified_frame_preservation_count": pass_count,
        "unresolved_count": len(ordered) - pass_count,
        "status_counts": dict(sorted(status_counts.items())),
        "special_routes": {
            "layered_configuration_nums": sorted(layered_special_nums),
            "4081_transport": use_special_transport,
        },
        "point_cap": arguments.point_cap,
        "matrix_cap": arguments.matrix_cap,
        "node_cap_special": arguments.node_cap_special,
        "full_certificates": arguments.full_certificates,
        "records_output": str(arguments.records_output.resolve()),
        "records_sha256": hashlib.sha256(payload.encode()).hexdigest(),
        "elapsed_seconds": time.monotonic() - started,
    }
    _atomic_write(arguments.summary_output, json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if pass_count == len(ordered) else 2


if __name__ == "__main__":
    raise SystemExit(main())
