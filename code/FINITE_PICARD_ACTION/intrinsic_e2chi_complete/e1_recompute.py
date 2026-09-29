#!/usr/bin/env python3
"""Recompute the integral E1 criterion from CICY configuration data.

This module is deliberately independent of the frozen E1 result ledgers.  Its
mathematical inputs are the configuration record, the favourable-rank gate
``h11 = number of ambient factors``, and the standard torsion-free theorem for
smooth non-product CICY threefolds.  Every remaining quantity is recomputed:

* projection/line/conic curve witnesses from the configuration matrix;
* the Smith normal form of the resulting divisor--curve pairing matrix;
* exact intersection, c2 and Hirzebruch--Riemann--Roch tests for possible
  finite-index overlattices;
* stable-map top-Chern witnesses when the preceding tests leave an
  overlattice candidate;
* the graph-isomorphism certificate when the displayed working matrix was
  obtained by the graph re-embedding used to repair E3.

No defining polynomial and no previously generated E1 certificate is read.
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
from pathlib import Path
from typing import Any, Iterable

from e1_generators.close_with_squarefree_curves import close_record
from e1_generators.e1_integral_snf_audit import audit_record as snf_audit_record
from e1_generators.e1_lattice_close import audit as augmented_lattice_audit
from e1_generators.e2_projection_scan import scan_record as projection_scan_record
from e1_generators.graph_provenance_audit import check_record as graph_check_record


PACKAGE_ROOT = Path(__file__).resolve().parent.parent
RELEASE_ROOT = PACKAGE_ROOT.parents[1]
DEFAULT_DATASET = RELEASE_ROOT / "data" / "cicy_criteria_input_7820.jsonl"
DEFAULT_TARGETS = RELEASE_ROOT / "data" / "extended_2635_nums.json"
DEFAULT_OUTPUT = RELEASE_ROOT / "reproduction_output" / "e1_complete_2635.jsonl"
DEFAULT_SUMMARY = RELEASE_ROOT / "reproduction_output" / "e1_complete_2635_summary.json"

TORSION_GATE = {
    "status": "pass_standard_nonproduct_CICY_Pic_torsion_free",
    "hypotheses": [
        "X is a smooth standard non-product CICY threefold",
        "H^1(X,O_X)=0",
    ],
    "argument": (
        "Standard smooth non-product CICY threefolds are simply connected. "
        "Hence H_1(X,Z)=0; the universal coefficient theorem makes H^2(X,Z) "
        "torsion-free, and the exponential sequence injects Pic(X) into H^2(X,Z)."
    ),
    "configuration_only": True,
    "special_defining_polynomial_used": False,
}


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def record_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            records.append(value)
    return records


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


def load_target_nums(path: Path) -> list[int]:
    value = json.loads(path.read_text(encoding="utf-8"))
    nums = [int(num) for num in value["nums"]]
    if int(value.get("count", -1)) != len(nums) or len(set(nums)) != len(nums):
        raise ValueError(f"invalid target inventory: {path}")
    return sorted(nums)


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


def _rational_favourability_gate(record: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    original = record["original_selected_conf"]
    h11 = int(record["official_fields"]["H11"])
    declared = record.get("hypotheses", {}).get(
        "rational_favourability_certified"
    )
    if declared is None:  # backward-compatible input reader
        declared = (
            record.get("E1_status", {}).get("original")
            == "pass_official_favourable_representative"
        )
    passed = (
        declared is True
        and h11 == len(original)
    )
    return passed, {
        "status": "pass" if passed else "UNKNOWN",
        "declared_source": record.get("source"),
        "declared_favourability": declared,
        "h11": h11,
        "ambient_factor_count": len(original),
        "note": (
            "The equality h11=m is a theorem/database input about the chosen favourable "
            "presentation; it is not inferred from a bare integer matrix."
        ),
    }


def _torsion_scope_gate(record: dict[str, Any]) -> bool:
    dimensions = [sum(row) - 1 for row in record["original_selected_conf"]]
    declared = record.get("hypotheses", {}).get("standard_nonproduct_cicy")
    if declared is None:  # backward-compatible input reader
        declared = record.get("official_fields", {}).get("IsProduct_original") is False
    return (
        declared is True
        and sum(dimensions) - len(record["original_selected_conf"][0]) == 3
    )


def _compute_snf_stage(record: dict[str, Any]) -> dict[str, Any]:
    projection = projection_scan_record(record)
    snf = snf_audit_record(record, projection)
    return {"record": record, "projection": projection, "snf": snf}


def _compute_augmented_stage(payload: dict[str, Any]) -> dict[str, Any]:
    augmented = augmented_lattice_audit(
        payload["record"], payload["projection"], "A6", do_hrr=True
    )
    return {**payload, "augmented": augmented}


def _compute_stable_map_stage(payload: dict[str, Any]) -> dict[str, Any]:
    return {**payload, "stable_map": close_record(payload["augmented"])}


def _finalize(payload: dict[str, Any]) -> dict[str, Any]:
    record = payload["record"]
    projection = payload["projection"]
    snf = payload["snf"]
    augmented = payload.get("augmented")
    stable_map = payload.get("stable_map")
    num = int(record["Num"])

    rational_pass, rational_gate = _rational_favourability_gate(record)
    torsion_pass = _torsion_scope_gate(record)
    graph = graph_check_record(record)
    graph_pass = graph["status"] in {"identity_presentation", "pass_graph_derivation"}

    factors = [int(value) for value in snf.get("smith_invariant_factors", [])]
    rank = len(record["original_selected_conf"])
    snf_pass = (
        int(snf.get("smith_rank", -1)) == rank
        and len(factors) == rank
        and all(value == 1 for value in factors)
        and snf.get("primitivity_mod_torsion_status") == "pass_snf_unit_invariants"
    )

    route: str | None = "SNF_unit_invariants" if snf_pass else None
    if route is None and augmented is not None and augmented.get("status") in {
        "pass_snf_unit_invariants",
        "pass_hrr_overlattice_exclusion",
    }:
        route = "augmented_curve_lattice_and_HRR"
    if route is None and stable_map is not None:
        stable_pass = (
            stable_map.get("status") == "pass_uniform_curve_overlattice_exclusion"
            and stable_map.get("unresolved_survivor_indices_0based") == []
            and len(stable_map.get("witnesses", [])) > 0
            and all(
                item.get("status") == "excluded_by_uniform_curve"
                for item in stable_map.get("survivors", [])
            )
        )
        if stable_pass:
            route = "uniform_stable_map_top_Chern_exclusion"

    passed = rational_pass and torsion_pass and graph_pass and route is not None
    frame_status = (
        "pass_identity_presentation"
        if graph["status"] == "identity_presentation"
        else (
            "pass_integral_identity_graph_transport"
            if graph_pass
            else "UNKNOWN_graph_transport"
        )
    )
    stages = [
        {"stage": "rational_favourability", **rational_gate},
        {
            "stage": "SNF",
            "status": "pass" if snf_pass else "not_saturated_by_initial_witnesses",
            "smith_invariant_factors": factors,
            "certificate_sha256": record_sha256(snf),
        },
    ]
    if augmented is not None:
        stages.append(
            {
                "stage": "augmented_lattice_HRR",
                "status": augmented.get("status"),
                "surviving_overlattice_count": len(
                    augmented.get("surviving_overlattices", [])
                ),
                "certificate_sha256": record_sha256(augmented),
            }
        )
    if stable_map is not None:
        stages.append(
            {
                "stage": "stable_map_top_Chern",
                "status": stable_map.get("status"),
                "witness_count": len(stable_map.get("witnesses", [])),
                "certificate_sha256": record_sha256(stable_map),
            }
        )
    stages.extend(
        [
            {
                "stage": "Picard_torsion_theorem",
                "status": "pass" if torsion_pass else "UNKNOWN",
            },
            {
                "stage": "working_frame_transport",
                "status": "pass" if graph_pass else "UNKNOWN",
                "detail": frame_status,
                "graph_certificate_sha256": record_sha256(graph),
            },
        ]
    )
    result = {
        "schema_version": "2.0.0",
        "Num": num,
        "source": record.get("source"),
        "canonical_hash_original": record.get("canonical_hash_original"),
        "canonical_hash_working": record.get("canonical_hash_working"),
        "integral_E1_status": "pass_integral_E1" if passed else "UNKNOWN_integral_E1",
        "E1_route": route,
        "stages": stages,
        "pic_torsion_free_gate": TORSION_GATE if torsion_pass else None,
        "working_frame_transport_status": frame_status,
        "E2chi_permitted": passed,
        "configuration_only": True,
        "special_defining_polynomial_used": False,
        "computed_from_configuration": True,
        "frozen_E1_result_read": False,
        "projection_certificate_sha256": record_sha256(projection),
    }
    if not passed:
        result["unresolved_reason"] = next(
            (
                stage["stage"]
                for stage in stages
                if str(stage.get("status", "")).startswith("UNKNOWN")
            ),
            "no_complete_integral_lattice_route",
        )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--targets", type=Path, default=DEFAULT_TARGETS)
    parser.add_argument("--only-nums", help="comma-separated Nums and inclusive ranges")
    parser.add_argument("--all-input", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    arguments = parser.parse_args()
    if arguments.workers < 1:
        parser.error("--workers must be positive")
    if arguments.only_nums and arguments.all_input:
        parser.error("use at most one of --only-nums and --all-input")

    started = time.monotonic()
    dataset = index_by_num(load_jsonl(arguments.dataset), "dataset")
    if arguments.only_nums:
        target_nums = sorted(parse_num_specification(arguments.only_nums))
    elif arguments.all_input:
        target_nums = sorted(dataset)
    else:
        target_nums = load_target_nums(arguments.targets)
    missing = sorted(set(target_nums) - set(dataset))
    if missing:
        raise ValueError(f"target Nums absent from dataset: {missing}")

    # The first stage is inexpensive and deterministic; keeping it ordered
    # makes progress and any failure easy to diagnose.
    payloads: dict[int, dict[str, Any]] = {}
    for position, num in enumerate(target_nums, start=1):
        payloads[num] = _compute_snf_stage(dataset[num])
        if position % 250 == 0 or position == len(target_nums):
            print(canonical_json({"stage": "SNF", "complete": position, "total": len(target_nums)}), flush=True)

    augmented_nums = [
        num
        for num in target_nums
        if payloads[num]["snf"].get("primitivity_mod_torsion_status")
        != "pass_snf_unit_invariants"
    ]
    if arguments.workers == 1:
        augmented_payloads = map(_compute_augmented_stage, (payloads[num] for num in augmented_nums))
    else:
        pool = concurrent.futures.ProcessPoolExecutor(max_workers=arguments.workers)
        augmented_payloads = pool.map(
            _compute_augmented_stage, (payloads[num] for num in augmented_nums)
        )
    try:
        for position, payload in enumerate(augmented_payloads, start=1):
            num = int(payload["record"]["Num"])
            payloads[num] = payload
            if position % 25 == 0 or position == len(augmented_nums):
                print(canonical_json({"stage": "augmented_lattice_HRR", "complete": position, "total": len(augmented_nums)}), flush=True)
    finally:
        if arguments.workers != 1:
            pool.shutdown()

    stable_nums = [
        num
        for num in augmented_nums
        if payloads[num].get("augmented", {}).get("status")
        == "unverified_overlattice_survives"
    ]
    if arguments.workers == 1:
        stable_payloads = map(_compute_stable_map_stage, (payloads[num] for num in stable_nums))
    else:
        pool = concurrent.futures.ProcessPoolExecutor(max_workers=arguments.workers)
        stable_payloads = pool.map(
            _compute_stable_map_stage, (payloads[num] for num in stable_nums)
        )
    try:
        for position, payload in enumerate(stable_payloads, start=1):
            num = int(payload["record"]["Num"])
            payloads[num] = payload
            print(canonical_json({"stage": "stable_map_top_Chern", "complete": position, "total": len(stable_nums), "Num": num}), flush=True)
    finally:
        if arguments.workers != 1:
            pool.shutdown()

    records = [_finalize(payloads[num]) for num in target_nums]
    output_payload = "".join(canonical_json(record) + "\n" for record in records)
    _atomic_write(arguments.output, output_payload)
    status_counts = Counter(record["integral_E1_status"] for record in records)
    route_counts = Counter(str(record["E1_route"]) for record in records)
    summary = {
        "schema_version": "2.0.0",
        "computation": "integral_E1_from_configuration",
        "configuration_only": True,
        "special_defining_polynomial_used": False,
        "frozen_E1_result_read": False,
        "input": {
            "dataset": str(arguments.dataset.resolve()),
            "dataset_sha256": sha256_file(arguments.dataset),
            "targets": None if arguments.all_input or arguments.only_nums else str(arguments.targets.resolve()),
            "targets_sha256": (
                None
                if arguments.all_input or arguments.only_nums
                else sha256_file(arguments.targets)
            ),
        },
        "target_count": len(target_nums),
        "status_counts": dict(sorted(status_counts.items())),
        "route_counts": dict(sorted(route_counts.items())),
        "augmented_lattice_case_count": len(augmented_nums),
        "stable_map_case_count": len(stable_nums),
        "output": str(arguments.output.resolve()),
        "output_sha256": hashlib.sha256(output_payload.encode()).hexdigest(),
        "elapsed_seconds": time.monotonic() - started,
    }
    _atomic_write(arguments.summary, json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if status_counts == {"pass_integral_E1": len(records)} else 2


if __name__ == "__main__":
    raise SystemExit(main())
