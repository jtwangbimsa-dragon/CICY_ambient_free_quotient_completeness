from __future__ import annotations

import json
from pathlib import Path

import sympy as sp

from intrinsic_e2chi_complete.candidate_enumerator import (
    DEFAULT_DATASET,
    analyze_case,
    c2_ellipsoid_certificate,
    shifted_ellipsoid_points,
)
from intrinsic_e2chi_complete.e1_recompute import (
    _compute_augmented_stage,
    _compute_snf_stage,
    _compute_stable_map_stage,
    _finalize,
)


DATASET = DEFAULT_DATASET


def record_by_num(path: Path, num: int) -> dict:
    with path.open() as stream:
        return next(record for record in map(json.loads, stream) if int(record["Num"]) == num)


def recompute_e1(source: dict) -> dict:
    payload = _compute_snf_stage(source)
    if payload["snf"].get("primitivity_mod_torsion_status") != "pass_snf_unit_invariants":
        payload = _compute_augmented_stage(payload)
        if payload["augmented"].get("status") == "unverified_overlattice_survives":
            payload = _compute_stable_map_stage(payload)
    return _finalize(payload)


def test_shifted_fincke_pohst_matches_brute_box() -> None:
    form = sp.Matrix([[3, 1], [1, 2]])
    center = sp.Matrix([sp.Rational(1, 3), sp.Rational(-2, 5)])
    radius = sp.Rational(19, 3)
    exact = set(shifted_ellipsoid_points(form, center, radius))
    brute = {
        (x, y)
        for x in range(-10, 11)
        for y in range(-10, 11)
        if ((sp.Matrix([x, y]) - center).T * form * (sp.Matrix([x, y]) - center))[0]
        <= radius
    }
    assert exact == brute


def test_num7037_complete_rank5_smoke_after_full_e1() -> None:
    source = record_by_num(DATASET, 7037)
    e1 = recompute_e1(source)
    result = analyze_case(source, e1, point_cap=10_000, matrix_cap=10_000)

    assert result["ellipsoid_certificate"]["negative_restriction_certified"] is True
    assert result["candidate_matrix_count"] == 4
    assert result["finite_order_frame_changing_survivor_count"] == 0
    assert result["H3_free_frame_preservation_certified"] is True
    assert result["E2chi_no_frame_changing_survivors"] is True
    assert result["status"] == "pass_E2chi_frame_preservation"


def test_num7303_known_reflection_is_excluded_in_complete_envelope() -> None:
    source = record_by_num(DATASET, 7303)
    e1 = recompute_e1(source)
    result = analyze_case(source, e1, point_cap=100_000, matrix_cap=100_000)

    assert result["ellipsoid_certificate"]["negative_restriction_certified"] is True
    assert result["candidate_screening_mode"] == "certified_group_conjugacy_classes"
    assert result["candidate_group_certificate"]["certified"] is True
    assert any(
        item["status"] in {"excluded_by_chi_Omega1", "excluded_by_E2chi_any_failure"}
        for item in result["finite_order_frame_changing_records"]
    )


def test_incomplete_e1_is_blocked_before_e2chi_starts() -> None:
    source = record_by_num(DATASET, 7037)
    e1 = recompute_e1(source)
    e1["integral_E1_status"] = "conditional_pass_requires_torsion_free_gate"
    result = analyze_case(source, e1, point_cap=100_000, matrix_cap=100_000)

    assert result["integral_E1_certified"] is False
    assert result["status"] == "blocked_E1_not_certified"
    assert result["E2chi_started"] is False
    assert result["H3_free_frame_preservation_certified"] is False


def test_working_graph_reembedding_is_bound_into_the_e1_certificate() -> None:
    source = record_by_num(DATASET, 414)
    e1 = recompute_e1(source)
    assert source["presentation_relation"] == "scheme_isomorphic_graph_embedding"
    assert source["obstruction_mappings"]
    assert e1["integral_E1_status"] == "pass_integral_E1"
    assert e1["working_frame_transport_status"] == "pass_integral_identity_graph_transport"
    assert e1["canonical_hash_working"] == source["canonical_hash_working"]


def test_negative_restriction_failure_is_not_promoted() -> None:
    tensor = [
        [[1, 0], [0, 0]],
        [[0, 0], [0, 1]],
    ]
    certificate, _kernel, _form = c2_ellipsoid_certificate(tensor, [1, 0])
    assert certificate.negative_restriction_certified is False
