from __future__ import annotations

import json

from intrinsic_e2chi_complete.candidate_enumerator import DEFAULT_DATASET
from intrinsic_e2chi_complete.e1_recompute import (
    _compute_augmented_stage,
    _compute_snf_stage,
    _compute_stable_map_stage,
    _finalize,
)


def source_record(num: int) -> dict:
    with DEFAULT_DATASET.open() as stream:
        return next(record for record in map(json.loads, stream) if int(record["Num"]) == num)


def recompute(num: int) -> dict:
    payload = _compute_snf_stage(source_record(num))
    if payload["snf"].get("primitivity_mod_torsion_status") != "pass_snf_unit_invariants":
        payload = _compute_augmented_stage(payload)
        if payload["augmented"].get("status") == "unverified_overlattice_survives":
            payload = _compute_stable_map_stage(payload)
    return _finalize(payload)


def test_each_integral_E1_route_is_recomputed_from_the_configuration() -> None:
    assert recompute(55)["E1_route"] == "SNF_unit_invariants"
    assert recompute(96)["E1_route"] == "augmented_curve_lattice_and_HRR"
    assert recompute(3398)["E1_route"] == "uniform_stable_map_top_Chern_exclusion"


def test_recomputed_e1_records_do_not_claim_frozen_input() -> None:
    for num in (55, 96, 3398):
        record = recompute(num)
        assert record["integral_E1_status"] == "pass_integral_E1"
        assert record["frozen_E1_result_read"] is False
        assert record["special_defining_polynomial_used"] is False
