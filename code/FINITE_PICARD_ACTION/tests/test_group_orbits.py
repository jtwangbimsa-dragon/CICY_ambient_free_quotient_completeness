from __future__ import annotations

from collections import deque

from intrinsic_e2chi_complete.candidate_enumerator import permutes_ambient_frame
from intrinsic_e2chi_complete.core import compute_e2chi_certificate, finite_matrix_order
from intrinsic_e2chi_complete.group_orbits import (
    certify_finite_matrix_group,
    conjugacy_class_records,
    identity_key,
    inverse_key,
    matrix_key,
    matrix_sha256,
    multiply_keys,
    screen_e2chi_by_conjugacy_or_fallback,
)


def generated_group(*generators):
    checked = tuple(matrix_key(generator) for generator in generators)
    rank = len(checked[0])
    alphabet = checked + tuple(inverse_key(generator) for generator in checked)
    result = {identity_key(rank)}
    queue = deque(result)
    while queue:
        element = queue.popleft()
        for generator in alphabet:
            product = multiply_keys(element, generator)
            if product not in result:
                result.add(product)
                queue.append(product)
    return tuple(sorted(result))


def zero_cubic(rank: int):
    return [[[0 for _ in range(rank)] for _ in range(rank)] for _ in range(rank)]


def sequential_statuses(matrices, d, c2, chi_omega1):
    statuses = {}
    for matrix in matrices:
        order = finite_matrix_order(matrix)
        if permutes_ambient_frame(matrix):
            continue
        if chi_omega1 % order:
            status = "excluded_by_chi_Omega1"
        else:
            certificate = compute_e2chi_certificate(d, c2, matrix)
            status = (
                "excluded_by_E2chi_any_failure"
                if certificate.e2chi_excludes_free_lift
                else "frame_changing_finite_order_survivor"
            )
        statuses[matrix_sha256(matrix)] = status
    return statuses


def test_s3_group_certificate_and_generator_conjugacy_orbits() -> None:
    rotation = ((0, -1), (1, -1))
    reflection = ((0, 1), (1, 0))
    group = generated_group(rotation, reflection)
    assert len(group) == 6

    certificate = certify_finite_matrix_group(group)
    assert certificate.certified is True
    assert certificate.group_order == 6
    assert certificate.identity_present is True
    assert certificate.every_inverse_present is True
    assert certificate.generator_closure_certified is True
    assert 1 <= certificate.generator_count <= 2

    repeated_certificate, classes = conjugacy_class_records(group)
    assert repeated_certificate.certified is True
    assert sorted(record.class_size for record in classes) == [1, 2, 3]
    assert all(record.class_size * record.centralizer_order == 6 for record in classes)


def test_non_group_with_identity_and_inverses_falls_back_without_loss() -> None:
    identity = ((1, 0), (0, 1))
    first = ((-1, 0), (0, 1))
    second = ((1, 0), (0, -1))
    candidates = (identity, first, second)

    certificate = certify_finite_matrix_group(candidates)
    assert certificate.certified is False
    assert certificate.identity_present is True
    assert certificate.every_inverse_present is True
    assert certificate.failure_stage == "generator_closure"

    result = screen_e2chi_by_conjugacy_or_fallback(
        candidates,
        zero_cubic(2),
        [0, 0],
        chi_omega1=1,
    )
    assert result["screening_mode"] == "elementwise_fallback"
    assert result["group_certificate"]["certified"] is False
    assert result["finite_order_frame_preserving_count"] == 1
    assert result["finite_order_frame_changing_count"] == 2
    assert result["finite_order_frame_changing_excluded_count"] == 2
    assert len(result["element_records"]) == 2


def test_group_acceleration_matches_independent_elementwise_s3_screen() -> None:
    rotation = ((0, -1), (1, -1))
    reflection = ((0, 1), (1, 0))
    group = generated_group(rotation, reflection)
    d = zero_cubic(2)
    c2 = [0, 0]

    result = screen_e2chi_by_conjugacy_or_fallback(group, d, c2, chi_omega1=0)
    assert result["screening_mode"] == "certified_group_conjugacy_classes"
    assert result["group_certificate"]["group_order"] == 6
    assert result["conjugacy_class_count"] == 3
    assert result["topological_generator_gate"] is True

    accelerated = {
        record["matrix_sha256"]: record["status"] for record in result["element_records"]
    }
    assert accelerated == sequential_statuses(group, d, c2, chi_omega1=0)
    assert result["finite_order_frame_changing_count"] == len(accelerated)
    # At least one S3 conjugacy class mixes the ambient-frame flag, so this
    # also exercises the required per-element marking inside a class.
    assert any(
        record["frame_changing_member_count"] and record["frame_preserving_member_count"]
        for record in result["class_records"]
    )


def test_duplicate_candidate_input_uses_elementwise_fallback_entries() -> None:
    identity = ((1,),)
    candidates = (identity, identity)
    result = screen_e2chi_by_conjugacy_or_fallback(
        candidates,
        zero_cubic(1),
        [0],
        chi_omega1=0,
    )
    assert result["screening_mode"] == "elementwise_fallback"
    assert result["group_certificate"]["failure_stage"] == "uniqueness"
    assert result["finite_order_frame_preserving_count"] == 2

