#!/usr/bin/env python3
"""Fail-closed conjugacy-class acceleration for E2-chi screening.

The ellipsoid enumerator returns a finite *candidate set* of integral
matrices.  In general that set need not be a group, so conjugacy-class
compression is unsafe unless group closure is proved first.  This module
implements the following exact gate:

1. the input is a duplicate-free finite set of square integral matrices;
2. the identity is present;
3. every matrix is unimodular and its exact inverse is in the set;
4. an incrementally constructed generator set reaches the whole candidate
   set, and right multiplication by every generator and inverse never leaves
   the set.

Only after all four checks pass do we compute conjugacy classes as orbits
under conjugation by the certified generators.  The E2-chi obstruction is
constant on such a class because every generator is separately checked to
preserve ``d`` and ``c2``.  Frame preservation is *not* assumed to be a class
property and is therefore still marked element by element.

If any group or topological-invariance check fails, the public screening
function automatically falls back to the original elementwise computation.
All arithmetic is exact and no defining polynomial is used.
"""

from __future__ import annotations

import hashlib
import json
from collections import deque
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Sequence

import sympy as sp

from intrinsic_e2chi_complete.core import compute_e2chi_certificate, finite_matrix_order


MatrixKey = tuple[tuple[int, ...], ...]
IntegerCubic = Sequence[Sequence[Sequence[int]]]


class GroupOrbitError(RuntimeError):
    """An internal exact group-orbit consistency check failed."""


def _exact_integer(value: object, label: str) -> int:
    exact = sp.sympify(value)
    if exact.is_Integer is not True:
        raise ValueError(f"{label} must be an exact integer, got {value!r}")
    return int(exact)


def matrix_key(matrix: Sequence[Sequence[int]] | sp.MatrixBase) -> MatrixKey:
    """Return a checked immutable row-major matrix key."""

    if isinstance(matrix, sp.MatrixBase):
        if matrix.rows == 0 or matrix.rows != matrix.cols:
            raise ValueError("matrix must be nonempty and square")
        return tuple(
            tuple(_exact_integer(matrix[i, j], f"matrix[{i}][{j}]") for j in range(matrix.cols))
            for i in range(matrix.rows)
        )
    rows = [tuple(row) for row in matrix]
    if not rows or any(len(row) != len(rows) for row in rows):
        raise ValueError("matrix must be nonempty and square")
    return tuple(
        tuple(_exact_integer(value, f"matrix[{i}][{j}]") for j, value in enumerate(row))
        for i, row in enumerate(rows)
    )


def identity_key(rank: int) -> MatrixKey:
    if rank < 1:
        raise ValueError("rank must be positive")
    return tuple(tuple(int(i == j) for j in range(rank)) for i in range(rank))


def multiply_keys(first: MatrixKey, second: MatrixKey) -> MatrixKey:
    """Multiply two equally sized integral matrices using Python integers."""

    rank = len(first)
    if len(second) != rank:
        raise ValueError("matrix ranks disagree")
    return tuple(
        tuple(sum(first[i][k] * second[k][j] for k in range(rank)) for j in range(rank))
        for i in range(rank)
    )


def inverse_key(matrix: MatrixKey) -> MatrixKey:
    """Return the exact integral inverse of a unimodular matrix."""

    exact = sp.Matrix(matrix)
    determinant = int(exact.det())
    if abs(determinant) != 1:
        raise ValueError(f"matrix determinant is {determinant}, not +/-1")
    inverse = exact.inv()
    if any(sp.denom(value) != 1 for value in inverse):
        raise ArithmeticError("a unimodular matrix unexpectedly has a nonintegral inverse")
    result = matrix_key(inverse)
    unit = identity_key(len(matrix))
    if multiply_keys(matrix, result) != unit or multiply_keys(result, matrix) != unit:
        raise ArithmeticError("exact inverse multiplication check failed")
    return result


def matrix_sha256(matrix: MatrixKey) -> str:
    flat = [value for row in matrix for value in row]
    payload = json.dumps(flat, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def permutes_ambient_frame(matrix: Sequence[Sequence[int]] | sp.MatrixBase) -> bool:
    """Test the ambient-frame permutation property element by element."""

    checked = matrix_key(matrix)
    rank = len(checked)
    columns = [tuple(checked[i][j] for i in range(rank)) for j in range(rank)]
    standard = [tuple(int(i == j) for i in range(rank)) for j in range(rank)]
    return sorted(columns) == sorted(standard)


@dataclass(frozen=True)
class FiniteGroupCertificate:
    certified: bool
    candidate_count: int
    distinct_candidate_count: int
    rank: int | None
    identity_present: bool
    all_unimodular: bool
    every_inverse_present: bool
    generator_closure_certified: bool
    group_order: int | None
    generator_count: int
    generator_sha256: tuple[str, ...]
    inverse_checks: int
    generator_edge_checks: int
    generated_element_count: int
    failure_stage: str | None
    failure_reason: str | None
    exact_arithmetic: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class _CertifiedGroup:
    certificate: FiniteGroupCertificate
    matrices: tuple[MatrixKey, ...]
    inverse_by_matrix: dict[MatrixKey, MatrixKey]
    generators: tuple[MatrixKey, ...]


def _failed_certificate(
    *,
    candidate_count: int,
    distinct_candidate_count: int,
    rank: int | None,
    identity_present: bool,
    all_unimodular: bool,
    every_inverse_present: bool,
    generator_closure_certified: bool,
    generators: Sequence[MatrixKey],
    inverse_checks: int,
    edge_checks: int,
    generated_count: int,
    stage: str,
    reason: str,
) -> FiniteGroupCertificate:
    return FiniteGroupCertificate(
        certified=False,
        candidate_count=candidate_count,
        distinct_candidate_count=distinct_candidate_count,
        rank=rank,
        identity_present=identity_present,
        all_unimodular=all_unimodular,
        every_inverse_present=every_inverse_present,
        generator_closure_certified=generator_closure_certified,
        group_order=None,
        generator_count=len(generators),
        generator_sha256=tuple(matrix_sha256(item) for item in generators),
        inverse_checks=inverse_checks,
        generator_edge_checks=edge_checks,
        generated_element_count=generated_count,
        failure_stage=stage,
        failure_reason=reason,
    )


def _certify_group(
    matrices: Iterable[Sequence[Sequence[int]] | sp.MatrixBase],
) -> _CertifiedGroup:
    """Produce a group witness or a fail-closed negative audit.

    Generator closure is proved without an all-pairs multiplication table.
    Starting from the identity, a missing candidate is added as a generator;
    breadth-first right multiplication by all generators and their inverses
    then computes the generated subgroup.  Every traversed edge is required
    to remain in the candidate set.  When the traversal reaches the whole set
    we have equality with the subgroup generated by the recorded generators,
    hence full multiplication closure.
    """

    raw = list(matrices)
    candidate_count = len(raw)
    if not raw:
        certificate = _failed_certificate(
            candidate_count=0,
            distinct_candidate_count=0,
            rank=None,
            identity_present=False,
            all_unimodular=False,
            every_inverse_present=False,
            generator_closure_certified=False,
            generators=(),
            inverse_checks=0,
            edge_checks=0,
            generated_count=0,
            stage="normalisation",
            reason="candidate set is empty",
        )
        return _CertifiedGroup(certificate, (), {}, ())

    try:
        checked = tuple(matrix_key(matrix) for matrix in raw)
    except (TypeError, ValueError) as exc:
        certificate = _failed_certificate(
            candidate_count=candidate_count,
            distinct_candidate_count=0,
            rank=None,
            identity_present=False,
            all_unimodular=False,
            every_inverse_present=False,
            generator_closure_certified=False,
            generators=(),
            inverse_checks=0,
            edge_checks=0,
            generated_count=0,
            stage="normalisation",
            reason=str(exc),
        )
        return _CertifiedGroup(certificate, (), {}, ())

    rank = len(checked[0])
    if any(len(matrix) != rank for matrix in checked):
        certificate = _failed_certificate(
            candidate_count=candidate_count,
            distinct_candidate_count=len(set(checked)),
            rank=rank,
            identity_present=False,
            all_unimodular=False,
            every_inverse_present=False,
            generator_closure_certified=False,
            generators=(),
            inverse_checks=0,
            edge_checks=0,
            generated_count=0,
            stage="normalisation",
            reason="candidate matrices have different ranks",
        )
        return _CertifiedGroup(certificate, checked, {}, ())

    candidate_set = set(checked)
    distinct_count = len(candidate_set)
    unit = identity_key(rank)
    identity_present = unit in candidate_set
    if distinct_count != candidate_count:
        certificate = _failed_certificate(
            candidate_count=candidate_count,
            distinct_candidate_count=distinct_count,
            rank=rank,
            identity_present=identity_present,
            all_unimodular=False,
            every_inverse_present=False,
            generator_closure_certified=False,
            generators=(),
            inverse_checks=0,
            edge_checks=0,
            generated_count=int(identity_present),
            stage="uniqueness",
            reason="candidate input contains duplicate matrices",
        )
        return _CertifiedGroup(certificate, tuple(sorted(candidate_set)), {}, ())
    if not identity_present:
        certificate = _failed_certificate(
            candidate_count=candidate_count,
            distinct_candidate_count=distinct_count,
            rank=rank,
            identity_present=False,
            all_unimodular=False,
            every_inverse_present=False,
            generator_closure_certified=False,
            generators=(),
            inverse_checks=0,
            edge_checks=0,
            generated_count=0,
            stage="identity",
            reason="identity matrix is absent",
        )
        return _CertifiedGroup(certificate, tuple(sorted(candidate_set)), {}, ())

    inverse_by_matrix: dict[MatrixKey, MatrixKey] = {}
    inverse_checks = 0
    for matrix in sorted(candidate_set):
        try:
            inverse = inverse_key(matrix)
        except (ArithmeticError, ValueError) as exc:
            certificate = _failed_certificate(
                candidate_count=candidate_count,
                distinct_candidate_count=distinct_count,
                rank=rank,
                identity_present=True,
                all_unimodular=False,
                every_inverse_present=False,
                generator_closure_certified=False,
                generators=(),
                inverse_checks=inverse_checks,
                edge_checks=0,
                generated_count=1,
                stage="unimodularity",
                reason=str(exc),
            )
            return _CertifiedGroup(certificate, tuple(sorted(candidate_set)), inverse_by_matrix, ())
        inverse_checks += 1
        if inverse not in candidate_set:
            certificate = _failed_certificate(
                candidate_count=candidate_count,
                distinct_candidate_count=distinct_count,
                rank=rank,
                identity_present=True,
                all_unimodular=True,
                every_inverse_present=False,
                generator_closure_certified=False,
                generators=(),
                inverse_checks=inverse_checks,
                edge_checks=0,
                generated_count=1,
                stage="inverse_membership",
                reason=f"inverse of {matrix_sha256(matrix)} is absent",
            )
            return _CertifiedGroup(certificate, tuple(sorted(candidate_set)), inverse_by_matrix, ())
        inverse_by_matrix[matrix] = inverse

    generators: list[MatrixKey] = []
    generated: set[MatrixKey] = {unit}
    edge_checks = 0
    while generated != candidate_set:
        generator = min(candidate_set - generated)
        generators.append(generator)
        alphabet = tuple(
            dict.fromkeys(
                item
                for selected in generators
                for item in (selected, inverse_by_matrix[selected])
            )
        )
        queue: deque[MatrixKey] = deque(sorted(generated))
        while queue:
            element = queue.popleft()
            for letter in alphabet:
                edge_checks += 1
                product = multiply_keys(element, letter)
                if product not in candidate_set:
                    certificate = _failed_certificate(
                        candidate_count=candidate_count,
                        distinct_candidate_count=distinct_count,
                        rank=rank,
                        identity_present=True,
                        all_unimodular=True,
                        every_inverse_present=True,
                        generator_closure_certified=False,
                        generators=generators,
                        inverse_checks=inverse_checks,
                        edge_checks=edge_checks,
                        generated_count=len(generated),
                        stage="generator_closure",
                        reason=(
                            f"right multiplication left the candidate set: "
                            f"{matrix_sha256(element)} * {matrix_sha256(letter)}"
                        ),
                    )
                    return _CertifiedGroup(
                        certificate,
                        tuple(sorted(candidate_set)),
                        inverse_by_matrix,
                        tuple(generators),
                    )
                if product not in generated:
                    generated.add(product)
                    queue.append(product)

    certificate = FiniteGroupCertificate(
        certified=True,
        candidate_count=candidate_count,
        distinct_candidate_count=distinct_count,
        rank=rank,
        identity_present=True,
        all_unimodular=True,
        every_inverse_present=True,
        generator_closure_certified=True,
        group_order=distinct_count,
        generator_count=len(generators),
        generator_sha256=tuple(matrix_sha256(item) for item in generators),
        inverse_checks=inverse_checks,
        generator_edge_checks=edge_checks,
        generated_element_count=len(generated),
        failure_stage=None,
        failure_reason=None,
    )
    return _CertifiedGroup(
        certificate,
        tuple(sorted(candidate_set)),
        inverse_by_matrix,
        tuple(generators),
    )


def certify_finite_matrix_group(
    matrices: Iterable[Sequence[Sequence[int]] | sp.MatrixBase],
) -> FiniteGroupCertificate:
    """Return the public exact group certificate for a candidate set."""

    return _certify_group(matrices).certificate


@dataclass(frozen=True)
class ConjugacyClassRecord:
    class_id: int
    representative_sha256: str
    member_sha256: tuple[str, ...]
    class_size: int
    centralizer_order: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _conjugacy_classes(group: _CertifiedGroup) -> tuple[tuple[MatrixKey, ...], ...]:
    if not group.certificate.certified:
        raise ValueError("conjugacy classes require a certified group")
    group_set = set(group.matrices)
    conjugators = tuple(
        dict.fromkeys(
            item
            for generator in group.generators
            for item in (generator, group.inverse_by_matrix[generator])
        )
    )
    unassigned = set(group.matrices)
    classes: list[tuple[MatrixKey, ...]] = []
    while unassigned:
        representative = min(unassigned)
        orbit = {representative}
        queue: deque[MatrixKey] = deque([representative])
        while queue:
            element = queue.popleft()
            for generator in conjugators:
                conjugate = multiply_keys(
                    multiply_keys(generator, element),
                    group.inverse_by_matrix[generator],
                )
                if conjugate not in group_set:
                    raise GroupOrbitError("generator conjugation escaped a certified group")
                if conjugate not in orbit:
                    orbit.add(conjugate)
                    queue.append(conjugate)
        if not orbit <= unassigned:
            raise GroupOrbitError("conjugacy orbits overlap before partition assignment")
        ordered = tuple(sorted(orbit))
        if len(group.matrices) % len(ordered):
            raise GroupOrbitError("conjugacy class size does not divide the group order")
        classes.append(ordered)
        unassigned.difference_update(orbit)
    if sum(map(len, classes)) != len(group.matrices):
        raise GroupOrbitError("conjugacy classes do not partition the group")
    return tuple(classes)


def conjugacy_class_records(
    matrices: Iterable[Sequence[Sequence[int]] | sp.MatrixBase],
) -> tuple[FiniteGroupCertificate, tuple[ConjugacyClassRecord, ...]]:
    """Certify the group and return generator-orbit conjugacy classes."""

    group = _certify_group(matrices)
    if not group.certificate.certified:
        return group.certificate, ()
    classes = _conjugacy_classes(group)
    order = len(group.matrices)
    records = tuple(
        ConjugacyClassRecord(
            class_id=index + 1,
            representative_sha256=matrix_sha256(members[0]),
            member_sha256=tuple(matrix_sha256(member) for member in members),
            class_size=len(members),
            centralizer_order=order // len(members),
        )
        for index, members in enumerate(classes)
    )
    return group.certificate, records


def _normalise_cubic(d: IntegerCubic, rank: int) -> tuple[tuple[tuple[int, ...], ...], ...]:
    if len(d) != rank:
        raise ValueError("d must have shape (rank, rank, rank)")
    tensor = tuple(
        tuple(
            tuple(_exact_integer(value, f"d[{i}][{j}][{k}]") for k, value in enumerate(row))
            for j, row in enumerate(plane)
        )
        for i, plane in enumerate(d)
    )
    if any(len(plane) != rank for plane in tensor) or any(
        len(row) != rank for plane in tensor for row in plane
    ):
        raise ValueError("d must have shape (rank, rank, rank)")
    for i in range(rank):
        for j in range(rank):
            for k in range(rank):
                orbit = {
                    tensor[i][j][k],
                    tensor[i][k][j],
                    tensor[j][i][k],
                    tensor[j][k][i],
                    tensor[k][i][j],
                    tensor[k][j][i],
                }
                if len(orbit) != 1:
                    raise ValueError(f"d is not symmetric at indices {(i, j, k)}")
    return tensor


def _normalise_c2(c2: Sequence[int], rank: int) -> tuple[int, ...]:
    if len(c2) != rank:
        raise ValueError("c2 must have length rank")
    return tuple(_exact_integer(value, f"c2[{i}]") for i, value in enumerate(c2))


def _preserves_c2(matrix: MatrixKey, c2: tuple[int, ...]) -> bool:
    rank = len(matrix)
    return all(sum(c2[i] * matrix[i][j] for i in range(rank)) == c2[j] for j in range(rank))


def _cubic_value(
    d: tuple[tuple[tuple[int, ...], ...], ...],
    first: Sequence[int],
    second: Sequence[int],
    third: Sequence[int],
) -> int:
    rank = len(d)
    return sum(
        d[i][j][k] * first[i] * second[j] * third[k]
        for i in range(rank)
        for j in range(rank)
        for k in range(rank)
    )


def _preserves_cubic(
    matrix: MatrixKey,
    d: tuple[tuple[tuple[int, ...], ...], ...],
) -> bool:
    rank = len(matrix)
    columns = [tuple(matrix[i][j] for i in range(rank)) for j in range(rank)]
    standard = [tuple(int(i == j) for i in range(rank)) for j in range(rank)]
    return all(
        _cubic_value(d, columns[i], columns[j], columns[k])
        == _cubic_value(d, standard[i], standard[j], standard[k])
        for i in range(rank)
        for j in range(i, rank)
        for k in range(j, rank)
    )


def _elementwise_screen(
    matrices: tuple[MatrixKey, ...],
    d: IntegerCubic,
    c2: Sequence[int],
    chi_omega1: int,
) -> dict[str, Any]:
    """The no-loss fallback, matching the original per-matrix semantics."""

    element_records: list[dict[str, Any]] = []
    infinite_order_count = 0
    frame_preserving_count = 0
    frame_changing_count = 0
    excluded_count = 0
    survivor_count = 0
    e2chi_evaluations = 0
    for matrix in matrices:
        try:
            order = finite_matrix_order(matrix)
        except ValueError:
            infinite_order_count += 1
            continue
        if permutes_ambient_frame(matrix):
            frame_preserving_count += 1
            continue
        frame_changing_count += 1
        record: dict[str, Any] = {
            "matrix_sha256": matrix_sha256(matrix),
            "matrix_order": order,
            "frame_changing": True,
            "evaluation_mode": "elementwise_fallback",
        }
        if int(chi_omega1) % order:
            excluded_count += 1
            record.update(
                status="excluded_by_chi_Omega1",
                chi_Omega1=int(chi_omega1),
                remainder=int(chi_omega1) % order,
            )
        else:
            certificate = compute_e2chi_certificate(d, c2, matrix)
            e2chi_evaluations += 1
            record["E2chi"] = certificate.to_dict()
            if certificate.e2chi_excludes_free_lift:
                excluded_count += 1
                record["status"] = "excluded_by_E2chi_any_failure"
            else:
                survivor_count += 1
                record["status"] = "frame_changing_finite_order_survivor"
        element_records.append(record)
    return {
        "screening_mode": "elementwise_fallback",
        "conjugacy_class_count": None,
        "conjugacy_class_evaluation_count": None,
        "e2chi_evaluation_count": e2chi_evaluations,
        "infinite_order_candidate_count": infinite_order_count,
        "finite_order_frame_preserving_count": frame_preserving_count,
        "finite_order_frame_changing_count": frame_changing_count,
        "finite_order_frame_changing_excluded_count": excluded_count,
        "finite_order_frame_changing_survivor_count": survivor_count,
        "class_records": [],
        "element_records": element_records,
    }


def screen_e2chi_by_conjugacy_or_fallback(
    matrices: Iterable[Sequence[Sequence[int]] | sp.MatrixBase],
    d: IntegerCubic,
    c2: Sequence[int],
    chi_omega1: int,
) -> dict[str, Any]:
    """Screen frame-changing candidates with a certified group fast path.

    If the candidate set is not proved to be a finite group, or if the
    certified generators are not all proved to preserve ``d`` and ``c2``,
    this function performs the original elementwise calculation instead.
    """

    raw = list(matrices)
    try:
        normalised_raw = tuple(matrix_key(matrix) for matrix in raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"candidate matrix normalisation failed: {exc}") from exc
    group = _certify_group(raw)
    audit = group.certificate.to_dict()
    if not group.certificate.certified:
        # Normalisation failures cannot be evaluated safely either.  In that
        # case surface the input error rather than silently changing it.
        fallback = _elementwise_screen(normalised_raw, d, c2, chi_omega1)
        return {
            "configuration_only": True,
            "defining_polynomial_used": False,
            "group_certificate": audit,
            "topological_generator_gate": False,
            **fallback,
        }

    rank = int(group.certificate.rank)
    tensor = _normalise_cubic(d, rank)
    c2_vector = _normalise_c2(c2, rank)
    generator_gate = all(
        _preserves_cubic(generator, tensor) and _preserves_c2(generator, c2_vector)
        for generator in group.generators
    )
    if not generator_gate:
        fallback = _elementwise_screen(group.matrices, d, c2, chi_omega1)
        return {
            "configuration_only": True,
            "defining_polynomial_used": False,
            "group_certificate": audit,
            "topological_generator_gate": False,
            **fallback,
        }

    classes = _conjugacy_classes(group)
    element_records: list[dict[str, Any]] = []
    class_records: list[dict[str, Any]] = []
    frame_preserving_count = 0
    frame_changing_count = 0
    excluded_count = 0
    survivor_count = 0
    e2chi_evaluations = 0
    evaluated_classes = 0
    group_order = len(group.matrices)
    for class_index, members in enumerate(classes, start=1):
        representative = members[0]
        representative_hash = matrix_sha256(representative)
        frame_flags = {member: not permutes_ambient_frame(member) for member in members}
        frame_changing_members = tuple(member for member in members if frame_flags[member])
        frame_preserving_count += len(members) - len(frame_changing_members)
        frame_changing_count += len(frame_changing_members)
        base_class_record: dict[str, Any] = {
            "class_id": class_index,
            "representative_sha256": representative_hash,
            "class_size": len(members),
            "centralizer_order": group_order // len(members),
            "frame_changing_member_count": len(frame_changing_members),
            "frame_preserving_member_count": len(members) - len(frame_changing_members),
            "member_sha256": [matrix_sha256(member) for member in members],
        }
        if not frame_changing_members:
            base_class_record["status"] = "frame_preserving_class_not_screened"
            class_records.append(base_class_record)
            continue

        evaluated_classes += 1
        order = finite_matrix_order(representative)
        if group_order % order:
            raise GroupOrbitError("representative order does not divide certified group order")
        base_class_record["matrix_order"] = order
        if int(chi_omega1) % order:
            status = "excluded_by_chi_Omega1"
            base_class_record.update(
                status=status,
                chi_Omega1=int(chi_omega1),
                remainder=int(chi_omega1) % order,
            )
        else:
            certificate = compute_e2chi_certificate(d, c2, representative)
            e2chi_evaluations += 1
            base_class_record["representative_E2chi"] = certificate.to_dict()
            status = (
                "excluded_by_E2chi_any_failure"
                if certificate.e2chi_excludes_free_lift
                else "frame_changing_finite_order_survivor"
            )
            base_class_record["status"] = status
        class_records.append(base_class_record)

        excluded = status in {"excluded_by_chi_Omega1", "excluded_by_E2chi_any_failure"}
        if excluded:
            excluded_count += len(frame_changing_members)
        else:
            survivor_count += len(frame_changing_members)
        for member in frame_changing_members:
            element_records.append(
                {
                    "matrix_sha256": matrix_sha256(member),
                    "matrix_order": order,
                    "frame_changing": True,
                    "status": status,
                    "evaluation_mode": "conjugacy_class_propagation",
                    "conjugacy_class_id": class_index,
                    "conjugacy_representative_sha256": representative_hash,
                }
            )

    if frame_preserving_count + frame_changing_count != group_order:
        raise GroupOrbitError("elementwise frame marks do not cover the certified group")
    return {
        "configuration_only": True,
        "defining_polynomial_used": False,
        "group_certificate": audit,
        "topological_generator_gate": True,
        "screening_mode": "certified_group_conjugacy_classes",
        "conjugacy_class_count": len(classes),
        "conjugacy_class_evaluation_count": evaluated_classes,
        "e2chi_evaluation_count": e2chi_evaluations,
        "infinite_order_candidate_count": 0,
        "finite_order_frame_preserving_count": frame_preserving_count,
        "finite_order_frame_changing_count": frame_changing_count,
        "finite_order_frame_changing_excluded_count": excluded_count,
        "finite_order_frame_changing_survivor_count": survivor_count,
        "class_records": class_records,
        "element_records": sorted(element_records, key=lambda item: item["matrix_sha256"]),
    }


__all__ = [
    "ConjugacyClassRecord",
    "FiniteGroupCertificate",
    "GroupOrbitError",
    "certify_finite_matrix_group",
    "conjugacy_class_records",
    "identity_key",
    "inverse_key",
    "matrix_key",
    "matrix_sha256",
    "multiply_keys",
    "permutes_ambient_frame",
    "screen_e2chi_by_conjugacy_or_fallback",
]
