#!/usr/bin/env python3
"""Exact finite candidate enumeration for the E2-chi completeness route.

For a favourable CICY Picard frame ``J_1,...,J_rho`` let

    B(x,y) = d(x,y,w),  w = J_1 + ... + J_rho.

If ``B`` is negative definite on ``ker(c2)``, then every image of an ambient
nef class under a holomorphic automorphism lies in a compact affine
ellipsoid.  This module enumerates every integral point in those ellipsoids,
assembles every unimodular matrix preserving ``d`` and ``c2``, and applies
the coefficient-independent E2-chi obstruction to every finite-order,
frame-changing matrix.

The candidate collection is deliberately called a *set*, not a group: the
outer nef inequalities used to make the enumeration finite need not be
closed under multiplication.  All calculations are exact.  No defining
polynomial is read or selected.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import sympy as sp

from intrinsic_e2chi_complete.core import (
    saturated_integer_kernel,
)
from intrinsic_e2chi_complete.configuration_geometry import (
    euler_characteristic,
)
from intrinsic_e2chi_complete.group_orbits import (
    screen_e2chi_by_conjugacy_or_fallback,
)


ROOT = Path(__file__).resolve().parent


def _portable_data_file(name: str) -> Path | None:
    for ancestor in ROOT.parents:
        candidate = ancestor / "data" / name
        if candidate.exists():
            return candidate
    return None


DEFAULT_DATASET = _portable_data_file("cicy_criteria_input_7820.jsonl") or (
    ROOT.parents[2] / "data" / "cicy_criteria_input_7820.jsonl"
)
DEFAULT_E1_RECORDS = ROOT.parents[2] / "reproduction_output" / "e1_complete_2635.jsonl"


class CandidateEnumerationError(RuntimeError):
    """Raised when an exact positive certificate cannot be produced."""


class EnumerationCapReached(CandidateEnumerationError):
    """Raised when a user-supplied safety cap prevents exhaustive search."""


FULL_E1_STATUSES = {
    "pass_integral_E1",
}

TORSION_GATE_STATUSES = {
    "pass_standard_nonproduct_CICY_Pic_torsion_free",
    "pass_standard_nonproduct_cicy_pic_torsion_free",
    "pass_standard_smooth_nonproduct_CICY_theorem",
}

WORKING_FRAME_STATUSES = {
    "pass_identity",
    "pass_identity_presentation",
    "pass_inherited_via_fixed-X_graph_isomorphism",
    "pass_integral_identity_graph_transport",
}


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_full_e1_certificate(
    record: dict[str, Any], e1_record: dict[str, Any] | None
) -> tuple[bool, str]:
    """Check the *completed* integral-E1 gate required before E2-chi.

    A unit Smith normal form by itself is only a primitivity statement modulo
    torsion.  It is therefore deliberately not accepted here.  The caller
    must supply the output of the preceding E1 stage, where the SNF or the
    augmented/stable-map route has already been combined with the independent
    Picard-torsion theorem and with transport to the displayed working frame.
    """

    if e1_record is None:
        return False, "missing_full_E1_certificate"
    if int(e1_record.get("Num", -1)) != int(record["Num"]):
        return False, "E1_Num_mismatch"
    if e1_record.get("source") != record.get("source"):
        return False, "E1_source_mismatch"
    if e1_record.get("canonical_hash_original") != record.get("canonical_hash_original"):
        return False, "E1_original_configuration_hash_mismatch"
    if e1_record.get("canonical_hash_working") != record.get("canonical_hash_working"):
        return False, "E1_working_configuration_hash_mismatch"
    if e1_record.get("integral_E1_status") not in FULL_E1_STATUSES:
        return False, "integral_E1_not_certified"

    torsion_gate = e1_record.get("pic_torsion_free_gate") or e1_record.get("torsion_gate")
    if not isinstance(torsion_gate, dict):
        return False, "missing_Picard_torsion_gate"
    if torsion_gate.get("status") not in TORSION_GATE_STATUSES:
        return False, "Picard_torsion_gate_not_certified"

    frame_status = e1_record.get("working_frame_transport_status")
    if frame_status not in WORKING_FRAME_STATUSES:
        return False, "working_frame_transport_not_certified"
    return True, "pass_full_integral_E1"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    with path.open() as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise CandidateEnumerationError(f"{path}:{line_number}: invalid JSON") from exc
            if not isinstance(value, dict):
                raise CandidateEnumerationError(f"{path}:{line_number}: expected JSON object")
            result.append(value)
    return result


def dimensions_from_conf(conf: Sequence[Sequence[int]]) -> list[int]:
    return [sum(int(value) for value in row) - 1 for row in conf]


def complete_intersection_class(
    conf: Sequence[Sequence[int]], dimensions: Sequence[int]
) -> dict[tuple[int, ...], int]:
    rank = len(conf)
    polynomial: dict[tuple[int, ...], int] = {(0,) * rank: 1}
    for column in zip(*conf):
        next_polynomial: dict[tuple[int, ...], int] = {}
        for exponent, coefficient in polynomial.items():
            for row, raw_degree in enumerate(column):
                degree = int(raw_degree)
                if degree == 0 or exponent[row] >= int(dimensions[row]):
                    continue
                target = list(exponent)
                target[row] += 1
                key = tuple(target)
                next_polynomial[key] = next_polynomial.get(key, 0) + coefficient * degree
        polynomial = next_polynomial
    return polynomial


def cubic_tensor(
    conf: Sequence[Sequence[int]], dimensions: Sequence[int]
) -> list[list[list[int]]]:
    rank = len(conf)
    ci_class = complete_intersection_class(conf, dimensions)
    tensor = [[[0 for _ in range(rank)] for _ in range(rank)] for _ in range(rank)]
    for i in range(rank):
        for j in range(i, rank):
            for k in range(j, rank):
                powers = [0] * rank
                powers[i] += 1
                powers[j] += 1
                powers[k] += 1
                if any(powers[q] > int(dimensions[q]) for q in range(rank)):
                    value = 0
                else:
                    complement = tuple(int(dimensions[q]) - powers[q] for q in range(rank))
                    value = int(ci_class.get(complement, 0))
                for a, b, c in {
                    (i, j, k),
                    (i, k, j),
                    (j, i, k),
                    (j, k, i),
                    (k, i, j),
                    (k, j, i),
                }:
                    tensor[a][b][c] = value
    return tensor


def c2_pairing_from_tensor(
    conf: Sequence[Sequence[int]],
    dimensions: Sequence[int],
    tensor: Sequence[Sequence[Sequence[int]]],
) -> list[int]:
    rank = len(conf)
    result: list[int] = []
    for target in range(rank):
        twice = 0
        for column in zip(*conf):
            for i in range(rank):
                for j in range(rank):
                    twice += int(column[i]) * int(column[j]) * int(tensor[i][j][target])
        for i, dimension in enumerate(dimensions):
            twice -= (int(dimension) + 1) * int(tensor[i][i][target])
        if twice % 2:
            raise CandidateEnumerationError("c2 pairing is unexpectedly half-integral")
        result.append(twice // 2)
    return result


def trilinear(
    sparse_tensor: Sequence[tuple[int, int, int, int]],
    first: Sequence[int],
    second: Sequence[int],
    third: Sequence[int],
) -> int:
    return sum(
        value * int(first[i]) * int(second[j]) * int(third[k])
        for i, j, k, value in sparse_tensor
    )


def sparse_cubic(
    tensor: Sequence[Sequence[Sequence[int]]],
) -> tuple[tuple[int, int, int, int], ...]:
    rank = len(tensor)
    return tuple(
        (i, j, k, int(tensor[i][j][k]))
        for i in range(rank)
        for j in range(rank)
        for k in range(rank)
        if int(tensor[i][j][k]) != 0
    )


def _ceil_sqrt_rational(value: sp.Rational) -> int:
    exact = sp.Rational(value)
    if exact < 0:
        raise CandidateEnumerationError("cannot take the square root of a negative radius")
    numerator = int(exact.p)
    denominator = int(exact.q)
    root = math.isqrt(numerator // denominator)
    while root * root * denominator < numerator:
        root += 1
    while root > 0 and (root - 1) * (root - 1) * denominator >= numerator:
        root -= 1
    return root


def _integer_interval(center: sp.Rational, squared_radius: sp.Rational) -> range:
    padding = _ceil_sqrt_rational(sp.Rational(squared_radius))
    lower = int(sp.ceiling(sp.Rational(center) - padding))
    upper = int(sp.floor(sp.Rational(center) + padding))
    return range(lower, upper + 1)


def shifted_ellipsoid_points(
    quadratic_form: sp.Matrix,
    center: sp.Matrix,
    radius: sp.Rational,
    *,
    point_cap: int | None = None,
) -> Iterator[tuple[int, ...]]:
    """Enumerate all integer points in a shifted positive-definite ellipsoid.

    The recursion is an exact shifted Fincke--Pohst enumeration using an LDL
    decomposition.  Rational interval bounds are rounded *outwards*, then the
    original quadratic inequality is checked exactly before a point is
    yielded.
    """

    dimension = quadratic_form.rows
    if quadratic_form.cols != dimension:
        raise CandidateEnumerationError("quadratic form must be square")
    if center.rows != dimension or center.cols != 1:
        raise CandidateEnumerationError("ellipsoid center has wrong shape")
    if radius < 0:
        return
    if dimension == 0:
        yield ()
        return

    lower, diagonal = quadratic_form.LDLdecomposition(hermitian=False)
    if lower * diagonal * lower.T != quadratic_form:
        raise CandidateEnumerationError("exact LDL decomposition identity failed")
    diagonal_entries = [sp.Rational(diagonal[i, i]) for i in range(dimension)]
    if any(value <= 0 for value in diagonal_entries):
        raise CandidateEnumerationError("ellipsoid form is not positive definite")

    chosen = [0] * dimension
    shifted = [sp.Rational(0)] * dimension
    yielded = 0

    def recurse(index: int, used: sp.Rational) -> Iterator[tuple[int, ...]]:
        nonlocal yielded
        if index < 0:
            vector = sp.Matrix(chosen)
            difference = vector - center
            exact_value = sp.Rational((difference.T * quadratic_form * difference)[0])
            if exact_value <= radius:
                yielded += 1
                if point_cap is not None and yielded > point_cap:
                    raise EnumerationCapReached(
                        f"ellipsoid point cap {point_cap} reached before exhaustion"
                    )
                yield tuple(chosen)
            return

        remaining = sp.Rational(radius - used)
        if remaining < 0:
            return
        tail = sum(
            sp.Rational(lower[j, index]) * shifted[j]
            for j in range(index + 1, dimension)
        )
        coordinate_center = sp.Rational(center[index] - tail)
        squared_bound = sp.Rational(remaining / diagonal_entries[index])
        for value in _integer_interval(coordinate_center, squared_bound):
            chosen[index] = int(value)
            shifted[index] = sp.Rational(value - center[index])
            transformed = shifted[index] + tail
            contribution = diagonal_entries[index] * transformed * transformed
            yield from recurse(index - 1, used + contribution)

    yield from recurse(dimension - 1, sp.Rational(0))


@dataclass(frozen=True)
class EllipsoidCertificate:
    picard_rank: int
    c2_kernel_rank: int
    c2_kernel_basis_columns: tuple[tuple[int, ...], ...]
    negative_restriction_leading_minors: tuple[str, ...]
    negative_restriction_certified: bool
    hodge_form_determinant: str
    exact_arithmetic: bool = True


def c2_ellipsoid_certificate(
    tensor: Sequence[Sequence[Sequence[int]]], c2: Sequence[int]
) -> tuple[EllipsoidCertificate, sp.Matrix, sp.Matrix]:
    rank = len(c2)
    if rank == 0 or len(tensor) != rank:
        raise CandidateEnumerationError("invalid Picard rank")
    c2_row = sp.Matrix([[int(value) for value in c2]])
    kernel = saturated_integer_kernel(c2_row)
    if kernel.cols != rank - 1:
        raise CandidateEnumerationError("c2 must be a nonzero linear form")
    if c2_row * kernel != sp.zeros(1, rank - 1):
        raise CandidateEnumerationError("recorded c2-kernel basis is invalid")

    hodge_form = sp.Matrix(
        rank,
        rank,
        lambda i, j: sum(int(tensor[i][j][k]) for k in range(rank)),
    )
    positive_form = -(kernel.T * hodge_form * kernel)
    leading_minors = tuple(
        str(sp.factor(positive_form[:size, :size].det()))
        for size in range(1, positive_form.rows + 1)
    )
    negative = all(sp.Rational(value) > 0 for value in leading_minors)
    basis_columns = tuple(
        tuple(int(kernel[row, column]) for row in range(rank))
        for column in range(kernel.cols)
    )
    certificate = EllipsoidCertificate(
        picard_rank=rank,
        c2_kernel_rank=kernel.cols,
        c2_kernel_basis_columns=basis_columns,
        negative_restriction_leading_minors=leading_minors,
        negative_restriction_certified=negative,
        hodge_form_determinant=str(sp.factor(hodge_form.det())),
    )
    return certificate, kernel, hodge_form


@dataclass(frozen=True)
class ColumnEnumerationRecord:
    target_index_1based: int
    c2_value: int
    ellipsoid_point_count: int
    primitive_point_count: int
    nef_outer_point_count: int
    self_cubic_point_count: int
    candidate_columns: tuple[tuple[int, ...], ...]


def enumerate_column_candidates(
    tensor: Sequence[Sequence[Sequence[int]]],
    c2: Sequence[int],
    target_index: int,
    kernel: sp.Matrix,
    hodge_form: sp.Matrix,
    *,
    point_cap: int | None = None,
) -> ColumnEnumerationRecord:
    rank = len(c2)
    if not 0 <= target_index < rank:
        raise CandidateEnumerationError("target column is out of range")
    sparse = sparse_cubic(tensor)
    standard = [tuple(int(i == j) for i in range(rank)) for j in range(rank)]
    particular = sp.eye(rank)[:, target_index]
    positive_form = -(kernel.T * hodge_form * kernel)
    linear = kernel.T * hodge_form * particular
    center = positive_form.inv() * linear
    constant = sp.Rational((particular.T * hodge_form * particular)[0])
    radius = sp.factor(constant + (linear.T * positive_form.inv() * linear)[0])
    if radius < 0:
        raise CandidateEnumerationError("ambient basis vector is missing from its own ellipsoid")

    ellipsoid_count = 0
    primitive_count = 0
    nef_outer_count = 0
    self_cubic_count = 0
    candidates: set[tuple[int, ...]] = set()
    for coordinates in shifted_ellipsoid_points(
        positive_form, center, sp.Rational(radius), point_cap=point_cap
    ):
        ellipsoid_count += 1
        vector_matrix = particular + kernel * sp.Matrix(coordinates)
        vector = tuple(int(value) for value in vector_matrix)
        if sum(int(c2[i]) * vector[i] for i in range(rank)) != int(c2[target_index]):
            raise CandidateEnumerationError("ellipsoid point escaped its c2 affine lattice")
        q_value = int((vector_matrix.T * hodge_form * vector_matrix)[0])
        if q_value < 0:
            raise CandidateEnumerationError("ellipsoid recursion emitted a point with q_w < 0")
        if math.gcd(*(abs(value) for value in vector)) != 1:
            continue
        primitive_count += 1

        # Every actual image of J_target is nef.  Intersections with the known
        # ambient nef classes are therefore nonnegative.  These are safe outer
        # filters and do not assume that the ambient orthant is the full nef cone.
        linear_nef = all(
            trilinear(sparse, vector, standard[i], standard[j]) >= 0
            for i in range(rank)
            for j in range(i, rank)
        )
        quadratic_nef = all(
            trilinear(sparse, vector, vector, standard[i]) >= 0
            for i in range(rank)
        )
        if not (linear_nef and quadratic_nef):
            continue
        nef_outer_count += 1
        if trilinear(sparse, vector, vector, vector) != int(
            tensor[target_index][target_index][target_index]
        ):
            continue
        self_cubic_count += 1
        candidates.add(vector)

    identity_column = standard[target_index]
    if identity_column not in candidates:
        raise CandidateEnumerationError("identity column was lost by the exact outer filters")
    ordered = tuple(sorted(candidates))
    return ColumnEnumerationRecord(
        target_index_1based=target_index + 1,
        c2_value=int(c2[target_index]),
        ellipsoid_point_count=ellipsoid_count,
        primitive_point_count=primitive_count,
        nef_outer_point_count=nef_outer_count,
        self_cubic_point_count=self_cubic_count,
        candidate_columns=ordered,
    )


def _rank_mod_prime(columns: Sequence[Sequence[int]], prime: int) -> int:
    if not columns:
        return 0
    rows = [
        [int(columns[column][row]) % prime for column in range(len(columns))]
        for row in range(len(columns[0]))
    ]
    pivot_row = 0
    for column in range(len(columns)):
        chosen = next(
            (row for row in range(pivot_row, len(rows)) if rows[row][column] % prime),
            None,
        )
        if chosen is None:
            continue
        rows[pivot_row], rows[chosen] = rows[chosen], rows[pivot_row]
        inverse = pow(rows[pivot_row][column], -1, prime)
        rows[pivot_row] = [(value * inverse) % prime for value in rows[pivot_row]]
        for row in range(len(rows)):
            if row == pivot_row:
                continue
            coefficient = rows[row][column]
            if coefficient:
                rows[row] = [
                    (value - coefficient * pivot) % prime
                    for value, pivot in zip(rows[row], rows[pivot_row])
                ]
        pivot_row += 1
        if pivot_row == len(rows):
            break
    return pivot_row


def permutes_ambient_frame(matrix: Sequence[Sequence[int]]) -> bool:
    rank = len(matrix)
    columns = [tuple(int(matrix[i][j]) for i in range(rank)) for j in range(rank)]
    standard = [tuple(int(i == j) for i in range(rank)) for j in range(rank)]
    return sorted(columns) == sorted(standard)


@dataclass(frozen=True)
class MatrixEnumerationStats:
    search_order_1based: tuple[int, ...]
    dynamic_mrv: bool
    backtrack_node_count: int
    pair_rejection_count: int
    cross_nef_rejection_count: int
    triple_rejection_count: int
    modular_rank_rejection_count: int
    forward_check_rejection_count: int
    pair_compatibility_cache_size: int
    triple_compatibility_cache_size: int
    determinant_rejection_count: int
    complete_candidate_matrix_count: int


def enumerate_candidate_matrices(
    tensor: Sequence[Sequence[Sequence[int]]],
    column_records: Sequence[ColumnEnumerationRecord],
    *,
    matrix_cap: int | None = None,
) -> tuple[list[tuple[tuple[int, ...], ...]], MatrixEnumerationStats]:
    rank = len(column_records)
    sparse = sparse_cubic(tensor)
    candidates = [record.candidate_columns for record in column_records]
    standard_basis = tuple(
        tuple(int(row == generator) for row in range(rank))
        for generator in range(rank)
    )
    search_order = tuple(sorted(range(rank), key=lambda index: (len(candidates[index]), index)))
    # Store both the candidate index and vector.  The indices give compact,
    # canonical keys for the exact pair/triple compatibility caches.
    selected: dict[int, tuple[int, tuple[int, ...]]] = {}
    matrices: list[tuple[tuple[int, ...], ...]] = []
    node_count = 0
    pair_rejections = 0
    cross_nef_rejections = 0
    triple_rejections = 0
    modular_rejections = 0
    forward_rejections = 0
    determinant_rejections = 0
    pair_cache: dict[tuple[int, int, int, int], int] = {}
    triple_cache: dict[tuple[tuple[int, int], ...], bool] = {}

    def pair_status(
        first_target: int,
        first_index: int,
        second_target: int,
        second_index: int,
    ) -> int:
        """Return 0 for compatible, 1 for cubic failure, 2 for nef failure."""

        if first_target > second_target:
            first_target, second_target = second_target, first_target
            first_index, second_index = second_index, first_index
        key = (first_target, first_index, second_target, second_index)
        cached = pair_cache.get(key)
        if cached is not None:
            return cached
        first = candidates[first_target][first_index]
        second = candidates[second_target][second_index]
        if trilinear(sparse, first, first, second) != int(
            tensor[first_target][first_target][second_target]
        ) or trilinear(sparse, first, second, second) != int(
            tensor[first_target][second_target][second_target]
        ):
            pair_cache[key] = 1
            return 1
        if any(
            trilinear(sparse, first, second, standard) < 0
            for standard in standard_basis
        ):
            pair_cache[key] = 2
            return 2
        pair_cache[key] = 0
        return 0

    def triple_ok(
        first_target: int,
        first_index: int,
        second_target: int,
        second_index: int,
        third_target: int,
        third_index: int,
    ) -> bool:
        labelled = tuple(
            sorted(
                (
                    (first_target, first_index),
                    (second_target, second_index),
                    (third_target, third_index),
                )
            )
        )
        cached = triple_cache.get(labelled)
        if cached is not None:
            return cached
        (a, a_index), (b, b_index), (c, c_index) = labelled
        compatible = trilinear(
            sparse,
            candidates[a][a_index],
            candidates[b][b_index],
            candidates[c][c_index],
        ) == int(tensor[a][b][c])
        triple_cache[labelled] = compatible
        return compatible

    def compatible_options(target: int) -> list[tuple[int, tuple[int, ...]]]:
        nonlocal node_count, pair_rejections, cross_nef_rejections
        nonlocal triple_rejections, modular_rejections
        existing = sorted(selected)
        options: list[tuple[int, tuple[int, ...]]] = []
        for candidate_index, vector in enumerate(candidates[target]):
            node_count += 1
            failed = False
            for other in existing:
                other_index, _ = selected[other]
                status = pair_status(other, other_index, target, candidate_index)
                if status == 1:
                    pair_rejections += 1
                    failed = True
                    break
                if status == 2:
                    cross_nef_rejections += 1
                    failed = True
                    break
            if failed:
                continue
            for first, second in itertools.combinations(existing, 2):
                first_index, _ = selected[first]
                second_index, _ = selected[second]
                if not triple_ok(
                    first,
                    first_index,
                    second,
                    second_index,
                    target,
                    candidate_index,
                ):
                    triple_rejections += 1
                    failed = True
                    break
            if failed:
                continue
            partial_columns = [selected[index][1] for index in existing] + [vector]
            if any(
                _rank_mod_prime(partial_columns, prime) != len(partial_columns)
                for prime in (2, 3, 5, 7)
            ):
                modular_rejections += 1
                continue
            options.append((candidate_index, vector))
        return options

    def recurse(depth: int) -> None:
        nonlocal forward_rejections, determinant_rejections
        if depth == rank:
            rows = tuple(
                tuple(selected[column][1][row] for column in range(rank))
                for row in range(rank)
            )
            determinant = int(sp.Matrix(rows).det())
            if abs(determinant) != 1:
                determinant_rejections += 1
                return
            matrices.append(rows)
            if matrix_cap is not None and len(matrices) > matrix_cap:
                raise EnumerationCapReached(
                    f"candidate matrix cap {matrix_cap} reached before exhaustion"
                )
            return

        # Forward checking plus dynamic minimum-remaining-values ordering.
        # All unassigned targets are tested, so an empty future domain closes
        # the branch immediately.  This changes only traversal order, never
        # the set of completed matrices.
        best_target: int | None = None
        best_options: list[tuple[int, tuple[int, ...]]] | None = None
        for target in search_order:
            if target in selected:
                continue
            options = compatible_options(target)
            if not options:
                forward_rejections += 1
                return
            if best_options is None or len(options) < len(best_options):
                best_target = target
                best_options = options
        if best_target is None or best_options is None:
            raise CandidateEnumerationError("dynamic MRV failed to select a target")
        for candidate_index, vector in best_options:
            selected[best_target] = (candidate_index, vector)
            recurse(depth + 1)
            del selected[best_target]

    recurse(0)
    unique = sorted(set(matrices))
    if len(unique) != len(matrices):
        raise CandidateEnumerationError("candidate matrix enumeration produced duplicates")
    stats = MatrixEnumerationStats(
        search_order_1based=tuple(index + 1 for index in search_order),
        dynamic_mrv=True,
        backtrack_node_count=node_count,
        pair_rejection_count=pair_rejections,
        cross_nef_rejection_count=cross_nef_rejections,
        triple_rejection_count=triple_rejections,
        modular_rank_rejection_count=modular_rejections,
        forward_check_rejection_count=forward_rejections,
        pair_compatibility_cache_size=len(pair_cache),
        triple_compatibility_cache_size=len(triple_cache),
        determinant_rejection_count=determinant_rejections,
        complete_candidate_matrix_count=len(unique),
    )
    return unique, stats


def analyze_case(
    record: dict[str, Any],
    e1_record: dict[str, Any] | None,
    *,
    point_cap: int | None = None,
    matrix_cap: int | None = None,
) -> dict[str, Any]:
    started = time.monotonic()
    # The frozen ``working`` presentation is the selected special conf.  It is
    # identical to the favourable presentation in ordinary cases and is the
    # certified graph re-embedding in the E3-repair cases.
    conf = record["working_conf"]
    dimensions = [int(value) for value in record["working_ambient_dimensions"]]
    if dimensions_from_conf(conf) != dimensions:
        raise CandidateEnumerationError("working ambient dimensions disagree with row sums")
    rank = len(conf)
    if int(record["official_fields"]["H11"]) != rank:
        raise CandidateEnumerationError("selected presentation is not favourable over Q")

    # Logical gate: E2-chi is a statement about integral Picard-lattice
    # actions.  Do not even construct the ellipsoids until the preceding E1
    # stage has certified that the displayed ambient frame is the full
    # integral Picard lattice.  In particular, a conditional SNF record is
    # not enough.
    e1_pass, e1_gate_reason = validate_full_e1_certificate(record, e1_record)
    gate_common = {
        "schema_version": "2.0.0",
        "computation_id": "cicy_e2chi_complete_ellipsoid_after_E1",
        "Num": int(record["Num"]),
        "source": record["source"],
        "canonical_hash_working": record.get("canonical_hash_working"),
        "picard_rank": rank,
        "configuration_only": True,
        "defining_polynomial_used": False,
        "integral_E1_certified": e1_pass,
        "integral_E1_status": None if e1_record is None else e1_record.get("integral_E1_status"),
        "E1_gate_reason": e1_gate_reason,
        "E2chi_started": False,
    }
    if not e1_pass:
        return gate_common | {
            "status": "blocked_E1_not_certified",
            "H3_free_frame_preservation_certified": False,
            "elapsed_seconds": time.monotonic() - started,
        }

    tensor = cubic_tensor(conf, dimensions)
    c2 = c2_pairing_from_tensor(conf, dimensions, tensor)
    official_c2 = record["official_fields"].get("C2_selected")
    if official_c2 is not None and [int(value) for value in official_c2] != c2:
        raise CandidateEnumerationError("computed c2 pairing disagrees with the supplied value")

    ellipsoid, kernel, hodge_form = c2_ellipsoid_certificate(tensor, c2)
    common = gate_common | {
        "E2chi_started": True,
        "c2_pairing": c2,
        "ellipsoid_certificate": asdict(ellipsoid),
    }
    if not ellipsoid.negative_restriction_certified:
        return common | {
            "status": "unverified_c2_kernel_not_negative_definite",
            "elapsed_seconds": time.monotonic() - started,
        }

    column_records = [
        enumerate_column_candidates(
            tensor,
            c2,
            target,
            kernel,
            hodge_form,
            point_cap=point_cap,
        )
        for target in range(rank)
    ]
    matrices, matrix_stats = enumerate_candidate_matrices(
        tensor, column_records, matrix_cap=matrix_cap
    )

    original_conf = record["original_selected_conf"]
    original_dimensions = dimensions_from_conf(original_conf)
    euler_original = euler_characteristic(original_conf, original_dimensions)
    euler_working = euler_characteristic(conf, dimensions)
    if euler_original != euler_working:
        raise CandidateEnumerationError("original and working Euler characteristics disagree")
    supplied_topology = record.get("topology_audit", {})
    for field, computed in (
        ("euler_original", euler_original),
        ("euler_working", euler_working),
    ):
        if field in supplied_topology and int(supplied_topology[field]) != computed:
            raise CandidateEnumerationError(
                f"supplied {field} disagrees with the configuration"
            )
    if euler_original % 2:
        raise CandidateEnumerationError("Euler characteristic is odd, so chi(Omega^1) is not integral")
    chi_omega1 = -euler_original // 2
    screening = screen_e2chi_by_conjugacy_or_fallback(
        matrices,
        tensor,
        c2,
        chi_omega1,
    )

    # Preserve the full matrix payload in the per-element audit even when a
    # conjugacy-class representative supplied the expensive E2-chi result.
    matrix_payload_by_hash: dict[str, tuple[list[list[int]], list[int]]] = {}
    for matrix in matrices:
        matrix_payload = [list(row) for row in matrix]
        matrix_flat = [value for row in matrix_payload for value in row]
        matrix_hash = sha256_bytes(canonical_json(matrix_flat).encode())
        if matrix_hash in matrix_payload_by_hash:
            raise CandidateEnumerationError("candidate matrix hashes are not unique")
        matrix_payload_by_hash[matrix_hash] = (matrix_payload, matrix_flat)
    finite_order_records: list[dict[str, Any]] = []
    for raw_element_record in screening["element_records"]:
        element_record = dict(raw_element_record)
        matrix_hash = str(element_record["matrix_sha256"])
        payload = matrix_payload_by_hash.get(matrix_hash)
        if payload is None:
            raise CandidateEnumerationError(
                "conjugacy screen returned an unknown candidate matrix hash"
            )
        element_record["matrix"] = payload[0]
        element_record["matrix_flat_row_major"] = payload[1]
        finite_order_records.append(element_record)

    infinite_order_count = int(screening["infinite_order_candidate_count"])
    frame_preserving_count = int(screening["finite_order_frame_preserving_count"])
    frame_changing_count = int(screening["finite_order_frame_changing_count"])
    excluded_count = int(screening["finite_order_frame_changing_excluded_count"])
    survivor_count = int(screening["finite_order_frame_changing_survivor_count"])
    no_frame_changing_survivors = survivor_count == 0
    h3_pass = no_frame_changing_survivors
    status = (
        "pass_E2chi_frame_preservation"
        if h3_pass
        else "frame_changing_finite_order_survivors_remain"
    )
    return common | {
        "status": status,
        "E2chi_no_frame_changing_survivors": no_frame_changing_survivors,
        "H3_conditional_on_integral_E1": False,
        "H3_free_frame_preservation_certified": h3_pass,
        "fixed_configuration_solver_H4_assumed": False,
        "column_enumeration_records": [asdict(item) for item in column_records],
        "matrix_enumeration_stats": asdict(matrix_stats),
        "candidate_matrix_count": len(matrices),
        "candidate_screening_mode": screening["screening_mode"],
        "candidate_group_certificate": screening["group_certificate"],
        "candidate_topological_generator_gate": screening[
            "topological_generator_gate"
        ],
        "candidate_conjugacy_class_count": screening["conjugacy_class_count"],
        "candidate_conjugacy_class_evaluation_count": screening[
            "conjugacy_class_evaluation_count"
        ],
        "E2chi_evaluation_count": screening["e2chi_evaluation_count"],
        "infinite_order_candidate_count": infinite_order_count,
        "finite_order_frame_preserving_count": frame_preserving_count,
        "finite_order_frame_changing_count": frame_changing_count,
        "finite_order_frame_changing_excluded_count": excluded_count,
        "finite_order_frame_changing_survivor_count": survivor_count,
        "finite_order_frame_changing_class_records": screening["class_records"],
        "finite_order_frame_changing_records": finite_order_records,
        "elapsed_seconds": time.monotonic() - started,
    }


def _index_by_num(records: Iterable[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    result: dict[int, dict[str, Any]] = {}
    for record in records:
        num = int(record["Num"])
        if num in result:
            raise CandidateEnumerationError(f"duplicate Num {num}")
        result[num] = record
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num", type=int, required=True, help="CICY Num to analyze")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--e1-records", type=Path, default=DEFAULT_E1_RECORDS)
    parser.add_argument("--point-cap", type=int)
    parser.add_argument("--matrix-cap", type=int)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()

    dataset = _index_by_num(load_jsonl(arguments.dataset))
    e1_records = _index_by_num(load_jsonl(arguments.e1_records))
    if arguments.num not in dataset:
        raise CandidateEnumerationError(f"Num {arguments.num} is absent from the dataset")
    try:
        result = analyze_case(
            dataset[arguments.num],
            e1_records.get(arguments.num),
            point_cap=arguments.point_cap,
            matrix_cap=arguments.matrix_cap,
        )
    except EnumerationCapReached as exc:
        result = {
            "schema_version": "1.0.0",
            "computation_id": "cicy_e2chi_complete_ellipsoid_20260824",
            "Num": arguments.num,
            "status": "unverified_enumeration_cap_reached",
            "reason": str(exc),
            "configuration_only": True,
            "defining_polynomial_used": False,
        }
    result["input_sha256"] = {
        str(arguments.dataset): sha256_file(arguments.dataset),
        str(arguments.e1_records): sha256_file(arguments.e1_records),
    }
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if arguments.output:
        arguments.output.write_text(rendered)
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
