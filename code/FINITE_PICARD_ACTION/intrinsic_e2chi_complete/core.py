#!/usr/bin/env python3
"""Exact arithmetic for the E2-chi cyclic free-action obstruction.

The module deliberately does not enumerate Picard-action candidates.  Given
one finite-order integral matrix ``M``, it computes the fixed Picard lattice

    Lambda_M = ker(M - I) intersect Z^rho

with a saturated integral basis.  On that lattice it computes the three gcds

    c_d, c_c2, c_chi,

where chi(D) = D^3/6 + c2.D/12.  A freely acting automorphism whose Picard
action is M would force ord(M) to divide all three gcds.  Consequently one
failed divisibility condition is enough to exclude a free lift of M.

All calculations are over ZZ/QQ.  No defining polynomial is used.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import sympy as sp
from sympy.matrices.normalforms import hermite_normal_form, smith_normal_form


IntegerMatrix = Sequence[Sequence[int]]
IntegerCubic = Sequence[Sequence[Sequence[int]]]


def _exact_integer(value: object, label: str) -> int:
    exact = sp.sympify(value)
    if exact.is_Integer is not True:
        raise ValueError(f"{label} must be an exact integer, got {value!r}")
    return int(exact)


def _square_integer_matrix(values: IntegerMatrix, label: str = "M") -> sp.Matrix:
    rows = [list(row) for row in values]
    if not rows:
        raise ValueError(f"{label} must be a nonempty square matrix")
    size = len(rows)
    if any(len(row) != size for row in rows):
        raise ValueError(f"{label} must be square")
    return sp.Matrix(
        [[_exact_integer(value, f"{label}[{i}][{j}]") for j, value in enumerate(row)]
         for i, row in enumerate(rows)]
    )


def _rectangular_integer_matrix(values: IntegerMatrix, label: str = "A") -> sp.Matrix:
    rows = [list(row) for row in values]
    if not rows:
        raise ValueError(f"{label} must have at least one row")
    width = len(rows[0])
    if any(len(row) != width for row in rows):
        raise ValueError(f"{label} must be rectangular")
    return sp.Matrix(
        [[_exact_integer(value, f"{label}[{i}][{j}]") for j, value in enumerate(row)]
         for i, row in enumerate(rows)]
    )


def _normalise_cubic(
    values: IntegerCubic,
    rank: int,
) -> tuple[tuple[tuple[int, ...], ...], ...]:
    if len(values) != rank:
        raise ValueError("d must have shape (rho, rho, rho)")
    tensor: list[list[list[int]]] = []
    for i, plane in enumerate(values):
        if len(plane) != rank:
            raise ValueError("d must have shape (rho, rho, rho)")
        tensor_plane: list[list[int]] = []
        for j, row in enumerate(plane):
            if len(row) != rank:
                raise ValueError("d must have shape (rho, rho, rho)")
            tensor_plane.append(
                [_exact_integer(value, f"d[{i}][{j}][{k}]") for k, value in enumerate(row)]
            )
        tensor.append(tensor_plane)

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
    return tuple(tuple(tuple(row) for row in plane) for plane in tensor)


def _normalise_c2(values: Sequence[int], rank: int) -> tuple[int, ...]:
    if len(values) != rank:
        raise ValueError("c2 must have length rho")
    return tuple(_exact_integer(value, f"c2[{i}]") for i, value in enumerate(values))


def finite_matrix_order(matrix: IntegerMatrix | sp.Matrix) -> int:
    """Return the exact order of a finite-order integral matrix.

    The characteristic polynomial of a finite-order rational matrix factors
    into cyclotomic polynomials and the matrix is semisimple.  If Phi_n occurs
    in rank ``rho``, then phi(n) <= rho.  The elementary inequality
    phi(n)^2 >= n/2 gives n <= 2*rho^2, so the cyclotomic lookup is finite and
    certified.  A final matrix-power equality rejects nontrivial Jordan blocks.
    """

    if isinstance(matrix, sp.MatrixBase):
        if matrix.rows != matrix.cols or matrix.rows == 0:
            raise ValueError("M must be a nonempty square matrix")
        checked = sp.Matrix(
            matrix.rows,
            matrix.cols,
            lambda i, j: _exact_integer(matrix[i, j], f"M[{i}][{j}]"),
        )
    else:
        checked = _square_integer_matrix(matrix)

    rank = checked.rows
    determinant = int(checked.det())
    if abs(determinant) != 1:
        raise ValueError("a finite-order integral matrix must be unimodular")

    variable = sp.Symbol("z")
    characteristic = sp.Poly(
        checked.charpoly(variable).as_expr(), variable, domain=sp.QQ
    ).monic()
    factors = sp.factor_list(characteristic)[1]

    index_bound = max(2, 2 * rank * rank)
    cyclotomic_indices: dict[tuple[sp.Rational, ...], int] = {}
    for index in range(1, index_bound + 1):
        polynomial = sp.Poly(sp.cyclotomic_poly(index, variable), variable, domain=sp.QQ)
        if polynomial.degree() <= rank:
            cyclotomic_indices[tuple(polynomial.all_coeffs())] = index

    eigenvalue_orders: list[int] = []
    for factor, _multiplicity in factors:
        key = tuple(factor.monic().all_coeffs())
        index = cyclotomic_indices.get(key)
        if index is None:
            raise ValueError("M is not finite order: a characteristic factor is not cyclotomic")
        eigenvalue_orders.append(index)

    candidate_order = math.lcm(*eigenvalue_orders)
    if checked**candidate_order != sp.eye(rank):
        raise ValueError("M is not finite order: its cyclotomic part has a nontrivial Jordan block")
    return candidate_order


def _determinantal_divisor(basis: sp.Matrix) -> int:
    """Return the gcd of the full-rank minors of a column basis."""

    column_count = basis.cols
    if column_count == 0:
        return 1
    if basis.rank() != column_count:
        raise ValueError("basis columns must be linearly independent")
    diagonal = smith_normal_form(basis, domain=sp.ZZ)
    nonzero = [abs(int(diagonal[i, i])) for i in range(column_count)]
    if any(value == 0 for value in nonzero):
        raise ArithmeticError("Smith form unexpectedly lost column rank")
    return math.prod(nonzero)


def _modular_kernel_vector(matrix: sp.Matrix, prime: int) -> list[int]:
    """Find a nonzero vector in the right kernel modulo ``prime``."""

    rows = [
        [int(matrix[i, j]) % prime for j in range(matrix.cols)]
        for i in range(matrix.rows)
    ]
    pivot_columns: list[int] = []
    pivot_row = 0
    for column in range(matrix.cols):
        chosen = next(
            (row for row in range(pivot_row, matrix.rows) if rows[row][column] % prime),
            None,
        )
        if chosen is None:
            continue
        rows[pivot_row], rows[chosen] = rows[chosen], rows[pivot_row]
        inverse = pow(rows[pivot_row][column], -1, prime)
        rows[pivot_row] = [(value * inverse) % prime for value in rows[pivot_row]]
        for row in range(matrix.rows):
            if row == pivot_row:
                continue
            coefficient = rows[row][column]
            if coefficient:
                rows[row] = [
                    (value - coefficient * pivot_value) % prime
                    for value, pivot_value in zip(rows[row], rows[pivot_row])
                ]
        pivot_columns.append(column)
        pivot_row += 1
        if pivot_row == matrix.rows:
            break

    free_columns = [column for column in range(matrix.cols) if column not in pivot_columns]
    if not free_columns:
        raise ArithmeticError("expected a modular rank drop while saturating the kernel")
    free = free_columns[0]
    vector = [0] * matrix.cols
    vector[free] = 1
    for row, pivot in enumerate(pivot_columns):
        vector[pivot] = (-rows[row][free]) % prime
    if all(value == 0 for value in vector):
        raise ArithmeticError("modular nullspace returned the zero vector")
    return vector


def _saturate_column_lattice(basis: sp.Matrix) -> sp.Matrix:
    """Saturate a full-column-rank sublattice of Z^rho."""

    if basis.cols == 0:
        return basis
    current = basis.applyfunc(int)
    while True:
        index = _determinantal_divisor(current)
        if index == 1:
            return current
        prime = min(sp.factorint(index))
        relation = _modular_kernel_vector(current, prime)
        pivot = next(i for i, value in enumerate(relation) if value % prime)
        scale = pow(relation[pivot], -1, prime)
        relation = [(value * scale) % prime for value in relation]
        if relation[pivot] != 1:
            raise ArithmeticError("failed to normalise a modular saturation relation")

        numerator = current * sp.Matrix(relation)
        if any(int(value) % prime for value in numerator):
            raise ArithmeticError("modular saturation relation did not lift integrally")
        enlarged = current.copy()
        enlarged[:, pivot] = numerator.applyfunc(lambda value: int(value) // prime)
        new_index = _determinantal_divisor(enlarged)
        if new_index * prime != index:
            raise ArithmeticError("a saturation step did not reduce the lattice index by prime")
        current = enlarged


def _canonical_column_basis(basis: sp.Matrix) -> sp.Matrix:
    """Put a saturated column lattice in deterministic Hermite form."""

    if basis.cols == 0:
        return basis
    canonical = hermite_normal_form(basis)
    if canonical.shape != basis.shape:
        raise ArithmeticError("column Hermite form unexpectedly changed the basis shape")
    coordinates = (basis.T * basis).inv() * basis.T * canonical
    if basis * coordinates != canonical:
        raise ArithmeticError("column Hermite form left the rational kernel")
    if any(sp.denom(value) != 1 for value in coordinates):
        raise ArithmeticError("column Hermite form is not integrally generated by the source basis")
    if abs(int(coordinates.det())) != 1:
        raise ArithmeticError("column Hermite form changed the lattice index")
    return canonical


def saturated_integer_kernel(matrix: IntegerMatrix | sp.Matrix) -> sp.Matrix:
    """Return a column basis of ``ker(matrix over Q) intersect Z^n``.

    Clearing denominators in a rational nullspace basis can produce a proper
    sublattice.  The subsequent prime-by-prime Smith saturation is therefore
    essential and is checked by a final determinantal divisor equal to one.
    """

    if isinstance(matrix, sp.MatrixBase):
        checked = sp.Matrix(
            matrix.rows,
            matrix.cols,
            lambda i, j: _exact_integer(matrix[i, j], f"A[{i}][{j}]"),
        )
    else:
        checked = _rectangular_integer_matrix(matrix)

    nullspace = checked.nullspace()
    if not nullspace:
        return sp.zeros(checked.cols, 0)

    columns: list[sp.Matrix] = []
    for vector in nullspace:
        denominator = math.lcm(*(int(sp.denom(value)) for value in vector))
        integral = [int(value * denominator) for value in vector]
        content = math.gcd(*(abs(value) for value in integral))
        if content == 0:
            raise ArithmeticError("rational nullspace unexpectedly contained the zero vector")
        columns.append(sp.Matrix([value // content for value in integral]))

    basis = _canonical_column_basis(
        _saturate_column_lattice(sp.Matrix.hstack(*columns))
    )
    if checked * basis != sp.zeros(checked.rows, basis.cols):
        raise ArithmeticError("computed integer basis is not in the kernel")
    if _determinantal_divisor(basis) != 1:
        raise ArithmeticError("computed integer kernel is not saturated")
    return basis


def _cubic_value(
    tensor: tuple[tuple[tuple[int, ...], ...], ...],
    first: Sequence[int],
    second: Sequence[int],
    third: Sequence[int],
) -> int:
    rank = len(tensor)
    return sum(
        tensor[i][j][k] * first[i] * second[j] * third[k]
        for i in range(rank)
        for j in range(rank)
        for k in range(rank)
    )


def _gcd(values: Iterable[int]) -> int:
    result = 0
    for value in values:
        result = math.gcd(result, abs(int(value)))
    return result


def _simplex_coefficients(rank: int) -> Iterable[tuple[int, ...]]:
    zero = (0,) * rank
    yield zero
    for total_degree in range(1, 4):
        for multiset in itertools.combinations_with_replacement(range(rank), total_degree):
            coefficients = [0] * rank
            for index in multiset:
                coefficients[index] += 1
            yield tuple(coefficients)


def _preserves_c2(matrix: sp.Matrix, c2: tuple[int, ...]) -> bool:
    return all(
        sum(c2[i] * int(matrix[i, j]) for i in range(matrix.rows)) == c2[j]
        for j in range(matrix.cols)
    )


def _preserves_cubic(
    matrix: sp.Matrix,
    tensor: tuple[tuple[tuple[int, ...], ...], ...],
) -> bool:
    columns = [tuple(int(matrix[i, j]) for i in range(matrix.rows)) for j in range(matrix.cols)]
    rank = matrix.rows
    standard = [tuple(int(i == j) for i in range(rank)) for j in range(rank)]
    return all(
        _cubic_value(tensor, columns[i], columns[j], columns[k])
        == _cubic_value(tensor, standard[i], standard[j], standard[k])
        for i in range(rank)
        for j in range(i, rank)
        for k in range(j, rank)
    )


@dataclass(frozen=True)
class E2ChiCertificate:
    """JSON-serialisable exact certificate for one Picard-action matrix."""

    picard_rank: int
    matrix_order: int
    invariant_rank: int
    invariant_basis_columns: tuple[tuple[int, ...], ...]
    c_d: int
    c_c2: int
    c_chi: int
    chi_simplex_size: int
    order_divides_c_d: bool
    order_divides_c_c2: bool
    order_divides_c_chi: bool
    failed_conditions: tuple[str, ...]
    any_divisibility_failure: bool
    e2chi_excludes_free_lift: bool
    preserves_cubic: bool
    preserves_c2: bool
    exact_arithmetic: bool = True
    defining_polynomial_used: bool = False

    def to_dict(self) -> dict:
        """Return a plain JSON-compatible dictionary."""

        return asdict(self)


def compute_e2chi_certificate(
    d: IntegerCubic,
    c2: Sequence[int],
    matrix: IntegerMatrix,
    *,
    require_topological_candidate: bool = True,
) -> E2ChiCertificate:
    """Compute the corrected any-failure E2-chi test for one matrix.

    By default ``M`` must preserve both ``d`` and ``c2``.  This prevents a
    non-candidate matrix from being mislabeled as an E2-chi exclusion.  Set
    ``require_topological_candidate=False`` only for diagnostics.
    """

    checked_matrix = _square_integer_matrix(matrix)
    rank = checked_matrix.rows
    tensor = _normalise_cubic(d, rank)
    c2_vector = _normalise_c2(c2, rank)
    order = finite_matrix_order(checked_matrix)
    preserves_d = _preserves_cubic(checked_matrix, tensor)
    preserves_c2 = _preserves_c2(checked_matrix, c2_vector)
    if require_topological_candidate and not (preserves_d and preserves_c2):
        failed = []
        if not preserves_d:
            failed.append("d")
        if not preserves_c2:
            failed.append("c2")
        raise ValueError(f"M is not a topological candidate: it does not preserve {', '.join(failed)}")

    invariant_basis = saturated_integer_kernel(checked_matrix - sp.eye(rank))
    invariant_rank = invariant_basis.cols
    basis_columns = [
        tuple(int(invariant_basis[row, column]) for row in range(rank))
        for column in range(invariant_rank)
    ]

    cubic_values = (
        _cubic_value(tensor, basis_columns[i], basis_columns[j], basis_columns[k])
        for i in range(invariant_rank)
        for j in range(i, invariant_rank)
        for k in range(j, invariant_rank)
    )
    c_d = _gcd(cubic_values)
    c_c2 = _gcd(
        sum(c2_vector[row] * vector[row] for row in range(rank))
        for vector in basis_columns
    )

    chi_values: list[int] = []
    for coefficients in _simplex_coefficients(invariant_rank):
        divisor = tuple(
            sum(coefficients[column] * basis_columns[column][row]
                for column in range(invariant_rank))
            for row in range(rank)
        )
        numerator = 2 * _cubic_value(tensor, divisor, divisor, divisor) + sum(
            c2_vector[row] * divisor[row] for row in range(rank)
        )
        if numerator % 12:
            raise ValueError(
                "Riemann--Roch is nonintegral on the invariant lattice; "
                f"12 does not divide {numerator} for divisor {divisor}"
            )
        chi_values.append(numerator // 12)
    c_chi = _gcd(chi_values)

    divisibility = {
        "c_d": c_d % order == 0,
        "c_c2": c_c2 % order == 0,
        "c_chi": c_chi % order == 0,
    }
    failed_conditions = tuple(name for name, passed in divisibility.items() if not passed)
    any_failure = bool(failed_conditions)
    is_topological_candidate = preserves_d and preserves_c2

    return E2ChiCertificate(
        picard_rank=rank,
        matrix_order=order,
        invariant_rank=invariant_rank,
        invariant_basis_columns=tuple(basis_columns),
        c_d=c_d,
        c_c2=c_c2,
        c_chi=c_chi,
        chi_simplex_size=len(chi_values),
        order_divides_c_d=divisibility["c_d"],
        order_divides_c_c2=divisibility["c_c2"],
        order_divides_c_chi=divisibility["c_chi"],
        failed_conditions=failed_conditions,
        any_divisibility_failure=any_failure,
        e2chi_excludes_free_lift=is_topological_candidate and any_failure,
        preserves_cubic=preserves_d,
        preserves_c2=preserves_c2,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="JSON file containing d, c2, and M")
    parser.add_argument("--output", type=Path, help="optional output JSON path")
    arguments = parser.parse_args()

    payload = json.loads(arguments.input.read_text())
    certificate = compute_e2chi_certificate(payload["d"], payload["c2"], payload["M"])
    rendered = json.dumps(certificate.to_dict(), indent=2, sort_keys=True) + "\n"
    if arguments.output:
        arguments.output.write_text(rendered)
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
