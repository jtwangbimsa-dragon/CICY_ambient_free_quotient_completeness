from __future__ import annotations

import math

import pytest
import sympy as sp

from intrinsic_e2chi_complete import (
    compute_e2chi_certificate,
    finite_matrix_order,
    saturated_integer_kernel,
)


def zero_cubic(rank: int) -> list[list[list[int]]]:
    return [[[0 for _ in range(rank)] for _ in range(rank)] for _ in range(rank)]


def test_saturated_kernel_repairs_joint_denominators() -> None:
    # Rational RREF gives (-3/2,1,0), (-5/2,0,1). Clearing the two
    # denominators separately has index 2, so a genuine saturation step is
    # required to recover all solutions, including (1,1,-1).
    equation = sp.Matrix([[2, 3, 5]])
    basis = saturated_integer_kernel(equation)
    assert equation * basis == sp.zeros(1, 2)
    target = sp.Matrix([1, 1, -1])
    coefficients = basis.gauss_jordan_solve(target)[0]
    assert all(value.is_Integer for value in coefficients)

    minors = [
        int(basis[list(rows), :].det())
        for rows in itertools_combinations(range(3), 2)
    ]
    assert math.gcd(*(abs(value) for value in minors)) == 1


def test_saturated_kernel_is_deterministic_under_repeated_evaluation() -> None:
    equation = [[2, 3, 5]]
    assert saturated_integer_kernel(equation) == saturated_integer_kernel(equation)


def itertools_combinations(values, length):
    # Kept local so the test states exactly which full-rank minors are used.
    import itertools

    return itertools.combinations(values, length)


def test_exact_order_uses_lcm_of_cyclotomic_blocks() -> None:
    order_three = sp.Matrix([[0, -1], [1, -1]])
    order_four = sp.Matrix([[0, -1], [1, 0]])
    matrix = sp.diag(order_three, order_four)
    assert finite_matrix_order(matrix) == 12


def test_nonsemisimple_cyclotomic_matrix_is_rejected() -> None:
    with pytest.raises(ValueError, match="Jordan block"):
        finite_matrix_order([[1, 1], [0, 1]])


def test_one_failed_divisibility_condition_is_enough() -> None:
    # M fixes e1 and negates e2. On the fixed lattice Z*e1:
    # c_d=2, c_c2=8, but chi(a*e1)=(a^3+2a)/3 has gcd 1.
    d = zero_cubic(2)
    d[0][0][0] = 2
    certificate = compute_e2chi_certificate(d, [8, 0], [[1, 0], [0, -1]])

    assert certificate.matrix_order == 2
    assert certificate.invariant_rank == 1
    assert certificate.c_d == 2
    assert certificate.c_c2 == 8
    assert certificate.c_chi == 1
    assert certificate.order_divides_c_d
    assert certificate.order_divides_c_c2
    assert not certificate.order_divides_c_chi
    assert certificate.failed_conditions == ("c_chi",)
    assert certificate.any_divisibility_failure
    assert certificate.e2chi_excludes_free_lift


def test_all_three_divisibilities_mean_survivor_not_existence() -> None:
    d = zero_cubic(2)
    d[0][0][0] = 2
    certificate = compute_e2chi_certificate(d, [20, 0], [[1, 0], [0, -1]])

    assert (certificate.c_d, certificate.c_c2, certificate.c_chi) == (2, 20, 2)
    assert certificate.failed_conditions == ()
    assert not certificate.any_divisibility_failure
    assert not certificate.e2chi_excludes_free_lift


def test_zero_rank_fixed_lattice_has_zero_gcds_and_survives() -> None:
    certificate = compute_e2chi_certificate([[[0]]], [0], [[-1]])
    assert certificate.matrix_order == 2
    assert certificate.invariant_rank == 0
    assert certificate.invariant_basis_columns == ()
    assert (certificate.c_d, certificate.c_c2, certificate.c_chi) == (0, 0, 0)
    assert certificate.chi_simplex_size == 1
    assert not certificate.any_divisibility_failure


def test_degree_three_simplex_gcd_matches_large_brute_box() -> None:
    d = zero_cubic(2)
    d[0][0][0] = 2
    d[1][1][1] = 2
    certificate = compute_e2chi_certificate(d, [20, 20], [[1, 0], [0, 1]])

    brute_values = []
    for x in range(-5, 6):
        for y in range(-5, 6):
            brute_values.append((x**3 + 5 * x + y**3 + 5 * y) // 3)
    assert certificate.chi_simplex_size == math.comb(2 + 3, 3)
    assert certificate.c_chi == math.gcd(*(abs(value) for value in brute_values)) == 2


def test_nonintegral_riemann_roch_data_fail_closed() -> None:
    d = zero_cubic(2)
    d[0][0][0] = 1
    with pytest.raises(ValueError, match="Riemann--Roch is nonintegral"):
        compute_e2chi_certificate(d, [0, 0], [[1, 0], [0, -1]])


def test_nonpreserving_matrix_is_not_mislabeled_as_e2chi() -> None:
    d = zero_cubic(2)
    d[0][0][0] = 2
    with pytest.raises(ValueError, match="not a topological candidate"):
        compute_e2chi_certificate(d, [20, 0], [[0, 1], [1, 0]])
