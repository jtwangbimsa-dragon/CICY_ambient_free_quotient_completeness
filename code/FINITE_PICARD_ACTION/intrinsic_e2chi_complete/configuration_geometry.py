"""Exact utilities derived only from a CICY configuration matrix."""

from __future__ import annotations

import itertools
from collections import Counter, defaultdict
from typing import Sequence


def _validate_conf(conf: Sequence[Sequence[int]]) -> tuple[int, int]:
    if not conf or not conf[0]:
        raise ValueError("configuration must be a nonempty matrix")
    width = len(conf[0])
    if any(len(row) != width for row in conf):
        raise ValueError("configuration must be rectangular")
    if any(
        not isinstance(value, int) or isinstance(value, bool) or value < 0
        for row in conf
        for value in row
    ):
        raise ValueError("configuration entries must be nonnegative integers")
    return len(conf), width


def _inverse_permutation(permutation: Sequence[int]) -> list[int]:
    if sorted(permutation) != list(range(len(permutation))):
        raise ValueError("not a permutation")
    inverse = [0] * len(permutation)
    for source, target in enumerate(permutation):
        inverse[target] = source
    return inverse


def _column_vectors(conf: Sequence[Sequence[int]]) -> list[tuple[int, ...]]:
    return [tuple(column) for column in zip(*conf)]


def find_configuration_isomorphism(
    source_dimensions: Sequence[int],
    source_conf: Sequence[Sequence[int]],
    target_dimensions: Sequence[int],
    target_conf: Sequence[Sequence[int]],
) -> tuple[list[int], list[int]]:
    """Return the least explicit dimension-preserving row/column relabelling."""

    source_rows, source_columns = _validate_conf(source_conf)
    target_rows, target_columns = _validate_conf(target_conf)
    if (source_rows, source_columns) != (target_rows, target_columns):
        raise ValueError("source and target configurations have different shapes")
    if len(source_dimensions) != source_rows or len(target_dimensions) != target_rows:
        raise ValueError("ambient-dimension vector has the wrong length")

    dimension_values = sorted(set(source_dimensions) | set(target_dimensions))
    source_groups = {
        dimension: [i for i, value in enumerate(source_dimensions) if value == dimension]
        for dimension in dimension_values
    }
    target_groups = {
        dimension: [i for i, value in enumerate(target_dimensions) if value == dimension]
        for dimension in dimension_values
    }
    if any(
        len(source_groups[dimension]) != len(target_groups[dimension])
        for dimension in dimension_values
    ):
        raise ValueError("ambient dimensions have different multiplicities")

    target_column_vectors = _column_vectors(target_conf)
    target_column_counter = Counter(target_column_vectors)
    best: tuple[tuple[int, ...], tuple[int, ...]] | None = None
    permutation_families = [
        itertools.permutations(target_groups[dimension])
        for dimension in dimension_values
    ]
    for grouped_images in itertools.product(*permutation_families):
        row_map = [-1] * source_rows
        for dimension, images in zip(dimension_values, grouped_images):
            for source_row, target_row in zip(source_groups[dimension], images):
                row_map[source_row] = target_row
        inverse_row_map = _inverse_permutation(row_map)
        transformed_columns = [
            tuple(
                source_conf[inverse_row_map[target_row]][source_column]
                for target_row in range(source_rows)
            )
            for source_column in range(source_columns)
        ]
        if Counter(transformed_columns) != target_column_counter:
            continue
        available: dict[tuple[int, ...], list[int]] = defaultdict(list)
        for target_column, vector in enumerate(target_column_vectors):
            available[vector].append(target_column)
        column_map = [available[vector].pop(0) for vector in transformed_columns]
        candidate = (tuple(row_map), tuple(column_map))
        if best is None or candidate < best:
            best = candidate
    if best is None:
        raise ValueError("no dimension-preserving row/column isomorphism exists")
    return list(best[0]), list(best[1])


def _normal_product(
    conf: Sequence[Sequence[int]], dimensions: Sequence[int]
) -> dict[tuple[int, ...], int]:
    rank = len(dimensions)
    polynomial: dict[tuple[int, ...], int] = {(0,) * rank: 1}
    for column in zip(*conf):
        updated: defaultdict[tuple[int, ...], int] = defaultdict(int)
        for exponent, coefficient in polynomial.items():
            for row, degree in enumerate(column):
                if degree == 0 or exponent[row] >= dimensions[row]:
                    continue
                target = list(exponent)
                target[row] += 1
                updated[tuple(target)] += coefficient * degree
        polynomial = dict(updated)
    return polynomial


def _c3_polynomial(
    conf: Sequence[Sequence[int]], dimensions: Sequence[int]
) -> dict[tuple[int, ...], int]:
    rank = len(dimensions)
    numerator: defaultdict[tuple[int, ...], int] = defaultdict(int)
    for row, dimension in enumerate(dimensions):
        exponent = [0] * rank
        exponent[row] = 3
        numerator[tuple(exponent)] += dimension + 1
    for column in zip(*conf):
        support = [row for row, degree in enumerate(column) if degree]
        for i in support:
            for j in support:
                for k in support:
                    exponent = [0] * rank
                    exponent[i] += 1
                    exponent[j] += 1
                    exponent[k] += 1
                    numerator[tuple(exponent)] -= column[i] * column[j] * column[k]
    answer: dict[tuple[int, ...], int] = {}
    for exponent, coefficient in numerator.items():
        if coefficient % 3:
            raise ArithmeticError("nonintegral c3 coefficient")
        coefficient //= 3
        if coefficient:
            answer[exponent] = coefficient
    return answer


def euler_characteristic(
    conf: Sequence[Sequence[int]], dimensions: Sequence[int]
) -> int:
    """Return the Euler characteristic by exact ambient intersection theory."""

    rows, _columns = _validate_conf(conf)
    if len(dimensions) != rows:
        raise ValueError("ambient-dimension vector has the wrong length")
    normal = _normal_product(conf, dimensions)
    answer = 0
    for exponent, coefficient in _c3_polynomial(conf, dimensions).items():
        target = tuple(n - e for n, e in zip(dimensions, exponent))
        if min(target) >= 0:
            answer += coefficient * normal.get(target, 0)
    return answer
