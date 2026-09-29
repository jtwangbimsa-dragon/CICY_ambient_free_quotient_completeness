#!/usr/bin/env python3
"""Low-memory search for one frame-changing E2-chi survivor.

This is a fail-closed escape hatch for cases whose complete candidate set is
too large to materialise.  Finding one survivor is enough to show that the
E2-chi obstruction does not certify H3 for that case.  Absence is reported
only when the exact matrix tree is completely exhausted.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import sympy as sp

from intrinsic_e2chi_complete.candidate_enumerator import (
    DEFAULT_DATASET,
    DEFAULT_E1_RECORDS,
    ColumnEnumerationRecord,
    _rank_mod_prime,
    c2_ellipsoid_certificate,
    c2_pairing_from_tensor,
    cubic_tensor,
    enumerate_column_candidates,
    load_jsonl,
    permutes_ambient_frame,
    sha256_file,
    sparse_cubic,
    trilinear,
)
from intrinsic_e2chi_complete.core import compute_e2chi_certificate, finite_matrix_order
from intrinsic_e2chi_complete.configuration_geometry import euler_characteristic


class SearchCapReached(RuntimeError):
    pass


class _SurvivorFound(RuntimeError):
    def __init__(self, payload: dict[str, Any]):
        super().__init__("frame-changing E2-chi survivor found")
        self.payload = payload


@dataclass(frozen=True)
class SearchStats:
    backtrack_node_count: int
    complete_unimodular_matrix_count: int
    finite_order_frame_preserving_count: int
    finite_order_frame_changing_count: int
    chi_Omega1_excluded_count: int
    E2chi_excluded_count: int
    E2chi_evaluation_count: int
    determinant_rejection_count: int
    exact_arithmetic: bool = True


def _matrix_sha256(matrix: Sequence[Sequence[int]]) -> str:
    flat = [int(value) for row in matrix for value in row]
    payload = json.dumps(flat, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def _order_dividing_index(
    matrix: Sequence[Sequence[int]], index: int
) -> int | None:
    """Return the exact order when it divides ``index``, else ``None``.

    A free cyclic action forces the matrix order to divide chi(Omega^1).
    Testing this power first avoids expensive characteristic-polynomial work
    on the overwhelmingly many candidates that are already impossible.
    """

    bound = abs(int(index))
    if bound == 0:
        try:
            return finite_matrix_order(matrix)
        except ValueError:
            return None
    key = tuple(tuple(int(value) for value in row) for row in matrix)
    rank = len(key)
    identity = tuple(tuple(int(i == j) for j in range(rank)) for i in range(rank))

    def multiply(
        first: tuple[tuple[int, ...], ...],
        second: tuple[tuple[int, ...], ...],
    ) -> tuple[tuple[int, ...], ...]:
        return tuple(
            tuple(
                sum(first[i][k] * second[k][j] for k in range(rank))
                for j in range(rank)
            )
            for i in range(rank)
        )

    def power(exponent: int) -> tuple[tuple[int, ...], ...]:
        result = identity
        base = key
        remaining = int(exponent)
        while remaining:
            if remaining & 1:
                result = multiply(result, base)
            remaining >>= 1
            if remaining:
                base = multiply(base, base)
        return result

    if power(bound) != identity:
        return None
    for divisor in sp.divisors(bound):
        if power(int(divisor)) == identity:
            return int(divisor)
    raise ArithmeticError("matrix power passed the index bound but no order divisor was found")


def search_first_survivor(
    tensor: Sequence[Sequence[Sequence[int]]],
    c2: Sequence[int],
    chi_omega1: int,
    column_records: Sequence[ColumnEnumerationRecord],
    *,
    node_cap: int | None = None,
    matrix_cap: int | None = None,
    random_seed: int | None = None,
    max_changed_columns: int | None = None,
    exact_changed_columns: int | None = None,
) -> dict[str, Any]:
    rank = len(column_records)
    sparse = sparse_cubic(tensor)
    candidates = [record.candidate_columns for record in column_records]
    standard_basis = tuple(
        tuple(int(row == generator) for row in range(rank)) for generator in range(rank)
    )
    search_order = tuple(sorted(range(rank), key=lambda index: (len(candidates[index]), index)))
    selected: dict[int, tuple[int, tuple[int, ...]]] = {}
    rng = None if random_seed is None else random.Random(random_seed)
    changed_cap = (
        exact_changed_columns if exact_changed_columns is not None else max_changed_columns
    )
    pair_cache: dict[tuple[int, int, int, int], int] = {}
    triple_cache: dict[tuple[tuple[int, int], ...], bool] = {}

    nodes = matrices = preserving = changing = 0
    chi_excluded = e2_excluded = e2_evaluations = determinant_rejections = 0

    def pair_status(a: int, ai: int, b: int, bi: int) -> int:
        if a > b:
            a, b, ai, bi = b, a, bi, ai
        key = (a, ai, b, bi)
        if key in pair_cache:
            return pair_cache[key]
        first, second = candidates[a][ai], candidates[b][bi]
        if (
            trilinear(sparse, first, first, second) != int(tensor[a][a][b])
            or trilinear(sparse, first, second, second) != int(tensor[a][b][b])
        ):
            pair_cache[key] = 1
        elif any(
            trilinear(sparse, first, second, standard) < 0
            for standard in standard_basis
        ):
            pair_cache[key] = 2
        else:
            pair_cache[key] = 0
        return pair_cache[key]

    def triple_ok(a: int, ai: int, b: int, bi: int, c: int, ci: int) -> bool:
        labelled = tuple(sorted(((a, ai), (b, bi), (c, ci))))
        if labelled not in triple_cache:
            (x, xi), (y, yi), (z, zi) = labelled
            triple_cache[labelled] = trilinear(
                sparse, candidates[x][xi], candidates[y][yi], candidates[z][zi]
            ) == int(tensor[x][y][z])
        return triple_cache[labelled]

    def compatible_options(target: int) -> list[tuple[int, tuple[int, ...]]]:
        nonlocal nodes
        existing = sorted(selected)
        changed_so_far = sum(
            vector != standard_basis[index] for index, (_candidate_index, vector) in selected.items()
        )
        options: list[tuple[int, tuple[int, ...]]] = []
        for candidate_index, vector in enumerate(candidates[target]):
            if (
                changed_cap is not None
                and vector != standard_basis[target]
                and changed_so_far >= changed_cap
            ):
                continue
            nodes += 1
            if node_cap is not None and nodes > node_cap:
                raise SearchCapReached(f"matrix node cap {node_cap} reached")
            if any(
                pair_status(other, selected[other][0], target, candidate_index) != 0
                for other in existing
            ):
                continue
            if any(
                not triple_ok(
                    first,
                    selected[first][0],
                    second,
                    selected[second][0],
                    target,
                    candidate_index,
                )
                for first, second in itertools.combinations(existing, 2)
            ):
                continue
            partial = [selected[index][1] for index in existing] + [vector]
            if any(_rank_mod_prime(partial, prime) != len(partial) for prime in (2, 3, 5, 7)):
                continue
            options.append((candidate_index, vector))
        # Search genuinely frame-changing branches first.  This affects only
        # time to a witness, never acceptance or exclusion.
        if rng is None:
            options.sort(key=lambda item: item[1] == standard_basis[target])
        else:
            rng.shuffle(options)
        return options

    def recurse(depth: int) -> None:
        nonlocal matrices, preserving, changing
        nonlocal chi_excluded, e2_excluded, e2_evaluations, determinant_rejections
        if exact_changed_columns is not None:
            changed_now = sum(
                vector != standard_basis[index]
                for index, (_candidate_index, vector) in selected.items()
            )
            remaining = rank - depth
            if changed_now > exact_changed_columns or changed_now + remaining < exact_changed_columns:
                return
        if depth == rank:
            rows = tuple(
                tuple(selected[column][1][row] for column in range(rank))
                for row in range(rank)
            )
            determinant = int(sp.Matrix(rows).det())
            if abs(determinant) != 1:
                determinant_rejections += 1
                return
            matrices += 1
            if matrix_cap is not None and matrices > matrix_cap:
                raise SearchCapReached(f"complete matrix cap {matrix_cap} reached")
            if permutes_ambient_frame(rows):
                preserving += 1
                return
            changing += 1
            order = _order_dividing_index(rows, int(chi_omega1))
            if order is None:
                chi_excluded += 1
                return
            certificate = compute_e2chi_certificate(tensor, c2, rows)
            e2_evaluations += 1
            if certificate.e2chi_excludes_free_lift:
                e2_excluded += 1
                return
            raise _SurvivorFound(
                {
                    "matrix": [list(row) for row in rows],
                    "matrix_sha256": _matrix_sha256(rows),
                    "matrix_order": order,
                    "E2chi": certificate.to_dict(),
                }
            )

        best_target: int | None = None
        best_options: list[tuple[int, tuple[int, ...]]] | None = None
        for target in search_order:
            if target in selected:
                continue
            options = compatible_options(target)
            if not options:
                return
            if best_options is None or len(options) < len(best_options):
                best_target, best_options = target, options
        if best_target is None or best_options is None:
            raise RuntimeError("dynamic MRV failed to select a target")
        for candidate_index, vector in best_options:
            selected[best_target] = (candidate_index, vector)
            recurse(depth + 1)
            del selected[best_target]

    survivor: dict[str, Any] | None = None
    exhausted = False
    try:
        recurse(0)
        exhausted = True
    except _SurvivorFound as found:
        survivor = found.payload

    stats = SearchStats(
        backtrack_node_count=nodes,
        complete_unimodular_matrix_count=matrices,
        finite_order_frame_preserving_count=preserving,
        finite_order_frame_changing_count=changing,
        chi_Omega1_excluded_count=chi_excluded,
        E2chi_excluded_count=e2_excluded,
        E2chi_evaluation_count=e2_evaluations,
        determinant_rejection_count=determinant_rejections,
    )
    return {
        "exhausted": exhausted,
        "survivor_found": survivor is not None,
        "survivor": survivor,
        "search_stats": asdict(stats),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num", type=int, required=True)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--e1-records", type=Path, default=DEFAULT_E1_RECORDS)
    parser.add_argument("--point-cap", type=int, default=2_000_000)
    parser.add_argument("--node-cap", type=int)
    parser.add_argument("--matrix-cap", type=int)
    parser.add_argument("--random-seed", type=int)
    parser.add_argument("--max-changed-columns", type=int)
    parser.add_argument("--exact-changed-columns", type=int)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    if arguments.max_changed_columns is not None and arguments.exact_changed_columns is not None:
        parser.error("use at most one of --max-changed-columns and --exact-changed-columns")

    started = time.monotonic()
    record = next(
        (item for item in load_jsonl(arguments.dataset) if int(item["Num"]) == arguments.num),
        None,
    )
    if record is None:
        raise ValueError(f"Num {arguments.num} is absent from the dataset")
    conf = record["working_conf"]
    dimensions = [int(value) for value in record["working_ambient_dimensions"]]
    tensor = cubic_tensor(conf, dimensions)
    c2 = c2_pairing_from_tensor(conf, dimensions, tensor)
    ellipsoid, kernel, hodge = c2_ellipsoid_certificate(tensor, c2)
    if not ellipsoid.negative_restriction_certified:
        raise ValueError("negative-restriction ellipsoid gate failed")
    columns = [
        enumerate_column_candidates(
            tensor, c2, target, kernel, hodge, point_cap=arguments.point_cap
        )
        for target in range(len(conf))
    ]
    result = search_first_survivor(
        tensor,
        c2,
        -euler_characteristic(conf, dimensions) // 2,
        columns,
        node_cap=arguments.node_cap,
        matrix_cap=arguments.matrix_cap,
        random_seed=arguments.random_seed,
        max_changed_columns=arguments.max_changed_columns,
        exact_changed_columns=arguments.exact_changed_columns,
    )
    payload = {
        "schema_version": "1.0.0",
        "computation_id": "cicy_e2chi_early_survivor_20260824",
        "Num": arguments.num,
        "configuration_only": True,
        "defining_polynomial_used": False,
        "ellipsoid_certificate": asdict(ellipsoid),
        "column_candidate_counts": [len(item.candidate_columns) for item in columns],
        "random_seed": arguments.random_seed,
        "max_changed_columns": arguments.max_changed_columns,
        "exact_changed_columns": arguments.exact_changed_columns,
        "input_sha256": {
            "dataset": sha256_file(arguments.dataset),
            "e1_records": sha256_file(arguments.e1_records),
        },
        "elapsed_seconds": time.monotonic() - started,
        **result,
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if arguments.output is None:
        print(rendered, end="")
    else:
        arguments.output.write_text(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
