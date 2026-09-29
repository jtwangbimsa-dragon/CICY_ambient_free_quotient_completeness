#!/usr/bin/env python3
"""Close residual integral-E1 cases with uniform curve certificates.

For every surviving prime-order overlattice representative v, search for a
square-free ambient curve degree beta such that

    <v,beta> is not an integer,     N_beta != 0.

The second condition is certified by ``stable_map_euler.py``.  It supplies an
honest integral curve class on every member of the CICY family, so the first
condition rules out v as an integral divisor class.  The public E1 program
passes its freshly computed surviving overlattices directly to
``close_record``; this module reads no saved result file and selects no
defining polynomial.
"""

from __future__ import annotations

import itertools
from fractions import Fraction
from typing import Any, Sequence

try:
    # Package import used by the public ``check_E1.py`` entry point.
    from .stable_map_euler import ambient_dimensions, stable_map_euler_number
except ImportError:  # pragma: no cover - retained for direct script execution
    from stable_map_euler import ambient_dimensions, stable_map_euler_number


def _fraction(value: str | int) -> Fraction:
    return Fraction(value)


def _fraction_json(value: Fraction) -> str:
    return str(value.numerator) if value.denominator == 1 else str(value)


def _pairing(rep: Sequence[Fraction], beta: Sequence[int]) -> Fraction:
    return sum((value * degree for value, degree in zip(rep, beta)), Fraction(0))


def _candidate_betas(rank: int, reps: Sequence[Sequence[Fraction]], unresolved: set[int]):
    candidates: list[tuple[int, int, tuple[int, ...], tuple[int, ...]]] = []
    for support_size in (2, 3):
        for support in itertools.combinations(range(rank), support_size):
            beta = tuple(int(index in support) for index in range(rank))
            killed = tuple(
                index
                for index in sorted(unresolved)
                if _pairing(reps[index], beta).denominator != 1
            )
            if killed:
                # Exhaust all bidegrees before any tridegree: the latter has
                # many more fixed trees.  Within one support size, prefer the
                # curve that excludes the largest number of survivors.
                candidates.append((support_size, -len(killed), support, killed))
    return sorted(candidates)


def close_record(record: dict[str, Any]) -> dict[str, Any]:
    num = record["Num"]
    conf = record["conf"]
    rank = len(conf)
    reps = [
        tuple(_fraction(value) for value in item["rep"])
        for item in record["surviving_overlattices"]
    ]
    if any(len(rep) != rank for rep in reps):
        raise ValueError(f"Num {num}: survivor rank mismatch")

    unresolved = set(range(len(reps)))
    witnesses: list[dict[str, Any]] = []
    computed: dict[tuple[int, ...], tuple[Fraction, int]] = {}

    while unresolved:
        progress = False
        for _, _, support, proposed_kills in _candidate_betas(rank, reps, unresolved):
            beta = tuple(int(index in support) for index in range(rank))
            if beta not in computed:
                computed[beta] = stable_map_euler_number(conf, beta)
            number, graph_count = computed[beta]
            if number == 0:
                continue
            killed = [index for index in proposed_kills if index in unresolved]
            if not killed:
                continue
            pairing_values = {
                str(index): _fraction_json(_pairing(reps[index], beta))
                for index in killed
            }
            witnesses.append(
                {
                    "beta": list(beta),
                    "support_rows_1based": [index + 1 for index in support],
                    "stable_map_euler_number": _fraction_json(number),
                    "fixed_graph_count": graph_count,
                    "excluded_survivor_indices_0based": killed,
                    "nonintegral_pairings": pairing_values,
                }
            )
            unresolved.difference_update(killed)
            progress = True
            break
        if not progress:
            break

    survivor_results = []
    for index, rep in enumerate(reps):
        witness_index = next(
            (
                position
                for position, witness in enumerate(witnesses)
                if index in witness["excluded_survivor_indices_0based"]
            ),
            None,
        )
        survivor_results.append(
            {
                "index_0based": index,
                "representative": [_fraction_json(value) for value in rep],
                "status": "excluded_by_uniform_curve" if witness_index is not None else "unresolved",
                "witness_index_0based": witness_index,
            }
        )

    closed = not unresolved
    return {
        "Num": num,
        "ambient_dimensions": list(ambient_dimensions(conf)),
        "conf": conf,
        "input_overlattice_survivor_count": len(reps),
        "computed_beta_count": len(computed),
        "witnesses": witnesses,
        "survivors": survivor_results,
        "status": "pass_uniform_curve_overlattice_exclusion" if closed else "unverified_curve_search_exhausted",
        "unresolved_survivor_indices_0based": sorted(unresolved),
        "scope": {
            "defining_polynomial_used": False,
            "family_uniform": True,
            "reason": (
                "A nonzero top Chern number on the proper convex stable-map "
                "space forces the induced section from every defining tuple "
                "to have a zero; the resulting integral curve degree pairs "
                "nonintegrally with the proposed overlattice divisor."
            ),
        },
    }
