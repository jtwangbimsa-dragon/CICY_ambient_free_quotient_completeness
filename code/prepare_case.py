#!/usr/bin/env python3
"""Create the minimal JSONL input record for one untransformed CICY matrix.

A bare configuration matrix determines the intersection-theoretic quantities,
but it does not by itself prove favourability.  The user must therefore supply
``--h11`` and explicitly assert ``--favourable-certified`` when using the E1
criteria.  Likewise, the Picard-torsion theorem is applied only after the user
asserts that this is a standard smooth non-product CICY threefold.

For graph re-embeddings use the distributed 7,820-case input file, which also
contains the original matrix and the exact graph-construction metadata.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

FINITE_PICARD_ROOT = Path(__file__).resolve().parent / "FINITE_PICARD_ACTION"
sys.path.insert(0, str(FINITE_PICARD_ROOT))

from intrinsic_e2chi_complete.candidate_enumerator import (  # noqa: E402
    c2_pairing_from_tensor,
    cubic_tensor,
    dimensions_from_conf,
)


def validate_conf(value: Any) -> list[list[int]]:
    if not isinstance(value, list) or not value or not isinstance(value[0], list) or not value[0]:
        raise ValueError("configuration must be a nonempty matrix")
    width = len(value[0])
    if any(not isinstance(row, list) or len(row) != width for row in value):
        raise ValueError("configuration must be rectangular")
    if any(not isinstance(x, int) or isinstance(x, bool) or x < 0 for row in value for x in row):
        raise ValueError("configuration entries must be nonnegative integers")
    dimensions = dimensions_from_conf(value)
    if any(n < 1 for n in dimensions):
        raise ValueError("every ambient projective factor must have positive dimension")
    if any(sum(row) != n + 1 for row, n in zip(value, dimensions)):
        raise ValueError("Calabi--Yau row-sum condition failed")
    if sum(dimensions) - width != 3:
        raise ValueError("the configuration does not have complex dimension three")
    return [[int(x) for x in row] for row in value]


def normal_product(conf: list[list[int]]) -> dict[tuple[int, ...], int]:
    dimensions = dimensions_from_conf(conf)
    polynomial: dict[tuple[int, ...], int] = {(0,) * len(conf): 1}
    for column in zip(*conf):
        updated: defaultdict[tuple[int, ...], int] = defaultdict(int)
        for exponent, coefficient in polynomial.items():
            for row, degree in enumerate(column):
                if degree and exponent[row] < dimensions[row]:
                    target = list(exponent)
                    target[row] += 1
                    updated[tuple(target)] += coefficient * int(degree)
        polynomial = dict(updated)
    return polynomial


def c3_polynomial(conf: list[list[int]]) -> dict[tuple[int, ...], int]:
    dimensions = dimensions_from_conf(conf)
    rank = len(conf)
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
                    numerator[tuple(exponent)] -= int(column[i] * column[j] * column[k])
    result: dict[tuple[int, ...], int] = {}
    for exponent, value in numerator.items():
        if value % 3:
            raise ValueError("c3 coefficient is not integral")
        if value // 3:
            result[exponent] = value // 3
    return result


def euler_characteristic(conf: list[list[int]]) -> int:
    dimensions = dimensions_from_conf(conf)
    normal = normal_product(conf)
    answer = 0
    for exponent, coefficient in c3_polynomial(conf).items():
        target = tuple(n - e for n, e in zip(dimensions, exponent))
        if min(target) >= 0:
            answer += coefficient * normal.get(target, 0)
    return answer


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--conf", help="configuration matrix as JSON")
    source.add_argument("--conf-file", type=Path, help="JSON file containing the matrix")
    parser.add_argument("--num", type=int, default=1)
    parser.add_argument("--h11", type=int, required=True)
    parser.add_argument("--favourable-certified", action="store_true")
    parser.add_argument("--standard-nonproduct-cicy", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    raw = json.loads(arguments.conf) if arguments.conf is not None else json.loads(arguments.conf_file.read_text())
    conf = validate_conf(raw)
    dimensions = dimensions_from_conf(conf)
    tensor = cubic_tensor(conf, dimensions)
    c2 = c2_pairing_from_tensor(conf, dimensions, tensor)
    euler = euler_characteristic(conf)
    payload_hash = hashlib.sha256(
        json.dumps(conf, separators=(",", ":")).encode()
    ).hexdigest()
    favourable = arguments.favourable_certified and arguments.h11 == len(conf)
    record = {
        "dataset_id": "user_supplied_configuration_v1",
        "Num": arguments.num,
        "source": "FavourConf" if favourable else "user_configuration",
        "original_selected_conf": conf,
        "working_conf": conf,
        "original_ambient_dimensions": dimensions,
        "working_ambient_dimensions": dimensions,
        "canonical_hash_original": payload_hash,
        "canonical_hash_working": payload_hash,
        "obstruction_mappings": [],
        "hypotheses": {
            "rational_favourability_certified": favourable,
            "standard_nonproduct_cicy": arguments.standard_nonproduct_cicy,
        },
        "official_fields": {
            "H11": arguments.h11,
        },
        "presentation_relation": "identity",
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
    print(json.dumps({
        "output": str(arguments.output.resolve()),
        "Num": arguments.num,
        "ambient_dimensions": dimensions,
        "h11": arguments.h11,
        "favourability_gate": "pass" if favourable else "UNKNOWN",
        "standard_nonproduct_gate": "pass" if arguments.standard_nonproduct_cicy else "UNKNOWN",
        "euler_characteristic": euler,
        "c2_pairing": c2,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
