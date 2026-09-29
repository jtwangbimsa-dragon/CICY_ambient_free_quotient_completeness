#!/usr/bin/env python3
"""Derivation certificate for the graph re-embeddings used as E3 repairs.

The dataset repairs an E3 obstruction column ``q_a = 2 e_r + e_i`` (ambient row ``r`` a
``P^1`` with homogeneous coordinates ``u_r, v_r``) by appending one coordinate ``z_a`` to the
target factor ``i`` and replacing the single equation

    p_a = a_a(x) u_r^2 + b_a(x) u_r v_r + c_a(x) v_r^2 = 0

by the pair

    E_1 := v_r z_a - 2 u_r a_a(x) - v_r b_a(x) = 0,
    E_2 := u_r z_a + u_r b_a(x) + 2 v_r c_a(x) = 0.

This module certifies that the resulting scheme is isomorphic to the *fixed* original X, so
that automorphism statements transport between the two presentations.  The certificate has
three parts, all coefficient independent.

(1) ELIMINATION.  u_r E_1 - v_r E_2 = -2 p_a identically, so the pair implies p_a = 0.

(2) GRAPH.  On {v_r != 0}, E_1 = 0 determines z_a = 2 a_a u_r / v_r + b_a uniquely and then
    E_2 = 2 p_a / v_r; on {u_r != 0}, E_2 = 0 determines z_a = -b_a - 2 c_a v_r / u_r and then
    E_1 = -2 p_a / u_r.  Since (u_r, v_r) never vanishes simultaneously, the two charts cover the
    P^1 row and agree, so the pair cuts out exactly the graph of a section over {p_a = 0}.

(3) THE PROJECTION IS A MORPHISM.  Each of a_a, b_a, c_a has multidegree e_i, hence is linear in
    the *original* coordinates of factor i.  If every original factor-i coordinate vanished at a
    point of the re-embedded scheme, then a_a = b_a = c_a = 0 there, so E_1 = v_r z_a = 0 and
    E_2 = u_r z_a = 0 force z_a = 0 for every appended coordinate targeting i, and all homogeneous
    coordinates of the enlarged factor would vanish, which is impossible.  Hence projecting away
    the appended coordinates is a morphism on the re-embedded scheme, and by (2) it inverts the
    graph embedding.

The script verifies, per record, that the structural hypotheses of (1)-(3) actually hold for the
supplied data, and emits a per-record derivation hash.  It does not re-derive (1)-(3); those are the
theorem, checked symbolically once.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


ELIMINATION_IDENTITY = "u_r*(v_r*z_a - 2*u_r*a_a - v_r*b_a) - v_r*(u_r*z_a + u_r*b_a + 2*v_r*c_a) = -2*(a_a*u_r^2 + b_a*u_r*v_r + c_a*v_r^2)"


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def unit(index: int, size: int) -> list[int]:
    return [1 if i == index else 0 for i in range(size)]


def check_record(record: dict[str, Any]) -> dict[str, Any]:
    num = record["Num"]
    mappings = record.get("obstruction_mappings") or []
    original = record["original_selected_conf"]
    working = record["working_conf"]
    odims = record["original_ambient_dimensions"]
    wdims = record["working_ambient_dimensions"]
    rho = len(odims)

    failures: list[str] = []

    if not mappings:
        return {
            "Num": num,
            "status": "identity_presentation",
            "obstruction_count": 0,
        }

    if len(wdims) != rho:
        failures.append("graph transformation must not create a new projective factor")

    appended_per_row = [0] * rho
    checked = []
    for mapping in mappings:
        r = mapping["p1_row_index_1based"] - 1
        i = mapping["target_row_index_1based"] - 1
        old = mapping["old_multidegree"]
        reps = mapping["replacement_multidegrees"]
        expected_old = [0] * rho
        expected_old[r] += 2
        expected_old[i] += 1
        expected_rep = [0] * rho
        expected_rep[r] += 1
        expected_rep[i] += 1

        if odims[r] != 1:
            failures.append(f"row {r+1} carries u,v but has dimension {odims[r]} != 1")
        if list(old) != expected_old:
            failures.append(f"column {mapping['original_column_index_1based']} is not 2e_r+e_i")
        if len(reps) != 2 or any(list(x) != expected_rep for x in reps):
            failures.append(
                f"replacement multidegrees for column "
                f"{mapping['original_column_index_1based']} are not two copies of e_r+e_i"
            )
        if r == i:
            failures.append("the P^1 row and the target row must differ")
        appended_per_row[i] += 1
        checked.append(
            {
                "p1_row_1based": r + 1,
                "target_row_1based": i + 1,
                "old_multidegree": list(old),
                "replacement_multidegree": expected_rep,
                "coefficients_linear_in_target_factor": True,
            }
        )

    for row in range(rho):
        if wdims[row] != odims[row] + appended_per_row[row]:
            failures.append(
                f"row {row+1}: working dimension {wdims[row]} != "
                f"{odims[row]} + {appended_per_row[row]} appended coordinates"
            )

    untouched_old = [
        col
        for col in range(len(original[0]))
        if col + 1 not in {m["original_column_index_1based"] for m in mappings}
    ]
    replaced_new = set()
    for mapping in mappings:
        replaced_new.update(mapping["replacement_working_column_indices_1based"])
    untouched_new = [
        col for col in range(len(working[0])) if col + 1 not in replaced_new
    ]
    if len(untouched_old) != len(untouched_new):
        failures.append("untouched column counts differ between the two presentations")
    else:
        for old_col, new_col in zip(untouched_old, untouched_new):
            old_deg = [original[row][old_col] for row in range(rho)]
            new_deg = [working[row][new_col] for row in range(rho)]
            if old_deg != new_deg:
                failures.append(
                    f"untouched column {old_col+1} changed multidegree under the transformation"
                )
                break

    for row in range(rho):
        if sum(working[row]) != wdims[row] + 1:
            failures.append(f"working row {row+1} violates the Calabi-Yau row-sum condition")

    if sum(wdims) - len(working[0]) != 3:
        failures.append("working presentation is not three dimensional")

    certificate = {
        "Num": num,
        "status": "pass_graph_derivation" if not failures else "fail",
        "obstruction_count": len(mappings),
        "appended_coordinates_per_row": appended_per_row,
        "columns": checked,
        "elimination_identity": ELIMINATION_IDENTITY,
        "graph_section_charts": [
            "v_r != 0 : z_a = 2 a_a u_r / v_r + b_a, and then E_2 = 2 p_a / v_r",
            "u_r != 0 : z_a = -b_a - 2 c_a v_r / u_r, and then E_1 = -2 p_a / u_r",
        ],
        "projection_is_a_morphism": (
            "a_a,b_a,c_a are linear in the original factor-i coordinates; if those all vanished "
            "then E_1 = v_r z_a and E_2 = u_r z_a force z_a = 0 for every appended coordinate, so "
            "all homogeneous coordinates of the enlarged factor would vanish"
        ),
        "conclusion": (
            "projection away from the appended coordinates is a morphism on the re-embedded "
            "scheme and is inverse to the graph embedding, so the two presentations carry the "
            "same fixed X and the same automorphism group"
        ),
    }
    if failures:
        certificate["failures"] = failures
    certificate["derivation_sha256"] = sha256_json(
        {k: v for k, v in certificate.items() if k != "derivation_sha256"}
    )
    return certificate


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="cicy_e1e3_fixed.jsonl")
    parser.add_argument("--out", default="graph_provenance_records.jsonl")
    parser.add_argument("--summary", default="graph_provenance_summary.json")
    args = parser.parse_args()

    records = [json.loads(line) for line in Path(args.dataset).read_text().splitlines() if line]
    counts: dict[str, int] = {}
    transformed = 0
    with open(args.out, "w") as handle:
        for record in records:
            certificate = check_record(record)
            counts[certificate["status"]] = counts.get(certificate["status"], 0) + 1
            if certificate["obstruction_count"]:
                transformed += 1
                handle.write(canonical_json(certificate) + "\n")

    summary = {
        "audit_id": "cicy_graph_provenance_20260823",
        "record_count": len(records),
        "graph_transformed_count": transformed,
        "status_counts": counts,
        "theorem": {
            "elimination": ELIMINATION_IDENTITY,
            "scope": (
                "coefficient independent: the identity and the two charts hold for arbitrary "
                "a_a, b_a, c_a, so the certificate is a property of the transformation, not of a "
                "particular defining polynomial"
            ),
        },
    }
    Path(args.summary).write_text(json.dumps(summary, indent=1, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
