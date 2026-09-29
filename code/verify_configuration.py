#!/usr/bin/env python3
"""Verify the completeness criteria for one supplied CICY presentation.

The program first checks E3 and the direct simultaneous E1--E2 criterion.  If
the direct route is inconclusive, it runs the integral-E1 calculation and then
the E2-chi alternative.  Every certificate is freshly written below the
chosen output directory.  A nonzero exit status means only that the supplied
data were not certified by these criteria; it is not a proof of failure.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
RELEASE_ROOT = HERE.parent
DEFAULT_CATALOGUE = RELEASE_ROOT / "data" / "cicy_criteria_input_7820.jsonl"
ORIGINAL_MATRIX_TABLES = (
    (RELEASE_ROOT / "data" / "cicy_certified_7509_original_and_working_confs.wl", 7509),
    (RELEASE_ROOT / "data" / "cicy_open_311_original_and_working_confs.wl", 311),
)
TABLE_ENTRY = re.compile(
    r'^\s*(\d+)-><\|"OriginalConf"->(\{\{[0-9{},]+\}\}),'
    r'"WorkingConf"->(\{\{[0-9{},]+\}\}),'
)


def _run(arguments: list[str], accepted: set[int] = {0}) -> int:
    completed = subprocess.run(
        [sys.executable, *arguments],
        cwd=RELEASE_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if completed.returncode not in accepted:
        raise RuntimeError(
            f"command failed with exit code {completed.returncode}: "
            + " ".join(arguments)
            + (f"\n{completed.stderr}" if completed.stderr else "")
        )
    return completed.returncode


def _read_one(path: Path, num: int | None = None) -> dict[str, Any]:
    records = [json.loads(line) for line in path.read_text().splitlines() if line]
    if num is not None:
        records = [record for record in records if int(record.get("Num", -1)) == num]
    if len(records) != 1:
        detail = "" if num is None else f" with Num={num}"
        raise ValueError(f"expected exactly one JSON record in {path}{detail}")
    return records[0]


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _parse_matrix(text: str) -> list[list[int]]:
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError("--conf must be a JSON array of integer rows") from exc
    if (
        not isinstance(value, list)
        or not value
        or any(not isinstance(row, list) or not row for row in value)
        or len({len(row) for row in value}) != 1
        or any(
            isinstance(entry, bool) or not isinstance(entry, int) or entry < 0
            for row in value
            for entry in row
        )
    ):
        raise ValueError(
            "--conf must be a nonempty rectangular matrix of nonnegative integers"
        )
    return value


def _original_matrix_index() -> dict[int, tuple[list[list[int]], list[list[int]]]]:
    """Read the original and working matrices from the two bundled WL tables.

    Only the two integer matrix fields are parsed; Mathematica code is never run.
    """
    index: dict[int, tuple[list[list[int]], list[list[int]]]] = {}
    for path, expected_count in ORIGINAL_MATRIX_TABLES:
        count = 0
        for line_number, line in enumerate(path.read_text().splitlines(), start=1):
            if re.match(r"^\s*\d+->", line) is None:
                continue
            match = TABLE_ENTRY.match(line)
            if match is None:
                raise ValueError(f"cannot read matrix entry at {path}:{line_number}")
            number = int(match.group(1))
            if number in index:
                raise ValueError(f"duplicate CICY Num={number} in the matrix tables")
            matrices = tuple(
                _parse_matrix(value.replace("{", "[").replace("}", "]"))
                for value in match.groups()[1:]
            )
            index[number] = matrices
            count += 1
        if count != expected_count:
            raise ValueError(
                f"expected {expected_count} matrix entries in {path}, found {count}"
            )
    return index


def _lookup_catalogue_record(
    matrix: list[list[int]], num: int | None
) -> tuple[dict[str, Any], list[str]]:
    records = [
        json.loads(line)
        for line in DEFAULT_CATALOGUE.read_text().splitlines()
        if line
    ]
    by_num = {int(record["Num"]): record for record in records}
    originals = _original_matrix_index()
    if len(records) != 7820 or len(by_num) != 7820 or set(by_num) != set(originals):
        raise ValueError("the JSON input and original-matrix tables disagree on CICY numbers")
    matches: list[tuple[dict[str, Any], list[str]]] = []
    for number, record in by_num.items():
        original, displayed_working = originals[number]
        if displayed_working != record.get("working_conf"):
            raise ValueError(f"working matrix differs between files for Num={number}")
        if num is not None and number != num:
            continue
        roles = []
        if matrix == original:
            roles.append("standard_original")
        if matrix == record.get("original_selected_conf"):
            roles.append("selected_presentation")
        if matrix == record.get("working_conf"):
            roles.append("working_presentation")
        if roles:
            matches.append((record, roles))
    if not matches:
        qualifier = "" if num is None else f" with Num={num}"
        raise ValueError(
            "the supplied matrix was not found in the bundled 7,820-case catalogue"
            f"{qualifier}; for a new matrix also supply --h11 and both hypothesis flags"
        )
    if len(matches) > 1:
        labels = ", ".join(str(record["Num"]) for record, _ in matches)
        raise ValueError(
            "the supplied matrix occurs in more than one catalogue entry "
            f"({labels}); repeat the command with --num"
        )
    return matches[0]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--record",
        type=Path,
        help="one-record JSONL file, including graph metadata when applicable",
    )
    source.add_argument(
        "--conf",
        help=(
            "original, selected, or working configuration matrix as JSON; "
            "it is looked up in the bundled catalogue unless --h11 is supplied"
        ),
    )
    source.add_argument(
        "--conf-file",
        type=Path,
        help="JSON file containing one configuration matrix",
    )
    parser.add_argument("--h11", type=int)
    parser.add_argument("--favourable-certified", action="store_true")
    parser.add_argument("--standard-nonproduct-cicy", action="store_true")
    parser.add_argument(
        "--num",
        type=int,
        help="label for --conf, or record selector when --record contains many cases",
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, default=Path("verification_output"))
    arguments = parser.parse_args()

    if arguments.workers < 1:
        parser.error("--workers must be positive")
    if arguments.record is not None and any(
        (
            arguments.h11 is not None,
            arguments.favourable_certified,
            arguments.standard_nonproduct_cicy,
        )
    ):
        parser.error(
            "--h11 and the hypothesis flags belong to --conf; a record already contains them"
        )

    output = arguments.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    case = output / "input_record.jsonl"
    if arguments.record is not None:
        record = _read_one(arguments.record.resolve(), arguments.num)
        input_matrix_roles = ["input_record"]
        case.write_text(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
    else:
        matrix_text = (
            arguments.conf
            if arguments.conf is not None
            else arguments.conf_file.resolve().read_text()
        )
        try:
            matrix = _parse_matrix(matrix_text)
        except ValueError as exc:
            parser.error(str(exc))
        if arguments.h11 is None:
            if arguments.favourable_certified or arguments.standard_nonproduct_cicy:
                parser.error(
                    "the hypothesis flags are unnecessary for catalogue lookup; "
                    "supply --h11 as well only for a matrix outside the catalogue"
                )
            try:
                record, input_matrix_roles = _lookup_catalogue_record(
                    matrix, arguments.num
                )
            except ValueError as exc:
                parser.error(str(exc))
            case.write_text(
                json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
            )
        else:
            input_matrix_roles = ["user_supplied_matrix"]
            prepare = [
                "code/prepare_case.py",
                "--conf",
                json.dumps(matrix, separators=(",", ":")),
                "--h11",
                str(arguments.h11),
                "--num",
                str(arguments.num if arguments.num is not None else 900001),
                "--output",
                str(case),
            ]
            if arguments.favourable_certified:
                prepare.append("--favourable-certified")
            if arguments.standard_nonproduct_cicy:
                prepare.append("--standard-nonproduct-cicy")
            _run(prepare)

    input_record = _read_one(case)

    e3_path = output / "E3.jsonl"
    _run(
        [
            "code/check_E3.py",
            "--input",
            str(case),
            "--expected-count",
            "1",
            "--certificates",
            str(e3_path),
            "--summary",
            str(output / "E3_summary.json"),
        ],
        {0, 1},
    )
    direct_path = output / "direct_E1_E2.jsonl"
    _run(
        [
            "code/check_E1_E2.py",
            "--input",
            str(case),
            "--expected-count",
            "1",
            "--certificates",
            str(direct_path),
            "--summary",
            str(output / "direct_E1_E2_summary.json"),
        ]
    )

    e3 = _read_one(e3_path)
    direct = _read_one(direct_path)
    e1: dict[str, Any] | None = None
    e2chi: dict[str, Any] | None = None
    if not direct.get("E1_and_E2_certified"):
        e1_path = output / "integral_E1.jsonl"
        _run(
            [
                "code/FINITE_PICARD_ACTION/check_E1.py",
                "--dataset",
                str(case),
                "--all-input",
                "--workers",
                str(arguments.workers),
                "--output",
                str(e1_path),
                "--summary",
                str(output / "integral_E1_summary.json"),
            ],
            {0, 2},
        )
        e1 = _read_one(e1_path)
        if e1.get("integral_E1_status") == "pass_integral_E1":
            e2chi_path = output / "E2chi.jsonl"
            _run(
                [
                    "code/FINITE_PICARD_ACTION/check_E2chi.py",
                    "--dataset",
                    str(case),
                    "--e1-records",
                    str(e1_path),
                    "--workers",
                    str(arguments.workers),
                    "--records-output",
                    str(e2chi_path),
                    "--summary-output",
                    str(output / "E2chi_summary.json"),
                ],
                {0, 2},
            )
            e2chi = _read_one(e2chi_path)

    direct_complete = bool(
        e3.get("E3_certified") and direct.get("E1_and_E2_certified")
    )
    extended_complete = bool(
        e3.get("E3_certified")
        and e1 is not None
        and e1.get("integral_E1_status") == "pass_integral_E1"
        and e2chi is not None
        and e2chi.get("E2chi_no_frame_changing_survivors")
    )
    if direct_complete:
        status = "CERTIFIED_BY_E1_E2_E3"
        route = "direct"
    elif extended_complete:
        status = "CERTIFIED_BY_E1_E3_E2CHI"
        route = "extended"
    else:
        status = "UNKNOWN_NOT_CERTIFIED_BY_THESE_CRITERIA"
        route = None

    summary = {
        "Num": int(input_record["Num"]),
        "input_matrix_roles": input_matrix_roles,
        "input_source": input_record.get("source"),
        "presentation_relation": input_record.get("presentation_relation"),
        "working_configuration_differs": (
            input_record.get("working_conf")
            != input_record.get("original_selected_conf")
        ),
        "status": status,
        "certificate_route": route,
        "E3_certified": bool(e3.get("E3_certified")),
        "direct_E1_E2_certified": bool(direct.get("E1_and_E2_certified")),
        "E1_certified": bool(
            direct.get("E1_and_E2_certified")
            or (
                e1 is not None
                and e1.get("integral_E1_status") == "pass_integral_E1"
            )
        ),
        "E2_certified": bool(direct.get("E1_and_E2_certified")),
        "integral_E1_certified": bool(
            direct.get("E1_and_E2_certified")
            or (
                e1 is not None
                and e1.get("integral_E1_status") == "pass_integral_E1"
            )
        ),
        "E2chi_frame_preservation_certified": bool(
            e2chi is not None and e2chi.get("E2chi_no_frame_changing_survivors")
        ),
        "configuration_only": True,
        "defining_polynomial_used": False,
        "output_directory": str(output),
    }
    _write_json(output / "verification_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if direct_complete or extended_complete else 2


if __name__ == "__main__":
    raise SystemExit(main())
