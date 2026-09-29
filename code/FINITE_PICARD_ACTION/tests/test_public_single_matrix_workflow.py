from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


RELEASE_ROOT = Path(__file__).resolve().parents[3]


def run(*arguments: str) -> None:
    subprocess.run(
        [sys.executable, *arguments],
        cwd=RELEASE_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )


def read_one(path: Path) -> dict:
    lines = [line for line in path.read_text().splitlines() if line]
    assert len(lines) == 1
    return json.loads(lines[0])


def test_bicubic_runs_through_every_public_entry_point(tmp_path: Path) -> None:
    case = tmp_path / "bicubic.jsonl"
    e3 = tmp_path / "e3.jsonl"
    direct = tmp_path / "direct.jsonl"
    e1 = tmp_path / "e1.jsonl"
    e2chi = tmp_path / "e2chi.jsonl"

    run(
        "code/prepare_case.py",
        "--conf",
        "[[3],[3]]",
        "--h11",
        "2",
        "--favourable-certified",
        "--standard-nonproduct-cicy",
        "--num",
        "900001",
        "--output",
        str(case),
    )
    run(
        "code/check_E3.py",
        "--input",
        str(case),
        "--expected-count",
        "1",
        "--certificates",
        str(e3),
        "--summary",
        str(tmp_path / "e3_summary.json"),
    )
    run(
        "code/check_E1_E2.py",
        "--input",
        str(case),
        "--expected-count",
        "1",
        "--certificates",
        str(direct),
        "--summary",
        str(tmp_path / "direct_summary.json"),
    )
    run(
        "code/FINITE_PICARD_ACTION/check_E1.py",
        "--dataset",
        str(case),
        "--all-input",
        "--output",
        str(e1),
        "--summary",
        str(tmp_path / "e1_summary.json"),
    )
    run(
        "code/FINITE_PICARD_ACTION/check_E2chi.py",
        "--dataset",
        str(case),
        "--e1-records",
        str(e1),
        "--records-output",
        str(e2chi),
        "--summary-output",
        str(tmp_path / "e2chi_summary.json"),
    )

    assert read_one(e3)["E3_certified"] is True
    assert read_one(direct)["E1_and_E2_certified"] is True
    assert read_one(e1)["integral_E1_status"] == "pass_integral_E1"
    assert read_one(e2chi)["E2chi_no_frame_changing_survivors"] is True


def test_missing_geometric_hypotheses_fail_closed(tmp_path: Path) -> None:
    case = tmp_path / "bicubic_without_hypotheses.jsonl"
    direct = tmp_path / "direct.jsonl"
    e1 = tmp_path / "e1.jsonl"
    run(
        "code/prepare_case.py",
        "--conf",
        "[[3],[3]]",
        "--h11",
        "2",
        "--num",
        "900002",
        "--output",
        str(case),
    )
    run(
        "code/check_E1_E2.py",
        "--input",
        str(case),
        "--expected-count",
        "1",
        "--certificates",
        str(direct),
        "--summary",
        str(tmp_path / "direct_summary.json"),
    )
    result = subprocess.run(
        [
            sys.executable,
            "code/FINITE_PICARD_ACTION/check_E1.py",
            "--dataset",
            str(case),
            "--all-input",
            "--output",
            str(e1),
            "--summary",
            str(tmp_path / "e1_summary.json"),
        ],
        cwd=RELEASE_ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert read_one(direct)["E1_and_E2_certified"] is False
    assert read_one(e1)["integral_E1_status"] == "UNKNOWN_integral_E1"


def test_one_command_wrapper_recomputes_direct_certificate(tmp_path: Path) -> None:
    output = tmp_path / "one_command"
    run(
        "code/verify_configuration.py",
        "--conf",
        "[[3],[3]]",
        "--h11",
        "2",
        "--favourable-certified",
        "--standard-nonproduct-cicy",
        "--output-dir",
        str(output),
    )
    summary = json.loads((output / "verification_summary.json").read_text())
    assert summary["status"] == "CERTIFIED_BY_E1_E2_E3"
    assert summary["certificate_route"] == "direct"
    assert summary["E1_certified"] is True
    assert summary["E2_certified"] is True
    assert summary["defining_polynomial_used"] is False


def test_one_command_wrapper_accepts_catalogue_matrix_alone(tmp_path: Path) -> None:
    output = tmp_path / "matrix_only"
    run(
        "code/verify_configuration.py",
        "--conf",
        "[[3],[3]]",
        "--output-dir",
        str(output),
    )
    summary = json.loads((output / "verification_summary.json").read_text())
    assert summary["status"] == "CERTIFIED_BY_E1_E2_E3"
    selected = read_one(output / "input_record.jsonl")
    assert selected["Num"] == 7884


def test_one_command_wrapper_accepts_matrix_json_file(tmp_path: Path) -> None:
    matrix = tmp_path / "matrix.json"
    matrix.write_text("[[3],[3]]\n")
    output = tmp_path / "matrix_file"
    run(
        "code/verify_configuration.py",
        "--conf-file",
        str(matrix),
        "--output-dir",
        str(output),
    )
    summary = json.loads((output / "verification_summary.json").read_text())
    assert summary["status"] == "CERTIFIED_BY_E1_E2_E3"


def test_matrix_lookup_carries_graph_reembedding_data(tmp_path: Path) -> None:
    output = tmp_path / "graph_case"
    run(
        "code/verify_configuration.py",
        "--conf",
        "[[0,2],[4,1]]",
        "--output-dir",
        str(output),
    )
    selected = read_one(output / "input_record.jsonl")
    summary = json.loads((output / "verification_summary.json").read_text())
    assert selected["Num"] == 7888
    assert selected["obstruction_mappings"]
    assert summary["status"] == "CERTIFIED_BY_E1_E2_E3"


def test_one_command_wrapper_selects_supplied_dataset_record(tmp_path: Path) -> None:
    output = tmp_path / "selected_record"
    run(
        "code/verify_configuration.py",
        "--record",
        "data/cicy_criteria_input_7820.jsonl",
        "--num",
        "232",
        "--output-dir",
        str(output),
    )
    summary = json.loads((output / "verification_summary.json").read_text())
    assert summary["status"] == "CERTIFIED_BY_E1_E2_E3"
    assert summary["certificate_route"] == "direct"


def test_one_command_wrapper_recomputes_extended_route(tmp_path: Path) -> None:
    output = tmp_path / "extended_record"
    run(
        "code/verify_configuration.py",
        "--conf",
        "[[1,1,1,0,0,0],[0,0,1,1,0,0],[0,1,0,0,1,0],"
        "[1,0,0,0,1,0],[0,0,0,2,1,2]]",
        "--workers",
        "2",
        "--output-dir",
        str(output),
    )
    summary = json.loads((output / "verification_summary.json").read_text())
    assert summary["status"] == "CERTIFIED_BY_E1_E3_E2CHI"
    assert summary["certificate_route"] == "extended"
    assert summary["direct_E1_E2_certified"] is False
    assert summary["E1_certified"] is True
    assert summary["E2_certified"] is False
    assert summary["integral_E1_certified"] is True
    assert summary["E2chi_frame_preservation_certified"] is True
