from __future__ import annotations

import json
from pathlib import Path


RELEASE_ROOT = Path(__file__).resolve().parents[3]
DATASET = RELEASE_ROOT / "data" / "cicy_criteria_input_7820.jsonl"
TARGETS = RELEASE_ROOT / "data" / "extended_2635_nums.json"


def test_distribution_dataset_contains_inputs_not_saved_criterion_answers() -> None:
    records = [json.loads(line) for line in DATASET.read_text().splitlines() if line]
    assert len(records) == 7820
    assert len({record["Num"] for record in records}) == 7820

    forbidden = {
        "E1_status",
        "E2_status",
        "E3_status",
        "E2chi_status",
        "E1_and_E2_certified",
        "E3_certified",
        "integral_E1_status",
        "E2chi_no_frame_changing_survivors",
        "certificate_route",
    }
    for record in records:
        assert forbidden.isdisjoint(record)
        assert set(record["official_fields"]) == {"H11"}
        assert record["hypotheses"] == {
            "rational_favourability_certified": True,
            "standard_nonproduct_cicy": True,
        }
        assert record["official_fields"]["H11"] == len(
            record["original_selected_conf"]
        )

    target_data = json.loads(TARGETS.read_text())
    targets = [int(value) for value in target_data["nums"]]
    assert target_data["count"] == len(targets) == 2635
    assert len(set(targets)) == len(targets)
    assert set(targets) <= {record["Num"] for record in records}
