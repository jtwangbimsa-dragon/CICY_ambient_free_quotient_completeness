from __future__ import annotations

import sys
from pathlib import Path

import pytest


RELEASE_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(RELEASE_ROOT / "code"))

from verify_configuration import _lookup_catalogue_record, _original_matrix_index


def test_both_original_matrix_tables_cover_all_7820_cases() -> None:
    originals = _original_matrix_index()
    assert len(originals) == 7820
    assert len({number for number in originals}) == 7820


def test_original_matrix_of_favourable_graph_case_is_found() -> None:
    original_414 = [
        [1, 1, 0, 0, 0, 0, 0],
        [0, 0, 1, 1, 0, 0, 0],
        [1, 0, 0, 0, 1, 0, 0],
        [1, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 2],
        [0, 1, 2, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 1, 1],
    ]
    record, roles = _lookup_catalogue_record(original_414, 414)
    assert record["Num"] == 414
    assert record["source"] == "FavourConf"
    assert record["original_selected_conf"] != original_414
    assert record["obstruction_mappings"]
    assert roles == ["standard_original"]


def test_original_matrix_in_open_311_is_found() -> None:
    original_1, _ = _original_matrix_index()[1]
    record, roles = _lookup_catalogue_record(original_1, 1)
    assert record["Num"] == 1
    assert "standard_original" in roles


def test_incorrect_cicy_number_does_not_select_another_matrix() -> None:
    original_414, _ = _original_matrix_index()[414]
    with pytest.raises(ValueError, match="not found"):
        _lookup_catalogue_record(original_414, 7884)
