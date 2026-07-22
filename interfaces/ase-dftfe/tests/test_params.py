# ---------------------------------------------------------------------
# Part of the DFT-FE code, released under LGPL v2.1 or later.
# ---------------------------------------------------------------------
#
# @author Mehul Darak
#
"""Unit tests for the single-source-of-truth parameter table."""

from dftfe_ase.params import (
    BY_KWARG,
    PARAM_TABLE,
    generate_mapping_markdown,
    to_prm_entries,
)


def test_int3_expands_to_indexed_keys():
    entries = to_prm_entries({"mp_grid": (2, 3, 4)})
    assert [(e["key"], e["value"]) for e in entries] == [
        ("SAMPLING POINTS 1", 2), ("SAMPLING POINTS 2", 3), ("SAMPLING POINTS 3", 4)
    ]


def test_bool_formats_lowercase_word():
    assert to_prm_entries({"use_single_prec_cheby": True})[0]["value"] == "true"
    assert to_prm_entries({"use_single_prec_cheby": False})[0]["value"] == "false"


def test_none_and_unknown_are_skipped():
    assert to_prm_entries({"xc": None, "not_a_real_param": 5}) == []


def test_entry_carries_section_and_key():
    (e,) = to_prm_entries({"mixing_scheme": "ANDERSON"})
    assert e["section"] == "SCF parameters" and e["key"] == "MIXING METHOD" and e["value"] == "ANDERSON"


def test_table_has_no_duplicate_kwargs_or_keys():
    kwargs = [p.kwarg for p in PARAM_TABLE]
    assert len(kwargs) == len(set(kwargs))
    assert len(BY_KWARG) == len(PARAM_TABLE)


def test_mapping_markdown_generates():
    md = generate_mapping_markdown()
    assert "MIXING METHOD" in md and "| `xc` |" in md and "auto-generated" in md
