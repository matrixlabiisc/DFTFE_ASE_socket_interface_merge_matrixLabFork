# ---------------------------------------------------------------------
# Part of the DFT-FE code, released under LGPL v2.1 or later.
# ---------------------------------------------------------------------
#
# @author Mehul Darak
#
"""``extra_prm``: the escape hatch, and the subsection trap inside it.

Any parameter without a typed kwarg reaches DFT-FE through ``extra_prm``. It
accepts three spellings of a key, and the risk lives in the bare one: a key is
just a string, subsections are not, so the resolution from key to subsection is
a lookup that can land in the wrong place. deal.II parses with
``skip_undefined=true``, so a wrong-subsection write is not an error -- the run
succeeds using a different value than the user asked for. That is the exact
failure this injection path was rewritten to eliminate, so it is worth pinning.
"""

from __future__ import annotations

from collections import defaultdict

import pytest

from dftfe_ase.calculator import _extra_prm_entries
from dftfe_ase.params import PARAM_TABLE


def only(entries):
    assert len(entries) == 1
    return entries[0]


# ── the three spellings ────────────────────────────────────────────────────


def test_tuple_form_is_taken_literally():
    e = only(_extra_prm_entries({("SCF parameters", "TOLERANCE"): 5e-5}))
    assert (e["section"], e["key"], e["value"]) == ("SCF parameters", "TOLERANCE", 5e-5)


def test_pipe_form_is_split():
    e = only(_extra_prm_entries({"Poisson problem parameters|||TOLERANCE": 1e-10}))
    assert e["section"] == "Poisson problem parameters"
    assert e["key"] == "TOLERANCE"


def test_bare_known_key_resolves_to_its_declared_subsection():
    e = only(_extra_prm_entries({"MIXING METHOD": "ANDERSON_WITH_RESTA"}))
    assert e["section"] == "SCF parameters"


def test_bare_top_level_key_stays_top_level():
    e = only(_extra_prm_entries({"MEM OPT MODE": True}))
    assert e["section"] == ""


def test_bare_unknown_key_falls_back_to_top_level():
    """Deliberate: an unknown key may be a build-specific entry we do not model.

    It is not silent -- applyPrm() in dftfeWrapper.cc reports any key it cannot
    find in the template rather than dropping it.
    """
    e = only(_extra_prm_entries({"SOME FUTURE KEY": 1}))
    assert e["section"] == ""


def test_none_section_is_normalised_to_empty_string():
    """The C++ side matches on "" for top level; None would never compare equal."""
    e = only(_extra_prm_entries({(None, "MEM OPT MODE"): True}))
    assert e["section"] == ""


# ── the trap ───────────────────────────────────────────────────────────────


AMBIGUOUS = sorted(
    k for k, sections in
    ((k, {p.section for p in PARAM_TABLE if p.prm_key == k})
     for k in {p.prm_key for p in PARAM_TABLE})
    if len(sections) > 1
)


def test_the_ambiguous_keys_are_the_ones_we_think():
    """If DFT-FE grows another duplicated key, this test says so."""
    assert AMBIGUOUS == ["MAXIMUM ITERATIONS", "TOLERANCE"]


@pytest.mark.parametrize("key", AMBIGUOUS)
def test_ambiguous_bare_key_raises_instead_of_guessing(key):
    """Both live in SCF parameters AND Poisson problem parameters.

    Resolving to whichever appears first in PARAM_TABLE meant asking for a
    Poisson tolerance and silently getting an SCF one -- a converged run with
    the wrong convergence criterion, and nothing in the output to show it.
    """
    with pytest.raises(ValueError, match="ambiguous"):
        _extra_prm_entries({key: 1e-12})


@pytest.mark.parametrize("key", AMBIGUOUS)
def test_ambiguous_key_is_fine_once_the_subsection_is_named(key):
    for section in ("SCF parameters", "Poisson problem parameters"):
        e = only(_extra_prm_entries({(section, key): 1e-12}))
        assert e["section"] == section


def test_error_message_names_both_candidate_subsections():
    with pytest.raises(ValueError) as exc:
        _extra_prm_entries({"TOLERANCE": 1e-12})
    text = str(exc.value)
    assert "SCF parameters" in text and "Poisson problem parameters" in text


# ── every bare key a user could reasonably pass ────────────────────────────


def test_every_unambiguous_table_key_resolves_to_its_own_section():
    """No bare key may land anywhere except where PARAM_TABLE declares it."""
    by_key = defaultdict(set)
    for p in PARAM_TABLE:
        by_key[p.prm_key].add(p.section)

    for key, sections in by_key.items():
        if len(sections) > 1:
            continue
        expected = (sections.pop() or "")
        got = only(_extra_prm_entries({key: 0}))["section"]
        assert got == expected, f"{key!r} resolved to {got!r}, declared in {expected!r}"
