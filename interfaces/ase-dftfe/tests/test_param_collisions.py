# ---------------------------------------------------------------------
# Part of the DFT-FE code, released under LGPL v2.1 or later.
# ---------------------------------------------------------------------
#
# @author Mehul Darak
#
"""What happens when the same DFT-FE parameter is set more than once.

Three routes reach one ``.prm`` key -- a typed kwarg, an ``extra_prm`` entry, and
a line in a ``prm_file`` deck -- and ``dftfeWrapper.cc:1017`` resolves a clash by
order: typed injection first, then the generic override list in sequence, so the
last writer wins. The precedence is deliberate; applying it silently was not.

These are constructor-only tests: no binary, no socket, no MPI. They build a
calculator, read back the override string it would send, and check that the value
DFT-FE ends up with is the one the caller asked for -- and that when two routes
disagree, somebody is told. Also covered here, because it is the same question
from the other side: an unset parameter must produce no entry at all, so DFT-FE
keeps its own default rather than being handed ours.
"""

import logging

import pytest

from dftfe_ase.calculator import DFTFE, _merge_entries
from dftfe_ase.params import BY_KWARG


# ── helpers ────────────────────────────────────────────────────────────────

def overrides(**kwargs):
    """The {(section, key): value} the calculator would send, from its wire form."""
    calc = DFTFE(command="/bin/true", **kwargs)
    out = {}
    if not calc._prm_overrides:
        return out
    for entry in calc._prm_overrides.split("@@@"):
        section, key, value = entry.split("|||", 2)
        out[(section, key)] = value
    return out


def deck(tmp_path, body, name="deck.prm"):
    path = tmp_path / name
    path.write_text(body)
    return str(path)


# ── plain parsing: the value asked for is the value sent ───────────────────

def test_typed_kwarg_travels_as_a_request_field_not_an_override():
    """`tolerance=` is "wired": it has its own C++ path, so it is not an entry."""
    calc = DFTFE(command="/bin/true", tolerance=1e-6, mesh_size=0.8)
    assert calc._params["tolerance"] == 1e-6
    assert calc._params["mesh_size"] == 0.8
    assert calc._prm_overrides == ""


def test_generic_kwarg_lands_in_its_declared_subsection():
    got = overrides(scalapack_procs=12)
    spec = BY_KWARG["scalapack_procs"]
    assert got == {(spec.section, spec.prm_key): "12"}


def test_bool_reaches_the_deck_as_a_prm_word_not_a_python_repr():
    """DFT-FE's parser wants true/false; 'True' is not a valid .prm bool."""
    got = overrides(adapt_anderson_mixing_parameter=True)
    assert set(got.values()) == {"true"}
    got = overrides(adapt_anderson_mixing_parameter=False)
    assert set(got.values()) == {"false"}


def test_nothing_asked_for_means_nothing_injected():
    """The defaults question: an unset parameter must leave no trace.

    An entry carrying our idea of a default would override DFT-FE's own, which is
    how a run ends up answering a different question than the deck describes.
    """
    assert overrides() == {}
    assert overrides(mesh_size=None, scalapack_procs=None) == {}


def test_unknown_parameter_is_rejected_at_construction():
    """A typo must fail here, not be silently dropped and never applied."""
    with pytest.raises(TypeError, match="unknown DFT-FE parameter"):
        DFTFE(command="/bin/true", tolarance=1e-6)


def test_a_deck_is_read_into_the_same_override_channel(tmp_path):
    got = overrides(prm_file=deck(tmp_path, """
        set VERBOSITY = 1
        subsection SCF parameters
          set TOLERANCE = 1e-8
        end
        """))
    assert got[("SCF parameters", "TOLERANCE")] == "1e-8"
    assert got[("", "VERBOSITY")] == "1"


# ── the failsafe: one parameter, two values ────────────────────────────────

def test_two_spellings_in_one_extra_prm_dict_raise(caplog):
    """No precedence exists inside one dict, so refuse rather than pick.

    "TOLERANCE" and ("SCF parameters", "TOLERANCE") are two spellings of one
    target. Whichever won would be decided by dict insertion order -- that is a
    coin toss deciding the SCF convergence criterion.
    """
    with pytest.raises(ValueError, match="twice"):
        DFTFE(command="/bin/true", extra_prm={
            ("SCF parameters", "TOLERANCE"): 1e-6,
            "SCF parameters|||TOLERANCE": 1e-9,
        })


def test_the_raise_names_both_values_so_it_can_be_acted_on():
    with pytest.raises(ValueError) as exc:
        DFTFE(command="/bin/true", extra_prm={
            ("SCF parameters", "TOLERANCE"): 1e-6,
            "SCF parameters|||TOLERANCE": 1e-9,
        })
    msg = str(exc.value)
    assert "1e-06" in msg and "1e-09" in msg
    assert "TOLERANCE" in msg and "SCF parameters" in msg


def test_the_same_value_twice_is_not_an_error():
    """Redundant is not contradictory: there is nothing to choose between."""
    got = overrides(extra_prm={
        ("SCF parameters", "TOLERANCE"): 1e-6,
        "SCF parameters|||TOLERANCE": 1e-6,
    })
    assert got[("SCF parameters", "TOLERANCE")] == "1e-06"


def test_deck_beats_extra_prm_and_says_so(tmp_path, caplog):
    """Documented precedence (kwarg < extra_prm < prm_file), applied out loud."""
    with caplog.at_level(logging.WARNING, logger="dftfe_ase"):
        got = overrides(
            extra_prm={("SCF parameters", "TOLERANCE"): 1e-6},
            prm_file=deck(tmp_path, """
                subsection SCF parameters
                  set TOLERANCE = 1e-9
                end
                """),
        )
    assert got[("SCF parameters", "TOLERANCE")] == "1e-9"
    warning = "\n".join(r.getMessage() for r in caplog.records)
    assert "set more than once" in warning
    assert "1e-06" in warning and "1e-9" in warning      # both values named
    assert "prm_file" in warning and "extra_prm" in warning  # both routes named


def test_a_typed_kwarg_overridden_by_a_deck_is_reported(tmp_path, caplog):
    """The easiest clash to create, and the one with no trace in the override list.

    `tolerance=` goes out as a typed request field, so it never appears among the
    entries; only DFT-FE ever sees both values. Before this, asking for 1e-6 in
    Python and loading a deck that sets 1e-8 produced a run at 1e-8 in silence.
    """
    with caplog.at_level(logging.WARNING, logger="dftfe_ase"):
        calc = DFTFE(command="/bin/true", tolerance=1e-6,
                     prm_file=deck(tmp_path, """
                         subsection SCF parameters
                           set TOLERANCE = 1e-8
                         end
                         """))
    warning = "\n".join(r.getMessage() for r in caplog.records)
    assert "TOLERANCE" in warning
    assert "1e-06" in warning and "1e-8" in warning
    assert "tolerance" in warning          # the kwarg that lost, by name
    # The typed field is still sent; the C++ side applies it first and the
    # override after, which is exactly what the warning says happens.
    assert calc._params["tolerance"] == 1e-6


def test_a_typed_kwarg_agreeing_with_a_deck_is_silent(tmp_path, caplog):
    """No warning when both routes ask for the same thing, or nobody reads them."""
    with caplog.at_level(logging.WARNING, logger="dftfe_ase"):
        DFTFE(command="/bin/true", tolerance=1e-8,
              prm_file=deck(tmp_path, """
                  subsection SCF parameters
                    set TOLERANCE = 1e-08
                  end
                  """))
    assert not [r for r in caplog.records if "set" in r.message]


def test_a_bool_kwarg_matching_a_deck_word_is_silent(caplog):
    """`True` vs `true` is the same request; comparison must be on the wire form."""
    with caplog.at_level(logging.WARNING, logger="dftfe_ase"):
        DFTFE(command="/bin/true", use_single_prec_cheby=True,
              extra_prm={("Eigen-solver parameters", "USE SINGLE PREC CHEBY"): "true"})
    assert not caplog.records


def test_distinct_parameters_sharing_a_key_do_not_collide():
    """TOLERANCE exists in two subsections; setting both is legitimate."""
    got = overrides(extra_prm={
        ("SCF parameters", "TOLERANCE"): 1e-6,
        ("Poisson problem parameters", "TOLERANCE"): 1e-10,
    })
    assert got[("SCF parameters", "TOLERANCE")] == "1e-06"
    assert got[("Poisson problem parameters", "TOLERANCE")] == "1e-10"


def test_only_one_value_per_parameter_reaches_the_wire(tmp_path):
    """The loser is dropped, not left in the string for the C++ side to re-resolve."""
    calc = DFTFE(command="/bin/true",
                 extra_prm={("SCF parameters", "TOLERANCE"): 1e-6},
                 prm_file=deck(tmp_path, """
                     subsection SCF parameters
                       set TOLERANCE = 1e-9
                     end
                     """))
    assert calc._prm_overrides.count("TOLERANCE") == 1


def test_merge_keeps_untouched_entries_in_order():
    """Deduplication must not reshuffle the rest of the deck."""
    entries = [
        {"section": "", "key": "A", "value": 1, "_src": "prm_file"},
        {"section": "S", "key": "B", "value": 2, "_src": "prm_file"},
        {"section": "", "key": "C", "value": 3, "_src": "prm_file"},
    ]
    assert [e["key"] for e in _merge_entries(list(entries))] == ["A", "B", "C"]


def test_a_deck_that_sets_one_key_twice_is_reported_not_refused(tmp_path, caplog):
    """A real deck must still load: warn, take the last, do not raise.

    deal.II's own parser takes the last occurrence, so refusing here would reject
    decks that already run.
    """
    with caplog.at_level(logging.WARNING, logger="dftfe_ase"):
        got = overrides(prm_file=deck(tmp_path, """
            subsection SCF parameters
              set TOLERANCE = 1e-6
              set TOLERANCE = 1e-9
            end
            """))
    assert got[("SCF parameters", "TOLERANCE")] == "1e-9"
    assert any("set more than once" in (r.getMessage()) for r in caplog.records)
