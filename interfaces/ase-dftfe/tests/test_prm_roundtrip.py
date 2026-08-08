# ---------------------------------------------------------------------
# Copyright (c) 2017-2025 The Regents of the University of Michigan and DFT-FE
# authors. Part of the DFT-FE code, released under LGPL v2.1 or later.
# ---------------------------------------------------------------------
#
# @author Mehul Darak
#
"""The two entry points must produce the same run.

A DFT-FE calculation can be set up either by handing the calculator a native
``.prm`` deck (``read_dftfe``) or by writing the same settings as Python kwargs.
If those drift apart, a socket-vs-native benchmark stops measuring the interface
and starts measuring the drift. These tests pin the behaviour that keeps them
equal, using the published Summit BCC-Mo 6x6x6 vacancy deck as the fixture --
the deck the scaling study runs.
"""

from __future__ import annotations

import os
import textwrap

import numpy as np
import pytest
from ase.units import Bohr

from dftfe_ase import read_dftfe_atoms, read_prm
from dftfe_ase.calculator import _serialize_overrides
from dftfe_ase.params import GEOMETRY_KEYS, LEGACY_KEYS

pytest.importorskip("ase")


# ── fixtures ────────────────────────────────────────────────────────────────

DECK = textwrap.dedent("""\
    set VERBOSITY = 4

    subsection GPU
      set USE GPU=true
      set USE ELPA GPU KERNEL=true
    end

    subsection Boundary conditions
      set PERIODIC1                       = true
      set PERIODIC2                       = true
      set PERIODIC3                       = true
      set SELF POTENTIAL RADIUS=4.0
    end

    subsection DFT functional parameters
      set EXCHANGE CORRELATION TYPE = GGA-PBE
      set PSEUDOPOTENTIAL FILE NAMES LIST = pseudo.inp
    end

    subsection Geometry
      set NATOMS=2
      set NATOM TYPES=1
      set ATOMIC COORDINATES FILE      = coordinates.inp
      set DOMAIN VECTORS FILE = domainVectors.inp
      subsection Optimization
        set ION FORCE = true
      end
    end

    subsection Helmholtz problem parameters
      set MAXIMUM ITERATIONS HELMHOLTZ = 10000
    end

    subsection Poisson problem parameters
      set MAXIMUM ITERATIONS = 10000
      set TOLERANCE          = 1e-8
    end

    subsection SCF parameters
      set MAXIMUM ITERATIONS               = 100
      set TOLERANCE                        = 1e-4
      subsection Eigen-solver parameters
          set USE MIXED PREC CHEBY=true
          set SPECTRUM SPLIT CORE EIGENSTATES = 2400
          set SCALAPACKPROCS=50
      end
    end
    """)


@pytest.fixture
def deck(tmp_path):
    """A minimal periodic deck: 2 atoms, cubic 10 Bohr cell, fractional coords."""
    (tmp_path / "parameterFile.prm").write_text(DECK)
    (tmp_path / "domainVectors.inp").write_text(
        "10.0 0.0 0.0\n0.0 10.0 0.0\n0.0 0.0 10.0\n")
    (tmp_path / "coordinates.inp").write_text(
        "42 14 0.00 0.00 0.00\n42 14 0.50 0.25 0.75\n")
    (tmp_path / "pseudo.inp").write_text("42 Mo_ONCV_PBE-1.0.upf\n")
    (tmp_path / "Mo_ONCV_PBE-1.0.upf").write_text("dummy\n")
    return str(tmp_path / "parameterFile.prm")


def entry(entries, section, key):
    hits = [e for e in entries if e["section"] == section and e["key"] == key]
    return hits[0]["value"] if hits else None


# ── geometry ────────────────────────────────────────────────────────────────

def test_atoms_come_from_the_coordinates_file(deck):
    atoms = read_dftfe_atoms(deck)
    assert atoms.get_chemical_symbols() == ["Mo", "Mo"]
    assert atoms.get_pbc().tolist() == [True, True, True]
    np.testing.assert_allclose(np.diag(atoms.get_cell()), 10.0 * Bohr, rtol=1e-12)


def test_fractional_coordinates_are_converted(deck):
    atoms = read_dftfe_atoms(deck)
    np.testing.assert_allclose(atoms.get_scaled_positions(),
                               [[0.0, 0.0, 0.0], [0.5, 0.25, 0.75]], atol=1e-12)


def test_atom_order_is_preserved(tmp_path):
    """Force parity is per-atom, so file order must survive the round trip."""
    (tmp_path / "parameterFile.prm").write_text(DECK)
    (tmp_path / "domainVectors.inp").write_text(
        "10.0 0.0 0.0\n0.0 10.0 0.0\n0.0 0.0 10.0\n")
    (tmp_path / "coordinates.inp").write_text(
        "8 6 0.10 0.00 0.00\n42 14 0.20 0.00 0.00\n8 6 0.30 0.00 0.00\n")
    (tmp_path / "pseudo.inp").write_text("8 O.upf\n42 Mo.upf\n")
    atoms = read_dftfe_atoms(str(tmp_path / "parameterFile.prm"))
    assert atoms.get_chemical_symbols() == ["O", "Mo", "O"]
    np.testing.assert_allclose(atoms.get_scaled_positions()[:, 0],
                               [0.1, 0.2, 0.3], atol=1e-12)


# ── what the deck is allowed to inject ──────────────────────────────────────

def test_geometry_keys_are_never_injected(deck):
    """A deck must not point DFT-FE back at its own coordinate files.

    The driver writes coordinates.inp / domainVectors.inp / pseudo.inp into its
    scratch folder from the ASE Atoms. If the deck's copies of these keys were
    injected they would overwrite those paths and the run would silently ignore
    everything ASE sent -- producing a parity result that proves nothing.
    """
    keys = {e["key"] for e in read_prm(deck)}
    assert not (keys & GEOMETRY_KEYS)


def test_both_poisson_and_scf_entries_survive(deck):
    """TOLERANCE and MAXIMUM ITERATIONS exist in two subsections each."""
    entries = read_prm(deck)
    assert entry(entries, "SCF parameters", "TOLERANCE") == "1e-4"
    assert entry(entries, "Poisson problem parameters", "TOLERANCE") == "1e-8"
    assert entry(entries, "SCF parameters", "MAXIMUM ITERATIONS") == "100"
    assert entry(entries, "Poisson problem parameters", "MAXIMUM ITERATIONS") == "10000"
    assert entry(entries, "Helmholtz problem parameters",
                 "MAXIMUM ITERATIONS HELMHOLTZ") == "10000"


def test_legacy_keys_are_renamed_or_dropped(deck):
    entries = read_prm(deck)
    keys = {e["key"] for e in entries}
    # renamed
    assert "USE MIXED PREC CHEBY" not in keys
    assert entry(entries, "Eigen-solver parameters", "USE SINGLE PREC CHEBY") == "true"
    # removed outright
    assert "SPECTRUM SPLIT CORE EIGENSTATES" not in keys
    # untouched
    assert entry(entries, "Eigen-solver parameters", "SCALAPACKPROCS") == "50"


@pytest.mark.parametrize("section", ["", "GPU"])
def test_use_gpu_is_driver_owned_in_any_subsection(tmp_path, section):
    """v1.0 decks nest USE GPU under `subsection GPU`, current ones put it at top
    level. Either way the calculator picks the device via use_device=."""
    body = "set USE GPU = true\n"
    if section:
        body = f"subsection {section}\n  {body}end\n"
    prm = tmp_path / "p.prm"
    prm.write_text(body)
    assert not [e for e in read_prm(str(prm)) if e["key"] == "USE GPU"]


def test_legacy_removal_is_warned_not_silent(deck, caplog):
    """deal.II parses with skip_undefined=true, so a stale key is free at parse
    time and wrong at run time. It has to be reported."""
    with caplog.at_level("WARNING", logger="dftfe_ase"):
        read_prm(deck)
    assert "SPECTRUM SPLIT CORE EIGENSTATES" in caplog.text


def test_every_legacy_target_is_a_real_current_key():
    """Guard against a migration entry that renames onto a key nothing declares."""
    template = os.path.join(os.path.dirname(__file__),
                            "..", "..", "..", "helpers", "parameterFile.prm")
    if not os.path.exists(template):
        pytest.skip("helpers/parameterFile.prm not available")
    declared, stack = set(), []
    for line in open(template):
        s = line.split("#", 1)[0].strip()
        if s.startswith("subsection"):
            stack.append(s[len("subsection"):].strip())
        elif s == "end":
            if stack:
                stack.pop()
        elif s.startswith("set "):
            key = s[4:].split("=", 1)[0].strip()
            declared.add((stack[-1] if stack else "", key))
    for source, target in LEGACY_KEYS.items():
        if target is not None:
            assert target in declared, f"{source} migrates to unknown key {target}"


# ── the two doors agree ─────────────────────────────────────────────────────

def test_deck_and_kwargs_produce_identical_overrides(deck):
    """Reading the deck and typing the same settings as kwargs must serialise
    to the same injection payload -- that equality is what makes an ASE-vs-native
    comparison a measurement of the interface rather than of a settings drift."""
    from dftfe_ase.params import to_prm_entries

    from_deck = read_prm(deck)

    from_python = to_prm_entries({
        "verbosity": 4,
        "use_elpa_gpu_kernel": True,
        "self_potential_radius": 4.0,
        "xc": "GGA-PBE",
        "poisson_max_iterations": 10000,
        "poisson_tolerance": "1e-8",
        "helmholtz_max_iterations": 10000,
        "max_scf_iterations": 100,
        "tolerance": "1e-4",
        "use_single_prec_cheby": True,
        "scalapack_procs": 50,
    })

    def norm(entries):
        return sorted((e["section"], e["key"], str(e["value"]).lower())
                      for e in entries
                      if e["key"] not in ("PERIODIC1", "PERIODIC2", "PERIODIC3",
                                          "ION FORCE"))

    assert norm(from_deck) == norm(from_python)


def test_serialized_payload_round_trips(deck):
    """The wire format must survive keys and values containing spaces."""
    entries = read_prm(deck)
    blob = _serialize_overrides(entries)
    decoded = [chunk.split("|||", 2) for chunk in blob.split("@@@")]
    assert decoded == [[e["section"], e["key"], str(e["value"])] for e in entries]
