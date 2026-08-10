# ---------------------------------------------------------------------
# Copyright (c) 2017-2025 The Regents of the University of Michigan and DFT-FE
# authors. Part of the DFT-FE code, released under LGPL v2.1 or later.
# ---------------------------------------------------------------------
#
# @author Mehul Darak
#
"""Read a native DFT-FE input deck into ``(Atoms, DFTFE)``.

The calculator has always supported the *Python* entry point -- describe the
system with ASE and pass DFT parameters as kwargs. This module adds the other
one: point it at an existing ``.prm`` and get back the same run.

    from dftfe_ase import read_dftfe

    atoms, calc = read_dftfe("6x6x6Vac/parameterFileGPU32NodesMPS.prm")
    atoms.calc = calc
    atoms.get_potential_energy()

Both doors have to arrive at the same calculation, so this reader deliberately
does *not* hand the deck to DFT-FE wholesale. It resolves the geometry into an
``Atoms`` object -- preserving file order, because per-atom force comparisons
against a native run are meaningless otherwise -- and routes every remaining
entry through the same validated injection path the kwargs use.

Deck conventions (see ``dftfeWrapper::reinit``):

* ``coordinates.inp`` columns are ``Z  valence  x  y  z``; for a periodic cell
  the coordinates are **fractional**, otherwise Cartesian Bohr measured from the
  cell centre.
* ``domainVectors.inp`` holds the three cell vectors, one per row, in Bohr.
* ``pseudo.inp`` maps atomic number to pseudopotential filename.

All paths are resolved relative to the ``.prm``.
"""

from __future__ import annotations

import logging
import os

import numpy as np
from ase import Atoms
from ase.data import chemical_symbols
from ase.units import Bohr

from .calculator import DFTFE, read_prm

log = logging.getLogger("dftfe_ase.io")

__all__ = ["read_dftfe", "read_dftfe_atoms"]


def _raw_entries(prm_path):
    """Every ``set`` line, including the ones :func:`read_prm` filters out."""
    import re

    entries, stack = [], []
    for line in open(prm_path):
        s = line.split("#", 1)[0].strip()
        m = re.match(r"subsection\s+(.+)", s)
        if m:
            stack.append(m.group(1).strip())
            continue
        if s == "end":
            if stack:
                stack.pop()
            continue
        m = re.match(r"set\s+([^=]+?)\s*=\s*(.*)", s)
        if m:
            entries.append((stack[-1] if stack else "", m.group(1).strip(),
                            m.group(2).strip()))
    return entries


def _lookup(entries, key, default=None):
    """Last value for ``key`` in any subsection (deck keys are unambiguous here)."""
    hit = default
    for _section, k, v in entries:
        if k == key:
            hit = v
    return hit


def _as_bool(value, default=False):
    if value is None:
        return default
    return value.strip().lower() in ("true", "1", "yes")


def _read_cell(path):
    cell = np.loadtxt(path, dtype=float)
    if cell.shape != (3, 3):
        raise ValueError(f"{path}: expected 3x3 domain vectors, got {cell.shape}")
    return cell


def _read_pseudo(path):
    """``pseudo.inp`` -> ``{chemical symbol: absolute path}``."""
    base = os.path.dirname(os.path.abspath(path))
    table = {}
    for line in open(path):
        parts = line.split()
        if len(parts) >= 2:
            table[chemical_symbols[int(parts[0])]] = os.path.join(base, parts[1])
    return table


def read_dftfe_atoms(prm_path):
    """Build the :class:`ase.Atoms` a DFT-FE deck describes, in file order."""
    prm_path = os.path.abspath(prm_path)
    base = os.path.dirname(prm_path)
    entries = _raw_entries(prm_path)

    pbc = [_as_bool(_lookup(entries, f"PERIODIC{i}")) for i in (1, 2, 3)]

    coords_file = _lookup(entries, "ATOMIC COORDINATES FILE", "coordinates.inp")
    cell_key = ("DOMAIN VECTORS FILE" if any(pbc)
                else "DOMAIN BOUNDING VECTORS FILE")
    cell_file = _lookup(entries, cell_key, "domainVectors.inp")

    raw = np.atleast_2d(np.loadtxt(os.path.join(base, coords_file), dtype=float))
    if raw.shape[1] < 5:
        raise ValueError(
            f"{coords_file}: expected 5 columns (Z valence x y z), got {raw.shape[1]}")
    numbers = raw[:, 0].astype(int)
    cell_bohr = _read_cell(os.path.join(base, cell_file))

    if any(pbc):
        positions_bohr = raw[:, 2:5] @ cell_bohr
    else:
        # Cartesian, measured from the cell centre (dftfeWrapper::reinit shifts
        # by -sum(cell[j][i])/2 when writing); undo that shift.
        positions_bohr = raw[:, 2:5] + cell_bohr.sum(axis=0) / 2.0

    natoms = _lookup(entries, "NATOMS")
    if natoms is not None and int(natoms) != len(numbers):
        log.warning("%s declares NATOMS=%s but %s holds %d atoms; using the file.",
                    os.path.basename(prm_path), natoms, coords_file, len(numbers))

    return Atoms(numbers=numbers,
                 positions=positions_bohr * Bohr,
                 cell=cell_bohr * Bohr,
                 pbc=pbc)


def read_dftfe(prm_path, **overrides):
    """Read a DFT-FE deck into ``(atoms, calculator)``.

    Every ``.prm`` entry reaches DFT-FE through the generic injector, except the
    handful that must be typed kwargs because the calculator acts on them
    *before* the run starts:

    ``mp_grid`` / ``mp_grid_shift``
        decide real vs complex binary (:func:`~dftfe_ase.binary.select_binary_kind`);
        left in the deck alone, a k-point calculation would launch the real binary.
    ``compute_forces`` / ``compute_stress``
        decide what the socket asks for each step.
    ``use_device``
        selects the GPU path; v1.0 decks spell this ``USE GPU`` inside
        ``subsection GPU``, which current DFT-FE no longer reads.

    ``overrides`` are passed straight to :class:`~dftfe_ase.calculator.DFTFE`
    (``nproc=``, ``launcher=``, ``cluster=``, ``debug_timing=`` …) and win over
    anything inferred from the deck.
    """
    prm_path = os.path.abspath(prm_path)
    base = os.path.dirname(prm_path)
    entries = _raw_entries(prm_path)

    atoms = read_dftfe_atoms(prm_path)

    kwargs = {
        "prm_file": prm_path,
        "compute_forces": _as_bool(_lookup(entries, "ION FORCE")),
        "compute_stress": _as_bool(_lookup(entries, "CELL STRESS")),
        "use_device": _as_bool(_lookup(entries, "USE GPU")),
    }

    # VERBOSITY is top level (no subsection). The generic override path emits it
    # with section="", and that entry does not reach dftParameters -- so a deck's
    # "set VERBOSITY = 4" was silently dropped and the run came out at DFT-FE's
    # socket default of -1 (completely silent). That made the socket arm write no
    # log at all while the file-driven arm wrote every diagnostic, which is both a
    # parameter-loss bug and a confound for any native-vs-socket timing
    # comparison. Lift it into a typed kwarg like the others above.
    verbosity = _lookup(entries, "VERBOSITY")
    if verbosity is not None:
        kwargs["verbosity"] = int(verbosity)

    grid = [_lookup(entries, f"SAMPLING POINTS {i}") for i in (1, 2, 3)]
    if all(g is not None for g in grid):
        kwargs["mp_grid"] = [int(g) for g in grid]
    shift = [_lookup(entries, f"SAMPLING SHIFT {i}") for i in (1, 2, 3)]
    if all(s is not None for s in shift):
        kwargs["mp_grid_shift"] = [int(s) for s in shift]

    psp_file = _lookup(entries, "PSEUDOPOTENTIAL FILE NAMES LIST")
    if psp_file:
        psp_path = os.path.join(base, psp_file)
        if os.path.exists(psp_path):
            kwargs["psp_path"] = _read_pseudo(psp_path)
        else:
            log.warning("%s: pseudopotential list %r not found; "
                        "pass psp_path= explicitly.", prm_path, psp_file)

    # read_prm() reports renamed/removed keys as it goes; run it once here so the
    # warnings surface at read time rather than on the first calculate().
    read_prm(prm_path)

    kwargs.update(overrides)
    return atoms, DFTFE(**kwargs)
