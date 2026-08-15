# ---------------------------------------------------------------------
#
# Copyright (c) 2017-2025 The Regents of the University of Michigan and DFT-FE
# authors.
#
# This file is part of the DFT-FE code.
#
# The DFT-FE code is free software; you can use it, redistribute
# it, and/or modify it under the terms of the GNU Lesser General
# Public License as published by the Free Software Foundation; either
# version 2.1 of the License, or (at your option) any later version.
# The full text of the license can be found in the file LICENSE at
# the top level of the DFT-FE distribution.
#
# ---------------------------------------------------------------------
#
# @author Mehul Darak
#

"""
CIF <-> DFT-FE input file converter.

Provides bidirectional conversion between CIF crystallographic files and
DFT-FE native input files (coordinates.inp, domainVectors.inp).

Unit conventions:
  - CIF / ASE: Angstrom
  - DFT-FE:    atomic units (Bohr for lengths)

Coordinate conventions (from DFT-FE documentation):
  - coordinates.inp (periodic / semi-periodic): fractional coordinates
  - coordinates.inp (fully non-periodic):       Cartesian in Bohr (origin at cell center)
  - domainVectors.inp:                          always in Bohr
"""

import os
import numpy as np
from ase import Atoms
from ase.io import read, write
from ase.units import Bohr
from ase.data import chemical_symbols

DEFAULT_PRM_TEMPLATE = """
set SOLVER MODE = GS
set VERBOSITY = 2
set USE GPU = {use_gpu}

subsection Geometry
  set NATOMS = {natoms}
  set NATOM TYPES = {ntypes}
  set ATOMIC COORDINATES FILE = coordinates.inp
  set DOMAIN VECTORS FILE = domainVectors.inp
end

subsection Boundary conditions
  set PERIODIC1 = {p1}
  set PERIODIC2 = {p2}
  set PERIODIC3 = {p3}
end

subsection DFT functional parameters
  set EXCHANGE CORRELATION TYPE = GGA-PBE
  set PSEUDOPOTENTIAL CALCULATION = true
  set PSEUDOPOTENTIAL FILE NAMES LIST = pseudo.inp
end

subsection SCF parameters
  set MIXING PARAMETER = 0.2
  set MAXIMUM ITERATIONS = 120
  set TOLERANCE = 1e-5
  set TEMPERATURE = 500
end

subsection Finite element mesh parameters
  set POLYNOMIAL ORDER = 7
  subsection Auto mesh generation parameters
    set ATOM BALL RADIUS = 8
    set MESH SIZE AROUND ATOM = 1.2
  end
end
""".strip()



# ═══════════════════════════════════════════════════════════════════════
#  CIF  -->  DFT-FE  (coordinates.inp + domainVectors.inp)
# ═══════════════════════════════════════════════════════════════════════

def cif_to_dftfe(cif_path: str,
                 output_dir: str = None,
                 valence_dict: dict = None,
                 pbc: list = None,
                 coord_filename: str = "coordinates.inp",
                 domain_filename: str = "domainVectors.inp",
                 write_prm: bool = False,
                 prm_template: str = None,
                 use_gpu: bool = True) -> dict:
    """
    Convert a CIF file into DFT-FE input files: coordinates.inp and domainVectors.inp.

    Parameters
    ----------
    cif_path : str
        Path to the input .cif file.
    output_dir : str, optional
        Directory where the output files will be written.
        Defaults to the same directory as the CIF file.
    valence_dict : dict, optional
        Mapping of element symbol -> valence electron count.
        If not provided, the full atomic number is used as the valence
        (suitable for all-electron calculations).
    pbc : list of bool, optional
        Periodic boundary conditions [pbc1, pbc2, pbc3].
        If not provided, inferred from the ASE Atoms object read from the CIF.
    coord_filename : str
        Name for the coordinates output file. Default: "coordinates.inp".
    domain_filename : str
        Name for the domain vectors output file. Default: "domainVectors.inp".

    Returns
    -------
    dict
        Dictionary with keys:
          - 'atoms':        the ASE Atoms object
          - 'pbc':          list of bools for periodicity
          - 'natoms':       number of atoms
          - 'ntypes':       number of unique atom types
          - 'coord_file':   absolute path to coordinates.inp
          - 'domain_file':  absolute path to domainVectors.inp
          - 'prm_file':     absolute path to parameterFile.prm (if generated, else None)

    Notes
    -----
    CIF files store positions in Angstrom. DFT-FE expects:
      * domainVectors.inp      : lattice vectors in Bohr
      * coordinates.inp        : fractional coordinates (periodic/semi-periodic)
                                  or Cartesian coordinates in Bohr (non-periodic)

    For the coordinate file format per atom line:
        AtomicNumber  ValenceCharge  x  y  z
    """
    if not os.path.isfile(cif_path):
        raise FileNotFoundError(f"CIF file not found: {cif_path}")

    atoms = read(cif_path)

    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(cif_path))
    os.makedirs(output_dir, exist_ok=True)

    # Determine periodicity
    if pbc is not None:
        pbc_flags = list(pbc)
    else:
        pbc_flags = atoms.get_pbc().tolist()

    is_fully_periodic = all(pbc_flags)
    is_any_periodic = any(pbc_flags)

    # --- domainVectors.inp (always in Bohr) ---
    cell_ang = np.array(atoms.get_cell())  # 3x3, Angstrom
    cell_bohr = cell_ang / Bohr            # 3x3, Bohr

    domain_path = os.path.join(output_dir, domain_filename)
    with open(domain_path, "w") as f:
        for vec in cell_bohr:
            f.write("{:24.16f} {:24.16f} {:24.16f}\n".format(*vec))

    # --- coordinates.inp ---
    if is_any_periodic:
        # Fractional coordinates for periodic / semi-periodic systems
        coords = atoms.get_scaled_positions()
    else:
        # Cartesian in Bohr, origin at cell center for non-periodic
        positions_ang = atoms.get_positions()
        center_ang = cell_ang.sum(axis=0) / 2.0
        coords = (positions_ang - center_ang) / Bohr

    coord_lines = []
    for atom, xyz in zip(atoms, coords):
        Z = atom.number
        symbol = atom.symbol
        valence = valence_dict.get(symbol, Z) if valence_dict else Z
        coord_lines.append(
            "{:d}  {:d}  {:24.16E}  {:24.16E}  {:24.16E}".format(
                Z, valence, xyz[0], xyz[1], xyz[2]
            )
        )

    coord_path = os.path.join(output_dir, coord_filename)
    with open(coord_path, "w") as f:
        f.write("\n".join(coord_lines) + "\n")

    natoms = len(atoms)
    ntypes = len(set(atom.number for atom in atoms))

    prm_path = None
    if write_prm:
        template = prm_template if prm_template is not None else DEFAULT_PRM_TEMPLATE
        prm_filled = template.format(
            natoms=natoms,
            ntypes=ntypes,
            p1="true" if pbc_flags[0] else "false",
            p2="true" if pbc_flags[1] else "false",
            p3="true" if pbc_flags[2] else "false",
            use_gpu=str(use_gpu).lower()
        )
        prm_path = os.path.join(output_dir, "parameterFile.prm")
        with open(prm_path, "w") as f:
            f.write(prm_filled)

    return {
        "atoms": atoms,
        "pbc": pbc_flags,
        "natoms": natoms,
        "ntypes": ntypes,
        "coord_file": os.path.abspath(coord_path),
        "domain_file": os.path.abspath(domain_path),
        "prm_file": os.path.abspath(prm_path) if prm_path else None,
    }


# ═══════════════════════════════════════════════════════════════════════
#  DFT-FE  -->  CIF  (coordinates.inp + domainVectors.inp  ->  .cif)
# ═══════════════════════════════════════════════════════════════════════

def dftfe_to_cif(coord_path: str,
                 domain_path: str,
                 cif_path: str,
                 pbc: list = None) -> Atoms:
    """
    Convert DFT-FE input files (coordinates.inp, domainVectors.inp) into a CIF file.

    Parameters
    ----------
    coord_path : str
        Path to the coordinates.inp file.
    domain_path : str
        Path to the domainVectors.inp file.
    cif_path : str
        Output path for the .cif file.
    pbc : list of bool, optional
        Periodic boundary conditions [pbc1, pbc2, pbc3].
        If not provided, defaults to [True, True, True].

    Returns
    -------
    ASE Atoms object reconstructed from the DFT-FE files.

    Notes
    -----
    The function auto-detects whether coordinates are fractional or Cartesian
    based on the periodicity (pbc). If all periodic, fractional is assumed;
    otherwise Cartesian in Bohr is assumed.
    """
    if not os.path.isfile(coord_path):
        raise FileNotFoundError(f"Coordinates file not found: {coord_path}")
    if not os.path.isfile(domain_path):
        raise FileNotFoundError(f"Domain vectors file not found: {domain_path}")

    # Read domain vectors (Bohr) -> convert to Angstrom
    cell_bohr = []
    with open(domain_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 3:
                cell_bohr.append([float(parts[0]), float(parts[1]), float(parts[2])])
    cell_bohr = np.array(cell_bohr)
    cell_ang = cell_bohr * Bohr  # Convert to Angstrom

    # Read coordinates
    symbols = []
    coords = []
    with open(coord_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 5:
                Z = int(float(parts[0]))
                # parts[1] = valence (not needed for reconstruction)
                x, y, z = float(parts[2]), float(parts[3]), float(parts[4])
                symbols.append(chemical_symbols[Z])
                coords.append([x, y, z])

    coords = np.array(coords)

    if pbc is None:
        pbc = [True, True, True]

    is_any_periodic = any(pbc)

    if is_any_periodic:
        # Coordinates are fractional -> convert to Cartesian Angstrom
        positions_ang = np.dot(coords, cell_ang)
    else:
        # Coordinates are Cartesian in Bohr with origin at cell center
        center_bohr = cell_bohr.sum(axis=0) / 2.0
        positions_ang = (coords + center_bohr) * Bohr

    atoms = Atoms(symbols=symbols, positions=positions_ang,
                  cell=cell_ang, pbc=pbc)

    # Ensure output directory exists
    out_dir = os.path.dirname(os.path.abspath(cif_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    write(cif_path, atoms, format="cif")
    return atoms


# ═══════════════════════════════════════════════════════════════════════
#  ASE Atoms  -->  DFT-FE  (convenience wrapper without CIF on disk)
# ═══════════════════════════════════════════════════════════════════════

def atoms_to_dftfe(atoms: Atoms,
                   output_dir: str,
                   valence_dict: dict = None,
                   pbc: list = None,
                   coord_filename: str = "coordinates.inp",
                   domain_filename: str = "domainVectors.inp",
                   write_prm: bool = False,
                   prm_template: str = None,
                   use_gpu: bool = True) -> dict:
    """
    Write DFT-FE input files directly from an ASE Atoms object.

    This is a convenience function when you already have an ASE Atoms object
    (e.g., built programmatically) and don't want to round-trip through CIF.

    Parameters
    ----------
    atoms : ase.Atoms
        The atomic structure.
    output_dir : str
        Directory where the output files will be written.
    valence_dict : dict, optional
        Element symbol -> valence electron count. Defaults to full Z.
    pbc : list of bool, optional
        Override periodicity. Defaults to atoms.get_pbc().
    coord_filename : str
        Name for coordinates output file.
    domain_filename : str
        Name for domain vectors output file.

    Returns
    -------
    dict  (same structure as cif_to_dftfe return value)
    """
    os.makedirs(output_dir, exist_ok=True)

    if pbc is not None:
        pbc_flags = list(pbc)
    else:
        pbc_flags = atoms.get_pbc().tolist()

    is_any_periodic = any(pbc_flags)

    # domainVectors.inp (Bohr)
    cell_ang = np.array(atoms.get_cell())
    cell_bohr = cell_ang / Bohr

    domain_path = os.path.join(output_dir, domain_filename)
    with open(domain_path, "w") as f:
        for vec in cell_bohr:
            f.write("{:24.16f} {:24.16f} {:24.16f}\n".format(*vec))

    # coordinates.inp
    if is_any_periodic:
        coords = atoms.get_scaled_positions()
    else:
        positions_ang = atoms.get_positions()
        center_ang = cell_ang.sum(axis=0) / 2.0
        coords = (positions_ang - center_ang) / Bohr

    coord_lines = []
    for atom, xyz in zip(atoms, coords):
        Z = atom.number
        symbol = atom.symbol
        valence = valence_dict.get(symbol, Z) if valence_dict else Z
        coord_lines.append(
            "{:d}  {:d}  {:24.16E}  {:24.16E}  {:24.16E}".format(
                Z, valence, xyz[0], xyz[1], xyz[2]
            )
        )

    coord_path = os.path.join(output_dir, coord_filename)
    with open(coord_path, "w") as f:
        f.write("\n".join(coord_lines) + "\n")

    natoms = len(atoms)
    ntypes = len(set(atom.number for atom in atoms))

    prm_path = None
    if write_prm:
        template = prm_template if prm_template is not None else DEFAULT_PRM_TEMPLATE
        prm_filled = template.format(
            natoms=natoms,
            ntypes=ntypes,
            p1="true" if pbc_flags[0] else "false",
            p2="true" if pbc_flags[1] else "false",
            p3="true" if pbc_flags[2] else "false",
            use_gpu=str(use_gpu).lower()
        )
        prm_path = os.path.join(output_dir, "parameterFile.prm")
        with open(prm_path, "w") as f:
            f.write(prm_filled)

    return {
        "atoms": atoms,
        "pbc": pbc_flags,
        "natoms": natoms,
        "ntypes": ntypes,
        "coord_file": os.path.abspath(coord_path),
        "domain_file": os.path.abspath(domain_path),
        "prm_file": os.path.abspath(prm_path) if prm_path else None,
    }


# ═══════════════════════════════════════════════════════════════════════
#  DFT-FE  -->  ASE Atoms  (read back without writing CIF)
# ═══════════════════════════════════════════════════════════════════════

def dftfe_to_atoms(coord_path: str,
                   domain_path: str,
                   pbc: list = None) -> Atoms:
    """
    Read DFT-FE input files and return an ASE Atoms object (without writing a CIF).

    Parameters
    ----------
    coord_path : str
        Path to coordinates.inp.
    domain_path : str
        Path to domainVectors.inp.
    pbc : list of bool, optional
        Periodicity flags. Defaults to [True, True, True].

    Returns
    -------
    ASE Atoms object with positions in Angstrom and cell in Angstrom.
    """
    if not os.path.isfile(coord_path):
        raise FileNotFoundError(f"Coordinates file not found: {coord_path}")
    if not os.path.isfile(domain_path):
        raise FileNotFoundError(f"Domain vectors file not found: {domain_path}")

    # Domain vectors: Bohr -> Angstrom
    cell_bohr = []
    with open(domain_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 3:
                cell_bohr.append([float(p) for p in parts[:3]])
    cell_bohr = np.array(cell_bohr)
    cell_ang = cell_bohr * Bohr

    # Coordinates
    symbols = []
    coords = []
    with open(coord_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 5:
                Z = int(float(parts[0]))
                x, y, z = float(parts[2]), float(parts[3]), float(parts[4])
                symbols.append(chemical_symbols[Z])
                coords.append([x, y, z])

    coords = np.array(coords)

    if pbc is None:
        pbc = [True, True, True]

    if any(pbc):
        positions_ang = np.dot(coords, cell_ang)
    else:
        center_bohr = cell_bohr.sum(axis=0) / 2.0
        positions_ang = (coords + center_bohr) * Bohr

    return Atoms(symbols=symbols, positions=positions_ang,
                 cell=cell_ang, pbc=pbc)
