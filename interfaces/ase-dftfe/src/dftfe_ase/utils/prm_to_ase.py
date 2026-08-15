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

"""
DFT-FE .prm + coordinates.inp + domainVectors.inp → ASE setup.

Provides:
  - ``parse_prm``         Parse a DFT-FE .prm parameter file into a dict.
  - ``dftfe_dir_to_atoms`` Build an ASE Atoms + DFTFE calculator directly from
                           a DFT-FE benchmark/run directory containing
                           coordinates.inp, domainVectors.inp, and a .prm file.
  - ``generate_ase_script`` Write a ready-to-run Python script mirroring what
                            dftfe_dir_to_atoms would do, but in standalone form.

Usage example
-------------
>>> from dftfe_ase.utils import dftfe_dir_to_atoms
>>> atoms, calc_kwargs = dftfe_dir_to_atoms("/path/to/Li2O_fcc/dftfe",
...                                          prm_file="Li2O_scf.prm")
>>> atoms.calc.close()   # (calculator is attached but not launched until needed)

Or generate a standalone script:
>>> from dftfe_ase.utils import generate_ase_script
>>> generate_ase_script("/path/to/Li2O_fcc/dftfe",
...                      prm_file="Li2O_scf.prm",
...                      output_script="my_calc.py")
"""

import os
import re
import textwrap
from typing import Optional, Dict, Any, Tuple

import numpy as np
from ase import Atoms
from ase.units import Bohr

from .cif_converter import dftfe_to_atoms as _dftfe_to_atoms


# ════════════════════════════════════════════════════════════════════════════
#  Internal helpers
# ════════════════════════════════════════════════════════════════════════════

def _strip_comment(val: str) -> str:
    """Remove trailing # comments from a value string."""
    return val.split("#")[0].strip()


def _parse_bool(val: str) -> bool:
    return val.strip().lower() in ("true", "yes", "1", "on")


def _parse_int_or_float(val: str):
    try:
        return int(val)
    except ValueError:
        return float(val)


# ════════════════════════════════════════════════════════════════════════════
#  Public: parse_prm
# ════════════════════════════════════════════════════════════════════════════

def parse_prm(prm_path: str) -> Dict[str, Any]:
    """
    Parse a DFT-FE ``.prm`` parameter file into a flat Python dictionary.

    Subsections are flattened with dot-notation keys, e.g.:
      ``"Geometry.NATOMS"`` or
      ``"Brillouin zone k point sampling options.Monkhorst-Pack (MP) grid generation.SAMPLING POINTS 1"``

    Parameters
    ----------
    prm_path : str
        Absolute or relative path to the ``.prm`` file.

    Returns
    -------
    dict
        Flat mapping of ``"[subsection.]KEY"`` → value (str, int, float, or bool).

    Examples
    --------
    >>> prm = parse_prm("Li2O_scf.prm")
    >>> prm["SOLVER MODE"]              # "GS"
    >>> prm["Geometry.NATOMS"]          # 95
    """
    if not os.path.isfile(prm_path):
        raise FileNotFoundError(f"PRM file not found: {prm_path}")

    params: Dict[str, Any] = {}
    section_stack = []   # list of active subsection names

    _set_re   = re.compile(r"^\s*set\s+(.+?)\s*=\s*(.+)$", re.IGNORECASE)
    _sub_re   = re.compile(r"^\s*subsection\s+(.+)$", re.IGNORECASE)
    _end_re   = re.compile(r"^\s*end\s*$", re.IGNORECASE)

    with open(prm_path, "r") as fh:
        for raw_line in fh:
            line = raw_line.strip()

            # Skip blank lines and full-line comments
            if not line or line.startswith("#"):
                continue

            m_sub = _sub_re.match(line)
            if m_sub:
                section_stack.append(m_sub.group(1).strip())
                continue

            if _end_re.match(line):
                if section_stack:
                    section_stack.pop()
                continue

            m_set = _set_re.match(line)
            if m_set:
                raw_key = m_set.group(1).strip()
                raw_val = _strip_comment(m_set.group(2))

                # Build hierarchical key
                if section_stack:
                    key = ".".join(section_stack) + "." + raw_key
                else:
                    key = raw_key

                # Attempt type coercion
                lv = raw_val.lower()
                if lv in ("true", "false"):
                    params[key] = _parse_bool(raw_val)
                else:
                    try:
                        params[key] = _parse_int_or_float(raw_val)
                    except ValueError:
                        params[key] = raw_val   # keep as string

    return params


# ════════════════════════════════════════════════════════════════════════════
#  Internal: prm dict → DFTFE calculator kwargs
# ════════════════════════════════════════════════════════════════════════════

_MP_PREFIX  = "Brillouin zone k point sampling options.Monkhorst-Pack (MP) grid generation."
_MESH_PRE   = "Finite element mesh parameters."
_MESH_AUTO  = "Finite element mesh parameters.Auto mesh generation parameters."
_SCF_PRE    = "SCF parameters."
_SCF_EIGEN  = "SCF parameters.Eigen-solver parameters."
_DFT_PRE    = "DFT functional parameters."
_GEO_OPT    = "Geometry.Optimization."


def _prm_to_calc_kwargs(prm: Dict[str, Any],
                        psp_dir: Optional[str] = None,
                        pseudo_inp: Optional[str] = None) -> Dict[str, Any]:
    """
    Translate a parsed PRM dict to keyword arguments for ``DFTFE(...)`` constructor.

    Parameters
    ----------
    prm        : dict from ``parse_prm``
    psp_dir    : directory that contains the ``.upf`` files (used when
                 ``PSEUDOPOTENTIAL FILE NAMES LIST`` is ``pseudo.inp``).
    pseudo_inp : full path to a pseudo.inp override (default: look beside prm).
    """
    kwargs: Dict[str, Any] = {}

    # ── Solver mode → compute_forces / compute_stress
    solver_mode = str(prm.get("SOLVER MODE", "GS")).upper()
    kwargs["compute_forces"] = bool(prm.get("Geometry.Optimization.ION FORCE", True))
    kwargs["compute_stress"] = bool(prm.get("Geometry.Optimization.CELL STRESS", False))

    # ── GPU
    kwargs["use_device"] = bool(prm.get("USE GPU", False))

    # ── Verbosity
    if "VERBOSITY" in prm:
        kwargs["verbosity"] = int(prm["VERBOSITY"])

    # ── PBC  (used externally, not passed to calculator, but returned)
    # Collected separately in dftfe_dir_to_atoms

    # ── k-point sampling
    sp1 = prm.get(_MP_PREFIX + "SAMPLING POINTS 1")
    sp2 = prm.get(_MP_PREFIX + "SAMPLING POINTS 2")
    sp3 = prm.get(_MP_PREFIX + "SAMPLING POINTS 3")
    if sp1 is not None and sp2 is not None and sp3 is not None:
        kwargs["mp_grid"] = (int(sp1), int(sp2), int(sp3))

    sh1 = prm.get(_MP_PREFIX + "SAMPLING SHIFT 1")
    sh2 = prm.get(_MP_PREFIX + "SAMPLING SHIFT 2")
    sh3 = prm.get(_MP_PREFIX + "SAMPLING SHIFT 3")
    if sh1 is not None and sh2 is not None and sh3 is not None:
        kwargs["mp_grid_shift"] = (int(sh1), int(sh2), int(sh3))

    trs = prm.get("Brillouin zone k point sampling options.USE TIME REVERSAL SYMMETRY")
    if trs is not None:
        kwargs["use_time_reversal_symmetry"] = bool(trs)

    # ── Mesh
    mesh = prm.get(_MESH_AUTO + "MESH SIZE AROUND ATOM")
    if mesh is not None:
        kwargs["mesh_size"] = float(mesh)

    abr = prm.get(_MESH_AUTO + "ATOM BALL RADIUS")
    if abr is not None:
        kwargs["atom_ball_radius"] = float(abr)

    po = prm.get(_MESH_PRE + "POLYNOMIAL ORDER")
    if po is None:
        po = prm.get("Finite element mesh parameters.POLYNOMIAL ORDER")  # no sub-indent
    if po is not None:
        kwargs["polynomial_order"] = int(po)

    # ── SCF
    tol = prm.get(_SCF_PRE + "TOLERANCE")
    if tol is not None:
        kwargs["tolerance"] = float(tol)

    temp = prm.get(_SCF_PRE + "TEMPERATURE")
    if temp is not None:
        kwargs["fermi_temp"] = float(temp)

    mix_m = prm.get(_SCF_PRE + "MIXING METHOD")
    if mix_m is not None:
        kwargs["mixing_scheme"] = str(mix_m).strip()

    mix_p = prm.get(_SCF_PRE + "MIXING PARAMETER")
    if mix_p is not None:
        kwargs["scf_mixing"] = float(mix_p)

    max_scf = prm.get(_SCF_PRE + "MAXIMUM ITERATIONS")
    if max_scf is not None:
        kwargs["max_scf_iterations"] = int(max_scf)

    dqr_key = "Finite element mesh parameters.DENSITY QUADRATURE RULE"
    dqr = prm.get(dqr_key)
    if dqr is not None:
        kwargs["density_quadrature_rule"] = int(dqr)

    spc_key = _SCF_PRE + "Eigen-solver parameters.USE SINGLE PREC CHEBY"
    spc = prm.get(spc_key)
    if spc is not None:
        kwargs["use_single_prec_cheby"] = bool(spc)

    # ── XC functional
    xc = prm.get(_DFT_PRE + "EXCHANGE CORRELATION TYPE")
    if xc is not None:
        kwargs["xc"] = str(xc).strip()

    # ── Pseudopotential
    pseudo_calc = prm.get(_DFT_PRE + "PSEUDOPOTENTIAL CALCULATION")
    if pseudo_calc is not None:
        kwargs["pseudopotential_calculation"] = bool(pseudo_calc)

    if psp_dir is not None:
        pseudo_list_name = str(prm.get(
            _DFT_PRE + "PSEUDOPOTENTIAL FILE NAMES LIST", "pseudo.inp"))
        if pseudo_inp is None:
            pseudo_inp = os.path.join(psp_dir, pseudo_list_name)

        if os.path.isfile(pseudo_inp):
            # Parse pseudo.inp: lines like "3 Li_ONCV_PBE-1.2.upf"
            psp_dict = {}
            from ase.data import chemical_symbols
            with open(pseudo_inp) as ph:
                for ln in ph:
                    ln = ln.strip()
                    if not ln or ln.startswith("#"):
                        continue
                    parts = ln.split()
                    if len(parts) >= 2:
                        z    = int(parts[0])
                        upf  = os.path.join(psp_dir, parts[1])
                        sym  = chemical_symbols[z]
                        psp_dict[sym] = upf
            if psp_dict:
                kwargs["psp_path"] = psp_dict

    return kwargs


# ════════════════════════════════════════════════════════════════════════════
#  Public: dftfe_dir_to_atoms
# ════════════════════════════════════════════════════════════════════════════

def dftfe_dir_to_atoms(
    dftfe_dir: str,
    prm_file: Optional[str] = None,
    coord_file: str = "coordinates.inp",
    domain_file: str = "domainVectors.inp",
    dftfe_binary: Optional[str] = None,
    np: int = 8,
    host: str = "localhost",
    port: int = 0,
    log_file: Optional[str] = None,
    extra_calc_kwargs: Optional[Dict[str, Any]] = None,
) -> Tuple[Atoms, Dict[str, Any]]:
    """
    Read a DFT-FE run directory and return a ready-to-use ``(atoms, calc_kwargs)``
    pair with the DFTFE calculator already attached to ``atoms``.

    Parameters
    ----------
    dftfe_dir : str
        Directory containing ``coordinates.inp``, ``domainVectors.inp``,
        and (optionally) a ``.prm`` file.
    prm_file : str, optional
        Name of the ``.prm`` file inside ``dftfe_dir``.
        If None, the first ``.prm`` file found in the directory is used.
    coord_file : str
        Name of the coordinates file (default: ``coordinates.inp``).
    domain_file : str
        Name of the domain vectors file (default: ``domainVectors.inp``).
    dftfe_binary : str, optional
        Full path to the DFT-FE executable.  If None, the value of the
        environment variable ``DFTFE_BIN`` is used, or a sensible default.
    np : int
        Number of MPI ranks to pass to ``mpirun``.
    host : str
        Socket host (default: ``"localhost"``).
    port : int
        Socket port (0 = auto-pick a free port).
    log_file : str, optional
        Log file name for DFT-FE output.  Defaults to ``<prm_stem>.log``.
    extra_calc_kwargs : dict, optional
        Additional keyword arguments forwarded to ``DFTFE(...)`` verbatim,
        overriding anything inferred from the ``.prm`` file.

    Returns
    -------
    atoms : ase.Atoms
        ASE Atoms object built from ``coordinates.inp`` + ``domainVectors.inp``.
    calc_kwargs : dict
        The full set of kwargs that were passed to ``DFTFE(…)`` for reference.

    Notes
    -----
    The DFTFE calculator is attached to ``atoms`` but **not yet launched**.
    The process starts on the first call to ``atoms.get_potential_energy()``
    (or any other computed property).

    Call ``atoms.calc.close()`` when done to cleanly terminate the DFTFE server.
    """
    dftfe_dir = os.path.abspath(dftfe_dir)

    # ── locate files ─────────────────────────────────────────────────────────
    coord_path  = os.path.join(dftfe_dir, coord_file)
    domain_path = os.path.join(dftfe_dir, domain_file)

    if not os.path.isfile(coord_path):
        raise FileNotFoundError(f"coordinates file not found: {coord_path}")
    if not os.path.isfile(domain_path):
        raise FileNotFoundError(f"domain vectors file not found: {domain_path}")

    # ── locate prm file ───────────────────────────────────────────────────────
    if prm_file is None:
        candidates = [f for f in os.listdir(dftfe_dir) if f.endswith(".prm")]
        if not candidates:
            raise FileNotFoundError(f"No .prm file found in {dftfe_dir}")
        prm_file = sorted(candidates)[0]   # pick alphabetically first
    prm_path = os.path.join(dftfe_dir, prm_file)

    # ── parse PRM ─────────────────────────────────────────────────────────────
    prm = parse_prm(prm_path)

    # ── PBC from prm ──────────────────────────────────────────────────────────
    pbc = [
        bool(prm.get("Boundary conditions.PERIODIC1", False)),
        bool(prm.get("Boundary conditions.PERIODIC2", False)),
        bool(prm.get("Boundary conditions.PERIODIC3", False)),
    ]

    # ── Build ASE Atoms ───────────────────────────────────────────────────────
    atoms = _dftfe_to_atoms(coord_path, domain_path, pbc=pbc)

    # ── Infer DFT-FE binary ───────────────────────────────────────────────────
    if dftfe_binary is None:
        dftfe_binary = os.environ.get("DFTFE_BIN", "dftfe")

    run_cmd = f"mpirun -np {np} {dftfe_binary}"

    # ── Build calc kwargs from PRM ────────────────────────────────────────────
    calc_kwargs = _prm_to_calc_kwargs(prm, psp_dir=dftfe_dir)

    # Default log file name from prm stem
    if log_file is None:
        log_file = os.path.splitext(prm_file)[0] + "_ase.log"

    calc_kwargs.update(dict(
        command=run_cmd,
        host=host,
        port=port,
        log_file=log_file,
    ))

    # User-supplied overrides win
    if extra_calc_kwargs:
        calc_kwargs.update(extra_calc_kwargs)

    # ── Attach calculator ─────────────────────────────────────────────────────
    from dftfe_ase import DFTFE
    atoms.calc = DFTFE(**calc_kwargs)

    return atoms, calc_kwargs


# ════════════════════════════════════════════════════════════════════════════
#  Public: generate_ase_script
# ════════════════════════════════════════════════════════════════════════════

def generate_ase_script(
    dftfe_dir: str,
    prm_file: Optional[str] = None,
    coord_file: str = "coordinates.inp",
    domain_file: str = "domainVectors.inp",
    dftfe_binary: str = "/path/to/dftfe",
    np: int = 8,
    output_script: Optional[str] = None,
    ase_dftfe_root: str = "/home/pa01/Mehul/DFTFE/interfaces/ase-dftfe",
) -> str:
    """
    Generate a standalone ASE Python script from a DFT-FE run directory.

    Reads ``coordinates.inp``, ``domainVectors.inp``, and the ``.prm`` file and
    writes a self-contained ``<name>_ase.py`` script that can be submitted directly.

    Parameters
    ----------
    dftfe_dir : str
        The DFT-FE run directory.
    prm_file : str, optional
        Name of the ``.prm`` file.  Auto-detected if None.
    coord_file : str
        Coordinates file name (default ``coordinates.inp``).
    domain_file : str
        Domain vectors file name (default ``domainVectors.inp``).
    dftfe_binary : str
        Full path to the DFT-FE executable inserted into the generated script.
    np : int
        Number of MPI ranks.
    output_script : str, optional
        Where to write the generated script.  Defaults to
        ``<dftfe_dir>/<prm_stem>_ase.py``.
    ase_dftfe_root : str
        Path to the ase-dftfe package root added to ``sys.path`` in the script.

    Returns
    -------
    str
        Path to the written script.
    """
    dftfe_dir = os.path.abspath(dftfe_dir)

    # Locate prm
    if prm_file is None:
        candidates = [f for f in os.listdir(dftfe_dir) if f.endswith(".prm")]
        if not candidates:
            raise FileNotFoundError(f"No .prm file found in {dftfe_dir}")
        prm_file = sorted(candidates)[0]

    prm_path = os.path.join(dftfe_dir, prm_file)
    prm      = parse_prm(prm_path)
    prm_stem = os.path.splitext(prm_file)[0]

    # PBC
    pbc = [
        bool(prm.get("Boundary conditions.PERIODIC1", False)),
        bool(prm.get("Boundary conditions.PERIODIC2", False)),
        bool(prm.get("Boundary conditions.PERIODIC3", False)),
    ]

    # Build calc kwargs
    calc_kw = _prm_to_calc_kwargs(prm, psp_dir=dftfe_dir)
    log_file = prm_stem + "_ase.log"

    # Read domain / coord for printing structure info in the script
    coord_path  = os.path.join(dftfe_dir, coord_file)
    domain_path = os.path.join(dftfe_dir, domain_file)

    # ── code-generate calc kwargs ──────────────────────────────────────────────
    def _repr(v):
        if isinstance(v, bool):
            return str(v)
        if isinstance(v, str):
            return repr(v)
        if isinstance(v, (tuple, list)):
            return repr(v)
        if isinstance(v, dict):
            return repr(v)
        return repr(v)

    kw_lines = []
    kw_lines.append(f'    command=run_cmd,')
    kw_lines.append(f'    host="localhost",')
    kw_lines.append(f'    port=0,')
    for k, v in calc_kw.items():
        if k in ("command", "host", "port", "log_file"):
            continue
        kw_lines.append(f'    {k}={_repr(v)},')
    kw_lines.append(f'    debug_timing=True,')
    kw_lines.append(f'    keep_scratch=False,')
    kw_lines.append(f'    log_file={repr(log_file)},')

    kw_block = "\n".join(kw_lines)

    # ── script template ────────────────────────────────────────────────────────
    script = textwrap.dedent(f"""\
    \"\"\"
    Auto-generated ASE script for DFT-FE run directory:
      {dftfe_dir}

    Source .prm : {prm_file}
    Generated by: dftfe_ase.utils.generate_ase_script
    \"\"\"

    import os
    import sys
    import numpy as np
    from ase.units import Bohr, Hartree

    # ── ase-dftfe package path ─────────────────────────────────────────────────
    ASE_DFTFE_ROOT = {repr(ase_dftfe_root)}
    if ASE_DFTFE_ROOT not in sys.path:
        sys.path.insert(0, ASE_DFTFE_ROOT)

    from dftfe_ase import DFTFE
    from dftfe_ase.utils import dftfe_to_atoms

    # ── input files ────────────────────────────────────────────────────────────
    DFTFE_DIR   = {repr(dftfe_dir)}
    COORD_FILE  = os.path.join(DFTFE_DIR, {repr(coord_file)})
    DOMAIN_FILE = os.path.join(DFTFE_DIR, {repr(domain_file)})

    atoms = dftfe_to_atoms(
        coord_path=COORD_FILE,
        domain_path=DOMAIN_FILE,
        pbc={pbc!r},
    )

    print(f"Loaded structure: {{len(atoms)}} atoms")
    print(f"  Symbols : {{set(atoms.get_chemical_symbols())}}")
    print(f"  Cell (Å): {{atoms.cell.lengths()}}")
    print(f"  PBC     : {{atoms.get_pbc()}}")

    # ── DFT-FE binary ─────────────────────────────────────────────────────────
    DFTFE_BIN = os.environ.get("DFTFE_BIN", {repr(dftfe_binary)})
    NP        = int(os.environ.get("SLURM_NTASKS", {np}))
    run_cmd   = f"mpirun -np {{NP}} {{DFTFE_BIN}}"

    # ── calculator ────────────────────────────────────────────────────────────
    calc = DFTFE(
{kw_block}
    )

    atoms.calc = calc

    # ── run ───────────────────────────────────────────────────────────────────
    print("\\nStarting DFT-FE ground-state calculation …")
    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()

    print("\\n" + "="*50)
    print("Results")
    print("="*50)
    print(f"  Total energy : {{energy:.8f}}  eV")
    print(f"  Total energy : {{energy / Hartree:.8f}}  Ha")
    print(f"  Max |force|  : {{np.max(np.linalg.norm(forces, axis=1)):.6e}}  eV/Å")
    print("="*50)
    """)

    if output_script is None:
        output_script = os.path.join(dftfe_dir, prm_stem + "_ase.py")

    with open(output_script, "w") as fh:
        fh.write(script)

    print(f"Generated: {output_script}")
    return output_script
