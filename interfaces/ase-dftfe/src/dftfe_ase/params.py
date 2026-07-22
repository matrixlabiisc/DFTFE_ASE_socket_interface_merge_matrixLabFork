# ---------------------------------------------------------------------
# Copyright (c) 2017-2025 The Regents of the University of Michigan and DFT-FE
# authors. Part of the DFT-FE code, released under LGPL v2.1 or later.
# ---------------------------------------------------------------------
#
# @author Mehul Darak
#
"""Single source of truth: ASE kwarg <-> DFT-FE .prm parameter mapping.

Every supported parameter is declared ONCE here (kwarg, .prm key, innermost
subsection, type). This table drives:
  * (B) full-Python control  — typed kwargs on the DFTFE calculator;
  * the generic .prm injection (kwarg + `extra_prm` passthrough (A) both become
    the same {section, key, value} entries — one inject path, no per-param sed);
  * `parameter_mapping.md`, generated from this table so it can never go stale.

`status`: "wired" = injected via the current C++ path AND two-value-verified;
"planned" = in the table for B, activated once the generic injector lands.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

# Subsection anchors (innermost). None = top-level (before any subsection).
FE = "Finite element mesh parameters"
AUTOMESH = "Auto mesh generation parameters"
SCF = "SCF parameters"
EIG = "Eigen-solver parameters"
BZ = "Brillouin zone k point sampling options"
MP = "Monkhorst-Pack (MP) grid generation"
DFTF = "DFT functional parameters"
BC = "Boundary conditions"
PAR = "Parallelization"
OPT = "Optimization"
DISP = "Dispersion Correction"
HUB = "Hubbard Parameters"


@dataclass(frozen=True)
class ParamSpec:
    kwarg: str
    prm_key: str
    section: Optional[str]
    dtype: str          # "int" | "float" | "bool" | "str" | "int3" | "float3"
    status: str         # "wired" | "planned"
    note: str = ""


# fmt: off
PARAM_TABLE = [
    # ── verified & wired (two-value tracking passed) ────────────────────────
    ParamSpec("xc",                         "EXCHANGE CORRELATION TYPE",            DFTF,     "str",   "wired"),
    ParamSpec("pseudopotential_calculation","PSEUDOPOTENTIAL CALCULATION",          DFTF,     "bool",  "wired"),
    ParamSpec("polynomial_order",           "POLYNOMIAL ORDER",                     FE,       "int",   "wired"),
    ParamSpec("density_quadrature_rule",    "DENSITY QUADRATURE RULE",              FE,       "int",   "wired"),
    ParamSpec("mesh_size",                  "MESH SIZE AROUND ATOM",                AUTOMESH, "float", "wired"),
    ParamSpec("atom_ball_radius",           "ATOM BALL RADIUS",                     AUTOMESH, "float", "wired"),
    ParamSpec("tolerance",                  "TOLERANCE",                            SCF,      "float", "wired"),
    ParamSpec("fermi_temp",                 "TEMPERATURE",                          SCF,      "float", "wired"),
    ParamSpec("scf_mixing",                 "MIXING PARAMETER",                     SCF,      "float", "wired"),
    ParamSpec("mixing_scheme",              "MIXING METHOD",                        SCF,      "str",   "wired"),
    ParamSpec("max_scf_iterations",         "MAXIMUM ITERATIONS",                   SCF,      "int",   "wired"),
    ParamSpec("num_bands",                  "NUMBER OF KOHN-SHAM WAVEFUNCTIONS",    EIG,      "int",   "wired"),
    ParamSpec("orthogonalization_type",     "ORTHOGONALIZATION TYPE",               EIG,      "str",   "wired"),
    ParamSpec("use_single_prec_cheby",      "USE SINGLE PREC CHEBY",                EIG,      "bool",  "wired"),
    ParamSpec("wfc_block_size",             "WFC BLOCK SIZE",                       EIG,      "int",   "wired"),
    ParamSpec("cheby_wfc_block_size",       "CHEBY WFC BLOCK SIZE",                 EIG,      "int",   "wired"),
    ParamSpec("mp_grid",                    "SAMPLING POINTS",                      MP,       "int3",  "wired", "-> SAMPLING POINTS 1/2/3"),
    ParamSpec("mp_grid_shift",              "SAMPLING SHIFT",                       MP,       "int3",  "wired", "-> SAMPLING SHIFT 1/2/3"),
    ParamSpec("use_time_reversal_symmetry", "USE TIME REVERSAL SYMMETRY",           BZ,       "bool",  "wired"),
    ParamSpec("use_group_symmetry",         "USE GROUP SYMMETRY",                   BZ,       "bool",  "wired"),
    ParamSpec("npkpt",                      "NPKPT",                                PAR,      "int",   "wired"),
    ParamSpec("smeared_nuclear_charges",    "SMEARED NUCLEAR CHARGES",              BC,       "bool",  "wired"),
    ParamSpec("spin_polarized",             "SPIN POLARIZATION",                    DFTF,     "int",   "wired"),
    ParamSpec("mixing_history",             "MIXING HISTORY",                       SCF,      "int",   "planned", "was mis-mapped to LBFGS HISTORY; fix to MIXING HISTORY"),
    ParamSpec("verbosity",                  "VERBOSITY",                            None,     "int",   "wired"),
    # ── planned (accuracy-relevant benchmark params; activate w/ generic inject) ─
    ParamSpec("base_mesh_size",             "BASE MESH SIZE",                       AUTOMESH, "float", "planned"),
    ParamSpec("mesh_size_at_atom",          "MESH SIZE AT ATOM",                    AUTOMESH, "float", "planned"),
    ParamSpec("auto_adapt_base_mesh_size",  "AUTO ADAPT BASE MESH SIZE",            AUTOMESH, "bool",  "planned"),
    ParamSpec("chebyshev_polynomial_degree","CHEBYSHEV POLYNOMIAL DEGREE",          EIG,      "int",   "planned"),
    ParamSpec("chebyshev_filter_tolerance", "CHEBYSHEV FILTER TOLERANCE",           EIG,      "float", "planned"),
    ParamSpec("highest_state_cheby",        "HIGHEST STATE OF INTEREST FOR CHEBYSHEV FILTERING", EIG, "int", "planned"),
    ParamSpec("total_magnetization",        "TOTAL MAGNETIZATION",                  DFTF,     "float", "planned"),
    ParamSpec("spin_mixing_enhancement",    "SPIN MIXING ENHANCEMENT FACTOR",       SCF,      "float", "planned"),
    ParamSpec("hubbard_parameters_file",    "HUBBARD PARAMETERS FILE",              HUB,      "str",   "planned", "DFT+U"),
    ParamSpec("kpoint_rule_file",           "kPOINT RULE FILE",                     BZ,       "str",   "planned"),
    ParamSpec("dispersion_correction_type", "DISPERSION CORRECTION TYPE",           DISP,     "int",   "planned", "needs Dispersion Correction subsection added"),
]
# fmt: on

BY_KWARG = {p.kwarg: p for p in PARAM_TABLE}


def format_value(dtype: str, value: Any):
    if dtype == "bool":
        return "true" if value else "false"
    return value


def to_prm_entries(kwargs: dict) -> list[dict]:
    """Turn typed kwargs into generic {section, key, value} injection entries."""
    entries = []
    for k, v in kwargs.items():
        if v is None or k not in BY_KWARG:
            continue
        spec = BY_KWARG[k]
        if spec.dtype in ("int3", "float3"):  # SAMPLING POINTS 1/2/3 etc.
            for i, comp in enumerate(v, start=1):
                entries.append({"section": spec.section, "key": f"{spec.prm_key} {i}", "value": comp})
        else:
            entries.append({"section": spec.section, "key": spec.prm_key,
                            "value": format_value(spec.dtype, v)})
    return entries


def generate_mapping_markdown() -> str:
    lines = [
        "# ASE-DFTFE Parameter Mapping (auto-generated from `src/dftfe_ase/params.py`)",
        "",
        "Do not edit by hand — regenerate with `python -m dftfe_ase.params`.",
        "Single source of truth for ASE kwarg ↔ DFT-FE `.prm` parameter, subsection, type, and status.",
        "`wired` = injected and two-value-verified; `planned` = defined for full-Python control,",
        "activated with the generic `.prm` injector. Any param not listed can still be passed via",
        "`extra_prm={...}` or a full `.prm` (option A).",
        "",
        "| ASE kwarg | DFT-FE `.prm` key | subsection | type | status | note |",
        "|---|---|---|---|---|---|",
    ]
    for p in PARAM_TABLE:
        sec = p.section or "(top-level)"
        lines.append(f"| `{p.kwarg}` | `{p.prm_key}` | {sec} | {p.dtype} | {p.status} | {p.note} |")
    lines += [
        "",
        "Also handled outside this table: geometry (coords/cell/numbers/pbc), `compute_forces` "
        "(`ION FORCE`), `compute_stress` (`CELL STRESS`), `use_device` (`USE GPU`), `keep_scratch`, "
        "`psp_path` (dict → per-element `pseudo.inp`). Optimizer/NEB/MD settings (FORCE TOL, OPTIMIZATION "
        "MODE, NUMBER OF IMAGES, spring constants, …) are **ASE-side** (ASE optimizers / `ase.mep.NEB`), "
        "not DFT-FE `.prm` params.",
    ]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    print(generate_mapping_markdown())
