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
GPU = "GPU"
POISSON = "Poisson problem parameters"
HELMHOLTZ = "Helmholtz problem parameters"


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
    # ── performance / solver knobs used by the published scaling decks ──────
    # These are what a Summit-style benchmark .prm tunes. Before they were added
    # here the only way to set them was extra_prm/prm_file, so a scaling study
    # could not be expressed in pure Python (option B).
    ParamSpec("auto_gpu_block_sizes",       "AUTO GPU BLOCK SIZES",                 GPU,      "bool",  "planned", "false = honour the explicit block sizes below"),
    ParamSpec("fine_grained_gpu_timings",   "FINE GRAINED GPU TIMINGS",             GPU,      "bool",  "planned"),
    ParamSpec("subspace_rot_full_cpu_mem",  "SUBSPACE ROT FULL CPU MEM",            GPU,      "bool",  "planned"),
    ParamSpec("use_gpudirect_mpi_allreduce","USE GPUDIRECT MPI ALL REDUCE",         GPU,      "bool",  "planned"),
    ParamSpec("use_dccl",                   "USE DCCL",                             GPU,      "bool",  "planned"),
    ParamSpec("use_elpa_gpu_kernel",        "USE ELPA GPU KERNEL",                  GPU,      "bool",  "planned"),
    # Top-level, and the one parameter whose default is *computed*: if the deck
    # leaves it unset, dftParameters.cc forces it to (solverMode is NSCF/BANDS),
    # i.e. false for a GS run -- see the get_entries_wrongly_not_set() branch.
    # Because the injector rewrites the template file before parse_input, an
    # explicit value here counts as "set" and survives that branch.
    ParamSpec("mem_opt_mode",               "MEM OPT MODE",                         None,     "bool",  "planned", "lower peak memory, marginal slowdown; unset => false for GS"),
    ParamSpec("self_potential_radius",      "SELF POTENTIAL RADIUS",                BC,       "float", "planned"),
    ParamSpec("polynomial_order_electrostatics", "POLYNOMIAL ORDER ELECTROSTATICS", FE,       "int",   "planned"),
    ParamSpec("poisson_tolerance",          "TOLERANCE",                            POISSON,  "float", "planned", "distinct from `tolerance` (SCF); same key, different subsection"),
    ParamSpec("poisson_max_iterations",     "MAXIMUM ITERATIONS",                   POISSON,  "int",   "planned", "distinct from `max_scf_iterations` (SCF)"),
    ParamSpec("helmholtz_tolerance",        "ABSOLUTE TOLERANCE HELMHOLTZ",         HELMHOLTZ,"float", "planned"),
    ParamSpec("helmholtz_max_iterations",   "MAXIMUM ITERATIONS HELMHOLTZ",         HELMHOLTZ,"int",   "planned"),
    ParamSpec("kerker_mixing_parameter",    "KERKER MIXING PARAMETER",              SCF,      "float", "planned"),
    ParamSpec("adapt_anderson_mixing_parameter", "ADAPT ANDERSON MIXING PARAMETER", SCF,      "bool",  "planned", "DFT-FE default false = hold `scf_mixing` fixed"),
    ParamSpec("resta_fermi_wavevector",     "RESTA FERMI WAVEVECTOR",               SCF,      "float", "planned", "only read by MIXING METHOD=ANDERSON_WITH_RESTA"),
    ParamSpec("resta_screening_length",     "RESTA SCREENING LENGTH",               SCF,      "float", "planned", "only read by MIXING METHOD=ANDERSON_WITH_RESTA"),
    ParamSpec("compute_energy_each_iter",   "COMPUTE ENERGY EACH ITER",             SCF,      "bool",  "planned"),
    ParamSpec("cheby_degree_scaling_first_scf", "CHEBYSHEV POLYNOMIAL DEGREE SCALING FACTOR FIRST SCF", EIG, "float", "planned"),
    ParamSpec("subspace_rot_dofs_block_size", "SUBSPACE ROT DOFS BLOCK SIZE",       EIG,      "int",   "planned"),
    ParamSpec("scalapack_procs",            "SCALAPACKPROCS",                       EIG,      "int",   "planned", "0 = auto; tuned per rank count in scaling decks"),
    ParamSpec("scalapack_block_size",       "SCALAPACK BLOCK SIZE",                 EIG,      "int",   "planned"),
    ParamSpec("reuse_lanczos_upper_bound",  "REUSE LANCZOS UPPER BOUND",            EIG,      "bool",  "planned"),
    ParamSpec("allow_multiple_passes_post_first_scf", "ALLOW MULTIPLE PASSES POST FIRST SCF", EIG, "bool", "planned"),
    ParamSpec("use_mixed_prec_cgs_sr",      "USE MIXED PREC CGS SR",                EIG,      "bool",  "planned"),
    ParamSpec("use_mixed_prec_xtox",        "USE MIXED PREC XTOX",                  EIG,      "bool",  "planned", "v1.0 decks: USE MIXED PREC CGS O"),
    ParamSpec("use_mixed_prec_xthx",        "USE MIXED PREC XTHX",                  EIG,      "bool",  "planned", "v1.0 decks: USE MIXED PREC XTHX SPECTRUM SPLIT"),
    ParamSpec("use_mixed_prec_rr_sr",       "USE MIXED PREC RR_SR",                 EIG,      "bool",  "planned"),
    ParamSpec("use_mixed_prec_commun_only_xtox_xthx", "USE MIXED PREC COMMUN ONLY XTOX XTHX", EIG, "bool", "planned"),
    ParamSpec("num_core_eigenstates_mixed_prec_rr", "NUMBER OF CORE EIGEN STATES FOR MIXED PREC RR", EIG, "int", "planned"),
    ParamSpec("npband",                     "NPBAND",                               PAR,      "int",   "planned"),
    ParamSpec("band_paral_opt",             "BAND PARAL OPT",                       PAR,      "bool",  "planned"),
]


# ── DFT-FE v1.0 -> current key migration ────────────────────────────────────
# Published benchmark decks (Summit/Frontier, DFT-FE v1.0) use key spellings
# that were later renamed or removed. deal.II parses with skip_undefined=true,
# so a stale key is silently ignored: the run succeeds and quietly uses a
# different algorithm. Reading a deck therefore rewrites what can be rewritten
# and *warns* about the rest rather than dropping it on the floor.
#
# (section, key) -> (section, key) | None  [None = removed from DFT-FE]
LEGACY_KEYS: dict[tuple[str, str], Optional[tuple[str, str]]] = {
    (EIG, "USE MIXED PREC CHEBY"):                (EIG, "USE SINGLE PREC CHEBY"),
    (EIG, "USE MIXED PREC CGS O"):                (EIG, "USE MIXED PREC XTOX"),
    (EIG, "USE MIXED PREC XTHX SPECTRUM SPLIT"):  (EIG, "USE MIXED PREC XTHX"),
    (EIG, "SPECTRUM SPLIT CORE EIGENSTATES"):     None,  # spectrum splitting removed
}

# Entries owned by the *driver*, not by dftParameters. These are declared in
# utils/runParameters.cc and consumed by the native file-based executable before
# dftParameters is even parsed; in socket mode the calculator decides them
# (`use_device=`, always solver mode GS, no restart files). Injecting them would
# at best be a no-op warning and at worst fight the calculator, so they are
# dropped from a deck in whichever subsection they appear -- v1.0 put USE GPU
# inside `subsection GPU`, current decks put it at top level.
DRIVER_OWNED_KEYS = frozenset({
    "USE GPU",
    "SOLVER MODE",
    "RESTART",
    "RESTART FOLDER",
})

# Geometry and pseudopotential paths are owned by the calculator: the atoms come
# from the ASE Atoms object and the driver writes coordinates.inp /
# domainVectors.inp / pseudo.inp into its scratch folder. A deck that also sets
# them would point DFT-FE back at its own files and silently ignore everything
# ASE sent, so these keys are never injected.
GEOMETRY_KEYS = frozenset({
    "NATOMS",
    "NATOM TYPES",
    "ATOMIC COORDINATES FILE",
    "DOMAIN VECTORS FILE",
    "PSEUDOPOTENTIAL FILE NAMES LIST",
})
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
        # Normalise top-level to "" (not None) so entries built from kwargs
        # compare equal to entries parsed out of a .prm, and so the C++ side
        # sees one spelling for "no subsection".
        section = spec.section or ""
        if spec.dtype in ("int3", "float3"):  # SAMPLING POINTS 1/2/3 etc.
            for i, comp in enumerate(v, start=1):
                entries.append({"section": section, "key": f"{spec.prm_key} {i}", "value": comp})
        else:
            entries.append({"section": section, "key": spec.prm_key,
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
