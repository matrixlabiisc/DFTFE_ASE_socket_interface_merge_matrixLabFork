# ---------------------------------------------------------------------
# Copyright (c) 2017-2025 The Regents of the University of Michigan and DFT-FE
# authors. Part of the DFT-FE code, released under LGPL v2.1 or later.
# ---------------------------------------------------------------------
#
# @author Mehul Darak
#
"""The DFT-FE ASE calculator.

Thin, backend-agnostic ASE ``Calculator``: it builds a request in DFT-FE atomic
units (Bohr, Hartree), hands it to a :class:`~dftfe_ase.backends.base.Backend`,
and converts the response back to ASE units (eV, eV/A, eV/A^3). All transport
concerns live in the backend; all wire-format concerns in :mod:`dftfe_ase.protocol`.
"""

from __future__ import annotations

import logging

import numpy as np
from ase.calculators.calculator import Calculator, all_changes
from ase.stress import full_3x3_to_voigt_6_stress
from ase.units import Bohr, Hartree

from .backends.socket import SocketBackend, DFTFEError

log = logging.getLogger("dftfe_ase")

# Optional DFT-FE parameters passed straight through to the socket driver.
# Wire keys match the C++ socketDriver's parse_request(). (A typed parameter
# table with validation replaces this passthrough in a later phase.)
_OPTIONAL_PARAMS = (
    "mp_grid", "mp_grid_shift", "spin_polarized", "start_magnetization",
    "fermi_temp", "npkpt", "mesh_size", "scf_mixing", "mixing_scheme",
    "polynomial_order", "tolerance", "xc", "atom_ball_radius", "num_bands",
    "orthogonalization_type", "wfc_block_size", "cheby_wfc_block_size",
    "density_quadrature_rule", "use_single_prec_cheby", "smeared_nuclear_charges",
    "use_group_symmetry", "use_time_reversal_symmetry", "mixing_history",
    "max_scf_iterations", "dispersion_correction_type",
    "pseudopotential_calculation", "pseudopotential_filename", "keep_scratch",
    "verbosity",
)


class DFTFE(Calculator):
    """ASE calculator driving a persistent DFT-FE process over a socket."""

    implemented_properties = ["energy", "forces", "stress"]

    def __init__(
        self,
        command: str = "dftfe",
        host: str = "127.0.0.1",
        port: int = 0,
        *,
        backend=None,
        env: dict | None = None,
        cwd: str | None = None,
        compute_forces: bool = True,
        compute_stress: bool = False,
        use_device: bool = False,
        psp_path=None,
        connect_timeout: float = 120.0,
        log_file: str = "dftfe.log",
        verbosity: int | None = None,
        **params,
    ):
        super().__init__()
        self.compute_forces = compute_forces
        self.compute_stress = compute_stress
        self.use_device = use_device
        self.verbosity = verbosity

        # Pseudopotential encoding: dict -> "DICT|El:path|..." ; else path/string.
        if isinstance(psp_path, dict):
            params["pseudopotential_filename"] = "DICT|" + "|".join(
                f"{k}:{v}" for k, v in psp_path.items()
            )
        elif psp_path is not None:
            params["pseudopotential_filename"] = psp_path
        if verbosity is not None:
            params.setdefault("verbosity", verbosity)

        unknown = set(params) - set(_OPTIONAL_PARAMS)
        if unknown:
            raise TypeError(f"unknown DFT-FE parameter(s): {sorted(unknown)}")
        self._params = {k: v for k, v in params.items() if v is not None}

        # Backend is injectable (tests pass a fake); default is the socket backend.
        self.backend = backend if backend is not None else SocketBackend(
            command, host=host, port=port, env=env, cwd=cwd,
            connect_timeout=connect_timeout, log_file=log_file, verbosity=verbosity,
        )

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        want_stress = self.compute_stress or ("stress" in properties)

        request = {
            "cmd": "run",
            "coords": (self.atoms.get_positions() / Bohr).tolist(),
            "cell": (self.atoms.get_cell()[:] / Bohr).tolist(),
            "numbers": self.atoms.get_atomic_numbers().tolist(),
            "pbc": self.atoms.get_pbc().tolist(),
            "compute_forces": bool(self.compute_forces),
            "compute_stress": bool(want_stress),
            "use_device": bool(self.use_device),
        }
        request.update(self._params)

        result = self.backend.compute(request)
        if isinstance(result, dict) and result.get("error"):
            raise DFTFEError(f"DFT-FE reported an error: {result['error']}")

        # Ha -> eV, Ha/Bohr -> eV/A, Ha/Bohr^3 -> eV/A^3.
        self.results["energy"] = float(result["energy"]) * Hartree
        if result.get("forces") is not None:
            self.results["forces"] = np.array(result["forces"], float) * (Hartree / Bohr)
        if want_stress and result.get("stress") is not None:
            stress = np.array(result["stress"], float) * (Hartree / Bohr**3)
            if stress.shape == (3, 3):
                stress = full_3x3_to_voigt_6_stress(stress)
            self.results["stress"] = stress

    def close(self) -> None:
        if getattr(self, "backend", None) is not None:
            self.backend.close()

    def __enter__(self) -> "DFTFE":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
