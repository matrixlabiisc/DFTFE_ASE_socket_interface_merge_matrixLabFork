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
and converts the response back to ASE units (eV, eV/A, eV/A^3).

Launch is zero-config where possible. In order of preference:
  * ``backend=`` — an explicit Backend (tests, custom transports);
  * ``command=`` — an explicit launch command (power-user escape hatch);
  * otherwise **auto**: resolve binaries (kwargs/env/``cluster=``/container/PATH),
    pick a launcher (``launcher=`` / cluster profile / autodetect), and build the
    command at run time — choosing the real vs complex binary from the k-points.
"""

from __future__ import annotations

import logging
import time

import numpy as np
from ase.calculators.calculator import Calculator, all_changes
from ase.stress import full_3x3_to_voigt_6_stress
from ase.units import Bohr, Hartree

from .backends.socket import SocketBackend, DFTFEError
from .binary import select_binary_kind
from .config import get_profile, resolve_binaries
from .launchers import detect_launcher, get_launcher

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
        command=None,
        host: str = "127.0.0.1",
        port: int = 0,
        *,
        backend=None,
        cluster: str | None = None,
        launcher: str | None = None,
        nproc: int | None = None,
        gpus_per_task: int | None = None,
        dftfe_real: str | None = None,
        dftfe_complex: str | None = None,
        bin_dir: str | None = None,
        env: dict | None = None,
        cwd: str | None = None,
        compute_forces: bool = True,
        compute_stress: bool = False,
        use_device: bool = False,
        psp_path=None,
        connect_timeout: float = 120.0,
        log_file: str = "dftfe.log",
        verbosity: int | None = None,
        debug_timing: bool = False,
        **params,
    ):
        super().__init__()
        self.compute_forces = compute_forces
        self.compute_stress = compute_stress
        self.use_device = use_device
        self.verbosity = verbosity
        self.debug_timing = debug_timing
        self.timing = None  # dict of the last call's timing breakdown

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

        # Backend connection settings retained for (possibly lazy) construction.
        self._host = host
        self._port = port
        self._cwd = cwd
        self._connect_timeout = connect_timeout
        self._log_file = log_file

        profile = get_profile(cluster) if cluster else None
        merged_env = dict(profile.env) if profile else {}
        if env:
            merged_env.update(env)
        self._env = merged_env or None

        if backend is not None:
            self.backend = backend
        elif command is not None:
            self.backend = self._make_socket_backend(command)
        else:
            # Auto-launch: defer command construction to run time (needs pbc for
            # the real/complex decision). Resolve binaries + launcher now.
            self.backend = None
            self._binaries = resolve_binaries(
                real=dftfe_real, complex=dftfe_complex, bin_dir=bin_dir, cluster=cluster
            )
            if launcher is not None:
                self._launcher = get_launcher(launcher, gpus_per_task=gpus_per_task)
            elif profile is not None:
                self._launcher = get_launcher(profile.launcher, gpus_per_task=gpus_per_task)
            else:
                self._launcher = detect_launcher(gpus_per_task=gpus_per_task)
            self._nproc = nproc

    # ── launch construction ────────────────────────────────────────────
    def _make_socket_backend(self, command):
        return SocketBackend(
            command, host=self._host, port=self._port, env=self._env, cwd=self._cwd,
            connect_timeout=self._connect_timeout, log_file=self._log_file,
            verbosity=self.verbosity,
        )

    def build_launch_argv(self, pbc):
        """Argv for launching DFT-FE for a system with the given ``pbc``.

        Pure/testable: picks real vs complex from k-points, resolves the binary
        path, and asks the launcher to build the command. Does not spawn anything.
        """
        kind = select_binary_kind(
            self._params.get("mp_grid"), self._params.get("mp_grid_shift"), pbc
        )
        binary = self._binaries.path_for(kind)
        nproc = self._nproc or self._launcher.default_nproc() or 1
        return self._launcher.build_command(binary, nproc)

    def _ensure_backend(self, atoms):
        if self.backend is None:
            argv = self.build_launch_argv(atoms.get_pbc())
            self.backend = self._make_socket_backend(argv)

    # ── ASE Calculator API ─────────────────────────────────────────────
    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        self._ensure_backend(self.atoms)
        t_start = time.perf_counter()
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

        # Timing breakdown (like the legacy debug_timing): the socket round-trip
        # (backend.last_wait_s) is DFT-FE compute + transport; the remainder is
        # ASE-side Python (unit conversion, request build). Overhead is that
        # remainder as a fraction of the whole call.
        total = time.perf_counter() - t_start
        wait = getattr(self.backend, "last_wait_s", None)
        if wait is not None:
            overhead = total - wait
            self.timing = {
                "total_s": total,
                "dftfe_wait_s": wait,
                "ase_overhead_s": overhead,
                "ase_overhead_pct": 100.0 * overhead / total if total > 0 else 0.0,
            }
            if self.debug_timing:
                log.info(
                    "ASE-DFTFE timing: total=%.4fs  DFT-FE wait=%.4fs  "
                    "ASE overhead=%.4fs (%.2f%%)",
                    total, wait, overhead, self.timing["ase_overhead_pct"],
                )
                print(
                    f"[dftfe_ase debug_timing] total={total:.4f}s  "
                    f"DFT-FE wait={wait:.4f}s  ASE overhead={overhead:.4f}s "
                    f"({self.timing['ase_overhead_pct']:.2f}%)",
                    flush=True,
                )

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
