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
from .params import BY_KWARG, to_prm_entries

log = logging.getLogger("dftfe_ase")

# Typed params kept on the existing per-key C++ path (verified "wired"). Note:
# mixing_history + dispersion_correction_type are intentionally NOT here — they
# route through the generic .prm-override path (mixing_history was mis-mapped to
# LBFGS HISTORY; the generic path sends the correct MIXING HISTORY).
_OPTIONAL_PARAMS = (
    "mp_grid", "mp_grid_shift", "spin_polarized", "start_magnetization",
    "fermi_temp", "npkpt", "mesh_size", "scf_mixing", "mixing_scheme",
    "polynomial_order", "tolerance", "xc", "atom_ball_radius", "num_bands",
    "orthogonalization_type", "wfc_block_size", "cheby_wfc_block_size",
    "density_quadrature_rule", "use_single_prec_cheby", "smeared_nuclear_charges",
    "use_group_symmetry", "use_time_reversal_symmetry",
    "max_scf_iterations",
    "pseudopotential_calculation", "pseudopotential_filename", "keep_scratch",
    "verbosity",
)

# Params routed through the generic .prm-override path ("planned" in params.py):
# the long tail + the ones the typed sed mishandled.
_GENERIC_KWARGS = {k for k, s in BY_KWARG.items() if s.status == "planned"}


def _serialize_overrides(entries):
    return "@@@".join(f"{e['section'] or ''}|||{e['key']}|||{e['value']}" for e in entries)


def _parse_prm_file(path):
    """Parse a DFT-FE .prm into generic {section,key,value} entries (option A)."""
    import re
    entries, stack = [], []
    for line in open(path):
        s = line.strip()
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
            entries.append({"section": stack[-1] if stack else "",
                            "key": m.group(1).strip(), "value": m.group(2).strip()})
    return entries


def _extra_prm_entries(extra):
    """extra_prm: {key: value} | {(section, key): value} | {'section|||key': value}."""
    out = []
    for k, v in extra.items():
        if isinstance(k, tuple):
            section, key = k
        elif "|||" in k:
            section, key = k.split("|||", 1)
        else:
            spec = next((s for s in BY_KWARG.values() if s.prm_key == k), None)
            section, key = (spec.section or "", spec.prm_key) if spec else ("", k)
        out.append({"section": section or "", "key": key, "value": v})
    return out


class DFTFE(Calculator):
    """ASE calculator driving a persistent DFT-FE process over a socket."""

    implemented_properties = ["energy", "forces", "stress"]

    def __init__(
        self,
        command=None,
        bind_host: str = "0.0.0.0",
        advertise_host: str | None = None,
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
        extra_prm: dict | None = None,
        prm_file: str | None = None,
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

        unknown = set(params) - set(_OPTIONAL_PARAMS) - _GENERIC_KWARGS
        if unknown:
            raise TypeError(f"unknown DFT-FE parameter(s): {sorted(unknown)}")
        self._params = {k: v for k, v in params.items()
                        if v is not None and k in _OPTIONAL_PARAMS}
        # Generic .prm overrides: "planned" kwargs (B) + extra_prm dict + full
        # .prm file (A) -> one {section,key,value} list applied by the C++
        # generic injector. No per-parameter sed, no silent defaulting.
        entries = to_prm_entries({k: v for k, v in params.items()
                                  if v is not None and k in _GENERIC_KWARGS})
        if extra_prm:
            entries += _extra_prm_entries(extra_prm)
        if prm_file:
            entries += _parse_prm_file(prm_file)
        self._prm_overrides = _serialize_overrides(entries)

        # Backend connection settings retained for (possibly lazy) construction.
        self._bind_host = bind_host
        self._advertise_host = advertise_host
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
            command, bind_host=self._bind_host, advertise_host=self._advertise_host,
            port=self._port, env=self._env, cwd=self._cwd,
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
        if self._prm_overrides:
            request["prm_overrides"] = self._prm_overrides

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

        # Timing breakdown. The interface overhead in a multi-step workflow has
        # TWO parts, both of which recur every step:
        #   ase_overhead  = total - round_trip     (Python: unit conv, request build)
        #   streaming     = round_trip - dft_compute  (serialize + send positions,
        #                                               recv forces; MPI bcast; etc.)
        # dft_compute (pure DFT-FE work) is reported by the C++ driver so it can
        # be subtracted out. interface_overhead = ase_overhead + streaming.
        total = time.perf_counter() - t_start
        round_trip = getattr(self.backend, "last_wait_s", None)
        if round_trip is not None:
            dft_compute = result.get("compute_time") if isinstance(result, dict) else None
            ase_overhead = total - round_trip
            streaming = (round_trip - dft_compute) if dft_compute is not None else None
            interface = ase_overhead + (streaming or 0.0)
            self.timing = {
                "total_s": total,
                "round_trip_s": round_trip,
                "dft_compute_s": dft_compute,
                "streaming_overhead_s": streaming,
                "ase_overhead_s": ase_overhead,
                "interface_overhead_s": interface,
                "interface_overhead_pct": 100.0 * interface / total if total > 0 else 0.0,
            }
            if self.debug_timing:
                stream_str = f"{streaming:.4f}s" if streaming is not None else "n/a (old binary)"
                dc_str = f"{dft_compute:.4f}s" if dft_compute is not None else "n/a"
                msg = (
                    f"[dftfe_ase debug_timing] total={total:.4f}s | "
                    f"DFT compute={dc_str} | streaming={stream_str} | "
                    f"ASE={ase_overhead:.4f}s | interface overhead="
                    f"{interface:.4f}s ({self.timing['interface_overhead_pct']:.2f}%)"
                )
                log.info(msg)
                print(msg, flush=True)

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
