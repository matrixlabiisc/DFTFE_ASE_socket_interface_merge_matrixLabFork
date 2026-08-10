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
from .params import (BY_KWARG, DRIVER_OWNED_KEYS, GEOMETRY_KEYS, LEGACY_KEYS,
                     to_prm_entries)

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


def read_prm(path):
    """Parse a DFT-FE ``.prm`` into ``{section, key, value}`` entries.

    ``section`` is the *innermost* subsection name (``""`` at top level), which
    is what the C++ injector matches on. Three classes of entry are handled
    specially so that pointing the calculator at a real benchmark deck produces
    the run the deck describes:

    * geometry / pseudopotential paths (:data:`~dftfe_ase.params.GEOMETRY_KEYS`)
      are dropped -- the atoms come from the ASE ``Atoms`` object;
    * DFT-FE v1.0 key spellings are rewritten via
      :data:`~dftfe_ase.params.LEGACY_KEYS`;
    * anything removed from DFT-FE is dropped with a warning.

    The warnings matter: deal.II parses ``.prm`` files with
    ``skip_undefined=true``, so an unrecognised key costs you nothing at parse
    time and silently changes the algorithm at run time.
    """
    import re

    entries, stack = [], []
    for line in open(path):
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
        if not m:
            continue

        section = stack[-1] if stack else ""
        key, value = m.group(1).strip(), m.group(2).strip()

        if key in GEOMETRY_KEYS:
            log.debug("%s: ignoring geometry entry %r (atoms come from ASE)", path, key)
            continue

        if key in DRIVER_OWNED_KEYS:
            log.debug("%s: ignoring driver entry %r (set by the calculator)", path, key)
            continue

        if (section, key) in LEGACY_KEYS:
            replacement = LEGACY_KEYS[(section, key)]
            if replacement is None:
                log.warning(
                    "%s: '%s' (subsection '%s') no longer exists in DFT-FE; ignored. "
                    "The run will use current defaults for this behaviour.",
                    path, key, section,
                )
                continue
            section, new_key = replacement
            log.warning("%s: '%s' renamed to '%s'; using the current spelling.",
                        path, key, new_key)
            key = new_key

        entries.append({"section": section, "key": key, "value": value})
    return entries


# Back-compat alias for the private name this used to have.
_parse_prm_file = read_prm


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
        launcher_args=(),
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
        # Backend.start() (mpiexec spawn + MPI init + handshake) runs lazily on
        # the first compute(). Track it so its cost is charged to startup once,
        # and never to per-step interface overhead.
        self._startup_counted = False

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
            # launcher_args land between the launcher's own flags and the binary
            # -- the slot for site placement options and rank wrappers, e.g.
            # Aurora's ["--ppn", "12", "gpu_tile_compact.sh"], which is how one
            # rank gets bound to each of the 12 GPU tiles on a node.
            launcher_kwargs = dict(gpus_per_task=gpus_per_task,
                                   extra_args=tuple(launcher_args))
            if launcher is not None:
                self._launcher = get_launcher(launcher, **launcher_kwargs)
            elif profile is not None:
                self._launcher = get_launcher(profile.launcher, **launcher_kwargs)
            else:
                self._launcher = detect_launcher(**launcher_kwargs)
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

        # Positions are sent unwrapped. DFT-FE folds out-of-cell atoms into the
        # cell itself, in reinit() and in updateAtomPositionsAndMoveMesh(),
        # using its own periodic wrap. Keeping the ASE-side positions
        # continuous is what lets LBFGS/NEB build coherent optimizer state:
        # wrapping here would make a boundary crossing look like a
        # full-lattice-vector jump to the optimizer.
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

        # Timing breakdown.
        #
        # On the first calculate() of a process, backend.compute() lazily calls
        # backend.start(), which spawns mpiexec, waits for MPI init across every
        # rank, and completes the TCP handshake. That cost lands inside this
        # timed window but outside last_wait_s, so it would otherwise be counted
        # as Python overhead. It is NOT interface overhead: a native DFT-FE run
        # spawns the identical mpiexec and pays the same launch, it simply does
        # not appear in DFT-FE's internal timer, which starts after MPI init.
        # Charging it to the interface would report a cost the interface does
        # not introduce, so it is subtracted out and reported separately.
        #
        #   startup      = backend.startup_s        (launch; common to both arms)
        #   python       = total - round_trip - startup  (unit conv, request build,
        #                                                 response parse)
        #   streaming    = round_trip - dft_compute (serialize + send, recv)
        #   interface    = python + streaming       (what ASE actually adds)
        #
        # Only `interface` recurs every step; `startup` is paid once per process.
        total = time.perf_counter() - t_start
        round_trip = getattr(self.backend, "last_wait_s", None)
        if round_trip is not None:
            dft_compute = result.get("compute_time") if isinstance(result, dict) else None
            b = self.backend
            if not self._startup_counted:
                self._startup_counted = True
                spawn = getattr(b, "spawn_s", 0.0) or 0.0
                boot = getattr(b, "boot_s", 0.0) or 0.0
                bind = getattr(b, "bind_s", 0.0) or 0.0
                shake = getattr(b, "handshake_s", 0.0) or 0.0
            else:
                spawn = boot = bind = shake = 0.0

            streaming = (round_trip - dft_compute) if dft_compute is not None else None
            python_overhead = total - round_trip - (bind + spawn + boot + shake)

            # BUCKET 1 -- cost a native DFT-FE run pays too. Not attributable to
            # the interface: same mpiexec, same rank count, same MPI_Init, same
            # solver. Invisible in a native log only because DFT-FE's internal
            # timer starts after MPI_Init.
            dftfe_s = spawn + boot + (dft_compute or 0.0)

            # BUCKET 2 -- cost that exists ONLY because of the interface.
            interface_s = bind + shake + (streaming or 0.0) + python_overhead

            self.timing = {
                "total_s": total,
                # bucket 1: also incurred by native
                "dftfe_s": dftfe_s,
                "dftfe_spawn_s": spawn,
                "dftfe_boot_s": boot,
                "dft_compute_s": dft_compute,
                # bucket 2: interface-only
                "interface_s": interface_s,
                "socket_bind_s": bind,
                "handshake_s": shake,
                "streaming_overhead_s": streaming,
                "python_overhead_s": python_overhead,
                # ratio of bucket 2 to the work itself
                "interface_pct": 100.0 * interface_s / dftfe_s if dftfe_s > 0 else 0.0,
                # legacy aliases
                "round_trip_s": round_trip,
                "ase_overhead_s": python_overhead,
                "interface_overhead_s": interface_s,
                "interface_overhead_pct": 100.0 * interface_s / dftfe_s if dftfe_s > 0 else 0.0,
            }
            if self.debug_timing:
                dc = f"{dft_compute:.4f}" if dft_compute is not None else "n/a"
                st = f"{streaming:.6f}" if streaming is not None else "n/a"
                msg = (
                    f"[dftfe_ase debug_timing] total={total:.4f}s\n"
                    f"  DFT-FE (native pays too) = {dftfe_s:.4f}s "
                    f"[spawn={spawn:.4f} boot={boot:.4f} compute={dc}]\n"
                    f"  INTERFACE (ours)         = {interface_s:.6f}s "
                    f"[bind={bind:.6f} handshake={shake:.6f} "
                    f"streaming={st} python={python_overhead:.6f}] "
                    f"= {self.timing['interface_pct']:.4f}% of DFT-FE"
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
