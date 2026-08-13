# ---------------------------------------------------------------------
# Copyright (c) 2017-2025 The Regents of the University of Michigan and DFT-FE
# authors. Part of the DFT-FE code, released under LGPL v2.1 or later.
# ---------------------------------------------------------------------
#
# @author Mehul Darak
#
"""File-based backend: run native file-in/file-out DFT-FE once per compute().

This is the *classic* way to run DFT-FE (write inputs, launch a fresh process,
read the output), wrapped as a :class:`Backend` so the SAME ASE optimizer/MD
integrator can drive it — enabling a true apples-to-apples comparison against
:class:`SocketBackend` with no symmetry tricks and no per-step state reuse.

Template-based: a reference ``parameterFile.prm`` (+ its ``domainVectors.inp`` /
``pseudo.inp`` and a ``coordinates.inp`` giving the atomic-number + valence
columns) supplies all DFT parameters. Each ``compute()`` rewrites only the atom
positions, runs ``<command> <prm>``, and parses energy + **all** per-atom forces
(and stress if present). Periodic systems get fractional coordinates; otherwise
Cartesian Bohr — matching DFT-FE's coordinates.inp convention.
"""

from __future__ import annotations

import logging
import os
import re
import shlex
import shutil
import subprocess
import tempfile
import time

import numpy as np

from .base import Backend
from .socket import DFTFEError

log = logging.getLogger("dftfe_ase.file")


class FileBackend(Backend):
    def __init__(
        self,
        command: str,
        template_dir: str,
        prm_name: str = "parameterFile.prm",
        *,
        env: dict | None = None,
        workroot: str | None = None,
        verbosity: int = 4,
    ):
        self.command = command
        self.template_dir = os.path.abspath(template_dir)
        self.prm_name = prm_name
        self.env = env
        self.workroot = workroot
        self.verbosity = verbosity
        self.last_wait_s = None
        self._z_val = self._read_z_valence()   # (Z, valence) per atom from template
        self._prm = self._prepare_prm()         # GS + ION FORCE + verbose, from template

    # ── template parsing ────────────────────────────────────────────────
    def _read_z_valence(self):
        """Read the atomic-number + valence columns from the template coordinates.inp."""
        rows = []
        with open(os.path.join(self.template_dir, "coordinates.inp")) as fh:
            for line in fh:
                p = line.split()
                if len(p) >= 5:
                    rows.append((int(float(p[0])), p[1]))  # Z, valence(str, preserved)
        return rows

    def _prepare_prm(self) -> str:
        """Force single-point GS + ION FORCE + high verbosity (ASE does the optimization)."""
        text = open(os.path.join(self.template_dir, self.prm_name)).read()
        text = re.sub(r"set\s+SOLVER MODE\s*=.*", "set SOLVER MODE = GS", text)
        if "set VERBOSITY" not in text:
            text = f"set VERBOSITY = {self.verbosity}\n" + text
        if "ION FORCE" not in text:  # ensure forces are printed
            text = text.replace("subsection Optimization",
                                "subsection Optimization\n    set ION FORCE = true", 1)
        return text

    # ── coordinates.inp writer (matches DFT-FE convention) ──────────────
    def _write_coordinates(self, work, coords_bohr, cell_bohr, pbc):
        coords = np.asarray(coords_bohr, float)
        cell = np.asarray(cell_bohr, float)
        periodic = bool(np.any(pbc))
        if periodic:
            rows = coords @ np.linalg.inv(cell)  # cell rows = vectors
        else:
            # Cartesian Bohr measured from the DOMAIN CENTRE, not the corner.
            # dft.cc prints this branch under "Cartesian coordinates of atoms
            # (origin at center of domain)" and uses atomLocations verbatim --
            # convertToCellCenteredCartesianCoordinates() is only called on the
            # periodic path. dftfeWrapper::reinit writes the same frame, shifting
            # by -sum(cell)/2, and io.read_dftfe_atoms undoes exactly that on the
            # way back. Writing corner-origin here put every atom half a cell
            # diagonal away from where the caller meant, which for a molecule
            # centred in its box lands it outside the domain entirely.
            rows = coords - cell.sum(axis=0) / 2.0
        with open(os.path.join(work, "coordinates.inp"), "w") as fh:
            for (z, val), r in zip(self._z_val, rows):
                fh.write(f"{z} {val} {r[0]:.14e} {r[1]:.14e} {r[2]:.14e}\n")

    # ── output parsing ──────────────────────────────────────────────────
    @staticmethod
    def _parse(text, natoms):
        energy = None
        for m in re.finditer(r"Total free energy:\s*([-\d.eE+]+)", text):
            energy = float(m.group(1))
        forces = []
        lines = text.splitlines()
        for i, ln in enumerate(lines):
            if "Ion forces (Hartree/Bohr)" in ln:
                forces = []
                for l2 in lines[i + 1:]:
                    m = re.match(
                        r"\s*(?:AtomId\s+)?\d+\s*:?\s+([-\d.eE+]+)\s+([-\d.eE+]+)\s+([-\d.eE+]+)",
                        l2)
                    if m:
                        forces.append([float(m.group(1)), float(m.group(2)), float(m.group(3))])
                    elif forces and (l2.strip().startswith("---")
                                     or "Maximum" in l2 or l2.strip() == ""):
                        break
        if energy is None:
            raise DFTFEError("FileBackend: could not parse 'Total free energy' from DFT-FE output")
        if forces and len(forces) != natoms:
            forces = forces[:natoms]  # guard against trailing stray matches
        return energy, (forces or None)

    # ── Backend API ─────────────────────────────────────────────────────
    def compute(self, request: dict) -> dict:
        root = self.workroot or tempfile.gettempdir()
        os.makedirs(root, exist_ok=True)
        work = tempfile.mkdtemp(prefix="dftfe_file_", dir=root)
        # inputs
        for fn in ("domainVectors.inp", "domainBoundingVectors.inp", "pseudo.inp"):
            src = os.path.join(self.template_dir, fn)
            if os.path.exists(src):
                shutil.copy(src, work)
        # copy any pseudopotential files referenced
        for fn in os.listdir(self.template_dir):
            if fn.lower().endswith((".upf", ".psp8")):
                shutil.copy(os.path.join(self.template_dir, fn), work)
        with open(os.path.join(work, self.prm_name), "w") as fh:
            fh.write(self._prm)
        self._write_coordinates(work, request["coords"], request["cell"], request["pbc"])

        argv = self.command if isinstance(self.command, (list, tuple)) else shlex.split(self.command)
        argv = list(argv) + [self.prm_name]
        env = None
        if self.env:
            env = dict(os.environ)
            env.update(self.env)

        t0 = time.perf_counter()
        proc = subprocess.run(argv, cwd=work, env=env, capture_output=True, text=True)
        self.last_wait_s = time.perf_counter() - t0
        text = proc.stdout + "\n" + proc.stderr
        with open(os.path.join(work, "dftfe.out"), "w") as fh:
            fh.write(text)
        natoms = len(request["numbers"])
        energy, forces = self._parse(text, natoms)
        resp = {"energy": energy, "compute_time": self.last_wait_s}
        if forces is not None:
            resp["forces"] = forces
        return resp

    def close(self) -> None:  # nothing persistent to tear down
        pass
