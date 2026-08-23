# ---------------------------------------------------------------------
# Part of the DFT-FE code, released under LGPL v2.1 or later.
# ---------------------------------------------------------------------
#
# @author Mehul Darak
#
"""A fake DFT-FE binary for testing — speaks the socket protocol, no physics.

Launched exactly like the real binary (``mock_dftfe.py --socket host:port``),
it connects back to the Python server, sends the ``READY`` handshake, then
answers each request with deterministic, unit-checkable values:

    energy (Ha)      = -1.0 * sum(atomic_numbers)
    forces (Ha/Bohr) = atom i -> [0.001*(i+1), 0.002*(i+1), -0.003*(i+1)]
    stress (Ha/Bohr^3) = diag(0.01, 0.02, 0.03)

Framing is hand-rolled (stdlib only) on purpose: an independent implementation
of the wire format is a stronger test double than reusing the code under test.

Test hooks via environment variables:
    MOCK_DFTFE_ERROR=1   -> reply with an {"error": ...} frame
    MOCK_DFTFE_NOCONNECT=1 -> exit immediately without connecting
    MOCK_DFTFE_ENGINE=1  -> keep DFT-FE's own copy of the geometry (see below)
    MOCK_DFTFE_SCRATCH=<dir>   -> write the wrapped deck (coordinates.inp) there
    MOCK_DFTFE_ENGINE_LOG=<f>  -> one JSON line per frame, for the tests to read
    MOCK_DFTFE_NO_MINIMUM_IMAGE=1 -> engine WITHOUT the minimum-image fold, i.e.
                                     DFT-FE as it behaved before the fix

The engine model (MOCK_DFTFE_ENGINE=1)
--------------------------------------
Everything above this line is physics-free bookkeeping. This part is not: it is a
deliberate model of what the C++ side does with coordinates, because the bug that
killed job 8761021 lives entirely in that bookkeeping and nowhere in Python.

    reinit()                    wraps the incoming positions into the cell and
      dftfeWrapper.cc:710-714   writes them to coordinates.inp as fractionals
      dftfeWrapper.cc:743-750
    every later frame           disp = new_coords - getAtomPositionsCart()
      socket_interface.cc:800
    updateAtomPositions()       minimum image on that disp, periodic axes only
      dftfeWrapper.cc:1342      -- THE FIX under test here
    updateAtomPositionsAnd-     applies it and wraps the positions again
      MoveMesh, moveAtoms.cc:269-298

WHAT THIS DOES AND DOES NOT PROVE. It is a re-implementation of the C++ rule in
Python, not the C++ rule itself, and no pytest can reach the C++ (there is no
unit-test seam in the library -- every ctest is a full DFT run on a deck). So
these tests catch a Python-side change that breaks the contract, and they pin
down what the C++ has to do, but they will keep passing if someone edits
dftfeWrapper.cc. test_wrap_unwrap.py::test_cpp_fold_still_matches_this_model is
the tripwire for that half; read its docstring before trusting any of this.
"""

import json
import math
import os
import socket
import sys


# ── the coordinate model (mirrors the C++ named above) ──────────────────
#
# stdlib only, like the rest of this file: an independent implementation is a
# stronger test double than importing numpy and, for the fold, than importing
# the very convention under test.


def _solve3(a, b):
    """Solve the 3x3 system a x = b by Gaussian elimination with partial pivoting.

    Stands in for LAPACK dgesv, which is what dftUtils::getFractionalCoordinates
    (include/dftfe/dftUtils.h:123-151) calls.
    """
    m = [list(a[r]) + [b[r]] for r in range(3)]
    for col in range(3):
        piv = max(range(col, 3), key=lambda r: abs(m[r][col]))
        if abs(m[piv][col]) < 1e-300:
            raise ValueError("singular lattice vectors")
        m[col], m[piv] = m[piv], m[col]
        for r in range(col + 1, 3):
            f = m[r][col] / m[col][col]
            for c in range(col, 4):
                m[r][c] -= f * m[col][c]
    x = [0.0, 0.0, 0.0]
    for r in (2, 1, 0):
        s = m[r][3] - sum(m[r][c] * x[c] for c in range(r + 1, 3))
        x[r] = s / m[r][r]
    return x


def fractional_coordinates(cell, vec):
    """Components of `vec` in the basis of the lattice vectors, which are ROWS of `cell`.

    The convention is not a free choice -- it is the one dgesv imposes on
    getFractionalCoordinates. That function passes cellVectorsFlattened with
    flat[3*i+j] = cell[i][j] (dftfeWrapper.cc:630-632) to a column-major solver
    with LDA=3, so LAPACK reads A(r,c) = flat[r+3c] = cell[c][r], i.e. it solves
    cell^T f = vec and returns coefficients of the lattice ROWS:

        vec = f0*cell[0] + f1*cell[1] + f2*cell[2]

    which is what makes `disp -= image * cell[idim]` the exact inverse of the
    fold. Getting this transposed is silent on any cell whose lattice matrix is
    symmetric (every orthorhombic one) and wrong on every other.
    """
    a = [[cell[c][r] for c in range(3)] for r in range(3)]
    return _solve3(a, list(vec))


def cartesian_coordinates(cell, frac):
    return [sum(frac[i] * cell[i][j] for i in range(3)) for j in range(3)]


def wrap_into_cell(cell, pbc, coord, tol=1e-8):
    """Fold a POINT into the cell -- model of internal::wrapAtomsAcrossPeriodicBc.

    moveAtoms.cc:80-119. Note it is a single conditional shift, not a floor():
    one lattice vector out is folded, further out is an error. Non-periodic
    directions are left alone -- there is no image to fold to there.
    """
    frac = fractional_coordinates(cell, coord)
    for i in range(3):
        if not pbc[i]:
            continue
        if frac[i] < -tol:
            frac[i] += 1.0
        elif frac[i] > 1.0 + tol:
            frac[i] -= 1.0
        if not (-2.0 * tol < frac[i] < 1.0 + 2.0 * tol):
            raise ValueError(
                "atom position does not lie inside the cell after wrapping "
                f"across the periodic boundary (fractional {frac[i]} along {i})"
            )
    return frac


def _round_half_away_from_zero(x):
    """std::round(), which is NOT Python's round().

    Python rounds halves to even, C++ rounds them away from zero. They differ
    only on an exact tie, but a tie here is a displacement of exactly half a
    cell -- the one input where the fold direction is genuinely ambiguous -- and
    a model that disagreed with the code there would be worse than no model.
    """
    return math.copysign(math.floor(abs(x) + 0.5), x)


def minimum_image_displacement(cell, pbc, disp):
    """Minimum image on a DISPLACEMENT -- model of dftfeWrapper.cc:1342 onwards.

    Returns (folded_disp, image, residual_frac).

    Three properties are load-bearing and each has a test in test_wrap_unwrap.py:

    * the fold goes through fractional coordinates, so subtracting one lattice
      vector moves ALL THREE Cartesian components. A per-component fold is
      right for an orthorhombic cell and silently destroys a triclinic one.
    * a zero image vector returns the caller's floats untouched -- no round trip
      through the solve, so an ordinary step is bit-identical to no fold at all.
    * ANY whole number of lattice vectors folds. What arrives is not the
      optimizer's step but step + n*a, where n is how far the driver's
      never-wrapped copy has drifted from DFT-FE's wrapped one; n grows by one
      per net crossing of the same face, so a diffusing ion reaches 2 and
      beyond. Subtracting whole lattice vectors is an exact symmetry, so the
      fold is correct for every n, and refusing n >= 2 would abort a correct
      trajectory mid-run.
    """
    frac = fractional_coordinates(cell, disp)
    image = [0, 0, 0]
    residual = [0.0, 0.0, 0.0]
    for i in range(3):
        if not pbc[i]:
            continue
        img = _round_half_away_from_zero(frac[i])
        image[i] = int(img)
        residual[i] = frac[i] - img
    if not any(image):
        return list(disp), image, residual
    folded = list(disp)
    for i in range(3):
        if image[i] == 0:
            continue
        for j in range(3):
            folded[j] -= image[i] * cell[i][j]
    return folded, image, residual


class PositionMismatch(RuntimeError):
    """Raised when DFT-FE's copy of the geometry is not what the driver asked for."""


def verify_positions_against_driver(cell, pbc, before, after, raw, frac_tol=1e-8):
    """Model of the invariant in dftfeWrapper::updateAtomPositions().

    after == before + raw, modulo whole lattice vectors along periodic
    directions and exactly along open ones. Checked against the RAW request, not
    the folded displacement, so the fold is allowed to change the answer by whole
    lattice vectors and by nothing else.

    Note what this does NOT check, because it took writing the model to see it:
    it does not detect the failure of job 8761021. There the driver asked for a
    24 Bohr step and DFT-FE applied a 24 Bohr step, so the two copies agreed
    afterwards and this invariant passes. What it catches is DFT-FE not doing
    what was asked -- a fold that subtracts something which is not a lattice
    vector (the naive per-Cartesian fold on a triclinic cell errs by 11.8 Bohr
    on the shipped ReS2 cell and would fire here), a fold applied along an open
    axis, or a wrap or mesh move that loses an atom.
    """
    worst_residual = 0.0
    worst_atom = 0
    worst_dim = 0
    worst_deviation = 0.0
    for i in range(len(after)):
        deviation = [after[i][j] - before[i][j] - raw[i][j] for j in range(3)]
        frac = fractional_coordinates(cell, deviation)
        for d in range(3):
            if pbc[d]:
                residual = abs(frac[d] - _round_half_away_from_zero(frac[d]))
            else:
                residual = abs(frac[d])
            if residual > worst_residual:
                worst_residual = residual
                worst_atom = i
                worst_dim = d
                worst_deviation = math.sqrt(sum(v * v for v in deviation))
    if worst_residual >= frac_tol:
        raise PositionMismatch(
            "atom %d, direction %d is off by %.3e of a lattice vector "
            "(tolerance %.1e), total deviation %.6f Bohr"
            % (worst_atom, worst_dim, worst_residual, frac_tol, worst_deviation))
    return worst_residual


class Engine:
    """DFT-FE's own copy of the geometry, and what it does to an incoming frame."""

    def __init__(self, cell, pbc, numbers, fold=True, verify=True, corrupt=None):
        self.cell = cell
        self.pbc = pbc
        self.numbers = numbers
        self.fold = fold
        # verify=False and corrupt are test hooks: corrupt injects a defect into
        # the move so the invariant can be shown to fire, which is the only way
        # to know the check is load bearing rather than decorative.
        self.verify = verify
        self.corrupt = corrupt
        self.frac = None
        self.coords = None

    def reinit(self, coords):
        """First frame: wrap into the cell and keep the wrapped copy."""
        self.frac = [wrap_into_cell(self.cell, self.pbc, c) for c in coords]
        self.coords = [cartesian_coordinates(self.cell, f) for f in self.frac]

    def update(self, coords):
        """Later frames: difference, fold, apply, wrap. Returns a log record."""
        raw = [[coords[i][j] - self.coords[i][j] for j in range(3)]
               for i in range(len(coords))]
        max_raw = max((abs(v) for d in raw for v in d), default=0.0)
        applied = []
        for d in raw:
            if self.fold:
                folded, _image, _res = minimum_image_displacement(
                    self.cell, self.pbc, d)
            else:
                folded = list(d)
            applied.append(folded)
        max_applied = max((abs(v) for d in applied for v in d), default=0.0)
        before = [list(c) for c in self.coords]
        moved = [[self.coords[i][j] + applied[i][j] for j in range(3)]
                 for i in range(len(coords))]
        if self.corrupt is not None:
            moved = self.corrupt(moved)
        self.frac = [wrap_into_cell(self.cell, self.pbc, c) for c in moved]
        self.coords = [cartesian_coordinates(self.cell, f) for f in self.frac]
        residual = None
        if self.verify:
            residual = verify_positions_against_driver(
                self.cell, self.pbc, before, self.coords, raw)
        return {"max_disp_raw": max_raw, "max_disp": max_applied,
                "invariant_residual": residual}

    def write_deck(self, path):
        """coordinates.inp as dftfeWrapper.cc:743-760 writes it: Z, valence, fractionals."""
        with open(path, "w") as fh:
            for z, f in zip(self.numbers, self.frac):
                fh.write("%d %d %.16f %.16f %.16f\n" % (z, 0, f[0], f[1], f[2]))


def parse_socket_arg(argv):
    for i, a in enumerate(argv):
        if a == "--socket" and i + 1 < len(argv):
            host, port = argv[i + 1].rsplit(":", 1)
            return host, int(port)
    raise SystemExit("mock_dftfe: missing --socket host:port")


def main():
    if os.environ.get("MOCK_DFTFE_NOCONNECT") == "1":
        return
    host, port = parse_socket_arg(sys.argv[1:])
    sock = socket.create_connection((host, port), timeout=30)
    sock.sendall(b"READY\n")

    buf = bytearray()

    engine = None
    frame = 0
    engine_on = os.environ.get("MOCK_DFTFE_ENGINE") == "1"
    engine_log = os.environ.get("MOCK_DFTFE_ENGINE_LOG")
    scratch = os.environ.get("MOCK_DFTFE_SCRATCH")
    fold = os.environ.get("MOCK_DFTFE_NO_MINIMUM_IMAGE") != "1"

    def record(entry):
        if not engine_log:
            return
        with open(engine_log, "a") as fh:
            fh.write(json.dumps(entry) + "\n")

    def read_line():
        while b"\n" not in buf:
            data = sock.recv(65536)
            if not data:
                return None
            buf.extend(data)
        nl = buf.index(b"\n")
        line = bytes(buf[:nl])
        del buf[: nl + 1]
        return line

    while True:
        line = read_line()
        if line is None:
            break
        req = json.loads(line.decode("utf-8"))
        if req.get("cmd") == "exit":
            break

        if os.environ.get("MOCK_DFTFE_ERROR") == "1":
            sock.sendall((json.dumps({"error": "mock failure"}) + "\n").encode())
            continue

        numbers = req.get("numbers", [])
        n = len(numbers)

        if engine_on:
            coords = req.get("coords", [])
            try:
                if engine is None:
                    engine = Engine(req.get("cell"), req.get("pbc"), numbers,
                                    fold=fold)
                    engine.reinit(coords)
                    info = {"max_disp_raw": 0.0, "max_disp": 0.0}
                    if scratch:
                        engine.write_deck(os.path.join(scratch, "coordinates.inp"))
                else:
                    info = engine.update(coords)
            except ValueError as exc:
                record({"frame": frame, "error": str(exc)})
                sock.sendall((json.dumps({"error": str(exc)}) + "\n").encode())
                frame += 1
                continue
            info.update({"frame": frame, "sent_coords": coords,
                         "engine_coords": engine.coords,
                         "engine_frac": engine.frac})
            record(info)
            frame += 1

        energy = -1.0 * sum(numbers)
        resp = {"energy": energy}
        if req.get("compute_forces", True):
            resp["forces"] = [
                [0.001 * (i + 1), 0.002 * (i + 1), -0.003 * (i + 1)] for i in range(n)
            ]
        if req.get("compute_stress", False):
            resp["stress"] = [[0.01, 0.0, 0.0], [0.0, 0.02, 0.0], [0.0, 0.0, 0.03]]
        resp["compute_time"] = 0.0  # mock does no real compute
        sock.sendall((json.dumps(resp) + "\n").encode())

    sock.close()


if __name__ == "__main__":
    main()
