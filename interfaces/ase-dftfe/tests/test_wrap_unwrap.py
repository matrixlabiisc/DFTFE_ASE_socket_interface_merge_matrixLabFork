# ---------------------------------------------------------------------
# Part of the DFT-FE code, released under LGPL v2.1 or later.
# ---------------------------------------------------------------------
#
# @author Mehul Darak
#
"""Regression tests for the wrapped-deck / unwrapped-socket mismatch.

THE BUG. The interface sends positions unwrapped along periodic axes on purpose
(calculator.py:519-527), so that an atom crossing a cell face does not look like
a full-lattice-vector jump to the optimizer holding the trajectory. DFT-FE wraps
its own copy (dftfeWrapper.cc:710-714 at reinit, moveAtoms.cc:269-298 on every
move). Those two conventions describe the same physical point, and everything is
fine until something SUBTRACTS them: socket_interface.cc:800 and
MDIEngine.cpp:619-621 both build the step as `new_coords - getAtomPositionsCart()`.
Once an atom sits outside the cell in the driver's frame, that difference is a
whole lattice vector.

A fresh run never sees it -- the deck is written at frame 0 with every atom
inside, so the wrap is the identity. A RESTART from a mid-trajectory frame does:
ASE frame 3 of job 8761021 carried atom 282 at fractional x = 1.000442,
coordinates.inp got 0.000442, and every ionic step then applied a 24.02 Bohr
displacement against a 24.0081 Bohr cell edge. That is 48x moveAtoms.cc:258's
0.5 Bohr threshold, so DFT-FE rebuilt the vself bins from scratch every step
instead of updating their boundary conditions, and died in the third rebuild
(rank 224, SIGSEGV, 32 nodes).

THE FIX under test is a minimum image on the displacement, periodic axes only,
in dftfeWrapper::updateAtomPositions (dftfeWrapper.cc:1342). Every external
driver funnels through it: socket, MDI, NEB.

WHAT THESE TESTS PROVE, AND WHAT THEY DO NOT. The fix is C++, and there is no
seam to unit-test it through: the library exposes no test target, and every
ctest is a full DFT run on a deck driven by DEAL_II_PICKUP_TESTS, which needs a
binary and compute nodes. So the tests below drive the real calculator, the real
socket protocol and the real wire format against a MODEL of DFT-FE's coordinate
bookkeeping (mock_dftfe.Engine), written from the C++ line by line.

That means:

  * they will catch a Python-side change that breaks the contract the C++ relies
    on -- e.g. someone "fixing" the interface by wrapping before send;
  * they pin down, executably, what the C++ has to do, including the two ways a
    naive implementation gets it wrong (per-Cartesian folding, and folding a
    non-periodic axis);
  * test_the_bug_as_it_was shows the model is discriminating: with the fold
    switched off it reproduces the 24 Bohr displacement, so a test passing here
    is a test that would have failed before the fix;
  * BUT they do not execute a single line of dftfeWrapper.cc. If someone edits
    the C++ fold, everything here still passes.
    test_cpp_fold_still_matches_this_model is the tripwire for exactly that, and
    it is the only test in this file that reads the C++ at all.

The end-to-end confirmation that no login-node test can give is the one PROJECT.md
17.6 names: replay the 6-evaluation state in relax_state_densityreuse, which
reproduces the original crash in about seven minutes on 32 nodes. That needs a
built binary and an approved allocation.
"""

import importlib.util
import json
import os

import numpy as np
import pytest
from ase import Atoms
from ase.units import Bohr

from dftfe_ase import DFTFE

_HERE = os.path.dirname(os.path.abspath(__file__))


def _load_mock():
    """Import mock_dftfe.py by path (it is not a package, and is not collected)."""
    spec = importlib.util.spec_from_file_location(
        "mock_dftfe_model", os.path.join(_HERE, "mock_dftfe.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


mock = _load_mock()

# Orthorhombic, the geometry that actually crashed: the LLZO x edge of job
# 8761021, in Bohr. The atom that killed it sat at fractional x = 1.000442.
ORTHO = np.array([[24.0081, 0.0, 0.0],
                  [0.0, 24.0081, 0.0],
                  [0.0, 0.0, 25.0]])

# Fully non-orthogonal, in the spirit of testsGPU/pseudopotential/real/
# domainVectors_ReS2.inp. Right-handed (det > 0), which is all dft.cc:517-546
# requires of a domain. Every off-diagonal element here is a component a
# per-Cartesian fold would get wrong.
TRICLINIC = np.array([[12.0, 0.0, 0.0],
                      [4.0, 11.0, 0.0],
                      [3.0, 2.5, 13.0]])

PERIODIC = [True, True, True]


# ══ 1. the fold arithmetic ═════════════════════════════════════════════
#
# These call the model directly. They are the five cases a C++ unit test would
# cover if the library had a seam to hang one on; see the module docstring.

def test_orthorhombic_fold_recovers_the_step():
    """The motivating case, to the digit: 24.0249 Bohr is really 0.0168 Bohr."""
    disp = [24.0249, 0.0, 0.0]
    folded, image, residual = mock.minimum_image_displacement(ORTHO, PERIODIC, disp)
    assert image == [1, 0, 0]
    assert folded[0] == pytest.approx(24.0249 - 24.0081, abs=1e-12)
    assert folded[0] == pytest.approx(0.0168, abs=1e-9)
    # A genuine boundary crossing leaves a tiny residual -- 7e-4 of a lattice
    # vector here. That is what distinguishes it from a displacement near half a
    # cell, where the fold direction would be a coin toss.
    assert abs(residual[0]) == pytest.approx(0.0007, abs=1e-4)
    # And it is now below the 0.5 Bohr threshold whose crossing rebuilt the bins.
    assert abs(folded[0]) < 0.5


def test_triclinic_fold_moves_all_three_components():
    """Where a naive implementation breaks, and by how much.

    Subtracting lattice vector b = (4, 11, 0) has to change x AND y. A fold done
    per Cartesian component against the cell's diagonal sees |x| = 4 < 12/2,
    leaves it alone, and lands the atom 4 Bohr from where it belongs -- silently,
    with plausible-looking forces.
    """
    step = np.array([0.013, -0.021, 0.008])
    raw = step + TRICLINIC[1]
    folded, image, _res = mock.minimum_image_displacement(
        TRICLINIC, PERIODIC, list(raw))
    assert image == [0, 1, 0]
    assert np.allclose(folded, step, atol=1e-12)

    naive = np.array(raw)
    for j in range(3):
        if abs(naive[j]) > TRICLINIC[j][j] / 2.0:
            naive[j] -= np.sign(naive[j]) * TRICLINIC[j][j]
    assert np.linalg.norm(naive - step) > 3.0


def test_triclinic_fold_survives_every_image_in_the_shell():
    """All 27 combinations of one lattice vector each way, on the hard cell."""
    rng = np.random.default_rng(20260820)
    worst = 0.0
    for n0 in (-1, 0, 1):
        for n1 in (-1, 0, 1):
            for n2 in (-1, 0, 1):
                step = rng.uniform(-0.2, 0.2, 3)
                raw = step + n0 * TRICLINIC[0] + n1 * TRICLINIC[1] + n2 * TRICLINIC[2]
                folded, image, _res = mock.minimum_image_displacement(
                    TRICLINIC, PERIODIC, list(raw))
                assert image == [n0, n1, n2]
                worst = max(worst, float(np.max(np.abs(np.array(folded) - step))))
    assert worst < 1e-12


def test_no_fold_is_a_bitwise_passthrough():
    """An ordinary step must come out the far side untouched, not reconstructed.

    getFractionalCoordinates is an LU solve. Round-tripping every displacement
    through it would perturb the last bits of every step DFT-FE takes on this
    path, and with them every stored reference energy in automated_tests.
    """
    disp = [0.1234567890123456, -0.9876543210987654, 0.5555555555555556]
    folded, image, _res = mock.minimum_image_displacement(
        TRICLINIC, PERIODIC, list(disp))
    assert image == [0, 0, 0]
    assert folded == disp                       # value equality
    for a, b in zip(folded, disp):
        assert a.hex() == b.hex()               # and bit-for-bit


def test_non_periodic_axis_is_never_folded():
    """A slab's open direction has no image to fold to -- folding it moves the
    atom through vacuum into a different structure, and dft.cc:1102-1109 would
    reject it anyway."""
    disp = [0.1, 0.1, 11.7]                     # 0.9 of the c vector's z
    open_z, image_open, _ = mock.minimum_image_displacement(
        TRICLINIC, [True, True, False], list(disp))
    assert image_open == [0, 0, 0]
    assert open_z == disp

    closed_z, image_closed, _ = mock.minimum_image_displacement(
        TRICLINIC, [True, True, True], list(disp))
    assert image_closed == [0, 0, 1]
    assert closed_z != disp


def test_many_lattice_vectors_fold_they_are_not_an_error():
    """What arrives here is not the optimizer's step. It is step + n*a, where n
    is how far the driver's copy has drifted from DFT-FE's: DFT-FE wraps its own
    copy back into the cell when a step forces it (moveAtoms.cc:246-298) and the
    driver deliberately never wraps (calculator.py:519-527; LAMMPS and i-PI keep
    unwrapped coordinates too). So n grows by one per net crossing of the same
    face and is unbounded under diffusion -- exactly the socket/MDI MD workload
    PROJECT.md 17.6 names as in scope.

    An earlier version of the fix aborted on |n| >= 2. That killed a correct
    trajectory: subtracting whole lattice vectors is an exact symmetry of the
    periodic system, so the fold recovers the true step for every n.
    """
    step = np.array([0.02, 0.0, 0.0])
    for n in (2, 3, -2, -7):
        disp = list(n * ORTHO[0] + step)
        folded, image, residual = mock.minimum_image_displacement(
            ORTHO, PERIODIC, disp)
        assert image == [n, 0, 0]
        assert folded[0] == pytest.approx(step[0], abs=1e-12)
        assert abs(residual[0]) < 1e-3

    # Same on a fully triclinic cell, where one lattice vector moves all three
    # Cartesian components.
    step = np.array([0.03, -0.02, 0.05])
    disp = list(-2 * TRICLINIC[0] + 3 * TRICLINIC[2] + step)
    folded, image, _res = mock.minimum_image_displacement(
        TRICLINIC, PERIODIC, disp)
    assert image == [-2, 0, 3]
    assert np.allclose(folded, step, atol=1e-10)


def test_a_diffusing_atom_does_not_kill_the_run():
    """The end-to-end version of the above, driven through the model engine.

    One atom walks +0.5 Bohr along x every frame while the driver keeps its
    positions unwrapped, as ASE/i-PI/LAMMPS do. Past two cell edges the raw
    difference spans two lattice vectors. Every applied step must stay 0.5 Bohr,
    which also keeps it under moveAtoms.cc:258's 0.5 Bohr bin-rebuild threshold
    that the unfolded displacement used to trip on every ionic step.
    """
    engine = mock.Engine(ORTHO.tolist(), PERIODIC, [3])
    pos = np.array([[1.0, 1.0, 1.0]])
    engine.reinit(pos.tolist())

    saw_image_two = False
    for _frame in range(120):          # 120 * 0.5 = 60 Bohr, 2.5 cell edges
        pos = pos + np.array([[0.5, 0.0, 0.0]])
        rec = engine.update(pos.tolist())
        assert rec["max_disp"] == pytest.approx(0.5, abs=1e-10)
        if rec["max_disp_raw"] > 2.0 * ORTHO[0][0]:
            saw_image_two = True
    assert saw_image_two, "the walk never reached a two-lattice-vector image"


# ══ 2. deck vs socket, end to end over the real wire ═══════════════════

def _atoms(cell_bohr, frac):
    cell = cell_bohr * Bohr
    return Atoms("H%d" % len(frac), cell=cell, pbc=True,
                 positions=np.asarray(frac) @ cell)


def _drive(atoms, steps, mock_command, tmp_path, extra_env=None):
    """Run one calculator over `1 + len(steps)` frames; return the engine's log.

    Every frame goes through the real DFTFE calculator, the real SocketBackend
    and the real wire protocol. Only the far end is a model.
    """
    log = os.path.join(str(tmp_path), "engine.jsonl")
    env = {"MOCK_DFTFE_ENGINE": "1",
           "MOCK_DFTFE_ENGINE_LOG": log,
           "MOCK_DFTFE_SCRATCH": str(tmp_path)}
    if extra_env:
        env.update(extra_env)
    with DFTFE(command=mock_command, env=env, connect_timeout=30,
               log_file=os.path.join(str(tmp_path), "dftfe.log")) as calc:
        atoms.calc = calc
        atoms.get_potential_energy()
        for delta in steps:
            atoms.set_positions(atoms.get_positions() + delta)
            atoms.get_potential_energy()
    with open(log) as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _same_point(cell_bohr, a, b):
    """True when two Cartesian positions differ by a whole lattice vector."""
    frac = np.linalg.solve(np.asarray(cell_bohr).T,
                           np.asarray(a) - np.asarray(b))
    return np.allclose(frac, np.round(frac), atol=1e-9)


# The restart geometry: atom 0 one thousandth of a cell OUTSIDE along x, exactly
# the way ASE frame 3 of job 8761021 carried atom 282 at fractional 1.000442.
RESTART_FRAC = [[1.000442, 0.4, 0.4],
                [0.2, 0.3, 0.5],
                [0.6, 0.7, 0.25]]


def test_deck_is_wrapped_and_describes_the_same_point_as_the_socket(
        mock_command, tmp_path):
    """The deck and the wire disagree by exactly one lattice vector -- by design.

    This is the state the whole bug grows from, so it is asserted rather than
    assumed: coordinates.inp holds a fractional inside [0,1), the socket holds
    1.000442, and the two are the same atom.
    """
    atoms = _atoms(ORTHO, RESTART_FRAC)
    frames = _drive(atoms, [], mock_command, tmp_path)
    assert len(frames) == 1

    deck = np.loadtxt(os.path.join(str(tmp_path), "coordinates.inp"))
    frac_deck = deck[:, 2:5]
    assert frac_deck[0][0] == pytest.approx(0.000442, abs=1e-9)
    assert np.all(frac_deck >= 0.0) and np.all(frac_deck < 1.0)

    sent = np.array(frames[0]["sent_coords"])
    frac_sent = np.linalg.solve(ORTHO.T, sent.T).T
    assert frac_sent[0][0] == pytest.approx(1.000442, abs=1e-9)

    # Same physical point: the deck's fractional and the socket's differ by a
    # whole number of lattice vectors, atom for atom.
    delta = frac_sent - frac_deck
    assert np.allclose(delta, np.round(delta), atol=1e-9)
    assert np.array_equal(np.round(delta), [[1, 0, 0], [0, 0, 0], [0, 0, 0]])


def test_restart_across_the_boundary_applies_the_step_not_the_lattice_vector(
        mock_command, tmp_path):
    """The regression itself.

    Boot from a geometry with one atom outside the cell, then take three small
    optimizer steps. Before the fix each of those steps arrived at DFT-FE as a
    24 Bohr displacement -- every step, not just the first, because the engine
    re-wraps after every move. After it, what arrives is the step.
    """
    atoms = _atoms(ORTHO, RESTART_FRAC)
    step_A = np.zeros((3, 3))
    step_A[0, 0] = 0.01 * Bohr                  # 0.01 Bohr along x, all three steps
    frames = _drive(atoms, [step_A] * 3, mock_command, tmp_path)
    assert len(frames) == 4

    for f in frames[1:]:
        # The raw difference really is a lattice vector: the bug is present in
        # the input, and the fold is what removes it.
        assert f["max_disp_raw"] > 20.0
        # What is applied is the optimizer's step, well under the 0.5 Bohr
        # threshold at moveAtoms.cc:258 that triggered the bin rebuild.
        assert f["max_disp"] == pytest.approx(0.01, abs=1e-9)
        assert f["max_disp"] < 0.5

    # And DFT-FE's copy still describes the same atoms ASE thinks it has.
    for f in frames:
        for sent, engine in zip(f["sent_coords"], f["engine_coords"]):
            assert _same_point(ORTHO, sent, engine)


def test_restart_across_the_boundary_on_a_triclinic_cell(mock_command, tmp_path):
    """Same restart, non-orthogonal cell: the case that punishes a per-Cartesian fold."""
    frac = [[0.3, 1.000442, 0.4], [0.2, 0.3, 0.5], [0.6, 0.7, 0.25]]
    atoms = _atoms(TRICLINIC, frac)
    step = np.zeros((3, 3))
    step[1, 2] = 0.02 * Bohr
    frames = _drive(atoms, [step, step], mock_command, tmp_path)

    for f in frames[1:]:
        assert f["max_disp_raw"] > 5.0
        assert f["max_disp"] == pytest.approx(0.02, abs=1e-9)
    for f in frames:
        for sent, engine in zip(f["sent_coords"], f["engine_coords"]):
            assert _same_point(TRICLINIC, sent, engine)


def test_the_bug_as_it_was(mock_command, tmp_path):
    """Switch the fold off and the crash conditions come straight back.

    This is the discrimination proof that keeps the tests above honest: it fails
    if the model has quietly stopped being able to express the bug -- which is
    the only way those tests could pass for the wrong reason.
    """
    atoms = _atoms(ORTHO, RESTART_FRAC)
    step = np.zeros((3, 3))
    step[0, 0] = 0.01 * Bohr
    frames = _drive(atoms, [step] * 3, mock_command, tmp_path,
                    extra_env={"MOCK_DFTFE_NO_MINIMUM_IMAGE": "1"})

    for f in frames[1:]:
        # ~24 Bohr, every single ionic step, against a 24.0081 Bohr edge.
        assert f["max_disp"] > 20.0
        assert f["max_disp"] == pytest.approx(f["max_disp_raw"], rel=1e-12)
        assert f["max_disp"] > 0.5              # the bin-rebuild threshold, 48x over


# ══ 3. the tripwire ════════════════════════════════════════════════════

_CPP = os.path.normpath(os.path.join(_HERE, "..", "..", "..", "src",
                                     "dftfeWrapper.cc"))


def _fold_source():
    if not os.path.isfile(_CPP):
        pytest.skip(f"C++ source not present at {_CPP} (installed-package layout)")
    with open(_CPP) as fh:
        text = fh.read()
    start = text.find("dftfeWrapper::updateAtomPositions")
    end = text.find("dftfeWrapper::deformCell")
    assert start != -1 and end > start, (
        "could not locate dftfeWrapper::updateAtomPositions in " + _CPP)
    return " ".join(text[start:end].split())


def test_cpp_fold_still_matches_this_model():
    """The only test here that reads the C++, and it reads it as text.

    Everything above tests a Python model of dftfeWrapper::updateAtomPositions.
    That model cannot notice the C++ changing under it, and a test suite that
    cannot notice is how a coordinate convention survived long enough to kill a
    32-node job (and how S14.3's force check passed for months against a key
    references.json never contained).

    So this asserts the five properties the model depends on are still in the
    source, structurally. It is deliberately coarse -- it survives reformatting,
    renaming of local variables it does not name, and any change to the
    diagnostics. If it fails, the C++ fold has been rewritten: re-derive the
    model in mock_dftfe.py against the new code and re-verify it, do not delete
    this test.
    """
    body = _fold_source()

    assert "getFractionalCoordinates" in body, (
        "the fold no longer goes through fractional coordinates. A per-Cartesian "
        "fold is correct for an orthorhombic cell and silently wrong for every "
        "other -- see test_triclinic_fold_moves_all_three_components, which "
        "measures the error at >3 Bohr.")

    assert "disp[i][jdim] -= image * cell[idim][jdim]" in body, (
        "the fold no longer subtracts whole lattice vectors from the original "
        "Cartesian components. One lattice vector moves all three components, "
        "and reconstructing from the fractionals instead would put every "
        "unfolded step through an LU solve.")

    assert all(k in body for k in ("periodicX", "periodicY", "periodicZ")), (
        "the fold no longer consults the PBC flags. Folding a non-periodic axis "
        "moves an atom through vacuum -- see "
        "test_non_periodic_axis_is_never_folded.")

    assert "image == 1 || image == -1" not in body, (
        "the fold has been re-restricted to a single lattice vector. The image "
        "count is the driver's accumulated offset, not the step, and it grows "
        "without bound under diffusion -- see "
        "test_a_diffusing_atom_does_not_kill_the_run.")

    assert "Minimum image applied to the displacement of" in body and (
        "d_dftfeParamsPtr->verbosity >= 1" in body), (
        "the fold no longer reports itself at default verbosity. Silent folding "
        "is the failure class this change exists to close: the numbers on that "
        "line (atoms folded, largest residual, largest applied displacement) "
        "are the only thing separating a boundary crossing from a driver "
        "sending a geometry where a step was expected.")

    assert "MPI_Bcast" in body, (
        "the image vector is no longer broadcast. The fold branches on a "
        "std::round() of an LU solve; two ranks disagreeing by one image would "
        "move the mesh differently and then diverge inside initNoRemesh()'s "
        "collectives.")


# ══ 4. the invariant on the two copies of the geometry ═════════════════
#
# dftfeWrapper::updateAtomPositions() now checks, after the mesh has moved, that
# what DFT-FE holds equals what the driver asked for -- modulo whole lattice
# vectors along periodic directions, exactly along open ones. These test the
# model of that check (mock_dftfe.verify_positions_against_driver).
#
# READ THIS BEFORE TRUSTING THE CHECK. It does NOT detect the failure of job
# 8761021, and test_invariant_does_not_catch_a_drifted_driver_frame below pins
# that down. There the driver asked for a 24 Bohr step and DFT-FE faithfully
# applied a 24 Bohr step, so the copies agreed afterwards. Repairing THAT is the
# fold's job, and reporting it is the fold's diagnostic. What the invariant adds
# is protection against DFT-FE not doing what was asked -- which is the failure
# mode the fold itself introduces the risk of.

def _engine(cell, pbc=PERIODIC, natoms=3, **kw):
    return mock.Engine(cell, pbc, [3] * natoms, **kw)


def _frame(cell, natoms=3, seed=0):
    rng = np.random.default_rng(seed)
    return [(rng.random(3) @ cell).tolist() for _ in range(natoms)]


def test_invariant_passes_on_an_ordinary_step():
    eng = _engine(ORTHO)
    coords = _frame(ORTHO)
    eng.reinit(coords)
    stepped = [[c[j] + 0.01 * (j + 1) for j in range(3)] for c in coords]
    rec = eng.update(stepped)
    assert rec["invariant_residual"] < 1e-12


def test_invariant_passes_on_a_boundary_crossing():
    """The fold is active here -- an image of one lattice vector is subtracted --
    and the invariant must still pass, because a whole lattice vector is exactly
    what it permits."""
    eng = _engine(ORTHO)
    coords = _frame(ORTHO)
    eng.reinit(coords)
    stepped = [list(c) for c in coords]
    stepped[0][0] += ORTHO[0][0] + 0.0168      # cross the x face
    rec = eng.update(stepped)
    assert rec["max_disp_raw"] > 24.0
    assert rec["max_disp"] < 0.5               # the fold did fire
    assert rec["invariant_residual"] < 1e-12   # and the invariant still holds


def test_invariant_passes_on_a_multi_image_step():
    """n = 3 lattice vectors, the diffusive-MD case the removed AssertThrow
    would have aborted. Modulo means modulo any integer, and the check has to
    agree with the fold about that."""
    eng = _engine(ORTHO)
    coords = _frame(ORTHO)
    eng.reinit(coords)
    stepped = [list(c) for c in coords]
    stepped[1][1] += 3 * ORTHO[1][1] - 0.02
    rec = eng.update(stepped)
    assert rec["invariant_residual"] < 1e-12


def test_invariant_passes_on_a_triclinic_cell():
    eng = _engine(TRICLINIC)
    coords = _frame(TRICLINIC, seed=7)
    eng.reinit(coords)
    stepped = [list(c) for c in coords]
    stepped[2] = [stepped[2][j] + TRICLINIC[1][j] + 0.013 for j in range(3)]
    rec = eng.update(stepped)
    assert rec["invariant_residual"] < 1e-10


def test_invariant_catches_a_small_wrong_move():
    """0.02 Bohr -- below moveAtoms.cc:258's 0.5 Bohr remesh threshold, so it
    would produce no symptom at all, no bin rebuild and no crash. Just wrong
    energies. This is the size of defect the check exists for."""
    def nudge(moved):
        moved[0][0] += 0.02
        return moved

    eng = _engine(ORTHO, corrupt=nudge)
    coords = _frame(ORTHO)
    eng.reinit(coords)
    with pytest.raises(mock.PositionMismatch) as exc:
        eng.update([[c[j] + 0.01 for j in range(3)] for c in coords])
    assert "atom 0, direction 0" in str(exc.value)


def test_invariant_catches_a_fold_that_is_not_a_lattice_vector():
    """The triclinic trap, caught in the binary rather than only in this file.

    A per-Cartesian fold on TRICLINIC subtracts (4, 0, 0) where it should
    subtract (4, 11, 0). The result is not a lattice vector, so the residual is
    not an integer and the check fires -- 11 Bohr of silently destroyed geometry
    that currently only a mock test stands between us and.
    """
    def wrong_fold(moved):
        moved[0][1] += TRICLINIC[1][1]     # put back the y part it should have kept
        return moved

    eng = _engine(TRICLINIC, corrupt=wrong_fold)
    coords = _frame(TRICLINIC, seed=3)
    eng.reinit(coords)
    with pytest.raises(mock.PositionMismatch):
        eng.update([[c[j] + 0.01 for j in range(3)] for c in coords])


def test_invariant_forbids_any_fold_along_an_open_axis():
    """Along z with pbc False there is no image to fold to: dft.cc:1102-1109
    requires the atom strictly inside, so a z displacement of one 'lattice
    vector' is not a symmetry, it is a different structure."""
    pbc = [True, True, False]
    def slide_z(moved):
        moved[1][2] += ORTHO[2][2]
        return moved

    eng = _engine(ORTHO, pbc=pbc, corrupt=slide_z)
    coords = _frame(ORTHO)
    eng.reinit(coords)
    with pytest.raises(mock.PositionMismatch) as exc:
        eng.update([[c[j] + 0.01 for j in range(3)] for c in coords])
    assert "direction 2" in str(exc.value)


def test_invariant_does_not_catch_a_drifted_driver_frame():
    """The honest limit of this check, pinned so nobody over-claims it.

    This is job 8761021 exactly: the driver's copy is a lattice vector away from
    DFT-FE's, so it asks for a 24 Bohr step. With the fold disabled DFT-FE
    applies all 24 Bohr of it -- faithfully -- and the invariant PASSES, because
    DFT-FE did what it was told. The 0.5 Bohr threshold is crossed, the bins get
    rebuilt every step, and eventually a rank dies.

    So the invariant is not a substitute for the fold, and not a detector for
    this failure class. The detector for this class is the fold's own diagnostic
    (a nonzero image, reported at verbosity >= 1); the repair is the fold.
    """
    eng = _engine(ORTHO, fold=False)
    coords = _frame(ORTHO)
    eng.reinit(coords)
    drifted = [list(c) for c in coords]
    drifted[0][0] += ORTHO[0][0] + 0.0168
    rec = eng.update(drifted)
    assert rec["max_disp"] > 24.0               # the pathological step was applied
    assert rec["invariant_residual"] < 1e-12    # and this check is blind to it
