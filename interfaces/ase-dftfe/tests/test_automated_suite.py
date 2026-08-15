# ---------------------------------------------------------------------
# Part of the DFT-FE code, released under LGPL v2.1 or later.
# ---------------------------------------------------------------------
#
# @author Mehul Darak
#
"""Unit tests for ``automated_tests/test_suite.py``'s relaxed-reference plumbing.

The suite itself needs a binary and a node; these do not. They cover the part
that is pure bookkeeping and that got a case's numbers compared against the
wrong reference: relax_o2 returned its RELAXED energy while references.json's
gated ``energy_ha`` is the native single point at the INITIAL geometry, so the
case would have failed by ~2.7e-3 Ha with nothing wrong.
"""

import importlib.util
import json
import os
import sys

import numpy as np
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_AUTO = os.path.join(os.path.dirname(_HERE), "automated_tests")


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def suite():
    return _load("auto_test_suite", os.path.join(_AUTO, "test_suite.py"))


def _ref(e=-31.764763711109943, ftol=1.944690395740863e-04, f=7.1e-05,
         native_steps=9):
    return {
        "relaxed": {
            "energy_ha": e,
            "forces_ha_per_bohr": [[f, 0.0, 0.0], [-f, 0.0, 0.0]],
            "force_tol_ha_bohr": ftol,
            "native_steps": native_steps,
        }
    }


def _got(e=-31.764763711109943, f=7.1e-05, steps=11):
    return {
        "relaxed": {
            "energy_ha": e,
            "forces_ha_per_bohr": [[f, 0.0, 0.0], [-f, 0.0, 0.0]],
            "steps": steps,
        }
    }


# ── check_relaxed ──────────────────────────────────────────────────────────

def test_no_relaxed_half_is_silent(suite):
    """A single-point case must produce no relaxed line at all."""
    assert suite.check_relaxed({}, {"energy_ha": -20.6}) == ""
    assert suite.check_relaxed(None, None) == ""


def test_reports_energy_gap_in_ha_and_mev(suite):
    note = suite.check_relaxed(_got(e=-31.764663711109943), _ref())
    assert "MEASURED" in note and "not gated" in note
    # 1.0e-04 Ha = 2.721 meV; both units must appear, since the whole point of
    # this line is that a human reads the gap and decides what it means.
    assert "1.000e-04 Ha" in note
    assert "2.721 meV" in note


def test_a_large_gap_is_still_not_a_failure(suite):
    """Nothing here may turn into a verdict: the caller does not gate on it."""
    note = suite.check_relaxed(_got(e=-31.0), _ref())
    assert "MEASURED" in note
    assert "FAIL" not in note


def test_step_counts_from_both_arms_are_reported(suite):
    note = suite.check_relaxed(_got(steps=11), _ref(native_steps=9))
    assert "ASE LBFGS 11" in note
    assert "LBFGS 9" in note


def test_unparsed_native_step_count_says_so(suite):
    note = suite.check_relaxed(_got(), _ref(native_steps=None))
    assert "not parsed" in note
    assert "None" not in note


def test_warns_when_an_arm_stopped_above_its_own_force_tolerance(suite):
    """Per-arm self-consistency: converged means max|F| <= the tol it was given."""
    tol = 1.944690395740863e-04
    clean = suite.check_relaxed(_got(f=7.1e-05), _ref(ftol=tol, f=7.1e-05))
    assert "WARNING" not in clean

    ase_bad = suite.check_relaxed(_got(f=3.0e-04), _ref(ftol=tol, f=7.1e-05))
    assert "WARNING" in ase_bad and "ASE arm" in ase_bad

    nat_bad = suite.check_relaxed(_got(f=7.1e-05), _ref(ftol=tol, f=3.0e-04))
    assert "WARNING" in nat_bad and "native arm" in nat_bad


def test_missing_reference_block_names_the_generator_to_run(suite):
    note = suite.check_relaxed(_got(), {"energy_ha": -31.76})
    assert "make_references.py --relax" in note


def test_missing_run_half_is_reported_not_silently_passed(suite):
    note = suite.check_relaxed({}, _ref())
    assert "reported none" in note


# ── the case/suite return-value contract ───────────────────────────────────

def test_relax_o2_returns_the_initial_point_and_the_relaxed_extra():
    """The gated value must be the INITIAL geometry's energy, not the relaxed one.

    Driven with a fake calculator, so no binary and no SCF: it records the
    positions it is asked about and returns a distinct energy for each, which is
    what makes "which one came back first" observable.
    """
    case = _load("case_relax_o2", os.path.join(_AUTO, "cases", "relax_o2.py"))

    class FakeCalc:
        """Enough ASE Calculator surface for LBFGS, with a known landscape."""

        def __init__(self, **kwargs):
            self.atoms = None
            self.nsteps = 0

        # A 1-D quadratic in the O-O separation along x: forces fall off as the
        # bond approaches r0, so LBFGS converges in a few steps and the relaxed
        # energy is strictly below the initial one.
        def _eval(self, atoms):
            r = atoms.get_positions()[1][0] - atoms.get_positions()[0][0]
            r0, k = 2.4, 5.0
            e = 0.5 * k * (r - r0) ** 2 - 800.0
            g = k * (r - r0)
            return e, np.array([[g, 0.0, 0.0], [-g, 0.0, 0.0]])

        def get_potential_energy(self, atoms=None, **_):
            self.nsteps += 1
            return self._eval(atoms)[0]

        def get_forces(self, atoms=None, **_):
            return self._eval(atoms)[1]

        def get_stress(self, atoms=None, **_):
            raise NotImplementedError

        def get_property(self, name, atoms=None, **_):
            if name == "energy":
                return self.get_potential_energy(atoms)
            if name == "forces":
                return self.get_forces(atoms)
            raise NotImplementedError(name)

        def calculation_required(self, atoms, quantities):
            return True

        def __getattr__(self, name):
            return lambda *a, **k: None

    case.DFTFE = FakeCalc
    out = case.run(dftfe_bin="/nonexistent",
                   psp_library=os.path.join(os.path.dirname(_AUTO), "psp_library"),
                   np_tasks=1, use_device=False)

    assert len(out) == 4, "relax_o2 must carry a fourth, non-gated element"
    energy, forces, stress, extra = out
    assert stress is None

    # The initial O-O separation is 2.4 Bohr; the fake minimum is at 2.4 Ang, so
    # the initial point sits well up the wall and relaxing must lower the energy.
    relaxed = extra["relaxed"]["energy_ha"]
    assert relaxed < energy, (
        "the gated energy is not the initial point -- it must be the higher of "
        "the two, since the relaxation lowers it")
    assert np.abs(forces).max() > np.abs(
        np.array(extra["relaxed"]["forces_ha_per_bohr"])).max()
    assert extra["relaxed"]["steps"] >= 1


# ── stress: parsing, units, and the check that used to not exist ───────────

_NATIVE_STRESS_BLOCK = """
Cell stress (Hartree/Bohr^3)
-------------------------------------------------------------------------
-1.103475612189628690e-04  -5.577887990917342057e-10  -1.217965722269959629e-09
-5.573254930353495430e-10  -1.103477489947552614e-04  -2.179469937732692617e-11
-1.217214208137916265e-09  -2.192087357062946655e-11  -1.103469100737820987e-04
-------------------------------------------------------------------------
Force and Stress computation, wall time: 7.73761s.
"""


def test_native_stress_block_is_parsed_as_a_3x3():
    from dftfe_ase.backends.file import FileBackend
    got = FileBackend._parse_stress(_NATIVE_STRESS_BLOCK)
    assert np.array(got).shape == (3, 3)
    assert got[0][0] == pytest.approx(-1.103475612189628690e-04, rel=0, abs=0)
    assert got[2][2] == pytest.approx(-1.103469100737820987e-04, rel=0, abs=0)


def test_no_stress_block_parses_to_none():
    """A run without CELL STRESS must give None, not an empty list or a crash."""
    from dftfe_ase.backends.file import FileBackend
    assert FileBackend._parse_stress("Total free energy:  -9.24\nIon forces\n") is None


def test_last_stress_block_wins():
    """A GEOOPT prints one per ionic step; the converged one is last."""
    from dftfe_ase.backends.file import FileBackend
    first = _NATIVE_STRESS_BLOCK.replace("-1.103475612189628690e-04",
                                         "-9.999999999999999000e-04")
    got = FileBackend._parse_stress(first + _NATIVE_STRESS_BLOCK)
    assert got[0][0] == pytest.approx(-1.103475612189628690e-04, rel=0, abs=0)


def test_voigt_to_matrix_conversion_is_symmetric_and_in_dftfe_units(suite):
    """ASE gives Voigt-6 eV/A^3; the reference is a 3x3 in Ha/Bohr^3."""
    from ase.units import Bohr, Hartree
    voigt = np.array([1.0, 2.0, 3.0, 0.4, 0.5, 0.6])   # xx yy zz yz xz xy
    got = suite.stress_to_ha_per_bohr3(voigt)
    unit = Hartree / Bohr ** 3
    assert got.shape == (3, 3)
    assert np.allclose(got, got.T)                      # symmetric
    assert got[0][0] == pytest.approx(1.0 / unit)
    assert got[1][2] == pytest.approx(0.4 / unit)       # yz
    assert got[0][2] == pytest.approx(0.5 / unit)       # xz
    assert got[0][1] == pytest.approx(0.6 / unit)       # xy
    # A 3x3 input must pass through in the same units, not be converted twice.
    assert np.allclose(suite.stress_to_ha_per_bohr3(got * unit), got)


def test_a_wrong_shape_stress_raises_rather_than_broadcasting(suite):
    with pytest.raises(ValueError, match="expected"):
        suite.stress_to_ha_per_bohr3(np.zeros(9))


def test_stress_check_passes_on_a_round_trip_and_fails_on_a_nudge(suite):
    """Round-trip the reference through ASE's units and back; then perturb it."""
    from ase.units import Bohr, Hartree
    import json
    ref = json.load(open(os.path.join(_AUTO, "references.json")))
    ref_stress = ref["al_bulk_periodic"]["stress_ha_per_bohr3"]
    assert ref_stress is not None, "al_bulk must carry a stress reference"

    unit = Hartree / Bohr ** 3
    # No negation: the printed reference is already ASE convention (job 8756125),
    # and the interface undoes the wrapper's flip on its own side.
    m = np.array(ref_stress) * unit
    # Voigt-6 the way ASE builds it: each off-diagonal is the average of the pair,
    # because DFT-FE's printed 3x3 is not exactly symmetric.
    voigt = np.array([m[0][0], m[1][1], m[2][2],
                      (m[1][2] + m[2][1]) / 2, (m[0][2] + m[2][0]) / 2,
                      (m[0][1] + m[1][0]) / 2])
    ok, diff, _, _, asym = suite.check_stress(voigt, ref_stress)
    assert ok
    assert diff < 1e-18, "symmetrised round trip must be exact to float precision"
    # The reference's own asymmetry is real and is the floor for the tolerance.
    assert 1e-13 < asym < 1e-11
    assert asym < suite.STRESS_TOL_HA_BOHR3

    # 1e-9 Ha/Bohr^3 is 1e-5 relative on a ~1e-4 stress -- must not pass.
    nudged = voigt.copy()
    nudged[0] += 1e-9 * unit
    bad, diff, wi, wj, _ = suite.check_stress(nudged, ref_stress)
    assert not bad
    assert (wi, wj) == (0, 0)
    assert diff == pytest.approx(1e-9, rel=1e-6)


def test_a_case_computing_stress_without_a_reference_fails(suite, tmp_path,
                                                          monkeypatch, capsys):
    """Unpinned stress must FAIL, on the same logic as a null energy reference."""
    import json
    refs = json.load(open(os.path.join(_AUTO, "references.json")))
    al = refs["al_bulk_periodic"]
    stripped = {k: v for k, v in al.items() if k != "stress_ha_per_bohr3"}
    monkeypatch.setattr(suite, "load_case", lambda name: (
        lambda **kw: (al["energy_ha"], None, np.zeros(6))))
    monkeypatch.setattr(suite, "REFS_FILE", str(tmp_path / "refs.json"))
    (tmp_path / "refs.json").write_text(json.dumps({"al_bulk_periodic": stripped}))
    monkeypatch.chdir(tmp_path)

    with pytest.raises(SystemExit):
        suite.run_suite(suite.parse_args([
            "--dftfe-real", sys.executable, "--dftfe-complex", sys.executable,
            "--psp-library", str(tmp_path), "--np", "64", "--no-device",
            "--tests", "al_bulk_periodic",
        ]))
    out = capsys.readouterr().out
    assert "NO STRESS REFERENCE" in out
    assert "FAIL=1" in out


def test_a_reference_richer_than_the_case_says_so_instead_of_staying_silent(
        suite, tmp_path, monkeypatch, capsys):
    """al_bulk carries forces it never computes; silence there reads as PASS."""
    import json
    refs = json.load(open(os.path.join(_AUTO, "references.json")))
    al = refs["al_bulk_periodic"]
    from ase.units import Bohr, Hartree
    unit = Hartree / Bohr ** 3
    m = np.array(al["stress_ha_per_bohr3"]) * unit
    voigt = np.array([m[0][0], m[1][1], m[2][2], m[1][2], m[0][2], m[0][1]])
    monkeypatch.setattr(suite, "load_case", lambda name: (
        lambda **kw: (al["energy_ha"], None, voigt)))
    monkeypatch.chdir(tmp_path)

    suite.run_suite(suite.parse_args([
        "--dftfe-real", sys.executable, "--dftfe-complex", sys.executable,
        "--psp-library", str(tmp_path), "--np", "64", "--no-device",
        "--tests", "al_bulk_periodic",
    ]))
    out = capsys.readouterr().out
    assert "PASS=1  FAIL=0" in out
    assert "compute_forces=False" in out     # the unchecked force reference, named
    assert "max |S_computed - S_ref|" in out  # and the stress actually compared


def test_suite_end_to_end_on_both_return_shapes(suite, tmp_path, monkeypatch,
                                                capsys):
    """Drive run_suite with fake cases that hand back the stored references.

    No binary and no MPI: the cases are replaced, so what is under test is the
    suite's own bookkeeping -- unpacking three- and four-element returns, the
    provenance gate at np=64, and whether the relaxed line reaches the report.
    Feeding the references straight back must give PASS; if it does not, the
    comparison path is broken independently of any physics.
    """
    import json

    refs = json.load(open(os.path.join(_AUTO, "references.json")))
    n2, o2 = refs["n2_nonperiodic"], refs["relax_o2"]

    def fake_load_case(name):
        if name == "n2_nonperiodic":
            return lambda **kw: (n2["energy_ha"],
                                 np.array(n2["forces_ha_per_bohr"]), None)
        return lambda **kw: (
            o2["energy_ha"], np.array(o2["forces_ha_per_bohr"]), None,
            {"relaxed": {"energy_ha": o2["relaxed"]["energy_ha"],
                         "forces_ha_per_bohr": o2["relaxed"]["forces_ha_per_bohr"],
                         "steps": 11}})

    monkeypatch.setattr(suite, "load_case", fake_load_case)
    monkeypatch.chdir(tmp_path)          # the results file lands in cwd

    args = suite.parse_args([
        "--dftfe-real", sys.executable, "--dftfe-complex", sys.executable,
        "--psp-library", str(tmp_path), "--np", "64", "--no-device",
        "--tests", "n2_nonperiodic", "relax_o2",
    ])
    suite.run_suite(args)                # sys.exit(1) on any FAIL
    out = capsys.readouterr().out

    assert "PASS=2  FAIL=0" in out
    assert "relaxed (MEASURED, not gated)" in out
    assert "ASE LBFGS 11" in out
    # references.json must not have been rewritten: nothing was seeded.
    assert not (tmp_path / "references.json").exists()


def test_provenance_gate_still_blocks_a_different_rank_count(suite, tmp_path,
                                                            monkeypatch, capsys):
    """The same run at np=8 must report MISMATCH, not PASS and not FAIL.

    This is what makes the 1e-10 bar honest: rank count alone moved these very
    energies by up to 2.7e-09 Ha with nothing wrong (PROJECT.md 14.6).
    """
    import json

    refs = json.load(open(os.path.join(_AUTO, "references.json")))
    n2 = refs["n2_nonperiodic"]
    monkeypatch.setattr(suite, "load_case", lambda name: (
        lambda **kw: (n2["energy_ha"], np.array(n2["forces_ha_per_bohr"]), None)))
    monkeypatch.chdir(tmp_path)

    suite.run_suite(suite.parse_args([
        "--dftfe-real", sys.executable, "--dftfe-complex", sys.executable,
        "--psp-library", str(tmp_path), "--np", "8", "--no-device",
        "--tests", "n2_nonperiodic",
    ]))
    out = capsys.readouterr().out
    assert "[MISMATCH]" in out
    assert "PASS=0  FAIL=0" in out


# ── one reference file per device ───────────────────────────────────────────
# A reference file holds exactly one configuration per case, and use_device is a
# CONFIG_KEY, so a GPU number stored in the CPU file would not just overwrite it:
# every subsequent CPU run would report MISMATCH instead of comparing.

def test_host_and_device_runs_read_different_files(suite):
    cpu = suite.parse_args(["--dftfe-real", "x", "--dftfe-complex", "y",
                            "--psp-library", "z", "--no-device"])
    gpu = suite.parse_args(["--dftfe-real", "x", "--dftfe-complex", "y",
                            "--psp-library", "z"])
    assert suite.refs_file_for(cpu).endswith("references.json")
    assert suite.refs_file_for(gpu).endswith("references.gpu.json")
    assert suite.refs_file_for(cpu) != suite.refs_file_for(gpu)


def test_refs_flag_overrides_both_defaults(suite):
    for extra in (["--no-device"], []):
        args = suite.parse_args(["--dftfe-real", "x", "--dftfe-complex", "y",
                                 "--psp-library", "z", "--refs", "/tmp/mine.json"]
                                + extra)
        assert suite.refs_file_for(args) == "/tmp/mine.json"


def test_missing_device_references_exit_with_instructions(suite, tmp_path,
                                                          monkeypatch, capsys):
    """Absent GPU file must say how to generate it, not crash on open()."""
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit):
        suite.run_suite(suite.parse_args([
            "--dftfe-real", sys.executable, "--dftfe-complex", sys.executable,
            "--psp-library", str(tmp_path), "--np", "12",
            "--refs", str(tmp_path / "absent.json"),
        ]))
    out = capsys.readouterr().out
    assert "no reference file" in out
    assert "make_references.py" in out


def test_cpu_reference_file_is_still_the_validated_one(suite):
    """Guard on the numbers themselves: the CPU set must stay host-generated."""
    import json
    refs = json.load(open(suite.REFS_FILE))
    for name, case in refs.items():
        if name.startswith("_"):
            continue
        assert case["provenance"]["use_device"] is False, (
            f"{name}: a device reference has been written into the CPU file")
        assert case["provenance"]["np"] == 64


def _build_deck(tmp_path, use_device):
    """Generate one native deck without running anything."""
    mr = _load("make_refs", os.path.join(_AUTO, "make_references.py"))
    atoms, kwargs = mr.capture("al_bulk_periodic")
    work = str(tmp_path / ("gpu" if use_device else "cpu"))
    mr.build_deck(work, atoms, kwargs,
                  os.path.join(os.path.dirname(_AUTO), "psp_library"),
                  use_device=use_device)
    return open(os.path.join(work, "parameterFile.prm")).read()


def test_gpu_deck_carries_use_gpu_at_top_level(tmp_path):
    """Without this line the native "GPU" reference is a CPU run mislabelled.

    USE GPU is driver-owned, so it is absent from the dftParameters-generated
    template and write_prm cannot inject it -- it has to be prepended, and it has
    to land before the first subsection to be a top-level entry.
    """
    deck = _build_deck(tmp_path, use_device=True)
    lines = [l.strip() for l in deck.splitlines() if l.strip()]
    assert "set USE GPU=true" in lines
    first_subsection = next(i for i, l in enumerate(lines)
                            if l.startswith("subsection"))
    assert lines.index("set USE GPU=true") < first_subsection


def test_host_deck_does_not_claim_gpu(tmp_path):
    # Match the entry, not the substring: the template declares
    # `USE GPUDIRECT MPI ALL REDUCE`, which contains "USE GPU".
    lines = [l.strip() for l in _build_deck(tmp_path, use_device=False).splitlines()]
    assert "set USE GPU=true" not in lines
    assert not [l for l in lines if l.startswith("set USE GPU=")]


def test_native_launch_command_pins_one_rank_per_tile():
    """GPU runs need gpu_tile_compact.sh, or every rank lands on tile 0."""
    mr = _load("make_refs2", os.path.join(_AUTO, "make_references.py"))

    class A:
        np, ppn, gpu_wrapper = 12, 12, "gpu_tile_compact.sh"
        use_device = True
    gpu = mr.native_cmd(A, "/bin/dftfe")
    assert gpu[:5] == ["mpiexec", "-n", "12", "--ppn", "12"]
    assert "gpu_tile_compact.sh" in gpu

    A.use_device = False
    cpu = mr.native_cmd(A, "/bin/dftfe")
    # The CPU references were made with plain mpirun; keep them reproducible.
    assert cpu == ["mpirun", "-np", "12", "/bin/dftfe", "parameterFile.prm"]
    assert "gpu_tile_compact.sh" not in cpu


# ── stress sign convention ──────────────────────────────────────────────────
# The native print and the wrapper API disagree in sign on purpose:
# configurationalForce.cc:811 prints the internal cellStressTensor, while
# dftfeWrapper::getCellStress() returns its negative "to use ASE style input
# output" (upstream b91c4bed8). references.json holds the printed convention.

def test_ase_convention_stress_matches_a_printed_convention_reference(suite):
    """A correct interface value EQUALS the stored reference, and must PASS.

    The printed reference is already ASE convention (job 8756125), and the
    calculator undoes dftfeWrapper's negation, so both sides agree directly.
    """
    from ase.units import Bohr, Hartree
    printed = [[-1.1034756121896287e-04, -5.5e-10, -1.2e-09],
               [-5.5e-10, -1.1034774899475526e-04, -2.2e-11],
               [-1.2e-09, -2.2e-11, -1.1034691007378210e-04]]
    unit = Hartree / Bohr ** 3
    m = np.array(printed) * unit          # what the calculator reports to ASE
    voigt = np.array([m[0][0], m[1][1], m[2][2],
                      (m[1][2] + m[2][1]) / 2, (m[0][2] + m[2][0]) / 2,
                      (m[0][1] + m[1][0]) / 2])
    ok, diff, _, _, _ = suite.check_stress(voigt, printed)
    assert ok, f"ASE-convention stress rejected against printed reference (diff {diff:.3e})"
    assert diff < 1e-18


def test_flipped_sign_is_rejected(suite):
    """The failure mode this cost a GPU run to find: a pure sign flip.

    If _WRAPPER_STRESS_SIGN ever returns to +1 -- e.g. because upstream fixed
    getCellStress() and this side was not updated with it -- max|dS| comes back at
    twice the stress magnitude with relative error ~2, which is exactly what the
    first GPU run reported. That must FAIL, not pass.
    """
    from ase.units import Bohr, Hartree
    printed = [[-1.1e-04, 0.0, 0.0], [0.0, -1.1e-04, 0.0], [0.0, 0.0, -1.1e-04]]
    unit = Hartree / Bohr ** 3
    m = -np.array(printed) * unit          # flipped: the bug this guards against
    voigt = np.array([m[0][0], m[1][1], m[2][2], 0.0, 0.0, 0.0])
    ok, diff, _, _, _ = suite.check_stress(voigt, printed)
    assert not ok
    assert diff == pytest.approx(2 * 1.1e-04, rel=1e-6), (
        "a sign flip must show up as twice the stress magnitude")


# ── provenance must not carry anyone's directory ───────────────────────────
#
# A reference file ships with the package. The energies and forces in it are the
# point; the absolute path of the binary that produced them is not, and it
# published one person's scratch directory to every reader. `_binary_id` keeps
# the half that identifies the build (`cpu_real/dftfe`) and drops the half that
# identifies the author. Nothing gates on it -- test_suite.CONFIG_KEYS is
# ("np", "use_device") -- so shortening it costs no strictness.

def test_binary_id_keeps_the_build_and_drops_the_owner(suite):
    assert suite._binary_id(
        "/lus/flare/projects/X/someone/install/dftfe_pgd/install/cpu_real/dftfe"
    ) == "cpu_real/dftfe"
    assert suite._binary_id("/opt/complex/dftfe") == "complex/dftfe"
    assert suite._binary_id("dftfe") == "dftfe"


@pytest.mark.parametrize("fname", ["references.json", "references.gpu.json"])
def test_no_reference_provenance_carries_an_absolute_path(fname):
    path = os.path.join(_AUTO, fname)
    if not os.path.isfile(path):
        pytest.skip(f"{fname} not present")
    with open(path) as fh:
        refs = json.load(fh)
    for name, entry in refs.items():
        if not isinstance(entry, dict):
            continue
        binary = (entry.get("provenance") or {}).get("binary")
        if binary is None:
            continue
        assert not os.path.isabs(binary), f"{fname}:{name} ships an absolute path: {binary}"
        assert binary.count(os.sep) <= 1, f"{fname}:{name} keeps too much path: {binary}"
