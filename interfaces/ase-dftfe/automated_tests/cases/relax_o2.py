# ---------------------------------------------------------------------
# Copyright (c) 2017-2025 The Regents of the University of Michigan and DFT-FE
# authors. Part of the DFT-FE code, released under LGPL v2.1 or later.
# ---------------------------------------------------------------------
#
# @author Mehul Darak
#
"""
Test case: O2 relaxation (non-periodic → real binary).
Mirrors exactly: examples/relax_o2.py
"""
import os
import numpy as np
from ase import Atoms
from ase.units import Bohr, Hartree
from ase.optimize import LBFGS
from dftfe_ase import DFTFE


def run(dftfe_bin: str, psp_library: str, np_tasks: int = 8,
        use_device: bool = True):
    # --- geometry (exactly as in examples/relax_o2.py) ---
    box_dims_bohr = np.array([40.0, 42.0, 38.0])
    cell_ang = np.diag(box_dims_bohr) * Bohr
    center = box_dims_bohr / 2.0
    rel_pos_bohr = np.array([
        [-1.2, 0.0, 0.0],
        [ 1.2, 0.0, 0.0],
    ])
    pos_ang = (center + rel_pos_bohr) * Bohr

    atoms = Atoms(
        symbols=['O', 'O'],
        positions=pos_ang,
        cell=cell_ang,
        pbc=[False, False, False],
    )

    # --- calculator (exactly as in examples/relax_o2.py) ---
    psp_dict = {"O": os.path.join(psp_library, "O.upf")}
    run_cmd = f"mpirun -np {np_tasks} {dftfe_bin}"

    calc = DFTFE(
        command=run_cmd,
        psp_path=psp_dict,
        mesh_size=0.6,
        polynomial_order=7,
        atom_ball_radius=6.0,
        tolerance=1e-6,
        scf_mixing=0.5,
        fermi_temp=500.0,
        num_bands=16,
        xc='GGA-PBE',
        compute_forces=True,
        compute_stress=False,
        use_device=use_device,
        verbosity=1,
        log_file="test_relax_o2.log",
    )

    atoms.calc = calc

    # ── the gated number: the single point at the INITIAL geometry ──
    # This, not the relaxed energy, is what the 1e-10 Ha reference holds. It is
    # the only quantity in this case that native pGD and the socket path compute
    # for the *same* geometry: make_references.py's recorder stops at the first
    # calculator call, which is optimizer step 0, so the native deck it writes is
    # this single point. Returning the relaxed energy here instead compared it
    # against the initial-point reference and failed by ~2.7e-3 Ha for no
    # physical reason.
    #
    # It costs no extra SCF. ASE caches results per configuration, so step 0
    # reuses this evaluation rather than repeating it.
    energy_ha = atoms.get_potential_energy() / Hartree
    forces_ha_per_bohr = atoms.get_forces() / (Hartree / Bohr)

    # ── matched to the native optimizer, on purpose ──
    # The native reference relaxes with DFT-FE's own LBFGS, so the ASE arm uses
    # LBFGS too, with its two settings matched to the deck make_references.py
    # writes -- otherwise the measured gap mixes an optimizer-family difference
    # (limited-memory vs full Hessian) into what is supposed to be an
    # implementation comparison:
    #
    #   LBFGS HISTORY            = 5     -> memory=5
    #   MAXIMUM ION UPDATE STEP  = 0.5   -> maxstep=0.5 * Bohr
    #
    # That 0.5 is in atomic units ("displacement in a.u.",
    # dftParameters.cc:327), i.e. 0.5 Bohr = 0.2646 A, NOT ASE's default 0.2 A.
    # fmax stays 0.01 eV/A, which is what FORCE TOL is derived from.
    opt = LBFGS(atoms, memory=5, maxstep=0.5 * Bohr,
                logfile="test_relax_o2_opt.log")
    opt.run(fmax=0.01)

    # ── the reported-but-not-gated number: where the relaxation lands ──
    # Matching the family and these two settings does not make the two codes the
    # same optimizer -- line search, initial inverse-Hessian scaling and the
    # convergence test are still each implementation's own -- so this stays a
    # measured quantity rather than a 1e-10 gate. What matching buys is that the
    # residual gap is now attributable to implementation, not to having asked two
    # different algorithms the same question.
    #
    # nsteps travels with it because it is the most sensitive signal here: a
    # change that perturbs forces below the energy tolerance can still move the
    # step count, and that shows up before the energy does.
    relaxed_energy_ha = atoms.get_potential_energy() / Hartree
    relaxed_forces = atoms.get_forces() / (Hartree / Bohr)
    nsteps = opt.get_number_of_steps()
    print(f"[relax_o2] converged in {nsteps} LBFGS steps "
          f"(memory=5, maxstep=0.5 Bohr, matched to the native deck)", flush=True)

    return energy_ha, forces_ha_per_bohr, None, {
        "relaxed": {
            "energy_ha": relaxed_energy_ha,
            "forces_ha_per_bohr": relaxed_forces.tolist(),
            "steps": nsteps,
        }
    }
