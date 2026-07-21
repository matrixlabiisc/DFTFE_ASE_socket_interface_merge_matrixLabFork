# ase-dftfe

An [ASE](https://wiki.fysik.dtu.dk/ase/) calculator for
[DFT-FE](https://github.com/dftfeDevelopers/dftfe), the massively-parallel
real-space finite-element DFT code.

The calculator drives a **persistent** DFT-FE process over a socket, so the
expensive MPI/GPU startup happens **once** and is reused across every energy /
force / stress evaluation — making DFT-in-the-loop workflows (relaxation,
molecular dynamics, active learning) practical.

> Status: `1.0.0.dev0` — the redesigned package lives in `src/dftfe_ase/`.
> `@author Mehul Darak`

## Install

```bash
pip install -e .            # from interfaces/ase-dftfe
# or, once released:  pip install ase-dftfe
```

Requires a built DFT-FE binary with socket support (`dftfe --socket host:port`).
See <https://github.com/dftfeDevelopers/install_DFTFE> for per-cluster builds.

## Quickstart

### Zero-config on a known cluster

Inside a job allocation, name your cluster and the calculator resolves the
binary (real vs complex, chosen from your k-points), the launcher, and the
process count for you:

```python
from ase.build import molecule
from dftfe_ase import DFTFE

atoms = molecule("N2")
atoms.center(vacuum=5.0)

with DFTFE(cluster="matrix", nproc=2, use_device=True,
           xc="GGA-PBE", mesh_size=1.0, polynomial_order=6) as calc:
    atoms.calc = calc
    print(atoms.get_potential_energy(), "eV")
```

### Explicit launch (any machine)

```python
calc = DFTFE(
    command="mpirun -np 8 /path/to/dftfe",   # --socket host:port is appended
    xc="GGA-PBE", mesh_size=1.0, polynomial_order=6,
    psp_path="/path/to/psp_library",         # dir, single .upf, or {El: path} dict
)
```

The calculator is a normal ASE `Calculator`: attach it to `atoms`, then use any
ASE optimizer (`BFGS`, `FIRE`), MD integrator, or workflow.

## How launch is resolved

1. `backend=` — an explicit `Backend` (tests / custom transports)
2. `command=` — an explicit launch command (power-user escape hatch)
3. **auto** — resolve binaries (`dftfe_real=`/`dftfe_complex=` > `DFTFE_BIN_*`
   env > `cluster=` profile > container path > `PATH`), pick a launcher
   (`launcher=` > cluster profile > autodetected from the scheduler), and build
   the command at run time. **Real vs complex** is chosen from the k-points
   (Gamma-only → real, k-point mesh → complex).

Built-in cluster profiles: `matrix`, `polaris`, `aurora`, `frontier`,
`pravega`, `siddhi` (see `src/dftfe_ase/config.py`).

## Architecture

```
src/dftfe_ase/
  calculator.py     ASE Calculator (unit conversion, request/response)
  protocol.py       wire format + framing (single source of truth)
  backends/         Backend ABC + SocketBackend (pluggable transport)
  binary.py         real/complex selection + path resolution
  config.py         layered config + cluster profiles
  launchers/        srun / mpirun / mpiexec / jsrun / local
```

## Testing

Unit and mock-integration tests need no DFT-FE binary (a fake DFT-FE speaks the
protocol):

```bash
pytest -q                         # locally
sbatch tests/run_pytest.slurm     # on an HPC login node (submits to a compute node)
```

An end-to-end smoke test against a real GPU build is provided in
`tests/smoke_real_n2.py` / `tests/run_smoke_real.slurm`.

## License

LGPL-2.1-or-later, as part of DFT-FE.
