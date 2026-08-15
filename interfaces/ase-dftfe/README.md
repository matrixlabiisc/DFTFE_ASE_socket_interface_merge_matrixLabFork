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
process count for you. Point it at your build once, via `bin_dir=`, the
`DFTFE_BIN_DIR` environment variable, or a one-line site file (below):

```python
from ase.build import molecule
from dftfe_ase import DFTFE

atoms = molecule("N2")
atoms.center(vacuum=5.0)

with DFTFE(cluster="aurora", nproc=2, use_device=True,
           xc="GGA-PBE", mesh_size=1.0, polynomial_order=6) as calc:
    atoms.calc = calc
    print(atoms.get_potential_energy(), "eV")
```

### Your machine, once

Cluster profiles ship scheduler, launcher and GPU backend only. Binary and
pseudopotential paths are per-user, so no profile carries one — a package that
hardcoded someone's `bin_dir` would resolve, for everyone else on that same
cluster, to a directory they cannot read.

Record yours in `~/.config/dftfe_ase/profiles.json` (or point `DFTFE_ASE_CONFIG`
anywhere you like):

```json
{
  "aurora":    {"bin_dir": "/lus/.../my_build", "psp_path": "/home/me/psp"},
  "mycluster": {"scheduler": "slurm", "launcher": "srun", "gpu_backend": "cuda",
                "bin_dir": "/home/me/dftfe/build"}
}
```

Fields override the shipped profile of the same name one at a time; a name that
is not shipped defines a new cluster, and then `scheduler` and `launcher` are
required. This is also where an institution-local machine belongs: its module
strings and dependency directories exist on one site and rot on that site's next
rebuild, so they are not shipped in the package.

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
   env > `bin_dir=` > site file > container path > `PATH`), pick a launcher
   (`launcher=` > cluster profile > autodetected from the scheduler), and build
   the command at run time. **Real vs complex** is chosen from the k-points
   (Gamma-only → real, k-point mesh → complex).

Shipped cluster profiles: `polaris`, `aurora`, `frontier` — the public
leadership systems, carrying scheduler, launcher and GPU backend and nothing
else. Anything institution-local goes in your own `profiles.json` (above).

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

```bash
pytest -q
```

A fake DFT-FE speaks the protocol, so the suite needs no binary, no MPI and no
scheduler, and finishes in seconds. It runs anywhere, including a login node.

End-to-end validation against a real build lives in `automated_tests/`: six
cases run twice, once through the socket and once as a native `.prm` deck on the
same binary, and compared. That needs an allocation, so the job scripts that
drive it are site-specific and live outside this repo, in
[install_DFTFE](https://github.com/dftfeDevelopers/install_DFTFE).

## License

LGPL-2.1-or-later, as part of DFT-FE.
