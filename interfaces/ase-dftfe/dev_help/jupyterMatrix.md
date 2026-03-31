To run the ASE-DFTFE interface in a Jupyter Notebook on a cluster like Matrix, you essentially need to move from a "Batch" mindset (submit and wait) to an "Interactive" mindset.

Here is the professional workflow to set this up.

Step 1: Request an Interactive Allocation
Do not run the Jupyter server on the login node. You need to request compute nodes first. Run this from your terminal:

bash
# Request 1 node with GPUs for 3 hours
salloc --nodes=1 --ntasks-per-node=16 --gres=gpu:8 --time=03:00:00
Wait until the allocation is granted and you are logged into a compute node (e.g., cn01).

Step 2: Set the Environment on the Compute Node
Once you are on the compute node, load the same modules you use in your SLURM scripts:

bash
# Load Modules
module load spack
module load openmpi/5.0.6-gcc-13.3.0-ytficip nccl/2.23.4-1-gcc-13.3.0-xyspmp2 gdrcopy/2.4.1-gcc-13.3.0-dvwa323
# Set Library Paths (Crucial for the backend to link correctly)
export CUDA_PATH="/apps/softwares/spack/opt/spack/linux-ubuntu24.04-cascadelake/gcc-13.3.0/cuda-12.6.3-wdrnyla44db6bmihvkefukg5f6axzprg"
export LIBRARY_PATH="/storage/dftfeDependenciesNoMKL/linAlgLibs/install/lib:${CUDA_PATH}/lib64:${LIBRARY_PATH}"
export LD_LIBRARY_PATH="/storage/dftfeDependenciesNoMKL/linAlgLibs/install/lib:${CUDA_PATH}/lib64:${LD_LIBRARY_PATH}"
# Activate Your Python Environment
source ~/.venvs/ase-env/bin/activate
Step 3: Launch Jupyter
Now, start the Jupyter server from the compute node.

bash
# Start jupyter without a browser and on a specific port
jupyter notebook --no-browser --port=8888 --ip=0.0.0.0
Step 4: Access the Notebook (SSH Tunneling)
If you are connecting from your local laptop, open a new terminal on your laptop and create a tunnel to the compute node:

bash
# Replace <user> and <compute-node> with your actual details
ssh -L 8888:<compute-node>:8888 <user>@matrix-server-address
Now, open your browser and go to http://localhost:8888.

Step 5: The Notebook Code
Inside your .ipynb, you can initialize the calculator exactly as you do in a script. The 

dftfe
 process will be spawned in the background and stay connected to your notebook's Python kernel.

python
import os
from dftfe import DFTFE
from ase.build import bulk
# IMPORTANT: Ensure the environment variables are visible to the kernel
# If you didn't set LIBRARY_PATH before starting jupyter, do it here:
os.environ["LIBRARY_PATH"] = "/storage/dftfeDependenciesNoMKL/linAlgLibs/install/lib:" + os.environ.get("LIBRARY_PATH", "")
# Initialize as usual
calc = DFTFE(
    command="mpirun -np 8 /home/pa01/Mehul/DFTFE/build_gpu/release/real/dftfe",
    psp_path="/home/pa01/Mehul/DFTFE/interfaces/ase-dftfe/psp_spms/",
    use_device=True,
    # ... other params
)
atoms = bulk("Cu", "fcc", a=3.6)
atoms.calc = calc
# This will trigger the socket transmission and the C++ backend will start
energy = atoms.get_potential_energy()
print(f"Energy: {energy}")
Pro Tip: Using VS Code Jupyter Extension
If you use VS Code, it’s even easier:

Connect VS Code to the server via Remote-SSH.
Open your .ipynb file.
Click "Select Kernel" -> "Existing Jupyter Server" and paste the URL (with token) provided when you ran jupyter notebook in Step 3.
VS Code handles the tunneling for you!

---

### Multi-Node / Multi-GPU Interactive Support

The architecture fully supports scaling your interactive sessions across multiple nodes. Since Python (the master) and DFT-FE (the worker) communicate via TCP sockets, you can launch a massively parallel job across the cluster and control it from a single notebook cell.

**1. Request a Multi-Node Allocation**
Modify your `salloc` to request more than one node:
```bash
# Request 2 nodes, 16 tasks per node (32 total), 8 GPUs per node
salloc --nodes=2 --ntasks-per-node=16 --gres=gpu:8 --time=03:00:00
```

**2. Update your Notebook Command**
Scale the `mpirun` count to match your full allocation:
```python
calc = DFTFE(
    # Launch on 32 total ranks across both nodes
    command="mpirun -np 32 /home/pa01/Mehul/DFTFE/build_gpu/release/real/dftfe",
    use_device=True,  # Automatically binds ranks to GPUs on all nodes
    # ...
)
```

**How it works:**
*   **MPI:** The `mpirun` command detects the SLURM allocation and automatically spreads the 32 processes across both nodes.
*   **GPU Binding:** Each rank detects its local GPU on its specific node.
*   **Networking:** Only **Rank 0** establishes the socket connection back to your Jupyter Notebook. As long as Rank 0 can reach the notebook's IP (usually `localhost` on the lead compute node), the multi-node communication is seamless.

