# ---------------------------------------------------------------------

# Copyright (c) 2017-2025 The Regents of the University of Michigan and DFT-FE
# authors.

# This file is part of the DFT-FE code.

# The DFT-FE code is free software; you can use it, redistribute
# it, and/or modify it under the terms of the GNU Lesser General
# Public License as published by the Free Software Foundation; either
# version 2.1 of the License, or (at your option) any later version.
# The full text of the license can be found in the file LICENSE at
# the top level of the DFT-FE distribution.

# ---------------------------------------------------------------------

# @author Mehul Darak

"""
The dftfe.py file is a Python interface for the DFT-FE code.
It allows you to use DFT-FE as a calculator in the Atomic Simulation Environment (ASE).

Client Server Architecture:
- ASE acts as the client.
- DFT-FE acts as the server.
- The client sends requests to the server and receives responses.

"""

import socket
import os
import json
import subprocess
import time
import os
import numpy as np
from ase.calculators.calculator import Calculator, all_changes
from ase.units import Bohr, Hartree

class dftfeSocketCalculator(Calculator):
    implemented_properties = ['energy', 'forces', 'stress']

    def __init__(self, command='dftfe', 
                 mp_grid=None,
                 mp_grid_shift=None,
                 spin_polarized=None,
                 start_magnetization=None,
                 fermi_temp=None,
                 npkpt=None,
                 mesh_size=None,
                 scf_mixing=None,
                 mixing_scheme=None,
                 polynomial_order=None,
                 tolerance=None,
                 xc=None, # Exchange Correlation Type
                 atom_ball_radius=None, # New explicit parameter
                 num_bands=None, # New explicit parameter (0 = auto)
                 orthogonalization_type=None, # New explicit parameter
                 # New parameters added in expansion
                 wfc_block_size=None, # New explicit parameter (0 = auto)
                 cheby_wfc_block_size=None, # New explicit parameter (0 = auto)
                 compute_forces=True, # New explicit parameter (Default True for standard ASE behavior)
                 compute_stress=False, # New explicit parameter
                 smeared_nuclear_charges=None,
                 use_group_symmetry=None,
                 use_time_reversal_symmetry=None,
                 mixing_history=None,
                 max_scf_iterations=None,
                 dispersion_correction_type=None,
                 pseudopotential_calculation=None,
                 psp_path=None,
                 pseudopotential_filename=None,
                 verbosity=None,
                 use_device=False,
                 host=None, port=0, timeout=300, 
                 keep_scratch=False, # New optional parameter
                 debug_timing=False, # New optional parameter for timing overhead
                 log_file='dftfe_output.log', **kwargs):
        """
        ASE Calculator for DFT-FE using a socket interface.
        """
        Calculator.__init__(self, **kwargs)
        self.launch_cmd = command
        self.host = host if host is not None else socket.gethostname()
        self.port = port
        self.timeout = timeout
        self.log_file_path = log_file
        
        # DFT-FE Parameters
        self.mp_grid = mp_grid
        self.mp_grid_shift = mp_grid_shift
        self.spin_polarized = spin_polarized
        self.start_magnetization = start_magnetization
        self.fermi_temp = fermi_temp
        self.npkpt = npkpt
        self.mesh_size = mesh_size
        self.mixing_scheme = mixing_scheme
        self.scf_mixing = scf_mixing
            
        self.polynomial_order = polynomial_order
        self.tolerance = tolerance
        self.xc = xc
        self.atom_ball_radius = atom_ball_radius
        self.num_bands = num_bands
        self.orthogonalization_type = orthogonalization_type
        
        # New parameters
        self.wfc_block_size = wfc_block_size
        self.cheby_wfc_block_size = cheby_wfc_block_size
        self.smeared_nuclear_charges = smeared_nuclear_charges
        self.use_group_symmetry = use_group_symmetry
        self.use_time_reversal_symmetry = use_time_reversal_symmetry
        self.mixing_history = mixing_history
        self.max_scf_iterations = max_scf_iterations
        self.dispersion_correction_type = dispersion_correction_type
        self.pseudopotential_calculation = pseudopotential_calculation
        
        if isinstance(psp_path, dict):
            self.pseudopotential_filename = "DICT|" + "|".join([f"{k}:{v}" for k, v in psp_path.items()])
        else:
            self.pseudopotential_filename = psp_path if psp_path is not None else pseudopotential_filename
            
        self.keep_scratch = keep_scratch
        self.compute_forces = compute_forces
        self.compute_stress = compute_stress
        
        self.verbosity = verbosity
        self.use_device = use_device
        self.debug_timing = debug_timing
        
        self.process = None
        self.server_socket = None
        self.conn = None
        self.addr = None
        self._cleanup = False

    def _get_free_port(self):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(('', 0))
        port = s.getsockname()[1]
        s.close()
        return port

    def _start_server(self):
        if self.port is None:
            self.port = self._get_free_port()
        
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((self.host, self.port))
        if self.port == 0:
            self.port = self.server_socket.getsockname()[1]
        self.server_socket.listen(1)
        if self.verbosity > 0:
            print(f"ASE: Listening on {self.host}:{self.port}")

    def _launch_dftfe(self):
        if self.process is not None:
            return

        if self.server_socket is None:
            self._start_server()

        # Construct command
        # We expect launch_cmd to be the base command (e.g. "mpirun -n 2 dftfe")
        # We append "--socket host:port"
        cmd = f"{self.launch_cmd} --socket {self.host}:{self.port}"
        if self.verbosity > 0:
            print(f"ASE: Launching: {cmd}")
        
        # Redirect output to file to avoid pipe buffer blocking
        # Respect self.directory if set by base Calculator
        cwd = self.directory if hasattr(self, 'directory') else None
        if cwd:
            os.makedirs(cwd, exist_ok=True)
            self.log_file_path = os.path.join(cwd, os.path.basename(self.log_file_path))

        self.log_file = open(self.log_file_path, "w", buffering=1)
        self.process = subprocess.Popen(cmd, shell=True, stdout=self.log_file, stderr=subprocess.STDOUT, cwd=cwd)
        
        # Accept connection
        self.server_socket.settimeout(120.0) # 120s timeout for startup
        try:
            self.conn, self.addr = self.server_socket.accept()
            if self.verbosity > 0:
                print(f"ASE: Connected to DFT-FE at {self.addr}")
            # Disable timeout for the connection, as DFT-FE initialization can be slow
            self.conn.settimeout(None)
            
            # Read handshake
            handshake = self.conn.recv(1024).decode().strip()
            if handshake != "READY":
                print(f"ASE: Warning: Unexpected handshake: {handshake}")
                
        except socket.timeout:
            self.close()
            raise RuntimeError("DFT-FE failed to connect within timeout.")

    def calculate(self, atoms=None, properties=['energy'], system_changes=['positions', 'numbers', 'cell', 'pbc', 'initial_magmoms', 'initial_charges']):
        if getattr(self, 'debug_timing', False):
            calc_start_time = time.time()

        Calculator.calculate(self, atoms, properties, system_changes)
        
        if not self.conn:
            self._launch_dftfe()
            
        # Convert positions to Bohr (ASE uses Angstrom)
        positions_bohr = atoms.get_positions() / Bohr
        cell_bohr = atoms.get_cell() / Bohr
        
        # Prepare JSON payload
        data = {
            "coords": positions_bohr.tolist(),
            "cell": cell_bohr.tolist(),
            "numbers": atoms.get_atomic_numbers().tolist(),
            "pbc": atoms.get_pbc().tolist(),
            "coords": positions_bohr.tolist(),
            "cell": cell_bohr.tolist(),
            "numbers": atoms.get_atomic_numbers().tolist(),
            "pbc": atoms.get_pbc().tolist(),
            "compute_forces": self.compute_forces,
            "use_device": self.use_device,
            "compute_stress": self.compute_stress or ('stress' in properties),
            "cmd": "run"
        }
        
        # Add optional parameters if defined
        optional_params = {
            "mp_grid": self.mp_grid,
            "mp_grid_shift": self.mp_grid_shift,
            "spin_polarized": self.spin_polarized,
            "start_magnetization": self.start_magnetization,
            "fermi_temp": self.fermi_temp,
            "npkpt": self.npkpt,
            "mesh_size": self.mesh_size,
            "scf_mixing": self.scf_mixing,
            "mixing_scheme": self.mixing_scheme,
            "polynomial_order": self.polynomial_order,
            "tolerance": self.tolerance,
            "xc": self.xc,
            "atom_ball_radius": self.atom_ball_radius,
            "num_bands": self.num_bands,
            "orthogonalization_type": self.orthogonalization_type,
            "wfc_block_size": self.wfc_block_size,
            "cheby_wfc_block_size": self.cheby_wfc_block_size,
            "smeared_nuclear_charges": self.smeared_nuclear_charges,
            "use_group_symmetry": self.use_group_symmetry,
            "use_time_reversal_symmetry": self.use_time_reversal_symmetry,
            "mixing_history": self.mixing_history,
            "max_scf_iterations": self.max_scf_iterations,
            "dispersion_correction_type": self.dispersion_correction_type,
            "pseudopotential_calculation": self.pseudopotential_calculation,
            "pseudopotential_filename": self.pseudopotential_filename,
            "keep_scratch": self.keep_scratch,
            "verbosity": self.verbosity,
        }
        
        for key, val in optional_params.items():
            if val is not None:
                data[key] = val
        
        # Send JSON
        msg = json.dumps(data) + "\n"
        if getattr(self, 'debug_timing', False):
            dftfe_start_time = time.time()
        self.conn.sendall(msg.encode())
        
        # Receive response
        # We need to read until newline or valid JSON
        response_data = ""
        while True:
            chunk = self.conn.recv(4096).decode()
            if not chunk:
                raise RuntimeError("Connection closed by DFT-FE")
            response_data += chunk
            if "\n" in response_data or "}" in response_data: # Simple check
                break
                
        try:
            # If multiple lines, take the last valid JSON? Or first?
            # Our C++ sends one line.
            result = json.loads(response_data.strip())
            if getattr(self, 'debug_timing', False):
                dftfe_end_time = time.time()
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Failed to parse response: {response_data}") from e
            
        # Store results
        # Energy: DFT-FE returns Hartree (implied by `computeDFTFreeEnergy` doc). ASE wants eV.
        self.results['energy'] = result['energy'] * Hartree
        
        # Forces: DFT-FE returns Hartree/Bohr. ASE wants eV/Angstrom.
        # F_eV_Ang = F_Ha_Bohr * (Hartree / Bohr)
        forces_ha_bohr = np.array(result['forces'])
        self.results['forces'] = forces_ha_bohr * (Hartree / Bohr)
        
        # Stress: DFT-FE returns Hartree/Bohr^3. ASE wants eV/Angstrom^3.
        # Stress_eV_Ang3 = Stress_Ha_Bohr3 * (Hartree / Bohr**3)
        # Note: ASE stress is usually Voigt (6 components) or 3x3.
        # Our C++ sends 3x3. ASE can handle 3x3.
        if 'stress' in result:
            stress_ha_bohr3 = np.array(result['stress'])
            self.results['stress'] = stress_ha_bohr3 * (Hartree / Bohr**3)

        if getattr(self, 'debug_timing', False):
            calc_end_time = time.time()
            dftfe_time = dftfe_end_time - dftfe_start_time
            total_time = calc_end_time - calc_start_time
            ase_overhead = total_time - dftfe_time
            print("\n" + "="*40)
            print("ASE-DFTFE Debug Timing:")
            print(f"  Total calculate() Time:  {total_time:.4f} s")
            print(f"  Strict DFT-FE Wait Time: {dftfe_time:.4f} s")
            print(f"  ASE Pre/Post Overhead:   {ase_overhead:.4f} s")
            print("="*40 + "\n")


    def close(self):
        if self.conn:
            try:
                self.conn.sendall(b'{"cmd": "exit"}\n')
                self.conn.close()
            except:
                pass
            self.conn = None
            
        if self.server_socket:
            self.server_socket.close()
            self.server_socket = None
            
        if self.process:
            # Wait for exit
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
            self.process = None
            
        if hasattr(self, 'log_file') and self.log_file:
            self.log_file.close()
            self.log_file = None

    def __del__(self):
        self.close()

# Alias for cleaner import
DFTFE = dftfeSocketCalculator
