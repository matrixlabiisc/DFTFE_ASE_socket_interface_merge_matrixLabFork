import os
import re
import numpy as np
from ase import Atoms
from ase.units import Bohr, Hartree

def parse_dftfe_log(log_path: str, symbols: list = None) -> Atoms:
    """
    Parses an offline DFT-FE output log (.op) and reconstructs an ML-ready ASE Atoms object.
    Automatically converts internal units (Hartree, Bohr) to ASE standard units (eV, Angstrom).
    
    Args:
        log_path: Path to the .op file.
        symbols: Optional list of atomic symbols. If not provided, it attempts to read
                 them from `coordinates.inp` in the same directory, or defaults to "X".
                 
    Returns:
        ASE Atoms object with .info['energy'] and .arrays['force'] populated.
    """
    if not os.path.exists(log_path):
        raise FileNotFoundError(f"Log file not found: {log_path}")

    positions_bohr = []
    forces_ha_bohr = []
    energy_ha = None
    
    with open(log_path, 'r') as f:
        lines = f.readlines()
        
    in_coords = False
    in_forces = False
    
    for line in lines:
        line = line.strip()
        
        # 1. Energy
        if "Total free energy:" in line:
            energy_ha = float(line.split(":")[-1].strip())
            
        # 2. Coordinates
        if "Cartesian coordinates of atoms" in line:
            in_coords = True
            positions_bohr = []
            continue
        if in_coords and line.startswith("AtomId"):
            parts = line.split(":")[-1].split()
            if len(parts) >= 3:
                positions_bohr.append([float(parts[0]), float(parts[1]), float(parts[2])])
        elif in_coords and line.startswith("---"):
            if len(positions_bohr) > 0:
                in_coords = False
                
        # 3. Forces
        if "Ion forces (Hartree/Bohr)" in line:
            in_forces = True
            forces_ha_bohr = []
            continue
        if in_forces and re.match(r"^\s*\d+\s+[-+]?\d*\.\d+[eE][-+]?\d+", line):
            parts = line.split()
            if len(parts) >= 4:
                forces_ha_bohr.append([float(parts[1]), float(parts[2]), float(parts[3])])
        elif in_forces and (line.startswith("---") or line.startswith("Maximum absolute")):
            if len(forces_ha_bohr) > 0:
                in_forces = False

    # Convert to standard units
    positions_ang = np.array(positions_bohr) * Bohr if positions_bohr else None
    
    # Auto-resolve symbols
    num_atoms = len(positions_ang) if positions_ang is not None else 0
    if symbols is None and num_atoms > 0:
        coord_file = os.path.join(os.path.dirname(log_path), "coordinates.inp")
        if os.path.exists(coord_file):
            try:
                symbols = []
                with open(coord_file, 'r') as cf:
                    header = True
                    for ln in cf:
                        ln = ln.strip()
                        if not ln or ln.startswith("#"): continue
                        if header:
                            header = False
                            continue
                        # format is: atom_type x y z
                        parts = ln.split()
                        # Real symbols are hard to get purely from type ID without pseudo.inp.
                        # For now, default to X if real symbols aren't explicitly passed.
                        symbols.append("X") 
            except Exception:
                symbols = ["X"] * num_atoms
        else:
            symbols = ["X"] * num_atoms
            
    if symbols is None or len(symbols) == 0:
        symbols = ["X"]

    # Reconstruct Atoms
    atoms = Atoms(symbols=symbols, positions=positions_ang)
    
    if energy_ha is not None:
        atoms.info['energy'] = energy_ha * Hartree
    if len(forces_ha_bohr) > 0:
        atoms.arrays['force'] = np.array(forces_ha_bohr) * (Hartree / Bohr)
        
    return atoms
