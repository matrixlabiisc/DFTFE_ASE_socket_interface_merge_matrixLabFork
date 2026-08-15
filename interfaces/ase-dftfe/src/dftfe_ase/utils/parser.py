import os
import re
import numpy as np
from ase import Atoms
from ase.units import Bohr, Hartree
from ase.data import chemical_symbols

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
    frac_positions = []
    cell_bohr = np.zeros((3, 3))
    pbc = [False, False, False]
    forces_ha_bohr = []
    energy_ha = None
    
    with open(log_path, 'r') as f:
        lines = f.readlines()
        
    in_coords = False
    in_frac = False
    in_forces = False
    
    for line in lines:
        line = line.strip()
        
        # 1. Energy
        if "Total free energy:" in line:
            energy_ha = float(line.split(":")[-1].strip())
            
        # PBC
        if line.startswith("set PERIODIC1"):
            pbc[0] = "true" in line.split("=")[-1].lower()
        elif line.startswith("set PERIODIC2"):
            pbc[1] = "true" in line.split("=")[-1].lower()
        elif line.startswith("set PERIODIC3"):
            pbc[2] = "true" in line.split("=")[-1].lower()

        # Cell Vectors (often printed above coordinates for periodic)
        if line.startswith("v1 :"):
            pts = line.split(":")[1].split()
            cell_bohr[0] = [float(p) for p in pts]
        elif line.startswith("v2 :"):
            pts = line.split(":")[1].split()
            cell_bohr[1] = [float(p) for p in pts]
        elif line.startswith("v3 :"):
            pts = line.split(":")[1].split()
            cell_bohr[2] = [float(p) for p in pts]
            
        # 2. Coordinates
        if "Cartesian coordinates of atoms" in line:
            in_coords = True
            in_frac = False
            positions_bohr = []
            continue
        elif "Fractional coordinates of atoms" in line:
            in_frac = True
            in_coords = False
            frac_positions = []
            continue

        if in_coords and line.startswith("AtomId"):
            parts = line.split(":")[-1].split()
            if len(parts) >= 3:
                positions_bohr.append([float(parts[0]), float(parts[1]), float(parts[2])])
        elif in_frac and line.startswith("AtomId"):
            parts = line.split(":")[-1].split()
            if len(parts) >= 3:
                frac_positions.append([float(parts[0]), float(parts[1]), float(parts[2])])

        elif (in_coords or in_frac) and line.startswith("---"):
            if len(positions_bohr) > 0:
                in_coords = False
            if len(frac_positions) > 0:
                in_frac = False
                
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

    # Convert Fractional to Cartesian if only Fractional was printed
    if len(positions_bohr) == 0 and len(frac_positions) > 0:
        positions_bohr = np.dot(np.array(frac_positions), cell_bohr)

    # Convert to standard units
    positions_ang = np.array(positions_bohr) * Bohr if len(positions_bohr) > 0 else None
    cell_ang = cell_bohr * Bohr if not np.allclose(cell_bohr, 0) else None
    
    # Auto-resolve symbols
    num_atoms = len(positions_ang) if positions_ang is not None else 0
    if symbols is None and num_atoms > 0:
        coord_file = os.path.join(os.path.dirname(log_path), "coordinates.inp")
        if os.path.exists(coord_file):
            try:
                parsed_symbols = []
                with open(coord_file, 'r') as cf:
                    for ln in cf:
                        ln = ln.strip()
                        if not ln or ln.startswith("#"): continue
                        parts = ln.split()
                        if len(parts) >= 5:
                            # In natively dumped coordinates.inp, Col 0 is Atomic Number
                            z = int(float(parts[0]))
                            parsed_symbols.append(chemical_symbols[z])
                if len(parsed_symbols) == num_atoms:
                    symbols = parsed_symbols
            except Exception:
                pass
            
    if symbols is None or len(symbols) != (num_atoms if num_atoms > 0 else 1):
        symbols = ["X"] * (num_atoms if num_atoms > 0 else 1)

    # Reconstruct Atoms
    atoms = Atoms(symbols=symbols, positions=positions_ang)
    if cell_ang is not None:
        atoms.set_cell(cell_ang)
        atoms.set_pbc(pbc)
    
    if energy_ha is not None:
        atoms.info['energy'] = energy_ha * Hartree
    if len(forces_ha_bohr) == len(atoms) and len(atoms) > 0:
        atoms.arrays['force'] = np.array(forces_ha_bohr) * (Hartree / Bohr)
        
    return atoms
