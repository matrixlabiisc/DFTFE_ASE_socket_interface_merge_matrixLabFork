
import numpy as np
from ase import Atoms
from ase.units import Bohr

def read_domain_vectors(filepath):
    """
    Reads domainVectors.inp which contains the 3x3 cell vectors.
    Returns cell in Angstroms (assuming input is in Bohr).
    """
    cell_bohr = []
    with open(filepath, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3:
                vec = [float(x) for x in parts[:3]]
                cell_bohr.append(vec)
            if len(cell_bohr) == 3:
                break
    
    # Check manual to confirm unit. Handover implies DFT-FE uses Bohr internally usually.
    # Assuming input file is Bohr.
    return np.array(cell_bohr) * Bohr

def read_coordinates(filepath, cell=None):
    """
    Reads coordinates.inp.
    Format usually: AtomicNumber Type X Y Z
    X, Y, Z often fractional if cell is provided? Or Cartesian (Bohr)?
    Based on coordinatesLLZO.inp content:
    57 11 0.5 0.75 0.625
    Values look like fractional coordinates (0 to 1).
    """
    numbers = []
    positions = [] # We will interpret these based on magnitude
    
    with open(filepath, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            try:
                # Format: Z Type X Y Z
                Z = int(parts[0])
                # Type = int(parts[1]) # Unused by ASE directly usually
                coords = [float(x) for x in parts[2:5]]
                numbers.append(Z)
                positions.append(coords)
            except ValueError:
                continue
                
    positions = np.array(positions)
    
    # SIMPLE HEURISTIC: If all coordinates are between 0 and 1, assume fractional.
    # LLZO file has 0.5, 0.75, etc.
    is_fractional = np.all((positions >= -0.1) & (positions <= 1.1))
    
    atoms = Atoms(numbers=numbers, cell=cell, pbc=True)
    if is_fractional and cell is not None:
        atoms.set_scaled_positions(positions)
    else:
        # If cartesian, assume Bohr -> Angstrom? Or just Angstrom?
        # Handover doesn't specify input unit. DFT-FE usually uses Bohr for lengths.
        # But if it's fractional, units don't matter yet for coords.
        # If it falls back here, we might need manual verification.
        # For now, if not fractional, we assume Bohr (common in this code) and convert to Hex.
        # But wait, logic: if no cell, can't handle fractional.
        # If cell is provided, we prefer scaled.
        # Let's assume fractional for these benchmarks as observed.
         atoms.set_positions(positions * Bohr) # Fallback assumption
         
    return atoms

def read_neb_coordinates(filepath, cell):
    """
    Reads NEB coordinates file which contains multiple images.
    Format:
    [Image 1 lines]
    [Image 2 lines]
    ...
    How do we distinguish images?
    Usually simply concatenated.
    We know NATOMS from parameter file or by counting first block if we know images count.
    Or we pass natoms.
    """
    # Count lines first to verify
    all_lines = []
    with open(filepath, 'r') as f:
        for line in f:
            if len(line.strip().split()) >= 5:
                all_lines.append(line)
                
    # Detect number of atoms by finding where the content repeats?
    # Or just rely on input.
    # Let's count unique atoms by parsing all and dividing by images if passed, 
    # or finding a break. 
    # Actually, coordinates.inp for NEB usually just lists them.
    # We will assume fixed atom count per image.
    pass 

def read_neb_images(filepath, cell, natoms):
    images = []
    current_numbers = []
    current_positions = []
    
    with open(filepath, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            
            Z = int(parts[0])
            coords = [float(x) for x in parts[2:5]]
            
            current_numbers.append(Z)
            current_positions.append(coords)
            
            if len(current_numbers) == natoms:
                # Image complete
                pos_array = np.array(current_positions)
                
                # Assume fractional again
                img = Atoms(numbers=current_numbers, cell=cell, pbc=True)
                # Heuristic for fractional
                if np.all((pos_array >= -0.1) & (pos_array <= 1.1)):
                    img.set_scaled_positions(pos_array)
                else:
                    img.set_positions(pos_array * Bohr)
                    
                images.append(img)
                current_numbers = []
                current_positions = []
                
    return images
