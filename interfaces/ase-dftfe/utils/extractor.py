from ase import Atoms
from ase.calculators.calculator import PropertyNotImplementedError

def extract_ml_frame(atoms: Atoms) -> Atoms:
    """
    Safely extracts ML-relevant properties (energy, forces, stress) from an ASE Atoms object
    that has completed a DFT calculation, ensuring they are stored as persistent arrays/info 
    dictionaries so they can be written to an ML-ready format like extxyz.
    
    ASE calculators wrap the properties. To bake them permanently into the Atoms object
    (decoupling it from the calculator), we read them and assign them to `info` and `arrays`.
    
    ASE standard units are inherently preserved (eV, eV/Angstrom).
    
    Args:
        atoms: An ASE Atoms object with an attached, computed calculator.
        
    Returns:
        A new detached ASE Atoms object containing all evaluated properties.
    """
    frame = atoms.copy()
    
    # 1. Total Energy (eV)
    try:
        energy = atoms.get_potential_energy()
        frame.info['energy'] = energy
    except (PropertyNotImplementedError, RuntimeError) as e:
        pass # Energy failed or unavailable
        
    # 2. Atomic Forces (eV/Angstrom)
    try:
        forces = atoms.get_forces()
        frame.arrays['force'] = forces # 'force' is standard extxyz keyword natively recognized by ASE
    except (PropertyNotImplementedError, RuntimeError) as e:
        pass # Forces failed or unavailable
        
    # 3. Cell Stress Tensor (eV/Angstrom^3)
    try:
        stress = atoms.get_stress(voigt=False) # 3x3 array is standard for extxyz cell tensors
        frame.info['stress'] = stress
    except (PropertyNotImplementedError, RuntimeError) as e:
        pass # Stress failed or unavailable
        
    # Remove the calculator to make this a purely static, data-holding frame
    frame.calc = None
    return frame
