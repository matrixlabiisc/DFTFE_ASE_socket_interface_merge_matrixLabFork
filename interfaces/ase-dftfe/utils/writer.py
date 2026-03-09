import os
from ase import Atoms
from ase.io import write

def append_to_extxyz(atoms: Atoms, filename: str):
    """
    Appends a single static ML-ready ASE Atoms frame to an Extended XYZ (.extxyz) dataset file.
    
    ASE's `write` function natively handles embedding `atoms.info` into the comment line
    and `atoms.arrays` into the atomic columns. It correctly maintains the 
    standard XYZ specifications.
    
    Args:
        atoms: The ASE Atoms object (already processed by extractor to decouple properties).
        filename: Destination .extxyz file path. Uses append mode.
    """
    # Simply route through ASE's native writer in append mode
    # ASE understands .extxyz extensions and uses the extxyz format automatically.
    write(filename, atoms, format='extxyz', append=True)
