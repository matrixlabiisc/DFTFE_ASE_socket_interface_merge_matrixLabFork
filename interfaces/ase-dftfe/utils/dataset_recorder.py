import os
import time
from typing import Optional
from ase import Atoms
from ase.calculators.calculator import PropertyNotImplementedError

from .extractor import extract_ml_frame
from .writer import append_to_extxyz

class DatasetRecorder:
    """
    An online active-learning logger for DFT-FE.
    
    This class is intended to sit inside geometry optimization or molecular 
    dynamics loops. When called, it securely extracts ML properties from
    ASE Atoms objects and streams them out to an `.extxyz` training file.
    """
    
    def __init__(self, output_file: str, max_force_threshold_ev_ang: float = 100.0):
        """
        Args:
            output_file: Path to the .extxyz dataset file.
            max_force_threshold_ev_ang: Automatically drop frames where forces exceed this threshold
                (avoids polluting ML datasets with unphysical crashed structures).
        """
        self.output_file = output_file
        self.max_force_threshold = max_force_threshold_ev_ang
        self.recorded_frames = 0
        
    def record(self, atoms: Atoms, step: Optional[int] = None, metadata: Optional[dict] = None) -> bool:
        """
        Record the current state of the Atoms object into the dataset.
        
        Args:
            atoms: Computed ASE Atoms object.
            step: Optional MD or Optimization step number (added as info['step']).
            metadata: Optional dictionary of extra tags to assign to the frame (e.g., source, temperature).
            
        Returns:
            bool: True if recorded, False if the frame was rejected due to thresholds or missing properties.
        """
        # Pull cleanly separated frame
        frame = extract_ml_frame(atoms)
        
        # We enforce strict requirements for ML datasets: must have Energy and Forces
        if 'energy' not in frame.info:
            print(f"Warning: DatasetRecorder rejecting frame (no energy found).")
            return False
        if 'force' not in frame.arrays:
            print(f"Warning: DatasetRecorder rejecting frame (no forces found).")
            return False
            
        # Optional Force Check (Active Learning safety)
        import numpy as np
        forces = frame.arrays['force']
        max_f = np.max(np.abs(forces))
        if max_f > self.max_force_threshold:
            print(f"Warning: DatasetRecorder rejecting frame (Max force {max_f:.2f} > threshold {self.max_force_threshold}).")
            return False
            
        # Append Metadata
        if metadata:
            for k, v in metadata.items():
                frame.info[k] = v
        frame.info['source'] = 'dftfe'
        if step is not None:
            frame.info['step'] = step
            
        # Write
        append_to_extxyz(frame, self.output_file)
        self.recorded_frames += 1
        return True
