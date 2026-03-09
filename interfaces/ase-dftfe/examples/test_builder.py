import os
import sys

# Add the parent directory to Python path to import utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import build_dataset

if __name__ == "__main__":
    demo_dir = "/home/pa01/Mehul/DFTFE/demo/ex1"
    output_xyz = "test_offline_dataset.extxyz"
    
    # Remove old one if exists
    if os.path.exists(output_xyz):
        os.remove(output_xyz)
        
    print(f"Building dataset from logs in {demo_dir}...")
    # Using explicit symbols because demo ex1 doesn't have an ASE structure generated
    # It is a N2 molecule ("AtomId 0", "AtomId 1") -> we specify ["N", "N"]
    build_dataset(input_dir=demo_dir, output_file=output_xyz, symbols=["N", "N"])
    
    if os.path.exists(output_xyz):
        print(f"Success! {output_xyz} created. Reading first few lines:")
        with open(output_xyz, 'r') as f:
            print("".join(f.readlines()[:10]))
    else:
        print("Failed: No dataset created.")
