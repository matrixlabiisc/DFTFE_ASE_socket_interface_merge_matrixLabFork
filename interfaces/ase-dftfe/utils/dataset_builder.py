import os
import glob
from .parser import parse_dftfe_log
from .writer import append_to_extxyz

def build_dataset(input_dir: str, output_file: str, log_extension: str = "*.op", symbols: list = None):
    """
    Crawls a directory for DFT-FE output logs, parses them all, and merges them into a single 
    Extended XYZ dataset.
    
    Args:
        input_dir: Root directory to search for logs.
        output_file: Destination .extxyz file.
        log_extension: Pattern of logs to search (default: *.op).
        symbols: Hardcoded symbol list if parsing raw logs without ASE metadata.
    """
    search_path = os.path.join(input_dir, "**", log_extension)
    log_files = glob.glob(search_path, recursive=True)
    
    if not log_files:
        print(f"No log files found matching {search_path}")
        return
        
    count = 0
    for file in log_files:
        try:
            frame = parse_dftfe_log(file, symbols=symbols)
            
            # Simple health check
            if 'energy' in frame.info and 'force' in frame.arrays:
                frame.info['source_file'] = file
                append_to_extxyz(frame, output_file)
                count += 1
            else:
                print(f"Skipping {file}: Missing energy or forces.")
        except Exception as e:
            print(f"Failed to parse {file}: {e}")
            
    print(f"Dataset successfully built with {count} frames written to {output_file}.")
