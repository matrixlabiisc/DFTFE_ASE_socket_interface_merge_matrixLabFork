import os
import glob
from .parser import parse_dftfe_log
from .writer import append_to_extxyz

def build_dataset(input_dir: str, output_file: str, log_extension: str = "*.op", symbols: list = None, extract_properties: list = None, append: bool = False, model: str = None):
    """
    Crawls a directory for DFT-FE output logs, parses them all, and merges them into a single 
    Extended XYZ dataset.
    
    Args:
        input_dir: Root directory to search for logs.
        output_file: Destination .extxyz file.
        log_extension: Pattern or list of patterns/extensions to search (default: "*.op").
                       Can be a comma-separated string like "*.op, *.out, *.log" or ".op, .out, .log".
        symbols: Hardcoded symbol list if parsing raw logs without ASE metadata.
        extract_properties: Optional list of properties to extract (e.g. ['energy', 'force']). 
                            If not specified, whatever properties are available in the log are extracted.
        append: If True, appends to the output_file if it already exists. If False, overwrites it. (default: False)
        model: Optional string naming a target ML architecture. E.g. passing 'MACE' renames energy/forces to REF_energy/REF_forces.
    """
    # Handle append mode: if not appending, remove existing file
    if not append and os.path.exists(output_file):
        os.remove(output_file)

    if isinstance(log_extension, str):
        extensions = [ext.strip() for ext in log_extension.split(',')]
    else:
        extensions = log_extension

    log_files = []
    for ext in extensions:
        if not ext.startswith('*'):
            ext = f"*{ext}" if ext.startswith('.') else f"*.{ext}"
            
        search_path = os.path.join(input_dir, "**", ext)
        log_files.extend(glob.glob(search_path, recursive=True))
        
    # Remove duplicates
    log_files = list(set(log_files))
    
    if not log_files:
        print(f"No log files found matching extensions in {input_dir}")
        return
        
    count = 0
    for file in log_files:
        try:
            frame = parse_dftfe_log(file, symbols=symbols)
            
            # If the user requested specific properties, filter out everything else
            if extract_properties is not None:
                for key in list(frame.info.keys()):
                    if key not in extract_properties and key not in ['source_file', 'filename']:
                        del frame.info[key]
                for key in list(frame.arrays.keys()):
                    if key not in extract_properties and key not in ['positions', 'numbers']:
                        del frame.arrays[key]

            # Model Specific transformations
            if model is not None and model.upper() == "MACE":
                if 'energy' in frame.info:
                    frame.info['REF_energy'] = frame.info.pop('energy')
                if 'force' in frame.arrays:
                    frame.arrays['REF_forces'] = frame.arrays.pop('force')
                elif 'forces' in frame.arrays:
                    frame.arrays['REF_forces'] = frame.arrays.pop('forces')

            # Check what properties we actually have after potentially filtering
            final_info = [k for k in frame.info.keys() if k not in ['source_file', 'filename']]
            final_arrays = [k for k in frame.arrays.keys() if k not in ['positions', 'numbers']]
            
            # Simple health check: keep the frame if we extracted at least something
            if len(final_info) > 0 or len(final_arrays) > 0:
                frame.info['source_file'] = file
                frame.info['filename'] = os.path.splitext(os.path.basename(file))[0]
                append_to_extxyz(frame, output_file)
                count += 1
            else:
                if extract_properties is not None:
                    print(f"Skipping {file}: None of the requested properties ({extract_properties}) were found.")
                else:
                    print(f"Skipping {file}: No parseable properties (like energy, forces) found.")
        except Exception as e:
            print(f"Failed to parse {file}: {e}")
            
    print(f"Dataset successfully built with {count} frames written to {output_file}.")
