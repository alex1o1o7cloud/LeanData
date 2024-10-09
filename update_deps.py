import os
import subprocess
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

def process_file(lean_file):
    source_directory = "NuminaMath/GPT"
    module_path = source_directory.replace('/', '.')
    module_name = lean_file[:-5]  # Remove the '.lean' extension

    command = ["lake", "build", f"{module_path}.{module_name}"]
    print(f"Running command: {' '.join(command)}")

    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=20)
        output = result.stdout + result.stderr
        if result.returncode != 0:
            # print(f"Error processing {lean_file}: {output}")
            print(f"**** Error processing {lean_file} ****")
            return None
        else:
            return f"import {module_path}.{module_name}\n"
    except subprocess.TimeoutExpired:
        print(f"**** Timeout expired for {lean_file} ****")
        return None

def main():
    source_directory = "NuminaMath/GPT"
    files = sorted(f for f in os.listdir(source_directory) if f.endswith('.lean'))
    print(f"==== Total number of .lean files: {len(files)} ====")

    num_processes = max(1, min(cpu_count() - 2, 20))  # Conservative process count
    with Pool(processes=num_processes) as pool:
        results = list(tqdm(pool.imap(process_file, files), total=len(files), desc="Processing .lean files"))

    with open("NuminaMath.lean", 'w') as file:
        for import_statement in filter(None, results):
            file.write(import_statement)

    errors_count = results.count(None)
    print(f"Total number of .lean files successfully built: {len(files) - errors_count}/{len(files)}")
    if errors_count:
        print("Errors detected in some files.")

if __name__ == "__main__":
    main()
