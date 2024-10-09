import os
import subprocess
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

def process_file(lean_file):
    source_directory = "NuminaMath/GPT"
    module_path = source_directory.replace('/', '.')
    module_name = lean_file[:-5]  # Remove the '.lean' extension
    import_statement = f"import {module_path}.{module_name}\n"

    # Run the 'lake build' command and capture output
    result = subprocess.run(["lake", "build"], capture_output=True, text=True)
    output = result.stdout + result.stderr

    if "error: " in output:
        return (None, lean_file)  # Return None to indicate an error and the file name
    return (import_statement, None)  # Return the import statement if successful

def main():
    source_directory = "NuminaMath/GPT"
    output_file = "NuminaMath.lean"

    # List all .lean files in the source directory
    files = sorted(f for f in os.listdir(source_directory) if f.endswith('.lean'))
    print(f"==== Total number of .lean files: {len(files)} ====")

    # Use multiprocessing to process files
    with Pool(processes=cpu_count()) as pool:
        results = list(tqdm(pool.imap(process_file, files), total=len(files), desc="Processing .lean files"))

    # Handle file writing and error reporting
    with open(output_file, 'w') as f:
        errors = []
        for content, error in results:
            if content:
                f.write(content)
            if error:
                errors.append(error)

    print(f"Total number of .lean files successfully added: {len(files) - len(errors)}/{len(files)}")
    if errors:
        print("Errors detected in files:", errors)

if __name__ == "__main__":
    main()
