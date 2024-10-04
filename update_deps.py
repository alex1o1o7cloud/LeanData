import os

source_directory = "NuminaMath/GPT"
output_file = "NuminaMath.lean"

files = os.listdir(source_directory)
lean_files = [f for f in files if f.endswith('.lean')]
module_path = source_directory.replace('/', '.')

with open(output_file, 'a') as file:
    file.write("\n")

    for lean_file in lean_files:
        module_name = lean_file[:-5]  # Remove the '.lean' extension
        import_statement = f"import {module_path}.{module_name}"
        file.write(import_statement + "\n")
