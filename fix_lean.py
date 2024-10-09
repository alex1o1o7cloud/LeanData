import os
import subprocess
import argparse

def convert_lean_files(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.lean'):
                file_path = os.path.join(root, file)
                try:
                    # Use sed to replace CRLF with LF
                    subprocess.run(['sed', '-i', 's/\r$//', file_path], check=True)
                    print(f"Converted: {file_path}")
                except subprocess.CalledProcessError as e:
                    print(f"Error converting {file_path}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Convert line endings of .lean files to Unix style using sed.")
    parser.add_argument("directory", help="The directory to search for .lean files")
    args = parser.parse_args()

    if not os.path.isdir(args.directory):
        print(f"Error: {args.directory} is not a valid directory.")
        return

    convert_lean_files(args.directory)

if __name__ == "__main__":
    main()