#!/bin/bash

# Run lake build and capture its output and exit status
build_output=$(lake build 2>&1)
build_status=$?

# Filter out warnings and linter notes
filtered_output=$(echo "$build_output" | grep -v -E "^warning:|^note: this linter can be disabled")

# Print the filtered output
echo "$filtered_output"

# Check if there were any errors (non-zero exit status)
if [ $build_status -ne 0 ]; then
    echo "Build failed with exit status $build_status"
    # Print the last few lines of the original output to see error messages
    echo "Last few lines of the original build output:"
    echo "$build_output" | tail -n 20
fi

# Exit with the original build status
exit $build_status