#!/bin/bash

# Usage: ./run_filter.sh <script.py> "<additional args>" "file1.jsonl,file2.jsonl,..."
# Example: ./run_filter.sh filter_for_hacks.py "--model claude-3-opus --temperature 0.5" "data1.jsonl,data2.jsonl"

if [ $# -lt 3 ]; then
    echo "Usage: $0 <script.py> \"<additional args>\" \"file1.jsonl,file2.jsonl,...\""
    echo "Example: $0 filter_for_hacks.py \"--model claude-3-opus --temperature 0.5\" \"data1.jsonl,data2.jsonl\""
    exit 1
fi

SCRIPT="$1"
ADDITIONAL_ARGS="$2"
FILES="$3"

# Check if script exists
if [ ! -f "$SCRIPT" ]; then
    echo "Error: Script '$SCRIPT' not found!"
    exit 1
fi

# Convert comma-separated files to array
IFS=',' read -ra FILE_ARRAY <<< "$FILES"

# Process each file
for file in "${FILE_ARRAY[@]}"; do
    # Trim whitespace
    file=$(echo "$file" | xargs)
    
    # Check if file exists
    if [ ! -f "$file" ]; then
        echo "Error: File '$file' not found!"
        exit 1
    fi
    
    # Get the stem (filename without extension)
    stem="${file%.jsonl}"
    stem="${stem%.JSONL}"
    output_folder="$stem"
    
    # Check if output folder already exists
    if [ -d "$output_folder" ]; then
        echo "Skipping $file - output folder '$output_folder' already exists"
        continue
    fi
    
    echo "=========================================="
    echo "Processing: $file"
    echo "Output folder: $output_folder"
    echo "=========================================="
    
    # Run the script with the file and additional arguments
    python "$SCRIPT" "$file" --output-folder "$output_folder" $ADDITIONAL_ARGS
    
    if [ $? -eq 0 ]; then
        echo "✓ Successfully processed $file"
    else
        echo "✗ Error processing $file"
        exit 1
    fi
    echo ""
done

echo "All files processed successfully!"