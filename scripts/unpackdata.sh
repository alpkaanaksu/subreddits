#!/bin/bash

# Check if the user provided a file as an argument
if [ -z "$1" ]; then
  echo "Usage: $0 filename.zst"
  exit 1
fi

# Get the input file name
input_file="$1"

# Check if the file exists
if [ ! -f "$input_file" ]; then
  echo "File not found!"
  exit 1
fi

# Unpack the .zst file using zstd
output_file="${input_file%.zst}.ndjson"

echo "Unpacking $input_file to $output_file..."

# Decompress the .zst file
zstd -d "$input_file" -o "$output_file"

# Check if decompression was successful
if [ $? -eq 0 ]; then
  echo "File successfully unpacked to $output_file"
else
  echo "Error occurred during decompression"
fi
