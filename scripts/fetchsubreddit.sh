#!/bin/bash

# Check if subreddit name is provided
if [ -z "$1" ]; then
  echo "Usage: $0 <subreddit name>"
  exit 1
fi

# Get the subreddit name from the argument
subreddit="$1"

# Base URL for downloading files
base_url="https://the-eye.eu/redarcs/files"

# Create a folder named after the subreddit if it doesn't exist
mkdir -p "../data/$subreddit"

# Define the two file types
files=("submissions" "comments")

# Loop through the file types and download each one
for file_type in "${files[@]}"; do
  file_url="$base_url/${subreddit}_${file_type}.zst"
  output_file="../data/${subreddit}/${subreddit}_${file_type}.zst"
  
  echo "Downloading $file_url to $output_file..."

  # Use curl to download the file
  curl -o "$output_file" "$file_url"

  # Check if the download was successful
  if [ $? -eq 0 ]; then
    echo "$file_type successfully downloaded and saved to $output_file"
  else
    echo "Failed to download $file_type for subreddit $subreddit"
  fi
done
