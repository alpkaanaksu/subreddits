#!/bin/bash

# Check if subreddit name is provided
if [ -z "$1" ]; then
  echo "Usage: $0 <subreddit name>"
  exit 1
fi

# Get the subreddit name
subreddit="$1"

echo "====== Fetching Data ======"

# Step 1: Download the files
./fetchsubreddit.sh "$subreddit"


echo "====== Unpacking .zst Files ======"
# Step 2: Unpack the downloaded .zst files
for zst_file in "../data/$subreddit"/*.zst; do
  ./unpackdata.sh "$zst_file"
done

echo "====== Removing .zst Files ======"

# Step 3: Clear .zst files
for zst_file in "../data/$subreddit"/*.zst; do
  rm "$zst_file"
done

echo "DONE."