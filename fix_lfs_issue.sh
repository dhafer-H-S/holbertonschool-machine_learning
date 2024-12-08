#!/bin/bash

# Untrack all LFS files
git lfs untrack "*"

# Remove all LFS files from the repository
git rm -r --cached .

# Add all files back to the repository
git add .

# Commit the changes
git commit -m "Remove all LFS files"

# Remove large files from the repository history
git filter-repo --strip-blobs-bigger-than 100M --force

# Re-add the remote
git remote add origin https://github.com/dhafer-H-S/holbertonschool-machine_learning.git

# Force push the changes
git push --force origin main