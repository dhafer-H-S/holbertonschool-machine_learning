# #!/bin/bash

# # List all LFS-tracked files
# lfs_files=$(git lfs ls-files -n)

# # Untrack all LFS files
# for file in $lfs_files; do
#     git lfs untrack "$file"
# done

# # Remove cached files
# for file in $lfs_files; do
#     git rm --cached "$file"
# done

# # Add files back
# git add .

# # Commit the changes
# git commit -m "Remove LFS tracking from all files"

# # Force push the changes
# git push --force origin main
#!/bin/bash

# Untrack all LFS files
git lfs untrack "*"

# Remove all LFS files from the repository
git rm -r --cached .

# Add all files back to the repository
git add .

# Commit the changes
git commit -m "Remove all LFS files"

# Force push the changes
git push --force origin main