#!/bin/bash

# Untrack the LFS files
git lfs untrack "supervised_learning/RNNs/0-rnn_cell.py"
git lfs untrack "supervised_learning/RNNs/1-rnn.py"
git lfs untrack "supervised_learning/RNNs/2-gru_cell.py"
git lfs untrack "supervised_learning/RNNs/3-lstm_cell.py"
git lfs untrack "supervised_learning/RNNs/4-deep_rnn.py"
git lfs untrack "supervised_learning/RNNs/5-bi_forward.py"
git lfs untrack "supervised_learning/RNNs/6-bi_backward.py"
git lfs untrack "supervised_learning/RNNs/7-bi_output.py"
git lfs untrack "supervised_learning/RNNs/8-bi_rnn.py"

# Remove the cached files
git rm --cached "supervised_learning/RNNs/0-rnn_cell.py"
git rm --cached "supervised_learning/RNNs/1-rnn.py"
git rm --cached "supervised_learning/RNNs/2-gru_cell.py"
git rm --cached "supervised_learning/RNNs/3-lstm_cell.py"
git rm --cached "supervised_learning/RNNs/4-deep_rnn.py"
git rm --cached "supervised_learning/RNNs/5-bi_forward.py"
git rm --cached "supervised_learning/RNNs/6-bi_backward.py"
git rm --cached "supervised_learning/RNNs/7-bi_output.py"
git rm --cached "supervised_learning/RNNs/8-bi_rnn.py"

# Add all Python files back
git add "*.py"

# Commit the changes
git commit -m "Remove LFS tracking from RNN files"

# Force push the changes
git push --force origin main