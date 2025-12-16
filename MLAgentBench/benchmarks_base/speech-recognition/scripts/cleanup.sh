#!/usr/bin/env bash
# run from scripts/ directory

# remove data downloaded by prepare.py
rm -r test_data ../env/data

# remove libribrain directories generated during training/evaluation
rm -r ../env/libribrain

# remove output files
rm -r ../env/output
