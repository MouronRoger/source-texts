#!/bin/bash
# Cleanup old scaffold files

cd /Users/james/GitHub/source-texts/scaffold-source-texts/scaffold

# Remove old multi-file structure
rm -f 01_scaffold_specification.yaml
rm -f 02_scaffold_roadmap.yaml
rm -f 03_scaffold_runner.yaml
rm -f scaffold.py
rm -f progress.json

echo "Old scaffold files removed. Current structure:"
ls -la
