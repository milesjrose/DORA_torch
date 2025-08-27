#!/bin/bash
echo "Cleaning Python cache files..."

# Remove __pycache__ directories
find . -type d -name "__pycache__" -exec rm -rf {} +

# Remove .pyc and .pyo files
find . -type f -name "*.pyc" -delete
find . -type f -name "*.pyo" -delete

echo "Done."
