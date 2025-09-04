#!/bin/bash
source .venv/bin/activate 

echo "Running tests..."
pytest DORA_tensorised/

echo "Clearing cache..."
./clear-cache.sh