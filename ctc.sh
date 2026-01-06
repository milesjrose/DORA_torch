#!/bin/bash
source .venv/bin/activate 

clear
echo ">Running unit tests..."
pytest DORA_tensorised/nodes/tests/unit/
./clear-cache.sh
echo ">Running functional tests..."
pytest DORA_tensorised/nodes/tests/funct/
./clear-cache.sh
echo ">Running end-to-end tests..."
pytest DORA_tensorised/nodes/tests/e2e/
./clear-cache.sh