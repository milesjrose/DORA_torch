#!/bin/bash
source .venv/bin/activate 

clear

pytest DORA_tensorised/

./clear-cache.sh