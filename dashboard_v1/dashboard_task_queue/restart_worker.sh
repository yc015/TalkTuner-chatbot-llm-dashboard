#!/bin/bash

python_script="worker/worker.py"

while true; do
    python "$python_script"
    echo "Python script terminated. Restarting..."
done
