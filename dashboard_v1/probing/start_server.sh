#!/bin/bash

# Startup script for the attribute probing Flask server

echo "=========================================="
echo "Starting Attribute Probing Flask Server"
echo "=========================================="
echo ""

# Check if we're in the right directory
if [ ! -f "app.py" ]; then
    echo "Error: app.py not found. Please run this script from the probing directory."
    exit 1
fi

# Check if CUDA is available
if command -v nvidia-smi &> /dev/null; then
    echo "✓ CUDA is available:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    echo ""
else
    echo "⚠ Warning: CUDA not available. Running on CPU (will be slow)."
    echo ""
fi

# Check if virtual environment is activated
if [ -z "$VIRTUAL_ENV" ]; then
    echo "⚠ Warning: No virtual environment detected."
    echo "  Consider creating one with: python -m venv venv && source venv/bin/activate"
    echo ""
fi

# Set default port and concurrency limit
PORT=${PORT:-5001}
MAX_CONCURRENT_TASKS=${MAX_CONCURRENT_TASKS:-2}

echo "Starting Flask server on port $PORT..."
echo "API will be available at: http://localhost:$PORT"
echo ""
echo "Configuration:"
echo "  Max concurrent probe tasks: $MAX_CONCURRENT_TASKS"
echo "  (Set MAX_CONCURRENT_TASKS env var to change)"
echo ""
echo "Endpoints:"
echo "  - GET  /           - Service info"
echo "  - GET  /health     - Health check"
echo "  - POST /probe      - Train attribute probes (async)"
echo "  - GET  /task/<id>  - Check task status"
echo "  - GET  /tasks      - List all tasks"
echo "  - GET  /available_probes - List available probes"
echo ""
echo "Press Ctrl+C to stop the server"
echo "=========================================="
echo ""

# Start the Flask app
export PORT=$PORT
export MAX_CONCURRENT_TASKS=$MAX_CONCURRENT_TASKS
python app.py

