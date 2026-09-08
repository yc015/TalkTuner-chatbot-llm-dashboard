from app import app, socketio
import argparse
from flask_socketio import SocketIO
from flask import Flask


# Argument parsing
parser = argparse.ArgumentParser(description="Run the web application")
parser.add_argument('--port', type=int, default=8510, help='Port to run the application on')
args = parser.parse_args()

if __name__ == '__main__':
    socketio.run(app, host="0.0.0.0", port=args.port, debug=False)