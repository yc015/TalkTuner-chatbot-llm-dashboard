from app import app
import argparse
# from app import socketio

# The port number should be the same as the front end
#try:
# socketio.run(app, host='localhost', port=8500, use_reloader=False, debug=True)
# app.run(host='localhost', port=8500, use_reloader=False, debug=True, ssl_context='adhoc')
# if __name__ == '__main__':
#     app.run(host="0.0.0.0", port=8505, threaded=True, debug=False)

# Argument parsing
parser = argparse.ArgumentParser(description="Run the web application")
parser.add_argument('--port', type=int, default=8505, help='Port to run the application on')
args = parser.parse_args()

if __name__ == '__main__':
    # Use the specified port if provided, else default to 8505
    app.run(host="0.0.0.0", port=args.port, threaded=True, debug=False)