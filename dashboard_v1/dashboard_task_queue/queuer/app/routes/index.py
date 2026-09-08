import json
import os
from app import app, socketio
from flask import send_file, request, jsonify, send_from_directory
import uuid
from pymongo import MongoClient
from datetime import datetime, timezone, timedelta
from bson import json_util, ObjectId
from flask_socketio import SocketIO, emit
import jwt 
from functools import wraps
import time

# connect to mongobd
client = MongoClient('localhost', 27017)
db = client.dashboardAI_db
task_collection = db.tasks_new
chat_collection = db.chats # for each chat session
msg_collection = db.messages
token_collection = db.tokens 

# secret key for /auto. see jwt package.
SECRET_KEY = ""

def get_client_ip():
    if request.headers.get('X-Forwarded-For'):
        ip = request.headers.get('X-Forwarded-For').split(',')[0]
    else:
        ip = request.remote_addr
    return ip

def get_est_waiting_time():
    position_in_queue = task_collection.count_documents({'status': 'Queued'}) # get the number of queued items 

    # TBD: 4 is the avg time for /chat
    return 4 * position_in_queue 

def to_json(records_list):
    # handle ObjectId and datetime types.
    def convert_document(document):
        for key, value in document.items():
            if isinstance(value, ObjectId):
                document[key] = str(value)
            elif isinstance(value, datetime):
                document[key] = value.isoformat()
        return document
    return [convert_document(r) for r in records_list]

@socketio.on('connect')
def handle_connect():
    print('Client connected')

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

def notify_data_change(data):
    socketio.emit('data_changed', data)

@app.route('/')
def _index():
    # headers = request.headers
    # print(headers)  # Log headers for debugging
    # if headers.get('X-Forwarded-For'):
    #     ip = headers.get('X-Forwarded-For').split(',')[0]
    # else:
    #     ip = request.remote_addr
    # return ip
    return json.dumps("back end")

def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get('x-access-token')
        if not token:
            return jsonify({'message': 'Token is missing!'}), 403

        try:
            data = jwt.decode(token, app.config['SECRET_KEY'], algorithms=["HS256"])
            current_user = data['username']
        except:
            return jsonify({'message': 'Token is invalid!'}), 403

        return f(current_user, *args, **kwargs)
    return decorated

@app.route('/loginAdmin', methods=['POST'])
def login():
    auth_data = request.json
    username = auth_data.get('username')
    password = auth_data.get('password')
    
    ## TBD - move to db
    users = {
        "your username here": "your password here",
    }
    
    if not username or not password:
        return jsonify({'message': 'Username and password are required!'}), 400
    
    user_password = users.get(username)
    
    if user_password and user_password == password:
        token = jwt.encode({
            'username': username,
            'exp': datetime.utcnow() + timedelta(minutes=60)
        }, app.config['SECRET_KEY'], algorithm='HS256')
        
        return jsonify({'token': token})
    
    return jsonify({'message': 'Invalid username or password!'}), 401

@app.route('/generateToken', methods=['POST'])
def generate_token():
    auth_data = request.json
    user_id = auth_data.get('user_id')
    valid_days = auth_data.get('valid_days')
    remark = auth_data.get('remark')

    payload = {
        'exp': datetime.utcnow() + timedelta(days=valid_days),
        'iat': datetime.utcnow(),
        'jti': str(ObjectId()),  # Token ID
        'sub': user_id 
    }
    token = jwt.encode(payload, SECRET_KEY, algorithm='HS256')

    # Store the token in the database
    token_data = {
        '_id': ObjectId(payload['jti']),
        'token': token,
        'revoked': False,
        'created_at': datetime.utcnow(),
        'expires_at': payload['exp'],
        'remark': remark,
        'user_id': user_id
    }
    token_collection.insert_one(token_data)
    return jsonify({'token': token})

@app.route('/auth', methods=['POST'])
def auth():
    token = request.headers.get('Authorization')
    if not token:
        return jsonify({"message": "Token is missing!"}), 403

    try:
        if token == f"Global Token":
            return jsonify({"message": "Token is valid!", "expires_at": datetime.utcnow() + timedelta(days=30)}), 200
        # Decode the token
        data = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        token_id = data['jti']

        # Check if the token exists in the database and is valid
        token_record = token_collection.find_one({'_id': ObjectId(token_id)})
        if token_record and not token_record['revoked']:
            return jsonify({"message": "Token is valid!", "expires_at": token_record['expires_at']}), 200
        else:
            return jsonify({"message": "Token is invalid or revoked!"}), 403
    except jwt.ExpiredSignatureError:
        return jsonify({"message": "Token has expired!"}), 403
    except jwt.InvalidTokenError:
        return jsonify({"message": "Invalid token!"}), 403


def push_task(post_data, client_ip, token, task_type):
    """
        Insert a task to task_db upon receiving API calls
    """
    # generate a task id
    task_id = str(uuid.uuid4())  

    task_data = {
        'task_id': task_id,
        'request_data': post_data,
        'status': 'Queued',
        'created_at': datetime.utcnow(),
        'client_ip': client_ip,
        'request_result': None,
        'processed_at': None,
        'finished_at': None,
        'retries': 0,  
        'priority': 'normal', 
        'error': None,
        'task_type': task_type,
        'token': token,
        'if_intervene': post_data['if_intervene']
    }

    task_collection.insert_one(task_data) # insert into db

    return task_id

@app.route('/chat', methods=['POST'])
def _chat():
    token = request.headers.get('Authorization')
    post_data = request.get_json()

    # Get client ip
    client_ip = get_client_ip()

    task_id = push_task(post_data, client_ip, token, 'chat')

    return jsonify({"task_id": task_id, 'est_time': get_est_waiting_time()}), 202 # 202 for accepted


@app.route('/regenerate_chat', methods=['POST'])
def _regenerate_chat():
    token = request.headers.get('Authorization')
    post_data = request.get_json()

    # Get client ip
    client_ip = get_client_ip()

    task_id = push_task(post_data, client_ip, token, 'regenerate_chat')

    return jsonify({"task_id": task_id, 'est_time': get_est_waiting_time()}), 202 # 202 for accepted

@app.route('/status/<task_id>', methods=['GET'])
def _get_status(task_id):
    task = task_collection.find_one({'task_id': task_id})
    if task:
        return jsonify({"task_id": task_id, "status": task['status'], "results": task['request_result']}), 200
    else:
        return jsonify({"task_id": task_id, "status": "Unknown"})
    
@app.route('/getChats', methods=['GET'])
def _getChats(nMax = 1000):
    # Retrieve a maximum of 1000 records
    records = chat_collection.find().limit(nMax)
    records_list = list(records)
    return to_json(records_list)

@app.route('/getMsgs/<chat_session_id>', methods=['GET'])
def _getMsgsByChat(chat_session_id):
    query = {'sessionId': chat_session_id}
    results = msg_collection.find(query)
    return to_json(list(results))

@app.route('/getTasks', methods=['GET'])
def _getTasks(nMax = 1000):
    # Retrieve a maximum of 1000 records
    records = task_collection.find().limit(nMax)

    def remove_keys(d):
        # reduce necessary returns
        del d['request_data']
        del d['request_result']
        return d

    records_list = list(records)
    updated_list_of_dicts = [remove_keys(d) for d in records_list]

    return to_json(updated_list_of_dicts)

@app.route('/getUsers', methods=['GET'])
def _getUsers(nMax = 1000):
    # Retrieve a maximum of 1000 records
    records = token_collection.find().limit(nMax)
    records_list = list(records)
    return to_json(records_list)