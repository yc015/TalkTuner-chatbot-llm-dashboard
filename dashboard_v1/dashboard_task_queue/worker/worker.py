from pymongo import MongoClient
import time
import aiohttp
import asyncio
from datetime import datetime, timezone, timedelta
import json

# Connect to MongoDB
client = MongoClient('localhost', 27017)
db = client.dashboardAI_db
task_collection = db.tasks_new
chat_collection = db.chats # for each chat session
msg_collection = db.messages

# Model backend API; can be extended if we have multiple later
backend_api = 'http://localhost:8505' #

def update_chat_db(task, request_result):
    """
        Update the chat collection when finishing a /chat task
        task: API input
        request_result: API output
    """
    task_id = task['task_id']
    created_at = task['created_at']
    client_ip = task['client_ip']

    request_data = task['request_data'] ## data fed into the POST
    # Check if post_data is still a string and try to decode it again
    if isinstance(request_data, str):
        request_data = json.loads(request_data)

    if 'msg' in request_data:
        msg = request_data['msg'] # user msg
    else:
        # /regenerate_chat
        msg = ''

    print(request_result)

    controlStatus = request_data['controlStatus']
    chat_session_id = request_result['id']

    bot_msg = request_result['msg']
    you_model = request_result['you_model']

    # Check if the chat session already exists in the database
    if not chat_collection.find_one({"sessionId": chat_session_id}):
        # If not, create a new chat session document
        chat_collection.insert_one({
            "sessionId": chat_session_id,
            "startTime": created_at,
            "client_ip": client_ip,
            "userToken": ""
        })

    # Insert the user/bot message into the Messages collection
    msg_collection.insert_one({
        "sessionId": chat_session_id,
        "timestamp": created_at,
        "user_msg": msg,
        "bot_msg": bot_msg,
        "you_model": you_model,
        "controlStatus": controlStatus,
        "client_ip": client_ip,
        "task_id": task_id
    })

def process_task(task):
    task_id = task['task_id']
    print(f"Processing task: {task_id}")
    
    # Update status to "Processing"
    task_collection.update_one({'task_id': task_id}, {'$set': {'status': 'Processing'}})
    
    task_type = task['task_type']
    finished_at = None
    request_result = None
    error_message = None
    request_result = None
    processed_at = None
    
    # Process task
    # try:
    if task_type == 'chat':
        processed_at, finished_at, request_result = asyncio.run(process_chat(task))

        # todo: check task result
        update_chat_db(task, request_result)
    elif task_type == 'regenerate_chat':
        processed_at, finished_at, request_result = asyncio.run(process_regenerate_chat(task))
        update_chat_db(task, request_result)
    else:
        raise ValueError(f"Unexpected task type: {task_type}")
    # except Exception as e:
    #     print('Error', e)
    #     error_message = str(e)

    # Prepare the update fields
    update_fields = {
        'status': 'Completed' if not error_message else 'Failed',
        'request_result': request_result,
        'processed_at': processed_at,
        'finished_at': finished_at,
        'error': error_message
    }

    # Update status to "Completed"
    task_collection.update_one(
        {'task_id': task_id}, 
        {'$set': update_fields}
    )
    
    print(f"Task completed: {task_id}")

def process_batch_task(tasks):
    # task_id = task['task_id']
    # print(f"Processing task: {task_id}")
    
    # Update status to "Processing"    
    task_ids = [task['_id'] for task in tasks]
    if task_ids:
        task_collection.update_many(
            {
                '_id': {'$in': task_ids}
            },
            {
                '$set': {'status': 'Processing'}
            }
        ) 

    task_type = list(tasks)[0]['task_type']
    finished_at = None
    request_result = None
    error_message = None
    request_result = None
    processed_at = None
    
    # Process task
    # try:
    if task_type == 'chat':
        processed_at, finished_at, request_results = asyncio.run(process_chat_batched(tasks))

        for task, request_result in zip(tasks, request_results['responses']):
            update_chat_db(task, request_result)
            task_id = task['task_id']

            # Prepare the update fields
            update_fields = {
                'status': 'Completed',
                'request_result': request_result,
                'processed_at': processed_at,
                'finished_at': finished_at,
                'error': error_message
            }

            # Update status to "Completed"
            task_collection.update_one(
                {'task_id': task_id}, 
                {'$set': update_fields}
            )

            print(f"Task completed batched: {task_id}")

    else:
        raise ValueError(f"Unexpected task type: {task_type}")
    # except Exception as e:
    #     print('Error', e)
    #     error_message = str(e)


    

async def async_post_service(url, data):
    """
        Send post request to the model host for processing task like /chat and /regenerate_chat
        Can be extended later when we have multiple model backend (e.g., some on cloud or more machines)
    """
    async with aiohttp.ClientSession() as session:
        processed_at = datetime.utcnow()
        async with session.post(url, json=data) as response:
            finished_at = datetime.utcnow()
            request_result = await response.json(content_type=response.content_type)
            print(request_result)

            return processed_at, finished_at, request_result


async def process_chat(task):
    request_data = task['request_data']
    return await async_post_service("{}/{}".format(backend_api, 'chat'), request_data)

async def process_chat_batched(tasks):
    request_data = [task['request_data'] for task in tasks]
    return await async_post_service("{}/{}".format(backend_api, 'chat_batched'), request_data)

async def process_regenerate_chat(task):
    request_data = task['request_data']
    return await async_post_service("{}/{}".format(backend_api, 'regenerate_chat'), request_data)

def pick_batch_task():
    # Define the limit for the number of tasks to work on
    limit = 3

    # batch on non-intervened chat only
    tasks = task_collection.find(
        {
            'status': 'Queued',
            'task_type': 'chat',
            'if_intervene': False
        }
    ).sort('created_at', 1).limit(limit)  # Sort by created_at and limit the results
    tasks = list(tasks) #!important

    # Extract the IDs of the found documents
    task_ids = [task['_id'] for task in tasks]

    if task_ids:
        print(task_ids)
        task_collection.update_many(
            {
                '_id': {'$in': task_ids}
            },
            {
                '$set': {'status': 'Picked'}
            }
        ) 
        process_batch_task(tasks)
        return True
    else:
        return False

def pick_non_batch_task():
    task = task_collection.find_one_and_update(
        {
            'status': 'Queued',
            '$nor': [
                {'task_type': 'chat', 'if_intervene': False}  # Exclude tasks where both conditions are true
            ]
        },        
        {'$set': {'status': 'Picked'}},
        sort=[('created_at', 1)]
    )
    if task:
        print('process task', task['task_id'])
        process_task(task)
        return True
    return False

def worker():
    while True:

        if_find_batch_task = pick_batch_task()
        if_find_non_batch_task = pick_non_batch_task()

        if if_find_batch_task or if_find_non_batch_task:
            continue
        else:
            # If no tasks are queued, wait for a bit before checking again
            time.sleep(0.5)

if __name__ == "__main__":
    worker()
