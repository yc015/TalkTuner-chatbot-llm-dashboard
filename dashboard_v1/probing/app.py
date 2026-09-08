#!/usr/bin/env python3
"""
Flask app for attribute probing endpoint.

This app provides an endpoint to generate synthetic conversations and train
linear probes on LLM residual stream representations.
"""

import os
import sys
import json
import torch
import threading
import uuid
from flask import Flask, request, jsonify
from flask_cors import CORS
from datetime import datetime
import traceback

# Add paths for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend', 'app'))

from transformers import AutoModelForCausalLM, AutoTokenizer
from generate_conversation_dataset import generate_dataset, save_dataset
from train_probes import train_probes_from_dataset, MODEL_CONFIGS

app = Flask(__name__)
CORS(app)

# Global variables for models (SHARED ACROSS ALL REQUESTS)
models = {}
tokenizers = {}
MODEL_NAMES = {
    "gemma-2-9b-it": "google/gemma-2-9b-it",
    "llama-3.1-8b-instruct": "meta-llama/Llama-3.1-8B-Instruct"
}

# Task tracking
tasks = {}
task_lock = threading.Lock()

# Concurrency control
MAX_CONCURRENT_TASKS = int(os.environ.get('MAX_CONCURRENT_TASKS', 2))
active_tasks_semaphore = threading.Semaphore(MAX_CONCURRENT_TASKS)


def initialize_models():
    """Initialize both models at startup (GLOBAL INSTANCES)."""
    print("="*60)
    print("Initializing GLOBAL model instances...")
    print("="*60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    for model_key, model_name in MODEL_NAMES.items():
        try:
            print(f"\nLoading {model_key}...")
            
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                use_auth_token=True,
                torch_dtype=torch.bfloat16
            )
            
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                use_auth_token=True,
                torch_dtype=torch.bfloat16,
                device_map=device
            )
            model.eval()
            
            # Add padding token if needed
            if '<pad>' not in tokenizer.get_vocab():
                tokenizer.add_special_tokens({"pad_token": "<pad>"})
                model.resize_token_embeddings(len(tokenizer))
                model.config.pad_token_id = tokenizer.pad_token_id
            
            models[model_key] = model
            tokenizers[model_key] = tokenizer
            
            print(f"✓ {model_key} loaded successfully!")
            
        except Exception as e:
            print(f"✗ Failed to load {model_key}: {e}")
            models[model_key] = None
            tokenizers[model_key] = None
    
    print("\n" + "="*60)
    print("Model initialization complete!")
    print(f"Models are SHARED across all requests (no duplication)")
    print(f"Max concurrent probe tasks: {MAX_CONCURRENT_TASKS}")
    print(f"  (Set MAX_CONCURRENT_TASKS env var to change)")
    print("="*60 + "\n")


def run_probing_task(
    task_id: str,
    openai_api_key: str,
    model_name: str,
    attribute1: str,
    attribute2: str,
    target: str,
    num_conversations: int,
    probe_type: str,
    meta_attribute: str = None,
    icon: list = None,
    system_prompt_template: str = None
):
    """
    Run the probing task in background thread.
    Uses GLOBAL model instances (no new model loading).
    Protected by semaphore to limit concurrent executions.
    """
    # Acquire semaphore (blocks if max concurrent tasks reached)
    active_tasks_semaphore.acquire()
    
    try:
        # Update task status
        with task_lock:
            tasks[task_id]["status"] = "generating_conversations"
            tasks[task_id]["progress"] = "Generating synthetic conversations..."
        
        # Step 1: Generate synthetic conversations
        print(f"\n[Task {task_id}] Generating conversations...")
        dataset = generate_dataset(
            openai_api_key,
            attribute1,
            attribute2,
            target,
            num_conversations,
            meta_attribute=meta_attribute,
            system_prompt_template=system_prompt_template
        )
        
        # Save dataset
        dataset_path = save_dataset(dataset, attribute1, attribute2)
        print(f"[Task {task_id}] Dataset saved to: {dataset_path}")
        
        with task_lock:
            tasks[task_id]["dataset_path"] = dataset_path
            tasks[task_id]["status"] = "training_probes"
            tasks[task_id]["progress"] = "Training probes on LLM activations..."
        
        # Step 2: Train probes using GLOBAL model instances
        print(f"\n[Task {task_id}] Training probes using SHARED model instance...")
        results = train_probes_from_dataset(
            dataset_path,
            model_name,
            probe_type=probe_type,
            device="cuda" if torch.cuda.is_available() else "cpu",
            model=models[model_name],  # Use GLOBAL model
            tokenizer=tokenizers[model_name],  # Use GLOBAL tokenizer
            icon=icon  # Pass icon to training function
        )
        
        # Update task with results
        with task_lock:
            tasks[task_id]["status"] = "completed"
            tasks[task_id]["progress"] = "Probing completed successfully!"
            tasks[task_id]["results"] = results
            tasks[task_id]["completed_at"] = datetime.now().isoformat()
        
        print(f"\n[Task {task_id}] ✓ Completed successfully!")
        
    except Exception as e:
        error_trace = traceback.format_exc()
        print(f"\n[Task {task_id}] ✗ Error: {e}")
        print(error_trace)
        
        with task_lock:
            tasks[task_id]["status"] = "failed"
            tasks[task_id]["error"] = str(e)
            tasks[task_id]["traceback"] = error_trace
            tasks[task_id]["failed_at"] = datetime.now().isoformat()
    
    finally:
        # Always release semaphore when done (success or failure)
        active_tasks_semaphore.release()


@app.route('/')
def index():
    """Health check endpoint."""
    available_models = [k for k, v in models.items() if v is not None]
    return jsonify({
        "status": "online",
        "service": "Attribute Probing API",
        "available_models": available_models,
        "endpoints": {
            "/probe": "POST - Start probe training (async)",
            "/task/<task_id>": "GET - Check task status",
            "/tasks": "GET - List all tasks",
            "/tasks/ongoing": "GET - List ongoing probe training tasks",
            "/health": "GET - Check service health"
        }
    })


@app.route('/health')
def health():
    """Health check with detailed model status."""
    model_status = {}
    for model_key in MODEL_NAMES.keys():
        model_status[model_key] = {
            "loaded": models.get(model_key) is not None,
            "tokenizer_loaded": tokenizers.get(model_key) is not None,
            "is_global_instance": True  # Emphasize shared instances
        }
    
    active_count = len([t for t in tasks.values() if t["status"] in ["generating_conversations", "training_probes"]])
    
    return jsonify({
        "status": "healthy",
        "models": model_status,
        "gpu_available": torch.cuda.is_available(),
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "active_tasks": active_count,
        "max_concurrent_tasks": MAX_CONCURRENT_TASKS,
        "available_slots": MAX_CONCURRENT_TASKS - active_count
    })


@app.route('/probe', methods=['POST'])
def start_probe_training():
    """
    Start probe training task (returns immediately).
    
    Expected JSON body:
    {
        "openai_api_key": "sk-...",
        "model": "gemma-2-9b-it" or "llama-3.1-8b-instruct",
        "attribute1": "happy",
        "attribute2": "sad" (optional, defaults to "non-{attribute1}"),
        "meta_attribute": "mood" (optional, meta label for reading prompt, defaults to attribute1),
        "target": "user" or "chatbot",
        "num_conversations": 50,
        "probe_type": "both" (optional, can be "control", "read", or "both"),
        "icon": ["FaHeart", "FaSadTear"] (optional, array of two icon names [icon1, icon2], defaults to ["FaQuestion", "FaQuestion"]),
        "system_prompt_template": "Custom prompt with {attribute} placeholder" (optional, custom template for conversation generation)
    }
    
    Returns task_id immediately for status checking.
    """
    try:
        # Parse request data
        data = request.get_json()
        
        # Validate required fields
        required_fields = ["openai_api_key", "model", "attribute1", "target", "num_conversations", "meta_attribute"]
        missing_fields = [field for field in required_fields if field not in data]
        
        if missing_fields:
            return jsonify({
                "status": "error",
                "message": f"Missing required fields: {', '.join(missing_fields)}"
            }), 400
        
        # Extract parameters
        openai_api_key = data["openai_api_key"]
        model_name = data["model"]
        attribute1 = data["attribute1"]
        attribute2 = data.get("attribute2", f"non-{attribute1}")
        meta_attribute = data.get("meta_attribute", None)  # Defaults to attribute1 if not provided
        target = data["target"]
        num_conversations = int(data["num_conversations"])
        probe_type = data.get("probe_type", "both")
        icon = data.get("icon", ["FaQuestion", "FaQuestion"])  # Default to array of FaQuestion if not provided
        system_prompt_template = data.get("system_prompt_template", None)  # Optional custom template
        
        # Ensure icon is a list with exactly 2 elements
        if not isinstance(icon, list):
            icon = ["FaQuestion", "FaQuestion"]
        elif len(icon) != 2:
            icon = ["FaQuestion", "FaQuestion"]
        
        # Validate system_prompt_template if provided
        if system_prompt_template and "{attribute}" not in system_prompt_template:
            return jsonify({
                "status": "error",
                "message": "system_prompt_template must contain '{attribute}' placeholder"
            }), 400
        
        # Validate model
        if model_name not in MODEL_NAMES:
            return jsonify({
                "status": "error",
                "message": f"Invalid model. Choose from: {list(MODEL_NAMES.keys())}"
            }), 400
        
        # Check if model is loaded
        if models.get(model_name) is None:
            return jsonify({
                "status": "error",
                "message": f"Model {model_name} is not loaded"
            }), 503
        
        # Validate target
        if target not in ["user", "chatbot"]:
            return jsonify({
                "status": "error",
                "message": "Target must be 'user' or 'chatbot'"
            }), 400
        
        # Validate probe_type
        if probe_type not in ["control", "read", "both"]:
            return jsonify({
                "status": "error",
                "message": "probe_type must be 'control', 'read', or 'both'"
            }), 400
        
        # Validate num_conversations
        if num_conversations <= 0 or num_conversations > 500:
            return jsonify({
                "status": "error",
                "message": "num_conversations must be between 1 and 500"
            }), 400
        
        # Check if probe with this meta_attribute already exists
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_config = MODEL_CONFIGS.get(model_name)
        if model_config:
            # Check both control and read probe directories
            # Folders are named: {meta_attribute}_{target}
            folder_name = f"{meta_attribute}_{target}"
            
            control_dir = os.path.join(base_dir, model_config["control_probe_dir"])
            read_dir = os.path.join(base_dir, model_config["read_probe_dir"])
            
            probe_exists = False
            existing_location = None
            
            # Check if folder exists in control or read directories
            if os.path.exists(control_dir):
                control_path = os.path.join(control_dir, folder_name)
                if os.path.exists(control_path) and os.path.isdir(control_path):
                    probe_exists = True
                    existing_location = "control probes"
            
            if os.path.exists(read_dir):
                read_path = os.path.join(read_dir, folder_name)
                if os.path.exists(read_path) and os.path.isdir(read_path):
                    probe_exists = True
                    if existing_location:
                        existing_location = "control and read probes"
                    else:
                        existing_location = "read probes"
            
            if probe_exists:
                return jsonify({
                    "status": "error",
                    "error_type": "duplicate_probe",
                    "message": f"A probe for '{meta_attribute}' (target: {target}) already exists in {existing_location}. Please use a different meta_attribute name or delete the existing probe first.",
                    "meta_attribute": meta_attribute,
                    "target": target
                }), 409  # 409 Conflict
        
        # Check if we can accept new tasks (concurrency limit)
        with task_lock:
            active_count = len([t for t in tasks.values() 
                              if t["status"] in ["generating_conversations", "training_probes"]])
        
        if active_count >= MAX_CONCURRENT_TASKS:
            return jsonify({
                "status": "error",
                "message": f"Server is at capacity. Maximum {MAX_CONCURRENT_TASKS} concurrent tasks allowed.",
                "active_tasks": active_count,
                "max_concurrent_tasks": MAX_CONCURRENT_TASKS,
                "suggestion": "Please wait for existing tasks to complete or check /tasks to see task status"
            }), 503  # Service Unavailable
        
        # Generate unique task ID
        task_id = str(uuid.uuid4())
        
        # Initialize task record
        with task_lock:
            tasks[task_id] = {
                "task_id": task_id,
                "status": "queued",
                "progress": "Task queued, starting soon...",
                "model": model_name,
                "attribute1": attribute1,
                "attribute2": attribute2,
                "meta_attribute": meta_attribute if meta_attribute else attribute1,
                "target": target,
                "num_conversations": num_conversations,
                "probe_type": probe_type,
                "icon": icon,
                "system_prompt_template": system_prompt_template,
                "created_at": datetime.now().isoformat(),
                "dataset_path": None,
                "results": None
            }
        
        # Start background thread
        thread = threading.Thread(
            target=run_probing_task,
            args=(task_id, openai_api_key, model_name, attribute1, 
                  attribute2, target, num_conversations, probe_type, meta_attribute, icon, system_prompt_template)
        )
        thread.daemon = True
        thread.start()
        
        print(f"\n{'='*60}")
        print(f"New probing task started:")
        print(f"  Task ID: {task_id}")
        print(f"  Model: {model_name} (using GLOBAL instance)")
        print(f"  Attribute 1: {attribute1}")
        print(f"  Attribute 2: {attribute2}")
        print(f"  Meta Attribute: {meta_attribute if meta_attribute else attribute1}")
        print(f"  Target: {target}")
        print(f"  Num conversations: {num_conversations}")
        print(f"  Probe type: {probe_type}")
        print(f"{'='*60}\n")
        
        return jsonify({
            "status": "success",
            "message": "Probing task started",
            "task_id": task_id,
            "check_status_url": f"/task/{task_id}"
        }), 202
        
    except Exception as e:
        error_trace = traceback.format_exc()
        print(f"\n{'='*60}")
        print("✗ Error starting probe task:")
        print(error_trace)
        print(f"{'='*60}\n")
        
        return jsonify({
            "status": "error",
            "message": str(e),
            "traceback": error_trace
        }), 500


@app.route('/task/<task_id>', methods=['GET'])
def get_task_status(task_id: str):
    """Get the status of a specific task."""
    with task_lock:
        if task_id not in tasks:
            return jsonify({
                "status": "error",
                "message": f"Task {task_id} not found"
            }), 404
        
        task_data = tasks[task_id].copy()
    
    # Format results for response
    if task_data["status"] == "completed" and task_data.get("results"):
        results = task_data["results"]
        formatted_results = {
            "dataset_path": task_data.get("dataset_path"),
            "model": task_data["model"],
            "attribute1": task_data["attribute1"],
            "attribute2": task_data["attribute2"],
            "target": task_data["target"],
            "num_conversations": task_data["num_conversations"],
            "probe_type": task_data["probe_type"]
        }
        
        # Add accuracy information
        if results.get("control_accuracies"):
            control_accs = results["control_accuracies"]
            formatted_results["control_probe"] = {
                "average_accuracy": float(sum(control_accs.values()) / len(control_accs)),
                "best_layer": int(max(control_accs, key=control_accs.get)),
                "best_accuracy": float(max(control_accs.values())),
                "probe_directory": MODEL_CONFIGS[task_data["model"]]["control_probe_dir"],
                "stats_directory": MODEL_CONFIGS[task_data["model"]]["control_probe_dir"].replace("_probes", "_probes_stats")
            }
        
        if results.get("read_accuracies"):
            read_accs = results["read_accuracies"]
            formatted_results["read_probe"] = {
                "average_accuracy": float(sum(read_accs.values()) / len(read_accs)),
                "best_layer": int(max(read_accs, key=read_accs.get)),
                "best_accuracy": float(max(read_accs.values())),
                "probe_directory": MODEL_CONFIGS[task_data["model"]]["read_probe_dir"],
                "stats_directory": MODEL_CONFIGS[task_data["model"]]["read_probe_dir"].replace("_probes", "_probes_stats")
            }
        
        task_data["results"] = formatted_results
    
    return jsonify(task_data), 200


@app.route('/tasks', methods=['GET'])
def list_tasks():
    """List all tasks."""
    with task_lock:
        all_tasks = list(tasks.values())
    
    # Optionally filter by status
    status_filter = request.args.get('status')
    if status_filter:
        all_tasks = [t for t in all_tasks if t["status"] == status_filter]
    
    return jsonify({
        "status": "success",
        "count": len(all_tasks),
        "tasks": all_tasks
    }), 200


@app.route('/tasks/ongoing', methods=['GET'])
def list_ongoing_tasks():
    """List all ongoing probe training tasks."""
    ongoing_statuses = ["queued", "generating_conversations", "training_probes"]
    
    with task_lock:
        ongoing_tasks = [t for t in tasks.values() if t["status"] in ongoing_statuses]
    
    return jsonify({
        "status": "success",
        "count": len(ongoing_tasks),
        "ongoing_tasks": ongoing_tasks
    }), 200


@app.route('/available_probes', methods=['GET'])
def list_available_probes():
    """List all available trained probes."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    probe_info = {}
    
    for model_key, config in MODEL_CONFIGS.items():
        probe_info[model_key] = {
            "control_probes": [],
            "read_probes": []
        }
        
        # Check control probes (now organized in subdirectories by attribute)
        control_dir = os.path.join(base_dir, config["control_probe_dir"])
        if os.path.exists(control_dir):
            # Look for subdirectories (each represents an attribute)
            try:
                subdirs = [d for d in os.listdir(control_dir) if os.path.isdir(os.path.join(control_dir, d))]
                # Each subdirectory name is an attribute
                probe_info[model_key]["control_probes"] = subdirs
            except Exception as e:
                print(f"Error listing control probe subdirectories: {e}")
                probe_info[model_key]["control_probes"] = []
        
        # Check read probes (now organized in subdirectories by attribute)
        read_dir = os.path.join(base_dir, config["read_probe_dir"])
        if os.path.exists(read_dir):
            # Look for subdirectories (each represents an attribute)
            try:
                subdirs = [d for d in os.listdir(read_dir) if os.path.isdir(os.path.join(read_dir, d))]
                # Each subdirectory name is an attribute
                probe_info[model_key]["read_probes"] = subdirs
            except Exception as e:
                print(f"Error listing read probe subdirectories: {e}")
                probe_info[model_key]["read_probes"] = []
    
    return jsonify({
        "status": "success",
        "probes": probe_info
    })


if __name__ == '__main__':
    # Initialize models before starting server
    initialize_models()
    
    # Start Flask server
    port = int(os.environ.get('PORT', 5001))
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)
