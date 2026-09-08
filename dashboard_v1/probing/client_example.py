#!/usr/bin/env python3
"""
Example client for the attribute probing Flask API (Async version).

This script demonstrates how to:
1. Start a probing task (returns immediately)
2. Poll for task status
3. Retrieve results when complete
"""

import requests
import json
import time
from typing import Dict, Optional


def start_probe_training(
    api_url: str,
    openai_api_key: str,
    model: str,
    attribute1: str,
    target: str,
    num_conversations: int,
    attribute2: Optional[str] = None,
    probe_type: str = "both"
) -> Dict:
    """
    Start probe training task (async - returns immediately).
    
    Returns:
        Response with task_id
    """
    endpoint = f"{api_url}/probe"
    
    payload = {
        "openai_api_key": openai_api_key,
        "model": model,
        "attribute1": attribute1,
        "target": target,
        "num_conversations": num_conversations,
        "probe_type": probe_type
    }
    
    if attribute2:
        payload["attribute2"] = attribute2
    
    print(f"Starting probe training task...")
    print(f"Payload: {json.dumps({k: v if k != 'openai_api_key' else '***' for k, v in payload.items()}, indent=2)}")
    
    response = requests.post(endpoint, json=payload)
    
    return response.json(), response.status_code


def get_task_status(api_url: str, task_id: str) -> Dict:
    """Check status of a specific task."""
    endpoint = f"{api_url}/task/{task_id}"
    response = requests.get(endpoint)
    return response.json(), response.status_code


def wait_for_task_completion(
    api_url: str,
    task_id: str,
    poll_interval: int = 5,
    timeout: int = 3600
) -> Dict:
    """
    Wait for task to complete by polling.
    
    Args:
        api_url: Base API URL
        task_id: Task ID to monitor
        poll_interval: Seconds between polls
        timeout: Maximum seconds to wait
        
    Returns:
        Final task data
    """
    start_time = time.time()
    
    print(f"\nMonitoring task {task_id}...")
    print(f"Polling every {poll_interval} seconds...")
    
    while True:
        elapsed = time.time() - start_time
        
        if elapsed > timeout:
            print(f"\n⚠ Timeout reached ({timeout}s)")
            break
        
        task_data, status_code = get_task_status(api_url, task_id)
        
        if status_code != 200:
            print(f"\n✗ Error checking task status: {task_data}")
            return task_data
        
        task_status = task_data.get("status")
        progress = task_data.get("progress", "")
        
        # Print progress
        elapsed_str = time.strftime("%M:%S", time.gmtime(elapsed))
        print(f"[{elapsed_str}] Status: {task_status} - {progress}")
        
        if task_status == "completed":
            print(f"\n✓ Task completed successfully!")
            return task_data
        elif task_status == "failed":
            print(f"\n✗ Task failed!")
            print(f"Error: {task_data.get('error')}")
            return task_data
        
        # Wait before next poll
        time.sleep(poll_interval)
    
    return task_data


def list_all_tasks(api_url: str, status_filter: Optional[str] = None) -> Dict:
    """List all tasks, optionally filtered by status."""
    endpoint = f"{api_url}/tasks"
    params = {}
    if status_filter:
        params["status"] = status_filter
    
    response = requests.get(endpoint, params=params)
    return response.json(), response.status_code


def check_health(api_url: str) -> Dict:
    """Check API health status."""
    endpoint = f"{api_url}/health"
    response = requests.get(endpoint)
    return response.json(), response.status_code


def main():
    """Example usage of the async probing API."""
    # Configuration
    API_URL = "http://localhost:5001"
    OPENAI_API_KEY = input("Enter your OpenAI API key: ").strip()
    
    print("\n" + "="*60)
    print("Attribute Probing API - Async Example Client")
    print("="*60)
    
    # Check health
    print("\n1. Checking API health...")
    health_data, status = check_health(API_URL)
    if status == 200:
        print("✓ API is healthy!")
        print(f"  GPU available: {health_data.get('gpu_available')}")
        print(f"  Active tasks: {health_data.get('active_tasks', 0)}")
        
        models = health_data.get('models', {})
        for model_name, model_info in models.items():
            if model_info.get('loaded'):
                print(f"  ✓ {model_name}: loaded (shared instance)")
    else:
        print("✗ API is not responding correctly")
        return
    
    # List existing tasks
    print("\n2. Listing existing tasks...")
    tasks_data, status = list_all_tasks(API_URL)
    if status == 200:
        task_count = tasks_data.get("count", 0)
        print(f"✓ Found {task_count} task(s)")
        if task_count > 0:
            for task in tasks_data.get("tasks", [])[:5]:  # Show first 5
                print(f"  - Task {task['task_id'][:8]}... : {task['status']}")
    
    # Start a new probe training task
    print("\n3. Starting a new probe training task...")
    print("\nExample: Training 'happy' vs 'sad' user attribute probe on Llama-3.1")
    
    # Ask user if they want to proceed
    proceed = input("\nDo you want to start this task? (y/n): ").strip().lower()
    if proceed != 'y':
        print("Exiting.")
        return
    
    # Start the task
    result, status = start_probe_training(
        api_url=API_URL,
        openai_api_key=OPENAI_API_KEY,
        model="llama-3.1-8b-instruct",
        attribute1="happy",
        attribute2="sad",
        target="user",
        num_conversations=10,  # Small number for testing
        probe_type="both"
    )
    
    if status == 503:
        print(f"\n⚠ Server is at capacity!")
        print(f"Message: {result.get('message')}")
        print(f"Active tasks: {result.get('active_tasks')}/{result.get('max_concurrent_tasks')}")
        print(f"Suggestion: {result.get('suggestion')}")
        return
    elif status != 202:
        print(f"\n✗ Failed to start task!")
        print(f"Error: {result.get('message')}")
        return
    
    task_id = result.get("task_id")
    print(f"\n✓ Task started successfully!")
    print(f"Task ID: {task_id}")
    print(f"Check status at: {result.get('check_status_url')}")
    
    # Ask if user wants to monitor
    monitor = input("\nDo you want to monitor this task? (y/n): ").strip().lower()
    if monitor != 'y':
        print(f"\nYou can check status later with:")
        print(f"  curl {API_URL}/task/{task_id}")
        return
    
    # Wait for completion
    print("\n4. Monitoring task progress...")
    final_data = wait_for_task_completion(
        API_URL,
        task_id,
        poll_interval=5,
        timeout=3600
    )
    
    # Display results
    if final_data.get("status") == "completed":
        print(f"\n{'='*60}")
        print("TASK COMPLETED SUCCESSFULLY")
        print(f"{'='*60}")
        
        results = final_data.get("results", {})
        
        print(f"\nDataset: {results.get('dataset_path')}")
        print(f"Model: {results.get('model')}")
        print(f"Attributes: {results.get('attribute1')} vs {results.get('attribute2')}")
        
        if results.get("control_probe"):
            cp = results["control_probe"]
            print(f"\nControl Probe:")
            print(f"  Average accuracy: {cp['average_accuracy']:.4f}")
            print(f"  Best layer: {cp['best_layer']}")
            print(f"  Best accuracy: {cp['best_accuracy']:.4f}")
            print(f"  Probes saved to: {cp['probe_directory']}/")
            print(f"  Stats saved to: {cp['stats_directory']}/")
        
        if results.get("read_probe"):
            rp = results["read_probe"]
            print(f"\nRead Probe:")
            print(f"  Average accuracy: {rp['average_accuracy']:.4f}")
            print(f"  Best layer: {rp['best_layer']}")
            print(f"  Best accuracy: {rp['best_accuracy']:.4f}")
            print(f"  Probes saved to: {rp['probe_directory']}/")
            print(f"  Stats saved to: {rp['stats_directory']}/")
        
        print(f"\n{'='*60}\n")


if __name__ == "__main__":
    main()
