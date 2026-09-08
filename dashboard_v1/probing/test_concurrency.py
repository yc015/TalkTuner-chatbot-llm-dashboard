#!/usr/bin/env python3
"""
Test script to demonstrate concurrency limit protection.

This script attempts to start multiple tasks simultaneously to verify
that the server properly rejects requests when at capacity.
"""

import requests
import time
import sys


def test_concurrency_limit(api_url: str = "http://localhost:5001"):
    """Test the concurrency limit protection."""
    
    print("="*60)
    print("Testing Concurrency Limit Protection")
    print("="*60)
    
    # Check server health and get max concurrent tasks
    print("\n1. Checking server configuration...")
    try:
        health_response = requests.get(f"{api_url}/health")
        health_data = health_response.json()
        
        max_concurrent = health_data.get("max_concurrent_tasks", "unknown")
        available_slots = health_data.get("available_slots", "unknown")
        active_tasks = health_data.get("active_tasks", 0)
        
        print(f"✓ Server is running")
        print(f"  Max concurrent tasks: {max_concurrent}")
        print(f"  Currently active: {active_tasks}")
        print(f"  Available slots: {available_slots}")
        
    except requests.exceptions.ConnectionError:
        print("✗ Server is not running. Start it with: bash start_server.sh")
        return
    except Exception as e:
        print(f"✗ Error checking server: {e}")
        return
    
    # Try to start multiple tasks
    print(f"\n2. Attempting to start {max_concurrent + 2} tasks...")
    print("   (This should start the first {max_concurrent}, then reject the rest)")
    
    if max_concurrent == "unknown":
        print("Cannot determine max concurrent tasks. Exiting.")
        return
    
    # Get OpenAI API key (won't actually be used in this test)
    openai_key = input("\nEnter OpenAI API key (or 'test' to simulate): ").strip()
    if not openai_key:
        print("No API key provided. Exiting.")
        return
    
    task_ids = []
    num_tasks_to_start = max_concurrent + 2
    
    print(f"\nStarting {num_tasks_to_start} tasks...")
    for i in range(num_tasks_to_start):
        try:
            response = requests.post(
                f"{api_url}/probe",
                json={
                    "openai_api_key": openai_key,
                    "model": "llama-3.1-8b-instruct",
                    "attribute1": f"attr{i}",
                    "attribute2": f"non-attr{i}",
                    "target": "user",
                    "num_conversations": 5,  # Very small for testing
                    "probe_type": "control"
                }
            )
            
            if response.status_code == 202:
                task_id = response.json().get("task_id")
                task_ids.append(task_id)
                print(f"  Task {i+1}: ✓ Started (ID: {task_id[:8]}...)")
            elif response.status_code == 503:
                result = response.json()
                print(f"  Task {i+1}: ⚠ REJECTED - Server at capacity")
                print(f"    Message: {result.get('message')}")
                print(f"    Active: {result.get('active_tasks')}/{result.get('max_concurrent_tasks')}")
            else:
                print(f"  Task {i+1}: ✗ Failed with status {response.status_code}")
                print(f"    {response.json().get('message')}")
        
        except Exception as e:
            print(f"  Task {i+1}: ✗ Error: {e}")
        
        # Small delay between requests
        time.sleep(0.1)
    
    # Summary
    print(f"\n{'='*60}")
    print("Summary:")
    print(f"  Tasks started: {len(task_ids)}")
    print(f"  Tasks rejected: {num_tasks_to_start - len(task_ids)}")
    print(f"  Expected rejections: {max(0, num_tasks_to_start - max_concurrent)}")
    print(f"{'='*60}")
    
    if len(task_ids) > 0:
        print(f"\nStarted task IDs:")
        for task_id in task_ids:
            print(f"  - {task_id}")
        
        print(f"\nYou can check task status with:")
        print(f"  curl {api_url}/task/<task_id>")
        print(f"\nOr list all tasks:")
        print(f"  curl {api_url}/tasks")
    
    # Check if protection worked correctly
    if len(task_ids) == max_concurrent and num_tasks_to_start - len(task_ids) > 0:
        print(f"\n✓ Concurrency limit protection is working correctly!")
    elif len(task_ids) < max_concurrent:
        print(f"\n⚠ Fewer tasks started than expected (may be API key issue)")
    else:
        print(f"\n⚠ More tasks started than expected - protection may not be working!")


if __name__ == "__main__":
    api_url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:5001"
    test_concurrency_limit(api_url)

