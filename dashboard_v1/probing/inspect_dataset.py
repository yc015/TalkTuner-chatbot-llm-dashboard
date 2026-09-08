#!/usr/bin/env python3
"""
Simple script to inspect and validate generated conversation datasets.
"""

import json
import argparse
import os
from pathlib import Path
from typing import Dict


def load_dataset(filepath: str) -> Dict:
    """Load a dataset from JSON file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def print_dataset_summary(dataset: Dict):
    """Print a summary of the dataset."""
    print("=" * 80)
    print("DATASET SUMMARY")
    print("=" * 80)
    
    metadata = dataset.get("metadata", {})
    print(f"\nCreated at: {metadata.get('created_at', 'N/A')}")
    print(f"Attribute 1: {metadata.get('attribute1', 'N/A')}")
    print(f"Attribute 2: {metadata.get('attribute2', 'N/A')}")
    print(f"Target: {metadata.get('target', 'N/A')}")
    print(f"Conversations per attribute: {metadata.get('num_conversations_per_attribute', 'N/A')}")
    print(f"Total conversations: {metadata.get('total_conversations', 'N/A')}")
    
    # Count conversations by attribute
    conversations = dataset.get("conversations", [])
    attribute_counts = {}
    for conv in conversations:
        attr = conv.get("attribute", "unknown")
        attribute_counts[attr] = attribute_counts.get(attr, 0) + 1
    
    print("\nConversation counts by attribute:")
    for attr, count in attribute_counts.items():
        print(f"  {attr}: {count}")
    
    # Calculate average conversation length
    total_turns = sum(len(conv.get("conversation", [])) for conv in conversations)
    avg_turns = total_turns / len(conversations) if conversations else 0
    print(f"\nAverage conversation length: {avg_turns:.1f} turns")
    print("=" * 80)


def print_sample_conversations(dataset: Dict, num_samples: int = 2):
    """Print sample conversations from the dataset."""
    conversations = dataset.get("conversations", [])
    
    print("\n" + "=" * 80)
    print("SAMPLE CONVERSATIONS")
    print("=" * 80)
    
    # Get samples from each attribute
    attributes = list(set(conv.get("attribute") for conv in conversations))
    
    for attr in attributes:
        attr_convs = [c for c in conversations if c.get("attribute") == attr]
        samples = attr_convs[:num_samples]
        
        print(f"\n{'─' * 80}")
        print(f"Attribute: {attr}")
        print(f"{'─' * 80}")
        
        for i, conv_data in enumerate(samples, 1):
            print(f"\nExample {i} (ID: {conv_data.get('id')})")
            print("-" * 80)
            conversation = conv_data.get("conversation", [])
            for turn in conversation:
                role = turn.get("role", "unknown")
                content = turn.get("content", "")
                print(f"\n[{role.upper()}]: {content}")
            print("-" * 80)


def validate_dataset(dataset: Dict) -> bool:
    """Validate the dataset structure."""
    print("\n" + "=" * 80)
    print("VALIDATION")
    print("=" * 80)
    
    errors = []
    warnings = []
    
    # Check metadata
    if "metadata" not in dataset:
        errors.append("Missing 'metadata' field")
    else:
        required_metadata = ["attribute1", "attribute2", "target", "num_conversations_per_attribute"]
        for field in required_metadata:
            if field not in dataset["metadata"]:
                warnings.append(f"Missing metadata field: {field}")
    
    # Check conversations
    if "conversations" not in dataset:
        errors.append("Missing 'conversations' field")
        return False
    
    conversations = dataset["conversations"]
    if not isinstance(conversations, list):
        errors.append("'conversations' should be a list")
        return False
    
    # Validate each conversation
    for i, conv in enumerate(conversations):
        if "id" not in conv:
            warnings.append(f"Conversation {i} missing 'id'")
        if "attribute" not in conv:
            errors.append(f"Conversation {i} missing 'attribute'")
        if "conversation" not in conv:
            errors.append(f"Conversation {i} missing 'conversation' field")
            continue
        
        turns = conv["conversation"]
        if not isinstance(turns, list) or len(turns) < 2:
            errors.append(f"Conversation {i} has invalid turns (needs at least 2)")
        
        for j, turn in enumerate(turns):
            if "role" not in turn:
                errors.append(f"Conversation {i}, turn {j} missing 'role'")
            if "content" not in turn:
                errors.append(f"Conversation {i}, turn {j} missing 'content'")
    
    # Print results
    if not errors and not warnings:
        print("\n✓ Dataset is valid!")
    else:
        if warnings:
            print(f"\n⚠ Warnings ({len(warnings)}):")
            for warning in warnings[:10]:  # Show first 10
                print(f"  - {warning}")
            if len(warnings) > 10:
                print(f"  ... and {len(warnings) - 10} more warnings")
        
        if errors:
            print(f"\n✗ Errors ({len(errors)}):")
            for error in errors[:10]:  # Show first 10
                print(f"  - {error}")
            if len(errors) > 10:
                print(f"  ... and {len(errors) - 10} more errors")
    
    print("=" * 80)
    return len(errors) == 0


def list_available_datasets(base_dir: str = None):
    """List all available datasets."""
    if base_dir is None:
        base_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "probing_datasets")
    
    print("\n" + "=" * 80)
    print("AVAILABLE DATASETS")
    print("=" * 80)
    
    if not os.path.exists(base_dir):
        print("\nNo datasets directory found.")
        return []
    
    datasets = []
    for folder in sorted(os.listdir(base_dir)):
        folder_path = os.path.join(base_dir, folder)
        if os.path.isdir(folder_path):
            files = [f for f in os.listdir(folder_path) if f.endswith('.json')]
            if files:
                print(f"\n{folder}/")
                for file in sorted(files):
                    filepath = os.path.join(folder_path, file)
                    datasets.append(filepath)
                    size = os.path.getsize(filepath)
                    print(f"  - {file} ({size / 1024:.1f} KB)")
    
    print(f"\nTotal: {len(datasets)} dataset(s)")
    print("=" * 80)
    return datasets


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Inspect and validate conversation datasets for attribute probing."
    )
    parser.add_argument("filepath", nargs="?", help="Path to the dataset JSON file")
    parser.add_argument("--list", action="store_true", help="List all available datasets")
    parser.add_argument("--no-samples", action="store_true", help="Don't show sample conversations")
    parser.add_argument("--num-samples", type=int, default=2, help="Number of sample conversations per attribute (default: 2)")
    
    args = parser.parse_args()
    
    if args.list or not args.filepath:
        datasets = list_available_datasets()
        if not args.filepath and datasets:
            print("\nUse: python inspect_dataset.py <filepath> to inspect a specific dataset")
        return
    
    # Load and inspect the dataset
    try:
        dataset = load_dataset(args.filepath)
        print_dataset_summary(dataset)
        validate_dataset(dataset)
        
        if not args.no_samples:
            print_sample_conversations(dataset, args.num_samples)
        
    except FileNotFoundError:
        print(f"Error: File not found: {args.filepath}")
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON file: {e}")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()

