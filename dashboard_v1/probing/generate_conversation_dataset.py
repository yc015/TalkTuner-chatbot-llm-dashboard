#!/usr/bin/env python3
"""
Generate conversation datasets with specific attributes for probing LLM representations.

This script generates conversations where either the user or chatbot exhibits certain attributes,
and saves them to a JSON file for later analysis.
"""

import os
import json
import argparse
from openai import OpenAI
from typing import List, Dict, Tuple
from datetime import datetime
from tqdm import tqdm


def get_user_input() -> Tuple[str, str, str, str, int]:
    """
    Get inputs from user interactively.
    
    Returns:
        Tuple of (api_key, attribute1, attribute2, target, num_conversations)
    """
    print("=" * 60)
    print("Conversation Dataset Generator for Attribute Probing")
    print("=" * 60)
    
    # Get API key
    api_key = input("\nEnter your OpenAI API key: ").strip()
    
    # Get first attribute
    attribute1 = input("\nEnter the first attribute (e.g., 'happy', 'formal', 'curious'): ").strip()
    
    # Get second attribute (optional)
    print("\nEnter the second attribute (leave empty if you want to generate non-attribute conversations):")
    attribute2 = input("Second attribute (opposite of first, e.g., 'sad', 'informal', 'indifferent'): ").strip()
    
    if not attribute2:
        attribute2 = f"non-{attribute1}"
    
    # Get target (user or chatbot)
    while True:
        target = input("\nIs this attribute about the 'user' or 'chatbot'? ").strip().lower()
        if target in ['user', 'chatbot']:
            break
        print("Please enter either 'user' or 'chatbot'")
    
    # Get number of conversations
    while True:
        try:
            num_conversations = int(input("\nHow many conversations to generate for each attribute? "))
            if num_conversations > 0:
                break
            print("Please enter a positive number")
        except ValueError:
            print("Please enter a valid number")
    
    return api_key, attribute1, attribute2, target, num_conversations


def generate_conversation(
    client: OpenAI,
    attribute: str,
    target: str,
    is_non_attribute: bool = False,
    system_prompt_template: str = None
) -> List[Dict[str, str]]:
    """
    Generate a single conversation with the specified attribute.
    
    Args:
        client: OpenAI client instance
        attribute: The attribute to exhibit
        target: Either 'user' or 'chatbot'
        is_non_attribute: Whether to generate a neutral conversation without the attribute
        system_prompt_template: Optional custom template for system prompt. Should contain "{attribute}" placeholder.
        
    Returns:
        List of conversation turns (each turn is a dict with 'role' and 'content')
    """
    # Create prompt based on target and attribute
    if system_prompt_template:
        # Use custom template and replace {attribute} placeholder
        system_prompt = system_prompt_template.replace("{attribute}", attribute)
    elif is_non_attribute:
        attribute = attribute.replace("non-", "")
        system_prompt = f"""Generate a natural conversation between a user and an AI assistant.
The conversation should NOT exhibit the attribute '{attribute}' in the {target}'s responses.
Keep the conversation natural but neutral regarding this attribute. You would like the readers of the conversation to tell the {target}'s {meta_attribute} is not {attribute} or the opposite of {attribute}.

Generate a realistic conversation with 4-8 turns (both user and assistant messages).
Return ONLY a JSON array of conversation turns in this format:
[
  {{"role": "user", "content": "..."}},
  {{"role": "assistant", "content": "..."}},
  ...
]
"""
    else:
        if target == "user":
            system_prompt = f"""Generate a natural conversation between a user and an AI assistant.
The USER should clearly exhibit the attribute: '{attribute}' in its language and tone, questions and requests, and other semantic or linguistic features. The assistant should respond naturally and appropriately to the user's tone and state. 

Generate a realistic conversation with 4-8 turns (both user and assistant messages).
The user's messages should strongly reflect the '{attribute}' attribute.
Return ONLY a JSON array of conversation turns in this format:
[
  {{"role": "user", "content": "..."}},
  {{"role": "assistant", "content": "..."}},
  ...
]
"""
        else:  # chatbot
            system_prompt = f"""Generate a natural conversation between a user and an AI assistant.
The ASSISTANT should clearly exhibit the attribute: '{attribute}' in its language and tone, responses, and other semantic or linguistic features. The user should ask natural questions or make natural requests.

Generate a realistic conversation with 4-8 turns (both user and assistant messages).
The assistant's responses should strongly reflect the '{attribute}' attribute.
Return ONLY a JSON array of conversation turns in this format:
[
  {{"role": "user", "content": "..."}},
  {{"role": "assistant", "content": "..."}},
  ...
]
"""
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": "Generate the conversation now."}
            ],
            temperature=0.9,  # Higher temperature for more variety
            max_tokens=2000
        )
        
        # Parse the response
        content = response.choices[0].message.content.strip()
        
        # Try to extract JSON from the response
        # Sometimes the model might wrap it in markdown code blocks
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
        
        conversation = json.loads(content)
        
        # Validate the conversation format
        if not isinstance(conversation, list) or len(conversation) < 2:
            raise ValueError("Invalid conversation format")
        
        for turn in conversation:
            if "role" not in turn or "content" not in turn:
                raise ValueError("Invalid turn format")
        
        return conversation
    
    except Exception as e:
        print(f"\nError generating conversation: {e}")
        print("Retrying...")
        # Retry once
        return generate_conversation(client, attribute, target, is_non_attribute, system_prompt_template)


def generate_dataset(
    api_key: str,
    attribute1: str,
    attribute2: str,
    target: str,
    num_conversations: int,
    meta_attribute: str = None,
    system_prompt_template: str = None
) -> Dict:
    """
    Generate the full dataset with conversations for both attributes.
    
    Args:
        api_key: OpenAI API key
        attribute1: First attribute
        attribute2: Second attribute (or non-attribute)
        target: Either 'user' or 'chatbot'
        num_conversations: Number of conversations per attribute
        meta_attribute: Meta attribute label (e.g., "mood" for "happy"/"sad")
                       If not provided, defaults to attribute1
        system_prompt_template: Optional custom template for system prompt. Should contain "{attribute}" placeholder.
        
    Returns:
        Dictionary containing the dataset
    """
    # Initialize OpenAI client
    client = OpenAI(api_key=api_key)
    
    # Determine if attribute2 is a non-attribute
    is_non_attribute = attribute2.startswith("non-")
    
    # Set default meta_attribute if not provided
    if meta_attribute is None:
        meta_attribute = attribute1
    
    dataset = {
        "metadata": {
            "created_at": datetime.now().isoformat(),
            "attribute1": attribute1,
            "attribute2": attribute2,
            "meta_attribute": meta_attribute,
            "target": target,
            "num_conversations_per_attribute": num_conversations,
            "total_conversations": num_conversations * 2
        },
        "conversations": []
    }
    
    print(f"\n{'=' * 60}")
    print(f"Generating {num_conversations} conversations for attribute: {attribute1}")
    print(f"{'=' * 60}")
    
    # Generate conversations for attribute1
    for i in tqdm(range(num_conversations), desc=f"Generating '{attribute1}' conversations"):
        conversation = generate_conversation(client, attribute1, target, is_non_attribute=False, system_prompt_template=system_prompt_template)
        dataset["conversations"].append({
            "id": f"{attribute1}_{i}",
            "attribute": attribute1,
            "target": target,
            "conversation": conversation
        })
    
    print(f"\n{'=' * 60}")
    print(f"Generating {num_conversations} conversations for attribute: {attribute2}")
    print(f"{'=' * 60}")
    
    # Generate conversations for attribute2
    for i in tqdm(range(num_conversations), desc=f"Generating '{attribute2}' conversations"):
        conversation = generate_conversation(client, attribute2, target, is_non_attribute=is_non_attribute, system_prompt_template=system_prompt_template)
        dataset["conversations"].append({
            "id": f"{attribute2}_{i}",
            "attribute": attribute2,
            "target": target,
            "conversation": conversation
        })
    
    return dataset


def save_dataset(dataset: Dict, attribute1: str, attribute2: str) -> str:
    """
    Save the dataset to a JSON file.
    
    Args:
        dataset: The dataset dictionary
        attribute1: First attribute (for folder naming)
        attribute2: Second attribute (for folder naming)
        
    Returns:
        Path to the saved file
    """
    # Create folder structure
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    folder_name = f"{attribute1}_vs_{attribute2}"
    dataset_dir = os.path.join(base_dir, "probing_datasets", folder_name)
    
    os.makedirs(dataset_dir, exist_ok=True)
    
    # Create filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"conversations_{timestamp}.json"
    filepath = os.path.join(dataset_dir, filename)
    
    # Save to JSON
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)
    
    return filepath


def main():
    """Main function to run the dataset generation."""
    parser = argparse.ArgumentParser(
        description="Generate conversation datasets with specific attributes for probing LLM representations."
    )
    parser.add_argument("--api-key", type=str, help="OpenAI API key")
    parser.add_argument("--attribute1", type=str, help="First attribute (e.g., 'happy')")
    parser.add_argument("--attribute2", type=str, help="Second attribute (e.g., 'sad' or leave empty for non-attribute)")
    parser.add_argument("--target", type=str, choices=["user", "chatbot"], help="Whether attribute is about user or chatbot")
    parser.add_argument("--num-conversations", type=int, help="Number of conversations to generate for each attribute")
    
    args = parser.parse_args()
    
    # Get inputs (either from args or interactively)
    if args.api_key and args.attribute1 and args.target and args.num_conversations:
        api_key = args.api_key
        attribute1 = args.attribute1
        attribute2 = args.attribute2 if args.attribute2 else f"non-{args.attribute1}"
        target = args.target
        num_conversations = args.num_conversations
    else:
        api_key, attribute1, attribute2, target, num_conversations = get_user_input()
    
    print(f"\n{'=' * 60}")
    print("Configuration:")
    print(f"  Attribute 1: {attribute1}")
    print(f"  Attribute 2: {attribute2}")
    print(f"  Target: {target}")
    print(f"  Conversations per attribute: {num_conversations}")
    print(f"  Total conversations: {num_conversations * 2}")
    print(f"{'=' * 60}\n")
    
    # Generate dataset
    try:
        dataset = generate_dataset(api_key, attribute1, attribute2, target, num_conversations)
        
        # Save dataset
        filepath = save_dataset(dataset, attribute1, attribute2)
        
        print(f"\n{'=' * 60}")
        print("✓ Dataset generation completed successfully!")
        print(f"{'=' * 60}")
        print(f"Dataset saved to: {filepath}")
        print(f"Total conversations: {len(dataset['conversations'])}")
        print(f"{'=' * 60}\n")
        
    except Exception as e:
        print(f"\n{'=' * 60}")
        print(f"✗ Error: {e}")
        print(f"{'=' * 60}\n")
        raise


if __name__ == "__main__":
    main()

