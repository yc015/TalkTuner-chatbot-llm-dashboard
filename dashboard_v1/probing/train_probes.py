#!/usr/bin/env python3
"""
Train linear probes on residual stream representations for attribute probing.

This script trains logistic regression probes on LLM activations extracted from
synthetic conversations to identify specific attributes.
"""

import os
import json
import torch
import torch.nn as nn
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm
from typing import List, Dict, Tuple, Optional
from datetime import datetime
import sys

# Add parent directory to path to import classifiers
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend', 'app'))
from chat.classifiers import LinearProbeClassification


# Model configurations
MODEL_CONFIGS = {
    "gemma-2-9b-it": {
        "model_name": "google/gemma-2-9b-it",
        "num_layers": 43,
        "hidden_size": 3584,
        "control_probe_dir": "gemma2_control_probes_extra",
        "read_probe_dir": "gemma2_read_probes_extra"
    },
    "llama-3.1-8b-instruct": {
        "model_name": "meta-llama/Llama-3.1-8B-Instruct",
        "num_layers": 33,
        "hidden_size": 4096,
        "control_probe_dir": "llama3_control_probes_extra",
        "read_probe_dir": "llama3_read_probes_extra"
    }
}


def load_model_and_tokenizer(model_name: str, device: str = "cuda"):
    """Load the specified model and tokenizer."""
    print(f"Loading model: {model_name}...")
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
    
    print(f"Model loaded successfully!")
    return model, tokenizer


def format_conversation_for_model(
    conversation: List[Dict[str, str]],
    tokenizer: AutoTokenizer,
    model_type: str
) -> str:
    """Format conversation based on model's chat template."""
    if model_type == "gemma-2-9b-it":
        # Gemma format
        formatted = ""
        for turn in conversation:
            role = turn["role"]
            content = turn["content"]
            if role == "user":
                formatted += f"<start_of_turn>user\n{content}<end_of_turn>\n"
            else:
                formatted += f"<start_of_turn>model\n{content}<end_of_turn>\n"
        return formatted
    else:  # Llama format
        # Use the tokenizer's apply_chat_template if available
        if hasattr(tokenizer, 'apply_chat_template'):
            convos = tokenizer.apply_chat_template(
                conversation,
                tokenize=False,
                add_generation_prompt=False
            )
            if convos.endswith("<s>"):
                convos = convos[:-3]
            return convos
        else:
            raise ValueError("Tokenizer does not support apply_chat_template")
            # Manual Llama formatting
            formatted = "<s>"
            for turn in conversation:
                role = turn["role"]
                content = turn["content"]
                if role == "user":
                    formatted += f"[INST] {content} [/INST]"
                else:
                    formatted += f" {content}</s><s>"
            return formatted.rstrip("<s>")


def extract_activations_control(
    conversation: List[Dict[str, str]],
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    model_type: str,
    num_layers: int,
    device: str = "cuda"
) -> torch.Tensor:
    """
    Extract activations for control probe (last token of last user message).
    
    Returns:
        Tensor of shape (num_layers, hidden_size)
    """
    # Find the last user message
    last_user_idx = -1
    for i in range(len(conversation) - 1, -1, -1):
        if conversation[i]["role"] == "user":
            last_user_idx = i
            break
    
    if last_user_idx == -1:
        raise ValueError("No user message found in conversation")
    
    # Format up to and including the last user message
    truncated_conv = conversation[:last_user_idx + 1]
    prompt = format_conversation_for_model(truncated_conv, tokenizer, model_type)
    
    # Tokenize
    with torch.no_grad():
        inputs = tokenizer(prompt, return_tensors='pt', return_token_type_ids=False)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Forward pass to get hidden states
        outputs = model(**inputs, output_hidden_states=True)
        hidden_states = outputs.hidden_states  # Tuple of (num_layers+1, batch_size, seq_len, hidden_size)
        
        # Extract last token activations for each layer
        activations = []
        for layer_idx in range(num_layers):
            # hidden_states[0] is embeddings, hidden_states[1] is layer 0, etc.
            layer_hidden = hidden_states[layer_idx][:, -1, :]  # Last token
            activations.append(layer_hidden.cpu().float())
        
        activations = torch.cat(activations, dim=0)  # Shape: (num_layers, hidden_size)
    
    return activations


def extract_activations_read(
    conversation: List[Dict[str, str]],
    meta_attribute: str,
    target: str,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    model_type: str,
    num_layers: int,
    device: str = "cuda"
) -> torch.Tensor:
    """
    Extract activations for read probe (incomplete reading message at the end).
    
    Args:
        meta_attribute: The meta attribute label to use in the prompt (e.g., "mood" for "happy"/"sad")
    
    Returns:
        Tensor of shape (num_layers, hidden_size)
    """
    # Create reading message based on target using meta_attribute
    if target == "user":
        reading_message = f"I think the {meta_attribute} of this user is"
    else:  # chatbot
        reading_message = f"I think the {meta_attribute} of myself is"
    
    # Add incomplete reading message as assistant message
    extended_conv = conversation + [{"role": "assistant", "content": reading_message}]
    
    # Format conversation
    prompt = format_conversation_for_model(extended_conv, tokenizer, model_type)
    
    # Tokenize
    with torch.no_grad():
        inputs = tokenizer(prompt, return_tensors='pt', return_token_type_ids=False)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Forward pass to get hidden states
        outputs = model(**inputs, output_hidden_states=True)
        hidden_states = outputs.hidden_states
        
        # Extract last token activations for each layer
        activations = []
        for layer_idx in range(num_layers):
            layer_hidden = hidden_states[layer_idx][:, -1, :]
            activations.append(layer_hidden.cpu().float())
        
        activations = torch.cat(activations, dim=0)
    
    return activations


def extract_all_activations(
    dataset: Dict,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    model_type: str,
    num_layers: int,
    probe_type: str = "control",
    device: str = "cuda"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract activations from all conversations in dataset.
    
    Returns:
        X: Array of shape (num_conversations, num_layers, hidden_size)
        y: Array of shape (num_conversations,) with attribute labels (0 or 1)
    """
    conversations = dataset["conversations"]
    attribute1 = dataset["metadata"]["attribute1"]
    attribute2 = dataset["metadata"]["attribute2"]
    meta_attribute = dataset["metadata"].get("meta_attribute", attribute1)  # Default to attribute1 if not present
    target = dataset["metadata"]["target"]
    
    X_list = []
    y_list = []
    
    print(f"\nExtracting {probe_type} probe activations...")
    for conv_data in tqdm(conversations):
        conversation = conv_data["conversation"]
        attribute = conv_data["attribute"]
        
        # Determine label (0 for attribute1, 1 for attribute2)
        label = 0 if attribute == attribute1 else 1
        
        try:
            if probe_type == "control":
                activations = extract_activations_control(
                    conversation, model, tokenizer, model_type, num_layers, device
                )
            else:  # read probe
                activations = extract_activations_read(
                    conversation, meta_attribute, target, model, tokenizer,
                    model_type, num_layers, device
                )
            
            X_list.append(activations.numpy())
            y_list.append(label)
        except Exception as e:
            print(f"\nWarning: Skipping conversation {conv_data['id']} due to error: {e}")
            continue
    
    X = np.array(X_list)  # Shape: (num_conversations, num_layers, hidden_size)
    y = np.array(y_list)  # Shape: (num_conversations,)
    
    return X, y


def train_probe_for_layer(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    layer_idx: int,
    hidden_size: int,
    num_classes: int = 2,
    device: str = "cuda",
    num_epochs: int = 100,
    learning_rate: float = 0.001
) -> Tuple[LinearProbeClassification, float]:
    """
    Train a logistic regression probe for a specific layer using sklearn.
    
    Returns:
        Trained probe model and test accuracy
    """
    # Extract activations for this layer
    X_train_layer = X_train[:, layer_idx, :]  # (num_samples, hidden_size)
    X_test_layer = X_test[:, layer_idx, :]    # (num_samples, hidden_size)
    
    # Train using sklearn's LogisticRegression
    # Use max_iter=1000 and increased tolerance for better convergence
    sklearn_probe = LogisticRegression(
        max_iter=1000,
        solver='lbfgs',
        multi_class='multinomial',
        random_state=42,
        verbose=0
    )
    sklearn_probe.fit(X_train_layer, y_train)
    
    # Get predictions and calculate accuracy
    predictions = sklearn_probe.predict(X_test_layer)
    accuracy = accuracy_score(y_test, predictions)
    
    # Create PyTorch probe model
    probe = LinearProbeClassification(
        device=device,
        probe_class=num_classes,
        input_dim=hidden_size,
        logistic=True
    )
    
    # Transfer sklearn coefficients to PyTorch model
    # sklearn coef_ shape: (n_classes, n_features) for multiclass, (1, n_features) for binary
    # sklearn intercept_ shape: (n_classes,) for multiclass, (1,) for binary
    # PyTorch Linear weight shape: (out_features, in_features)
    # PyTorch Linear bias shape: (out_features,)
    
    with torch.no_grad():
        # Get the linear layer from the Sequential module
        linear_layer = probe.proj[0]  # First element is nn.Linear
        
        # Handle binary classification case
        if num_classes == 2 and sklearn_probe.coef_.shape[0] == 1:
            # For binary classification, sklearn only stores coefficients for one class
            # We need to create coefficients for both classes
            # Class 0: negative of the coefficients (implicit in sklearn)
            # Class 1: the stored coefficients
            sklearn_coef = sklearn_probe.coef_[0]  # Shape: (n_features,)
            sklearn_intercept = sklearn_probe.intercept_[0]  # Scalar
            
            # Create weight and bias for both classes
            weight_tensor = torch.stack([
                torch.from_numpy(-sklearn_coef).float(),  # Class 0
                torch.from_numpy(sklearn_coef).float()    # Class 1
            ])  # Shape: (2, n_features)
            
            bias_tensor = torch.tensor([
                -sklearn_intercept,  # Class 0
                sklearn_intercept    # Class 1
            ]).float()  # Shape: (2,)
        else:
            # For multiclass, sklearn stores coefficients for all classes
            weight_tensor = torch.from_numpy(sklearn_probe.coef_).float()
            bias_tensor = torch.from_numpy(sklearn_probe.intercept_).float()
        
        # Set the parameters
        linear_layer.weight.copy_(weight_tensor)
        linear_layer.bias.copy_(bias_tensor)
    
    probe.eval()
    
    return probe, accuracy


def train_all_probes(
    X: np.ndarray,
    y: np.ndarray,
    num_layers: int,
    hidden_size: int,
    probe_type: str,
    attribute: str,
    probe_dir: str,
    attribute1: str,
    attribute2: str,
    meta_attribute: str = None,
    icon: list = None,
    target: str = "user",
    device: str = "cuda",
    test_size: float = 0.2
) -> Dict[int, float]:
    """
    Train probes for all layers and save them.
    
    Returns:
        Dictionary mapping layer number to test accuracy
    """
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    print(f"\nTraining {probe_type} probes for attribute: {attribute}")
    print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
    
    # Create probe directory if it doesn't exist
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    full_probe_dir = os.path.join(base_dir, probe_dir)
    os.makedirs(full_probe_dir, exist_ok=True)
    
    # Create attribute-specific subdirectory for probes with target suffix
    # Use meta_attribute if provided, otherwise fall back to attribute
    # Format: {meta_attribute}_{target} (e.g., ConfidenceANDUncertain_user or positive_negative_chatbot)
    attr_for_folder = meta_attribute if meta_attribute else attribute
    folder_name = f"{attr_for_folder}_{target}"
    attribute_probe_dir = os.path.join(full_probe_dir, folder_name)
    os.makedirs(attribute_probe_dir, exist_ok=True)
    
    # Create stats directory with attribute subdirectory
    stats_dir = os.path.join(base_dir, probe_dir.replace("_probes", "_probes_stats"))
    os.makedirs(stats_dir, exist_ok=True)
    attribute_stats_dir = os.path.join(stats_dir, folder_name)
    os.makedirs(attribute_stats_dir, exist_ok=True)
    
    accuracies = {}
    
    # Train probe for each layer
    for layer_idx in tqdm(range(num_layers), desc="Training probes"):
        probe, accuracy = train_probe_for_layer(
            X_train, y_train, X_test, y_test,
            layer_idx, hidden_size, num_classes=2, device=device
        )
        
        accuracies[layer_idx] = accuracy
        
        # Save probe with proper naming in attribute subdirectory
        # Use meta_attribute for filename consistency
        filename_base = meta_attribute if meta_attribute else attribute
        probe_path = os.path.join(
            attribute_probe_dir,
            f"{filename_base}_at_layer_{layer_idx}.pth"
        )
        torch.save(probe.state_dict(), probe_path)
    
    # Save stats JSON file
    # Ensure icon is a list with 2 elements
    icon_array = icon if (icon and isinstance(icon, list) and len(icon) == 2) else ["FaQuestion", "FaQuestion"]
    
    stats_data = {
        "attribute": attribute,
        "attribute1": attribute1,
        "attribute2": attribute2,
        "meta_attribute": meta_attribute if meta_attribute else attribute1,
        "icon": icon_array,
        "target": target,
        "probe_type": probe_type,
        "num_layers": num_layers,
        "hidden_size": hidden_size,
        "training_samples": len(X_train),
        "test_samples": len(X_test),
        "layer_accuracies": {int(k): float(v) for k, v in accuracies.items()},
        "average_accuracy": float(np.mean(list(accuracies.values()))),
        "best_layer": int(max(accuracies, key=accuracies.get)),
        "best_accuracy": float(max(accuracies.values())),
        "timestamp": datetime.now().isoformat()
    }
    
    # Save stats in attribute subdirectory
    # Use meta_attribute for filename consistency
    stats_path = os.path.join(
        attribute_stats_dir,
        f"{filename_base}_at_layer.json"
    )
    with open(stats_path, 'w') as f:
        json.dump(stats_data, f, indent=2)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"{probe_type.upper()} PROBE TRAINING SUMMARY")
    print(f"{'='*60}")
    print(f"Average accuracy: {np.mean(list(accuracies.values())):.4f}")
    print(f"Best layer: {max(accuracies, key=accuracies.get)} "
          f"(accuracy: {max(accuracies.values()):.4f})")
    print(f"Probes saved to: {attribute_probe_dir}")
    print(f"Stats saved to: {stats_path}")
    print(f"{'='*60}\n")
    
    return accuracies


def train_probes_from_dataset(
    dataset_path: str,
    model_name: str,
    probe_type: str = "both",
    device: str = "cuda",
    model: Optional[AutoModelForCausalLM] = None,
    tokenizer: Optional[AutoTokenizer] = None,
    icon: list = None
) -> Dict:
    """
    Main function to train probes from a dataset.
    
    Args:
        dataset_path: Path to the JSON dataset file
        model_name: Name of the model (e.g., "gemma-2-9b-it")
        probe_type: Type of probe to train ("control", "read", or "both")
        device: Device to use for training
        model: Pre-loaded model (optional, will load if not provided)
        tokenizer: Pre-loaded tokenizer (optional, will load if not provided)
        
    Returns:
        Dictionary with training results
    """
    # Load dataset
    print(f"Loading dataset from: {dataset_path}")
    with open(dataset_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    
    attribute1 = dataset["metadata"]["attribute1"]
    attribute2 = dataset["metadata"]["attribute2"]
    meta_attribute = dataset["metadata"].get("meta_attribute", attribute1)  # Default to attribute1 if not present
    
    # Determine attribute naming: single attribute or "attribute1ANDattribute2"
    if attribute2.startswith("non-"):
        # Single attribute with non-attribute baseline
        attribute_name = attribute1
    else:
        # Two distinct attributes
        attribute_name = f"{attribute1}AND{attribute2}"
    
    # Get model config
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model: {model_name}. "
                        f"Supported models: {list(MODEL_CONFIGS.keys())}")
    
    config = MODEL_CONFIGS[model_name]
    
    # Load model if not provided
    if model is None or tokenizer is None:
        model, tokenizer = load_model_and_tokenizer(config["model_name"], device)
        should_cleanup = True
    else:
        should_cleanup = False
    
    results = {
        "dataset": dataset_path,
        "model": model_name,
        "attribute1": attribute1,
        "attribute2": attribute2,
        "meta_attribute": meta_attribute,
        "attribute_name": attribute_name,
        "control_accuracies": None,
        "read_accuracies": None
    }
    
    # Train control probes
    if probe_type in ["control", "both"]:
        X_control, y_control = extract_all_activations(
            dataset, model, tokenizer, model_name,
            config["num_layers"], probe_type="control", device=device
        )
        
        control_accuracies = train_all_probes(
            X_control, y_control,
            config["num_layers"],
            config["hidden_size"],
            probe_type="control",
            attribute=attribute_name,
            probe_dir=config["control_probe_dir"],
            attribute1=attribute1,
            attribute2=attribute2,
            meta_attribute=meta_attribute,
            icon=icon,
            target=dataset["metadata"]["target"],
            device=device
        )
        results["control_accuracies"] = control_accuracies
    
    # Train read probes
    if probe_type in ["read", "both"]:
        X_read, y_read = extract_all_activations(
            dataset, model, tokenizer, model_name,
            config["num_layers"], probe_type="read", device=device
        )
        
        read_accuracies = train_all_probes(
            X_read, y_read,
            config["num_layers"],
            config["hidden_size"],
            probe_type="read",
            attribute=attribute_name,
            probe_dir=config["read_probe_dir"],
            attribute1=attribute1,
            attribute2=attribute2,
            meta_attribute=meta_attribute,
            icon=icon,
            target=dataset["metadata"]["target"],
            device=device
        )
        results["read_accuracies"] = read_accuracies
    
    # Clean up only if we loaded the model ourselves
    if should_cleanup:
        del model, tokenizer
        torch.cuda.empty_cache()
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Train linear probes on LLM activations for attribute probing."
    )
    parser.add_argument("dataset_path", type=str, help="Path to the conversation dataset JSON file")
    parser.add_argument(
        "--model",
        type=str,
        choices=["gemma-2-9b-it", "llama-3.1-8b-instruct"],
        required=True,
        help="Model to use for probe training"
    )
    parser.add_argument(
        "--probe-type",
        type=str,
        choices=["control", "read", "both"],
        default="both",
        help="Type of probe to train (default: both)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (default: cuda)"
    )
    
    args = parser.parse_args()
    
    results = train_probes_from_dataset(
        args.dataset_path,
        args.model,
        args.probe_type,
        args.device
    )
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)

