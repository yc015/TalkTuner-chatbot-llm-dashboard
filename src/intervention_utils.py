from probes import ProbeClassification, ProbeClassificationMixScaler, LinearProbeClassification, LinearProbeClassificationMixScaler
import os
import torch.nn.functional as F

import torch
from tqdm.auto import tqdm

from dataset import llama_v2_prompt
import numpy as np

from torch import nn
device = "cuda"
torch_device = "cuda"


def load_probe_classifier(model_func, input_dim, num_classes, weight_path, **kwargs):
    """
    Instantiate a ProbeClassification model and load its pretrained weights.
    
    Args:
    - input_dim (int): Input dimension for the classifier.
    - num_classes (int): Number of classes for classification.
    - weight_path (str): Path to the pretrained weights.
    
    Returns:
    - model: The ProbeClassification model with loaded weights.
    """

    # Instantiate the model
    model = model_func(device, num_classes, input_dim, **kwargs)
    
    # Load the pretrained weights into the model
    model.load_state_dict(torch.load(weight_path))
    
    return model


num_classes = {"age": 4,
               "gender": 2,
               "education": 3,
               "socioeco": 3,}


def return_classifier_dict(directory, model_func, chosen_layer=None, mix_scaler=False, sklearn=False, **kwargs):
    checkpoint_paths = os.listdir(directory)
    # file_paths = [os.path.join(directory, file) for file in checkpoint_paths if file.endswith("pth")]
    classifier_dict = {}
    for i in range(len(checkpoint_paths)):
        category = checkpoint_paths[i][:checkpoint_paths[i].find("_")]
        weight_path = os.path.join(directory, checkpoint_paths[i])
        num_class = num_classes[category]
        if category == "gender" and sklearn:
            num_class = 1
        if category not in classifier_dict.keys():
            classifier_dict[category] = {}
        if mix_scaler:
            classifier_dict[category]["all"] = load_probe_classifier(model_func, 5120, 
                                                                     num_classes=num_class,
                                                                     weight_path=weight_path, **kwargs)
        else:
            layer_num = int(checkpoint_paths[i][checkpoint_paths[i].rfind("_") + 1: checkpoint_paths[i].rfind(".pth")])

            if chosen_layer is None or layer_num == chosen_layer:
                try:
                    classifier_dict[category][layer_num] = load_probe_classifier(model_func, 5120, 
                                                                                 num_classes=num_class,
                                                                                 weight_path=weight_path, **kwargs)
                except Exception as e:
                    print(category)
                    # print(e)
                        
    return classifier_dict


def split_into_messages(text: str) -> list[str]:
    # Constants used for splitting
    B_INST, E_INST = "[INST]", "[/INST]"

    # Use the tokens to split the text
    parts = []
    current_message = ""

    for word in text.split():
        # If we encounter a start or end token, and there's a current message, store it
        if word in [B_INST, E_INST] and current_message:
            parts.append(current_message.strip())
            current_message = ""
        # If the word is not a token, add it to the current message
        elif word not in [B_INST, E_INST]:
            current_message += word + " "

    # Append any remaining message
    if current_message:
        parts.append(current_message.strip())

    return parts


def llama_v2_reverse(prompt: str) -> list[dict]:
    # Constants used in the LLaMa style
    B_INST, E_INST = "[INST]", "[/INST]"
    B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
    BOS, EOS = "<s>", "</s>"
    messages = []
    sys_start = prompt.find(B_SYS)
    sys_end = prompt.rfind(E_SYS)
    if sys_start != -1 and sys_end != -1:
        system_msg = prompt[sys_start + len(B_SYS): sys_end]
    messages.append({"role": "system", "content": system_msg})
    prompt = prompt[sys_end + len(E_SYS):]
    
    user_ai_msgs = split_into_messages(prompt)
    
    user_turn = True
    for message in user_ai_msgs:
        if user_turn:
            messages.append({"role": "user", "content": message})
        else:
            messages.append({"role": "assistant", "content": message})
        
        if user_turn:
            user_turn = False
        else:
            user_turn = True

    return messages


def optimize_one_inter_rep(inter_rep, layer_name, target, probe,
                           lr=1e-2, max_epoch=128, 
                           loss_func=nn.CrossEntropyLoss(), 
                           verbose=False, simplified=False, N=4, normalized=False):
    global first_time
    tensor = (inter_rep.clone()).to(torch_device).requires_grad_(True)
    rep_f = lambda: tensor
    target_clone = target.clone().to(torch_device).to(torch.float)

    optimizer = torch.optim.Adam([tensor], lr=lr)

    cur_loss = 1000
    begin_tensor = rep_f().clone().detach()
    cur_input_tensor = rep_f().clone().detach()
    if not simplified:
        if verbose:
            bar = tqdm(range(max_epoch), leave=False)
        else:
            bar = range(max_epoch)
        for i in bar:
            input_tensor = rep_f()
            optimizer.zero_grad()
            probe_seg_out = probe(input_tensor)[0]
            # Compute the loss
            if first_time:
                if logistic:
                    print(probe_seg_out)
                else:
                    print(torch.nn.functional.softmax(probe_seg_out))
                first_time = False
            loss = loss_func(probe_seg_out, target_clone)
            # Call gradient descent
            loss.backward()
            optimizer.step()
            if verbose:
                bar.set_description(f'At layer {layer_name} [{i + 1}/{max_epoch}]; Loss: {loss.item():.3f}') 
            if early_stop and abs(cur_input_tensor - input_tensor).mean() < stopping_eta:
                break
            # dist = torch.sqrt(torch.sum((begin_tensor - input_tensor) ** 2))
            dist = torch.sum(torch.sqrt((begin_tensor - input_tensor) ** 2))
            if dist > dist_thresh:
                break
            # cur_loss = loss.item()
            cur_input_tensor = input_tensor.clone().detach()
    else:
        if normalized:
            cur_input_tensor = rep_f() + target_clone.view(1, -1) @ probe.proj[0].weight * N * 100 / rep_f().norm() 
        else:
            cur_input_tensor = rep_f() + target_clone.view(1, -1) @ probe.proj[0].weight * N
        dist = torch.sum(torch.sqrt((begin_tensor - cur_input_tensor.clone()) ** 2))
        if layer_name not in distance.keys():
            distance[layer_name] = []
        distance[layer_name].append(dist.item())
        return cur_input_tensor.clone()
    
    dist = torch.sum(torch.sqrt((begin_tensor - input_tensor) ** 2))
    if layer_name not in distance.keys():
        distance[layer_name] = []
    distance[layer_name].append(dist.item())
    
    return rep_f().clone()


def edit_inter_rep_multi_layers(output, layer_name):
    """
    This function must be called inside the script, given classifier dict and other hyperparameters are undefined in this function
    """
    if residual:
        layer_num = layer_name[layer_name.rfind("model.layers.") + len("model.layers."):]
    else:
        layer_num = layer_name[layer_name.rfind("model.layers.") + len("model.layers."):layer_name.rfind(".mlp")]
    layer_num = int(layer_num)
    probe = classifier_dict[attribute][layer_num]
    cloned_inter_rep = output[0][0][-1].unsqueeze(0).detach().clone().to(torch.float)
    with torch.enable_grad():
        cloned_inter_rep = optimize_one_inter_rep(cloned_inter_rep, layer_name, 
                                                  cf_target, probe,
                                                  lr=lr, max_epoch=max_epoch, 
                                                  loss_func=loss_func,
                                                  simplified=simplified,
                                                  N=N,
                                                  normalized=normalized)
    # output[1] = cloned_inter_rep.to(torch.float16)
    # print(len(output))
    output[0][0][-1] = cloned_inter_rep[0].to(torch.float16)
    return output
