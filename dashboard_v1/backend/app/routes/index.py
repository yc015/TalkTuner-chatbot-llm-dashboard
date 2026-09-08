# -*- coding: UTF-8 -*-
from app import app
from app.chat.utils import *
import json
import os
import numpy as np
from flask import send_file, request, jsonify, send_from_directory
from datetime import datetime
from os.path import dirname, abspath, join
from flask_cors import CORS, cross_origin
import pickle
import traceback

import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from torch import nn
from app.chat.classifiers import return_classifier_dict, LinearProbeClassification, load_probe_classifier
from app.chat.attribute import attribution, output_sensitivity, attribution_multi_msg, output_uncertainty
status_code = 404 # default error code

special_prompts = ["age", "gender", "education level", "ethnicity", "socioeconomic status", "marital status", "spoken language", "religious belief", "political view"]
multi_special_prompt = True

# Alter: abspath('') is called from back/run.py
rootDir = dirname(abspath(''))
# print('here', rootDir)

cate_labels = {"gender": ["male", "female"],
                "age": ["child", "adolescent", "adult", "olderAdult"],
                "education": ["someschool", "highschool", "collegemore"],
                "socioEco": ["low", "middle", "high"],

              }

translate_keys = {"gender": "gender",
                  "age": "age",
                  "education": "education",
                  "ethnics": "ethnicity",
                  "socioeco": "socioEco",
                  "marital": "marital",
                  "language": "language",
                  "religion": "religion",
                  "political": "political",
                  "sycophancy": "sycophancy",
                  "hallucination": "hallucination"}

attribute_to_prompt = {"age": "age",
                       "gender": "gender",
                       "education": "education level",
                       "ethnics": "ethnicity",
                       "socioeco": "socioeconomic status",
                       "marital": "marital status",
                       "language": "spoken language",
                       "religion": "religious belief",
                       "political": "political view",}

model = None
tokenizer = None
classifier_dict = None

# Global cache for extra probes
# loaded_files now stores "probe_type/filename" to distinguish files with same name in different folders
extra_probes_cache = {
    "llama3": {"control": {}, "read": {}, "loaded_files": set()},
    "gemma2": {"control": {}, "read": {}, "loaded_files": set()},
    "mistral": {"control": {}, "read": {}, "loaded_files": set()}
}

# Global dict to store attribute targets (user vs chatbot)
# Key: attribute name (meta_attribute), Value: "user" or "chatbot"
# Default probes have target="user", extra probes load from JSON
attribute_targets = {}

# Extra probe directories
EXTRA_PROBE_DIRS = {
    "llama3": {
        "control": "llama3_control_probes_extra",
        "read": "llama3_read_probes_extra"
    },
    "gemma2": {
        "control": "gemma2_control_probes_extra",
        "read": "gemma2_read_probes_extra"
    },
    "mistral": {
        "control": "mistral_control_probes_extra",
        "read": "mistral_read_probes_extra"
    }
}

# global tokenizer, model, classifier_dict
try:
    print('start loading model')



    # Download the model and tokenizer directly to the specified directory
    llama3_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct", use_auth_token=True, torch_dtype=torch.bfloat16)
    llama3_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B-Instruct", use_auth_token=True, torch_dtype=torch.bfloat16, device_map={"": 1})
    llama3_model.eval();


    # Download the model and tokenizer directly to the specified directory
    gemma2_tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-9b-it", use_auth_token=True,  torch_dtype=torch.bfloat16)
    gemma2_model = AutoModelForCausalLM.from_pretrained("google/gemma-2-9b-it", use_auth_token=True, torch_dtype=torch.bfloat16, device_map={"": 0})
    gemma2_model.eval();
    

    

    llama3_classifier_dict = return_classifier_dict(
        "llama3_read_probes",
                                             LinearProbeClassification, 
                                             chosen_layer=None,
                                             mix_scaler=False,
                                             hidden_neurons=4096,
                                             logistic=True
                                             )
    llama3_intervene_classifier_dict = return_classifier_dict(
        "llama3_control_probes",     
                                             LinearProbeClassification, 
                                             chosen_layer=None,
                                             mix_scaler=False,
                                             hidden_neurons=4096,
                                             logistic=True
                                             )
    
    llama3_attribution_classifier_dict = return_classifier_dict(
        # "probe_read_1",    
        "llama3_read_probes",
                                             LinearProbeClassification, 
                                             chosen_layer=None,
                                             mix_scaler=False,
                                             hidden_neurons=4096,
                                             logistic=False
                                             )
    

    gemma2_classifier_dict = return_classifier_dict(
        "gemma2_read_probes",
                                             LinearProbeClassification, 
                                             chosen_layer=None,
                                             mix_scaler=False,
                                             hidden_neurons=3584,
                                             logistic=True
                                             )
    gemma2_intervene_classifier_dict = return_classifier_dict(
        "gemma2_control_probes",     
                                             LinearProbeClassification, 
                                             chosen_layer=None,
                                             mix_scaler=False,
                                             hidden_neurons=3584,
                                             logistic=True
                                             )
    
    gemma2_attribution_classifier_dict = return_classifier_dict(
        # "probe_read_1",    
        "gemma2_read_probes",
                                             LinearProbeClassification, 
                                             chosen_layer=None,
                                             mix_scaler=False,
                                             hidden_neurons=3584,
                                             logistic=False
                                             )
    
    print("Available attributes (LLaMa3) for probe", llama3_classifier_dict.keys())
    print("Available attributes (LLaMa3) for intervention", llama3_intervene_classifier_dict.keys())
    print("Available attributes (Gemma2)", gemma2_classifier_dict.keys())
    print("Available attributes (Gemma2) for intervention", gemma2_intervene_classifier_dict.keys())
#     print("Available attributes (LLaMa2)", classifier_dict.keys())
    print('finish loading model')
    
    # Initialize default probes with target="user"
    # All built-in probes are for user attributes
    for attr in llama3_classifier_dict.keys():
        attribute_targets[attr] = "user"
    print(f"Initialized {len(attribute_targets)} default attributes with target='user'")
    for attr in gemma2_classifier_dict.keys():
        attribute_targets[attr] = "user"
    print(f"Initialized {len(attribute_targets)} default attributes with target='user'")
    
#     for single_model, single_tokenizer in zip([llama3_model, gemma2_model], [llama3_tokenizer, gemma2_tokenizer]):
    for single_model, single_tokenizer in zip([llama3_model], [llama3_tokenizer]):
        if '<pad>' not in single_tokenizer.get_vocab():
            single_tokenizer.add_special_tokens({"pad_token":"<pad>"})

        single_model.resize_token_embeddings(len(single_tokenizer))
        single_model.config.pad_token_id = single_tokenizer.pad_token_id
        assert single_model.config.pad_token_id == single_tokenizer.pad_token_id, "The model's pad token ID does not match the tokenizer's pad token ID!"
        print('Tokenizer pad token ID:', single_tokenizer.pad_token_id)
        print('Model pad token ID:', single_model.config.pad_token_id)
        print('Model config pad token ID:', single_model.config.pad_token_id)
except Exception as e:
    print(e)
        

def process_prompt_based_results(prompt_based_results):
    """
    Process prompt-based results into you_model format.
    The prompt-based results are already in the correct format (dict of dicts),
    so we just need to ensure consistency and add uncertainty if needed.
    """
    if not prompt_based_results:
        return {}
    
    # Convert attribute names if needed (e.g., ethnicity -> ethnics)
    processed = {}
    for attribute, labels_dict in prompt_based_results.items():
        # Map attribute names to match frontend expectations
        if attribute == "ethnicity":
            processed_attr = "ethnics"
        elif attribute == "socioEco":
            processed_attr = "socioEco"
        else:
            processed_attr = attribute
        
        processed[processed_attr] = labels_dict
    
    # Add uncertainty as 0 if not present (prompt-based doesn't compute uncertainty)
    if "uncertainty" not in processed:
        processed["uncertainty"] = {"uncertainty": 0.0}
    
    return processed
        
        
def default(o):
    if isinstance(o, np.int_):
        return int(o)
    raise TypeError


@app.route('/')
def _index():
    return json.dumps("back end")

@app.route('/auth', methods=['POST'])
def authenticate():
    data = request.json
    token = data.get('token')
    
    if not token:
        return jsonify({"error": "Token is required"}), 400
    
    with open('../../tokens.txt', 'r') as file:
        valid_tokens = [line.strip() for line in file]
    
    if token in valid_tokens:
        response = jsonify({'error': "Valid Token"})
        status_code = 200;
    else:
        response = jsonify({'error': "Invalid Token"})
        status_code = 400;

    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,PATCH,OPTIONS')
    return response, status_code


@app.route('/id', methods=['POST'])
def _get_session_id():
    status = "failed"
    try:
        chat_files = os.listdir("chat_history")
        chat_files = [filename for filename in chat_files if filename.endswith(".json")]
        chat_files_num = [int(filename[:-5]) for filename in chat_files]
        if len(chat_files_num) > 0:
            user_id = str(max(chat_files_num) + 1)
        else:
            user_id = str(1)
        with open(os.path.join("chat_history", user_id + ".json"), "w") as outfile:
            pass
        outfile.close()
        status = "success"
        status_code = 201
    except Exception as e:
        user_id = "error"
        status_code = 500
        print('/id fails', e)

    # return json.dumps({'id': user_id,
    #                    'status': status})
    response = jsonify({'id': user_id,
                        'verbose_status': status})
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,PATCH,OPTIONS')
    return response, status_code
        

@app.route('/inst', methods=['POST'])
def _instantiate_model():
    global tokenizer, model, classifier_dict, mistral_model, mistral_tokenizer
    status = "failed"
    try:
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-13b-chat-hf", use_auth_token=True)
        model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-13b-chat-hf", use_auth_token=True)
        model.half().cuda();
        model.eval();
        classifier_dict = return_classifier_dict("probe_weights",
                                                 LinearProbeClassification, 
                                                 chosen_layer=None,
                                                 mix_scaler=False,
                                                 # hidden_neurons=2560
                                                 logistic=True
                                                 )
        status = "success"
        status_code = 201
    except Exception as e:
        status_code = 500
        print(e)

    # return json.dumps({'id': user_id,
    #                    'status': status})
    response = jsonify({'verbose_status': status})
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,PATCH,OPTIONS')
    return response, status_code

# TODO, add a catch all failure resp like the other endpoints
@app.route('/summary/<int:id>', methods=['GET'])
def _get_session_log(id: int):
    user_id = str(id)
    chat_files = os.listdir("chat_history")
    chat_files = [filename for filename in chat_files if filename.endswith(".json")]

    if user_id is None or not (user_id + ".json" in chat_files):
        response = jsonify({'msg': "Invalid user ID"})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,PATCH,OPTIONS')
        return response, 400
    else:
        with open(os.path.join("chat_history", user_id + ".json"), "r") as infile:
            log = json.load(infile)
        infile.close()
    
    response = jsonify({'summary': log,
                        'id': user_id})
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,PATCH,OPTIONS')
    return response, 200


# Scale the intervention strength for different user attributes.
# This is similar to N in the paper. In the original dashboard, users could dynamically adjust
# the intervention strength rather than applying a fixed intervention or constant activation shift.
# To support fine-grained control over intervention strength, we use scale_factor_dict to scale
# the translation distance of the intervention vector as a function of:
# (user-specified intervened attribute probability - 50) ^ scale_factor / 10
# This nonlinear scaling is motivated by the sigmoid relationship between the model's activation
# and the predicted attribute probability: as the probability approaches the extremes, achieving
# the same change in probability generally requires a larger change in the input activation.

# NOTE: in the current implementation, we apply fix translation to each attribute when they are intervened (front-end logic) so you can safely replace the scale_factor_dict with a fixed constant, and in the comparing_models function, replace the difference_model[diff_attr][subattr] ** (scale_factor) with that constant.
scale_factor_dict = {
    "gender": 0.67,
    "age": 0.72,
    # "age": 0.96,
    "education": 0.72,
    "ethnics": 1.1,
    "socioEco": 0.85,
    "marital": 0.80,
    "language": 0.80,
    "religion": 0.80,
    "political": 0.95,
    "hallucination": 0.95,
}


mistral_scale_factor_dict = {
    "gender": 0.70,
    "age": 0.70,
    "education": 0.65,
    "ethnics": 0.7,
    "socioEco": 0.50,
    "marital": 0.50,
    "language": 0.50,
    "religion": 0.60,
    "political": 0.60,
    "hallucination": 0.60,
}

llama3_scale_factor_dict = {
    "gender": 0.20,
    "age": 0.15,
    "education": 0.15,
    "socioEco": 0.15,
}

gemma2_scale_factor_dict = {
    "gender": 1.15,
    "age": 1.00,
    "education": 1.00,
    "socioEco": 1.15,
}

# Global dict to store scale factors for extra probes
# Key: meta_attribute, Value: scale factor
extra_probe_scale_factors = {}

# Hardcoded mapping for default attributes (handles special name cases)
# Frontend sends these exact labels, backend uses different keys in cate_labels
DEFAULT_ATTRIBUTE_DICT = {
    "gender": ["Male", "Female", "Other"],
    "age": ["Child", "Adolescent", "Adult", "Older Adult"],
    "education": ["Some Education", "High School", "College & More"],
    "ethnics": ["Asian", "African", "White", "Hispanic", "Native Americans", "Arab", "Jewish"],
    "socioeco": ["Lower", "Middle", "Upper"],
    "marital": ["Single", "Married", "Divorced", "Widowed"],
    "language": ["Chinese", "Japanese", "English", "German", "Spanish", "Portuguese", "Arabic", "Russian"],
    "religion": ["Christian", "Islam", "Buddhism", "Hinduism", "Judaism", "Atheism"],
    "political": ["Left", "Right", "Moderate"],
    "sycophancy": ["Sycophant", "nonSycohant"]
}


def find_attribute_for_subattribute(subattribute, cate_labels):
    """
    Find attribute category for a given subattribute.
    Handles both default attributes (hardcoded) and extra probes (case-insensitive).
    
    Args:
        subattribute: The trait name sent from frontend (e.g., "Positive", "Older Adult")
        cate_labels: Global cate_labels dict (includes both default and extra probes)
    
    Returns:
        attribute name (e.g., "age", "positive_negative") or None if not found
    """
    # First, try hardcoded default attributes (handles complex cases like "Older Adult" -> "olderAdult")
    for attr, labels in DEFAULT_ATTRIBUTE_DICT.items():
        if subattribute in labels:
            return attr
    
    # For extra probes: simple case-insensitive match
    # Extra probes use attribute1/attribute2 with simple capitalization
    # Backend has "positive", frontend sends "Positive"
    subattribute_lower = subattribute.lower()
    for attr, labels in cate_labels.items():
        # Skip default attributes (already checked above)
        if attr in DEFAULT_ATTRIBUTE_DICT:
            continue
        
        # For extra probes, check case-insensitive match
        if subattribute_lower in [label.lower() for label in labels]:
            return attr
    
    return None


def merge_extra_probes(base_classifier_dict, model_type, probe_type='read'):
    """
    Merge extra probes from cache with base classifier dict.
    
    Args:
        base_classifier_dict: The base classifier dictionary (e.g., llama3_classifier_dict)
        model_type: Model type (llama3, gemma2, mistral)
        probe_type: Type of probes to merge ('read' or 'control')
    
    Returns:
        Merged classifier dictionary with both base and extra probes
    """
    global extra_probes_cache
    
    if model_type not in extra_probes_cache:
        return base_classifier_dict
    
    # Create a merged dict with base probes + extra probes
    merged_classifier_dict = dict(base_classifier_dict)
    
    if probe_type in extra_probes_cache[model_type]:
        for attribute_name, layers in extra_probes_cache[model_type][probe_type].items():
            if attribute_name not in merged_classifier_dict:
                merged_classifier_dict[attribute_name] = {}
            merged_classifier_dict[attribute_name].update(layers)
    
    return merged_classifier_dict


def comparing_models(original, control, status=None, scale_factor_dict=scale_factor_dict, model_type=None):
    global extra_probe_scale_factors
    
    # Merge extra probe scale factors with base scale factors
    merged_scale_factors = dict(scale_factor_dict)
    merged_scale_factors.update(extra_probe_scale_factors)
    
    difference_model = {}
    need_intervene = False
    if model_type == "mistral":
        up_limit = 4
    elif model_type == "gemma2":
        up_limit = 18
    else:
        up_limit = 15

    if model_type == "mistral":
        sum_limit = 4
    elif model_type == "gemma2":
        sum_limit = 18
    else:
        sum_limit = 15
    if model_type == "mistral":
        total_sum_limit = 4
    elif model_type == "gemma2":
        total_sum_limit = 18
    else:
        total_sum_limit = 15
    total_sum = 0
    for attr in list(original.keys()):
        if attr == "ethnicity":
            diff_attr = "ethnics"
        else:
            diff_attr = attr
        if attr == "uncertainty" or attr == "sycophancy":
            continue
        difference_model[diff_attr] = {}
        for subattr in list(original[attr].keys()):
            if (control[attr][subattr] is None) or (original[attr][subattr] is None):
                continue
            degree = (control[attr][subattr]) / 10

            if degree < 5:
                degree = (5 - degree) * (-2)
            else:
                degree = degree * 2
            difference_model[diff_attr][subattr] = degree
            if model_type == "mistral":
                difference_model[diff_attr][subattr] /= 3
                
            if difference_model[diff_attr][subattr] > up_limit:
                difference_model[diff_attr][subattr] = up_limit
            elif difference_model[diff_attr][subattr] < -up_limit:
                difference_model[diff_attr][subattr] = -up_limit 
            
            if model_type == "mistral":
                scale_factor = mistral_scale_factor_dict.get(diff_attr, 0.60)
            else:
                # Use merged scale factors (includes both base and extra probes)
                scale_factor = merged_scale_factors.get(diff_attr, 1.0)
            if difference_model[diff_attr][subattr] > 0:
                difference_model[diff_attr][subattr] = difference_model[diff_attr][subattr] ** (scale_factor)
            elif difference_model[diff_attr][subattr] < 0:
                difference_model[diff_attr][subattr] = -((-difference_model[diff_attr][subattr]) ** (scale_factor))
            if subattr == "low":
                difference_model[diff_attr][subattr] *= 1.6
            if subattr == "child":
                difference_model[diff_attr][subattr] *= 1.8

                
            if abs(difference_model[diff_attr][subattr]) < 1:
                difference_model[diff_attr][subattr] = 0
            if subattr == "unknown":
                difference_model[diff_attr][subattr] = 0
                
            if (status is not None) and (subattr in status[attr]) and (not status[attr][subattr]):
                difference_model[diff_attr][subattr] = 0
                
            total_sum += abs(difference_model[diff_attr][subattr])
            if abs(difference_model[diff_attr][subattr]) >= 1:
                need_intervene = True
        if sum([abs(value) for value in difference_model[diff_attr].values()]) >= sum_limit:
            cur_sum = sum([abs(value) for value in difference_model[diff_attr].values()])
            for subattr in list(original[attr].keys()):
                difference_model[diff_attr][subattr] *= sum_limit / cur_sum
            total_sum = total_sum - cur_sum + sum_limit
          
    # Prevent over-intervention
    if total_sum > total_sum_limit:
        for attr in list(original.keys()):
            if attr == "ethnicity":
                diff_attr = "ethnics"
            else:
                diff_attr = attr
            if attr == "uncertainty" or attr == "sycophancy":
                continue
            for subattr in list(original[attr].keys()):
                difference_model[diff_attr][subattr] *= total_sum_limit / total_sum
            
    return difference_model, need_intervene
    

@app.route('/chat', methods=['POST'])
def _chat_with_bot(data=None):
    global model, tokenizer, mistral_model, mistral_tokenizer
    if data:
        post_data = data
    else:
        post_data = json.loads(request.data.decode())
    user_id = str(post_data["id"])
    
    sys_prompt = None
    if "sysPrompt" in post_data.keys():
        sys_prompt = post_data["sysPrompt"]
        
    chat_files = os.listdir("chat_history")
    chat_files = [filename for filename in chat_files if filename.endswith(".json")]
    
    if ("model" in post_data.keys()):
        model_type = post_data["model"].lower()
    print(model_type)
    
    if not ("controlStatus" in post_data.keys()):
        post_data["controlStatus"] = None
    
    chosen_scale_factor_dict = scale_factor_dict
    if model_type == "llama3":
        chosen_scale_factor_dict = llama3_scale_factor_dict
    elif model_type == "gemma2":
        chosen_scale_factor_dict = gemma2_scale_factor_dict
    difference_model, need_intervene = comparing_models(post_data["currentYouModel"], 
                                                        post_data["controlSetting"],
                                                        post_data["controlStatus"],
                                                        scale_factor_dict=chosen_scale_factor_dict,
                                                        model_type=model_type)
    print(difference_model, need_intervene)
    print(model_type)
    responded = False
    parsed = False
    
    if need_intervene:
        if user_id is None or not (user_id + ".json" in chat_files):
            messages = [{"role": "user", "content": post_data["msg"]}]
            chat_files_num = [int(filename[:-5]) for filename in chat_files]
            if len(chat_files_num) > 0:
                user_id = str(max(chat_files_num) + 1)
            else:
                user_id = str(1)
            with open(os.path.join("chat_history", user_id + ".json"), "w") as outfile:
                json.dump(messages, outfile)
            outfile.close()
        else:
            # print(os.path.join("chat_history", user_id + ".json"))
            if os.path.getsize(os.path.join("chat_history", user_id + ".json")):
                with open(os.path.join("chat_history", user_id + ".json"), "r") as infile:
                    messages = json.load(infile)
                    for msg in messages:
                        msg["content"] = msg["content"].strip().replace("\\n", "\n")
                infile.close()
                messages = messages + [{"role": "user", "content": post_data["msg"]}]
            else:
                messages = [{"role": "user", "content": post_data["msg"]}]
        if model_type == "mistral":
            used_model = mistral_model
            used_tokenizer = mistral_tokenizer
            used_intervene_classifier_dict = mistral_intervene_classifier_dict
            from_idx = 20
            to_idx = 25
        elif model_type == "llama3":
            used_model = llama3_model
            used_tokenizer = llama3_tokenizer
            used_intervene_classifier_dict = llama3_intervene_classifier_dict
            from_idx = 20
            to_idx = 25
        elif model_type == "gemma2":
            used_model = gemma2_model
            used_tokenizer = gemma2_tokenizer
            used_intervene_classifier_dict = gemma2_intervene_classifier_dict
            from_idx = 20
            to_idx = 30
        else:
            used_model = model
            used_tokenizer = tokenizer
            used_intervene_classifier_dict = intervene_classifier_dict
            from_idx = 20
            to_idx = 30
        
        # Merge extra control probes with base intervene classifier dict
        used_intervene_classifier_dict = merge_extra_probes(used_intervene_classifier_dict, model_type, probe_type='control')
        print(f"Merged control probes for {model_type}, total attributes: {list(used_intervene_classifier_dict.keys())}")

        responses, scores, last_acts, prompt_based_results = control_chat_with_llama(messages, used_model, 
                                                    used_tokenizer, classifiers=used_intervene_classifier_dict,
                                                    attributes_info=difference_model,
                                                    layer_num_min=from_idx,
                                                    layer_num_max=to_idx,
                                                    residual_stream=True, 
                                                    include_inst=True, 
                                                    return_user_msg_last_act=True, 
                                                    context_length=2048,
                                                    sys_prompt=sys_prompt,
                                                    special_prompt=special_prompts,
                                                    multi_special_prompt=multi_special_prompt,
                                                    system_special_prompt={"sycophancy": '<s> [INST] Were you sycophantic with my view when answering the previous question? Reply "I think yes" or "I think no" [/INST] I think',
                                                                     "hallucination": '<s> [INST] Is your answer to my previous question hallucinated or factually coherent? [/INST] My previous answer is'},
                                                               mistral= model_type == "mistral",
                                                    model_type=model_type,
                                                    attribute_targets=attribute_targets,
                                                    cate_labels=cate_labels)
        torch.cuda.empty_cache()
        responded = True
        ai_response = []
        try:
            for response in responses:
                print(response)
                if model_type == "mistral":
                    ai_response.append(extract_mistral_response(response)["content"])
                else:
                    ai_response.append(extract_chatbot_response(response)["content"])
        except Exception as e:
            print(e)

        new_chat_history = messages + [{"role":"assistant", "content": ai_response[0]}]
        last_acts["uncertainty"] = scores
        with open(os.path.join("chat_history", user_id + ".json"), "w") as outfile:
            json.dump(new_chat_history, outfile)
        with open(os.path.join("chat_history", "last_token_act_" + user_id + ".pkl"), "wb") as outfile:
            pickle.dump(last_acts, outfile)
        
        # Save prompt-based results
        with open(os.path.join("chat_history", "prompt_based_act_" + user_id + ".pkl"), "wb") as outfile:
            pickle.dump(prompt_based_results, outfile)

        you_model, _ = query_you_model_llama(post_data)
        
        # Process prompt-based results into you_model format
        you_model_prompt = process_prompt_based_results(prompt_based_results)

        parsed = True
        status = "success"

        response = jsonify({'msg': ai_response[0],
                            'id': user_id,
                            'verbose_status': status,
                            'you_model': json.loads(you_model.get_data(as_text=True)),
                            'you_model_prompt': you_model_prompt})

        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,PATCH,OPTIONS')
        return response, 201
    
    else:
        if user_id is None or not (user_id + ".json" in chat_files):
            messages = [{"role": "user", "content": post_data["msg"]}]
            chat_files_num = [int(filename[:-5]) for filename in chat_files]
            if len(chat_files_num) > 0:
                user_id = str(max(chat_files_num) + 1)
            else:
                user_id = str(1)
            with open(os.path.join("chat_history", user_id + ".json"), "w") as outfile:
                json.dump(messages, outfile)
            outfile.close()
        else:
            # print(os.path.join("chat_history", user_id + ".json"))
            if os.path.getsize(os.path.join("chat_history", user_id + ".json")):
                with open(os.path.join("chat_history", user_id + ".json"), "r") as infile:
                    messages = json.load(infile)
                    for msg in messages:
                        msg["content"] = msg["content"].replace("\\n", "\n")
                infile.close()
                messages = messages + [{"role": "user", "content": post_data["msg"]}]
            else:
                messages = [{"role": "user", "content": post_data["msg"]}]
        try:
            if model_type == "mistral":
                used_model = mistral_model
                used_tokenizer = mistral_tokenizer
            elif model_type == "llama3":
                used_model = llama3_model
                used_tokenizer = llama3_tokenizer
            elif model_type == "gemma2":
                used_model = gemma2_model
                used_tokenizer = gemma2_tokenizer
            else:
                used_model = model
                used_tokenizer = tokenizer
            response, last_acts, prompt_based_results = chat_with_llama(messages, used_model, used_tokenizer, residual_stream=True, include_inst=True,
                                                  return_user_msg_last_act=True, context_length=2048, special_prompt=special_prompts,
                                                  multi_special_prompt=multi_special_prompt,
                                                  system_special_prompt={"sycophancy": '<s> [INST] Were you sycophantic with my view when answering the previous question? Reply "I think yes" or "I think no" [/INST] I think',
                                                                         "hallucination": '<s> [INST] Is your answer to my previous question hallucinated or factually coherent? [/INST] My previous answer is'},
                                                  sys_prompt=sys_prompt,
                                                  mistral= model_type == "mistral",
                                                  model_type=model_type,
                                                  attribute_targets=attribute_targets,
                                                  cate_labels=cate_labels)
            print(response)
            responded = True
            if model_type == "mistral":
                ai_response = extract_mistral_response(response)
            else:
                ai_response = extract_chatbot_response(response)
            ai_response = ai_response
            print(ai_response)
            parsed = True
            status = "success"

        except Exception as e:
            print(e)
            status = "failed"
            error_msg = ""
            if not responded:
                error_msg = "LLaMa failed to generate the message"
                status_code = 404
            if not parsed:
                error_msg = "A parsing error occured when we tried to extract the AI's message from LLaMa's response"
                status_code = 500
            response = jsonify({'msg': error_msg,
                                'id': user_id,
                                'verbose_status': status})
            response.headers.add('Access-Control-Allow-Origin', '*')
            response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,PATCH,OPTIONS')
            return response, status_code


        new_chat_history = messages + [{"role": ai_response["role"], "content": ai_response["content"].replace("\n", "\\n")}]
        with open(os.path.join("chat_history", user_id + ".json"), "w") as outfile:
            json.dump(new_chat_history, outfile)
        with open(os.path.join("chat_history", "last_token_act_" + user_id + ".pkl"), "wb") as outfile:
            pickle.dump(last_acts, outfile)
        
        # Save prompt-based results
        with open(os.path.join("chat_history", "prompt_based_act_" + user_id + ".pkl"), "wb") as outfile:
            pickle.dump(prompt_based_results, outfile)

        you_model, _ = query_you_model_llama(post_data)
        
        # Process prompt-based results into you_model format
        you_model_prompt = process_prompt_based_results(prompt_based_results)

        response = jsonify({'msg': ai_response["content"],
                            'id': user_id,
                            'verbose_status': status,
                            'you_model':  json.loads(you_model.get_data(as_text=True)),
                            'you_model_prompt': you_model_prompt})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,PATCH,OPTIONS')
    return response, 201


@app.route('/chat_batched', methods=['POST'])
def _chat_with_bot_batched(data=None):
    global model, tokenizer, mistral_model, mistral_tokenizer 
    if data:
        post_data = data
    else:
        post_datas = json.loads(request.data.decode())
    
    messages_array = []
    ai_responses = []
    for post_data in post_datas:
        user_id = str(post_data["id"])
        sys_prompt = None
        if "sysPrompt" in post_data.keys():
            sys_prompt = post_data["sysPrompt"]

        chat_files = os.listdir("chat_history")
        chat_files = [filename for filename in chat_files if filename.endswith(".json")]

        model_type = ""
        if ("model" in post_data.keys()):
            model_type = post_data["model"].lower()
        print(model_type)

        if not ("controlStatus" in post_data.keys()):
            post_data["controlStatus"] = None
        chosen_scale_factor_dict = scale_factor_dict
        if model_type == "llama3":
            chosen_scale_factor_dict = llama3_scale_factor_dict
        elif model_type == "gemma2":
            chosen_scale_factor_dict = gemma2_scale_factor_dict
        difference_model, need_intervene = comparing_models(post_data["currentYouModel"], 
                                                            post_data["controlSetting"],
                                                            post_data["controlStatus"],
                                                            scale_factor_dict=chosen_scale_factor_dict,
                                                            model_type=model_type)
        print(difference_model, need_intervene)

        responded = False
        parsed = False
        if user_id is None or not (user_id + ".json" in chat_files):
            messages = [{"role": "user", "content": post_data["msg"]}]
            chat_files_num = [int(filename[:-5]) for filename in chat_files]
            if len(chat_files_num) > 0:
                user_id = str(max(chat_files_num) + 1)
            else:
                user_id = str(1)
            with open(os.path.join("chat_history", user_id + ".json"), "w") as outfile:
                json.dump(messages, outfile)
            outfile.close()
        else:
            # print(os.path.join("chat_history", user_id + ".json"))
            if os.path.getsize(os.path.join("chat_history", user_id + ".json")):
                with open(os.path.join("chat_history", user_id + ".json"), "r") as infile:
                    messages = json.load(infile)
                    for msg in messages:
                        msg["content"] = msg["content"].strip().replace("\\n", "\n")
                infile.close()
                messages = messages + [{"role": "user", "content": post_data["msg"]}]
            else:
                messages = [{"role": "user", "content": post_data["msg"]}]
        
        messages_array.append(messages)
    
    # try:
    if model_type == "mistral":
        used_model = mistral_model
        used_tokenizer = mistral_tokenizer
    elif model_type == "llama3":
        used_model = llama3_model
        used_tokenizer = llama3_tokenizer
    elif model_type == "gemma2":
        used_model = gemma2_model
        used_tokenizer = gemma2_tokenizer
    else:
        used_model = model
        used_tokenizer = tokenizer
    responses, last_acts, prompt_based_results_array = chat_with_llama_batched(messages_array, used_model, used_tokenizer, residual_stream=True, include_inst=True,
                                                    return_user_msg_last_act=True, context_length=4096, special_prompt=special_prompts,
                                                    multi_special_prompt=multi_special_prompt,
                                                    system_special_prompt={"sycophancy": '<s> [INST] Were you sycophantic with my view when answering the previous question? Reply "I think yes" or "I think no" [/INST] I think',
                                                                            "hallucination": '<s> [INST] Is your answer to my previous question hallucinated or factually coherent? [/INST] My previous answer is'},
                                                    sys_prompt=sys_prompt,
                                                    mistral= model_type == "mistral",
                                                    model_type=model_type,
                                                    attribute_targets=attribute_targets,
                                                    cate_labels=cate_labels)
    print(responses)
    
    for response in responses:
        responded = True
        if model_type == "mistral":
            ai_response = extract_mistral_response(response)
        else:
            ai_response = extract_chatbot_response(response)
        ai_response = ai_response
        print(ai_response)
        ai_responses.append(ai_response)
    parsed = True
    status = "success"

    response_array = []
    for post_data, messages, ai_response, last_act, prompt_based_results in zip(post_datas, messages_array, ai_responses, last_acts, prompt_based_results_array):
        user_id = str(post_data["id"])
        
        new_chat_history = messages + [{"role": ai_response["role"], "content": ai_response["content"].replace("\n", "\\n")}]
        with open(os.path.join("chat_history", user_id + ".json"), "w") as outfile:
            json.dump(new_chat_history, outfile)
        with open(os.path.join("chat_history", "last_token_act_" + user_id + ".pkl"), "wb") as outfile:
            pickle.dump(last_act, outfile)
        
        # Save prompt-based results
        with open(os.path.join("chat_history", "prompt_based_act_" + user_id + ".pkl"), "wb") as outfile:
            pickle.dump(prompt_based_results, outfile)
            
        you_model, _ = query_you_model_llama(post_data)

        # Process prompt-based results into you_model format
        you_model_prompt = process_prompt_based_results(prompt_based_results)

        response_array.append(
            {'msg': ai_response["content"],
            'id': user_id,
            'verbose_status': status,
            'you_model':  json.loads(you_model.get_data(as_text=True)),
            'you_model_prompt': you_model_prompt}
        )

    response = jsonify({"responses": response_array})
    
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,PATCH,OPTIONS')
    return response, 201


@app.route('/regenerate_chat', methods=['POST'])
def _regenerate_chat():
    global model, tokenizer
    # print(request)
    # print(request.data)
    post_data = json.loads(request.data.decode())
    # Check if post_data is still a string and try to decode it again
    if isinstance(post_data, str):
        post_data = json.loads(post_data)

    user_id = str(post_data["id"])

    chat_files = os.listdir("chat_history")
    chat_files = [filename for filename in chat_files if filename.endswith(".json")]

    if os.path.getsize(os.path.join("chat_history",  "{}.json".format(user_id))):
        with open(os.path.join("chat_history", "{}.json".format(user_id)), "r") as infile:
            messages = json.load(infile)
            for msg in messages:
                msg["content"] = msg["content"].replace("\\n", "\n")
        infile.close()
    
    print("-" * 20, "OLD CHATHISTORY", "-" * 20, "\n", messages)
    
    new_chat_history = messages[:post_data["mid"] - 1]
                
    with open(os.path.join("chat_history", user_id + ".json"), "w") as outfile:
        json.dump(new_chat_history, outfile)
        
    post_data["msg"] = messages[post_data["mid"] - 1]["content"]
    print("-" * 20, "NEW CHATHISTORY", "-" * 20, "\n", new_chat_history)
        
    response, code = _chat_with_bot(post_data)
    return response, code


@app.route('/remove_history', methods=['POST'])
def _remove_history():
    global model, tokenizer
    # print(request)
    # print(request.data)
    post_data = json.loads(request.data.decode())
    user_id = str(post_data["id"])
    # print(user_id)
    chat_files = os.listdir("chat_history")
    chat_files = [filename for filename in chat_files if filename.endswith(".json")]

    if os.path.getsize(os.path.join("chat_history", user_id + ".json")):
        with open(os.path.join("chat_history", user_id + ".json"), "r") as infile:
            messages = json.load(infile)
            for msg in messages:
                msg["content"] = msg["content"].replace("\\n", "\n")
        infile.close()
    
    
    new_chat_history = messages[:post_data["mid"]]
                
    with open(os.path.join("chat_history", user_id + ".json"), "w") as outfile:
        json.dump(new_chat_history, outfile)
        
    status = "success"
        
    response = jsonify({'id': user_id,
                        'verbose_status': status})
    
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,PATCH,OPTIONS')
    return response, 201


def query_you_model_llama(post_data):
    global extra_probes_cache
    
    model_type = ""
    if ("model" in post_data.keys()):
        model_type = post_data["model"].lower()
        
    if model_type == "mistral":
        used_classifier_dict = mistral_classifier_dict
        from_idx = 25
        to_idx = 32
    elif model_type == "llama3":
        used_classifier_dict = llama3_classifier_dict
        from_idx = 20
        to_idx = 31
    elif model_type == "gemma2":
        used_classifier_dict = gemma2_classifier_dict
        from_idx = 20
        to_idx = 42
    else:
        used_classifier_dict = classifier_dict
        from_idx = 25
        to_idx = 35
    
    # Merge extra probes with base classifier dict
    if model_type in extra_probes_cache:
        # Create a merged dict with base probes + extra probes
        merged_classifier_dict = dict(used_classifier_dict)
        for attribute_name, layers in extra_probes_cache[model_type]["read"].items():
            if attribute_name not in merged_classifier_dict:
                merged_classifier_dict[attribute_name] = {}
            merged_classifier_dict[attribute_name].update(layers)
        used_classifier_dict = merged_classifier_dict
    
    # Filter by visible attributes if provided
    visible_attributes = post_data.get("visibleAttributes", [])
    if visible_attributes and len(visible_attributes) > 0:
        print(f"Filtering probes to visible attributes: {visible_attributes}")
        # Create filtered classifier dict with only visible attributes
        filtered_classifier_dict = {}
        for attr in visible_attributes:
            # Handle attribute name mapping (e.g., ethnicity -> ethnics)
            mapped_attr = attr
            if attr == "ethnicity":
                mapped_attr = "ethnics"
            elif attr == "socioEco":
                mapped_attr = "socioeco"
        
            if mapped_attr in used_classifier_dict:
                filtered_classifier_dict[mapped_attr] = used_classifier_dict[mapped_attr]
            elif attr in used_classifier_dict:
                filtered_classifier_dict[attr] = used_classifier_dict[attr]
        
        used_classifier_dict = filtered_classifier_dict
        print(f"Using probes for attributes: {list(used_classifier_dict.keys())}")
    
    # questions_dict = post_data.get("questions")
    user_id = str(post_data["id"])
    if user_id is None:
        json_object = {}
        for attribute in cate_labels.keys():
            json_object[attribute] = {}
            for counter in range(0, len(cate_labels[attribute])):
                json_object[attribute][cate_labels[attribute][counter]] = 1 / len(cate_labels[attribute])
                
        response = jsonify(json_object)
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,PATCH,OPTIONS')
        return response, 400
    
    chat_files = os.listdir("chat_history")
    token_files = [filename for filename in chat_files if filename.endswith(".pkl")]
    
    if user_id is None or not ("last_token_act_" + user_id + ".pkl" in token_files):
        response = jsonify({'data': "Chat history file not found",
                            'verbose_status': "failed"})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,PATCH,OPTIONS')
        return response, 404
    else:
        if os.path.getsize(os.path.join("chat_history", "last_token_act_" + user_id + ".pkl")):
            with open(os.path.join("chat_history", "last_token_act_" + user_id + ".pkl"), "rb") as infile:
                last_token_act = pickle.load(infile)
            infile.close()
        else:
            response = jsonify({'data': "Corrupted last token activation",
                                'verbose_status': "failed"})
            response.headers.add('Access-Control-Allow-Origin', '*')
            response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,PATCH,OPTIONS')
            return response, 500
        
    if isinstance(last_token_act, dict):
        score_dict = {}
        for attribute in used_classifier_dict.keys():
            one_last_token_act = last_token_act[attribute].to("cuda")
            aggregated_output = None
            start_idx = from_idx
            if attribute == "sycophancy":
                start_idx = 20

            print(len(used_classifier_dict[attribute].keys()), one_last_token_act.shape[1])
            for layer_num in range(start_idx, to_idx):
                output = used_classifier_dict[attribute][layer_num](one_last_token_act[:, layer_num])
                output = output[0].detach().cpu()
                output = output.numpy()
                # output /= output.sum(axis=1)
                if aggregated_output is None:
                    aggregated_output = output
                else:
                    aggregated_output += output
            aggregated_output /= (to_idx - start_idx)

            attribute = translate_keys[attribute]
            score_dict[attribute] = {}
            counter = 0
            aggregated_output = aggregated_output[0]

            for prob in aggregated_output:
                score_dict[attribute][cate_labels[attribute][counter]] = float(prob)
                counter += 1
            
        print("It's a dict!")
    else:
        last_token_act = last_token_act.to("cuda")
        score_dict = {}
        for attribute in used_classifier_dict.keys():
            aggregated_output = None
            for layer_num in range(from_idx, to_idx):
                output = used_classifier_dict[attribute][layer_num](last_token_act[:, layer_num])
                output = output[0].detach().cpu()
                output = output.numpy()
                # output /= output.sum(axis=1)
                if aggregated_output is None:
                    aggregated_output = output
                else:
                    aggregated_output += output
            # aggregated_output /= aggregated_output.sum(axis=1)
            aggregated_output /= (to_idx - from_idx)

            attribute = translate_keys[attribute]
            score_dict[attribute] = {}
            counter = 0
            aggregated_output = aggregated_output[0]

            for prob in aggregated_output:
                score_dict[attribute][cate_labels[attribute][counter]] = float(prob)
                counter += 1
    
    json_object = {}
    for attribute in score_dict.keys():
        json_object[attribute] = {}
        # for counter in range(0, len(cate_labels[attribute])):
        for counter in range(0, len(score_dict[attribute].keys())):
            # json_object[attribute]["values"].append({"answer": cate_labels[attribute][counter],
            #                                          "value": score_dict[attribute][cate_labels[attribute][counter]]})
            json_object[attribute][cate_labels[attribute][counter]] = score_dict[attribute][cate_labels[attribute][counter]]

    json_object["uncertainty"] = torch.mean(last_token_act["uncertainty"]).item() / 1.2
    if json_object["uncertainty"] > 1:
        json_object["uncertainty"] = 1
    print(json_object)
    response = jsonify(json_object)
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,PATCH,OPTIONS')
    return response, 201

@app.route('/query_you_model_llama', methods=['POST'])
def _query_you_model_llama():
    global classifier_dict, cate_labels, mistral_classifier_dict
    post_data = json.loads(request.data.decode())
    
    return query_you_model_llama(post_data)

    
@app.route("/post_chat_exploration", methods=['POST'])
def _post_chat_with_intervention():
    global model, tokenizer, mistral_model, mistral_tokenizer
    post_data = json.loads(request.data.decode())
    user_id = str(post_data["id"])
    attribute = None
    if "attribute" in post_data.keys():
        attribute = post_data["attribute"]
    subattribute = post_data["subattribute"]
    samples = 5
    if "samples" in post_data.keys():
        samples = int(post_data["samples"])
    N_min = None
    if "minN" in post_data.keys():
        N_min = int(post_data["minN"])
    N_max = None
    if "maxN" in post_data.keys():
        N_max = int(post_data["maxN"])
    chat_files = os.listdir("chat_history")
    chat_files = [filename for filename in chat_files if filename.endswith(".json")]

    responded = False
    parsed = False
    model_type = ""
    if ("model" in post_data.keys()):
        model_type = post_data["model"].lower()
    if user_id is None or not (user_id + ".json" in chat_files):
        status = "failed"
        error_msg = "User id does not exist."
        response = jsonify({'msg': error_msg,
                            'id': user_id,
                            'verbose_status': status})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,PATCH,OPTIONS')
        return response, status_code
    else:
        # print(os.path.join("chat_history", user_id + ".json"))
        if os.path.getsize(os.path.join("chat_history", user_id + ".json")):
            with open(os.path.join("chat_history", user_id + ".json"), "r") as infile:
                messages = json.load(infile)
                for msg in messages:
                    msg["content"] = msg["content"].replace("\\n", "\n")
            infile.close()
            messages = messages[:-1]
        else:
            status = "failed"
            error_msg = "No chat history"
            response = jsonify({'msg': error_msg,
                            'id': user_id,
                            'verbose_status': status})
            response.headers.add('Access-Control-Allow-Origin', '*')
            response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,PATCH,OPTIONS')
            return response, status_code
    # try:
    if model_type == "mistral":
        used_model = mistral_model
        used_tokenizer = mistral_tokenizer
        used_intervene_classifier_dict = mistral_intervene_classifier_dict
        from_idx = 20
        to_idx = 25
    elif model_type == "llama3":
        used_model = llama3_model
        used_tokenizer = llama3_tokenizer
        used_intervene_classifier_dict = llama3_intervene_classifier_dict
        from_idx = 20
        to_idx = 25
    elif model_type == "gemma2":
        used_model = gemma2_model
        used_tokenizer = gemma2_tokenizer
        used_intervene_classifier_dict = gemma2_intervene_classifier_dict
        from_idx = 20
        to_idx = 30
    else:
        used_model = model
        used_tokenizer = tokenizer
        used_intervene_classifier_dict = intervene_classifier_dict
        from_idx = 20
        to_idx = 30
    
    # Merge extra control probes with base intervene classifier dict
    used_intervene_classifier_dict = merge_extra_probes(used_intervene_classifier_dict, model_type, probe_type='control')
    
    responses = chat_with_llama_with_intervention(messages, used_model, 
                                                  used_tokenizer, classifiers=used_intervene_classifier_dict,
                                                  attribute=attribute,
                                                  subattribute=subattribute,
                                                  N_min=N_min,
                                                  N_max=N_max,
                                                  layer_num_min=from_idx,
                                                  layer_num_max=to_idx,
                                                  samples=samples,
                                                  residual_stream=True, 
                                                  include_inst=True, 
                                                  attribute_targets=attribute_targets,
                                                  return_user_msg_last_act=True, 
                                                  context_length=2048,
                                                  mistral=model_type == "mistral",
                                                  model_type=model_type,
                                                  cate_labels=cate_labels)
    responded = True
    ai_response = []
    try:
        for response in responses:
            print(response)
            if model_type == "mistral":
                ai_response.append(extract_mistral_response(response)["content"])
            else:
                ai_response.append(extract_chatbot_response(response)["content"])
    except Exception as e:
        print(e)
    parsed = True
    status = "success"

    response = jsonify({subattribute: ai_response,
                        'id': user_id,
                        'verbose_status': status})
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,PATCH,OPTIONS')
    return response, 201


@app.route("/pre_chat_exploration", methods=['POST'])
def _pre_chat_with_intervention():
    global model, tokenizer
    post_data = json.loads(request.data.decode())
    user_id = str(post_data["id"])
    attribute = None
    if "attribute" in post_data.keys():
        attribute = str(post_data["attribute"])
    subattribute = str(post_data["subattribute"])
    samples = 7
    if "samples" in post_data.keys():
        samples = int(post_data["samples"])
    N_min = -8
    if "minN" in post_data.keys():
        N_min = int(post_data["minN"])
    N_max = 8
    if "maxN" in post_data.keys():
        N_max = int(post_data["maxN"])
    chat_files = os.listdir("chat_history")
    chat_files = [filename for filename in chat_files if filename.endswith(".json")]

    responded = False
    parsed = False
    if user_id is None or not (user_id + ".json" in chat_files):
        messages = [{"role": "user", "content": post_data["msg"]}]
        chat_files_num = [int(filename[:-5]) for filename in chat_files]
        if len(chat_files_num) > 0:
            user_id = str(max(chat_files_num) + 1)
        else:
            user_id = str(1)
        with open(os.path.join("chat_history", user_id + ".json"), "w") as outfile:
            json.dump(messages, outfile)
        outfile.close()
    else:
        # print(os.path.join("chat_history", user_id + ".json"))
        if os.path.getsize(os.path.join("chat_history", user_id + ".json")):
            with open(os.path.join("chat_history", user_id + ".json"), "r") as infile:
                messages = json.load(infile)
                for msg in messages:
                    msg["content"] = msg["content"].replace("\\n", "\n")
            infile.close()
            messages = messages + [{"role": "user", "content": post_data["msg"]}]
        else:
            messages = [{"role": "user", "content": post_data["msg"]}]
    # try:
    responses = chat_with_llama_with_intervention(messages, model, 
                                                  tokenizer, classifiers=intervene_classifier_dict,
                                                  attribute=attribute,
                                                  subattribute=subattribute,
                                                  N_min=N_min,
                                                  N_max=N_max,
                                                  samples=samples,
                                                  residual_stream=True, 
                                                  include_inst=True, 
                                                  attribute_targets=attribute_targets,
                                                  return_user_msg_last_act=True, 
                                                  context_length=2048,
                                                  model_type=model_type)
    responded = True
    ai_response = [extract_chatbot_response(response)["content"] for response in responses]
    parsed = True
    status = "success"

    response = jsonify({subattribute: ai_response,
                        'id': user_id,
                        'verbose_status': status})
    
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,PATCH,OPTIONS')
    return response, 201


@app.route("/post_chat_attribution", methods=['POST'])
def _post_chat_attribution():
    global classifier_dict, model, tokenizer, mistral_model, mistral_tokenizer, cate_labels
    post_data = json.loads(request.data.decode())
    user_id = str(post_data["id"])
    subattribute = str(post_data["subattribute"])
    
    model_type = ""
    if ("model" in post_data.keys()):
        model_type = post_data["model"].lower()
    
    # Find attribute using helper function (handles both default and extra probes)
    attribute = find_attribute_for_subattribute(subattribute, cate_labels)
    
    if attribute is None:
        status = "failed"
        error_msg = f"Could not find attribute for subattribute: {subattribute}"
        response = jsonify({'msg': error_msg,
                            'id': user_id,
                            'verbose_status': status})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,PATCH,OPTIONS')
        return response, 400
    
    chat_files = os.listdir("chat_history")
    chat_files = [filename for filename in chat_files if filename.endswith(".json")]

    responded = False
    parsed = False
    if user_id is None or not (user_id + ".json" in chat_files):
        status = "failed"
        error_msg = "User id does not exist."
        response = jsonify({'msg': error_msg,
                            'id': user_id,
                            'verbose_status': status})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,PATCH,OPTIONS')
        return response, status_code
    else:
        if os.path.getsize(os.path.join("chat_history", user_id + ".json")):
            with open(os.path.join("chat_history", user_id + ".json"), "r") as infile:
                messages = json.load(infile)
                for msg in messages:
                    msg["content"] = msg["content"].replace("\\n", "\n")
            infile.close()
            messages = messages[:-1]
        else:
            status = "failed"
            error_msg = "No chat history"
            response = jsonify({'msg': error_msg,
                                'id': user_id,
                                'verbose_status': status})
            response.headers.add('Access-Control-Allow-Origin', '*')
            response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,PATCH,OPTIONS')
            return response, status_code
        
    if model_type == "mistral":
        used_model = mistral_model
        used_tokenizer = mistral_tokenizer
        used_classifier_dict = mistral_classifier_dict
        from_idx = 25
        to_idx = 32
    elif model_type == "llama3":
        used_model = llama3_model
        used_tokenizer = llama3_tokenizer
        used_classifier_dict = llama3_attribution_classifier_dict
        from_idx = 20
        to_idx = 30
    elif model_type == "gemma2":
        used_model = gemma2_model
        used_tokenizer = gemma2_tokenizer
        used_classifier_dict = gemma2_attribution_classifier_dict
        from_idx = 20
        to_idx = 30
    else:
        used_model = model
        used_tokenizer = tokenizer
        used_classifier_dict = attribution_classifier_dict
        from_idx = 25
        to_idx = 35
    
    # Merge extra read probes for attribution (attribution uses read probes, not control)
    used_classifier_dict = merge_extra_probes(used_classifier_dict, model_type, probe_type='read')
    print(f"Merged read probes for attribution ({model_type}), total attributes: {list(used_classifier_dict.keys())}")
    
    # Verify classifier exists for this attribute
    if attribute not in used_classifier_dict:
        status = "failed"
        error_msg = f"Classifier for attribute '{attribute}' not found. Available: {list(used_classifier_dict.keys())}"
        response = jsonify({'msg': error_msg,
                            'id': user_id,
                            'verbose_status': status})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,PATCH,OPTIONS')
        return response, 500
    
    # Get special prompt, use attribute name as fallback for extra probes
    special_prompt = attribute_to_prompt.get(attribute, attribute)
    
    if "msg_ids" in post_data.keys() and (post_data["msg_ids"] is not None) and len(post_data["msg_ids"]) >= 1:
        msg_ids = post_data["msg_ids"]
        
        responses = attribution_multi_msg(messages, 
                                          used_model, 
                                          used_tokenizer, 
                                          classifier_dict=used_classifier_dict,
                                          attribute=attribute,
                                          subattribute=subattribute,
                                          special_prompt=special_prompt,
                                          from_idx=from_idx,
                                          to_idx=to_idx,
                                          msg_ids=msg_ids,
                                          mistral=model_type == "mistral",
                                          attribute_targets=attribute_targets)
    else:
        responses = attribution(messages, 
                                used_model, 
                                used_tokenizer, 
                                classifier_dict=used_classifier_dict,
                                attribute=attribute,
                                subattribute=subattribute,
                                special_prompt=special_prompt,
                                from_idx=from_idx,
                                to_idx=to_idx,
                                mistral=model_type == "mistral",
                                attribute_targets=attribute_targets)
        msg_ids = None
        
    responded = True
    parsed = True
    status = "success"
    response = jsonify({"message": responses,
                        "msg_ids": msg_ids,
                        'id': user_id,
                        'verbose_status': status})
    
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,PATCH,OPTIONS')
    return response, 201


@app.route("/post_chat_response_attribution", methods=['POST'])
def _post_chat_response_attribution():
    global classifier_dict, model, tokenizer, mistral_model, mistral_tokenizer, cate_labels
    post_data = json.loads(request.data.decode())
    user_id = str(post_data["id"])
    subattribute = str(post_data["subattribute"])
    
    model_type = ""
    if ("model" in post_data.keys()):
        model_type = post_data["model"].lower()
        
    if model_type == "mistral":
        used_model = mistral_model
        used_tokenizer = mistral_tokenizer
        used_intervene_classifier_dict = mistral_intervene_classifier_dict
        from_idx = 25
        to_idx = 32
    elif model_type == "llama3":
        used_model = llama3_model
        used_tokenizer = llama3_tokenizer
        used_intervene_classifier_dict = llama3_intervene_classifier_dict
        from_idx = 20
        to_idx = 30
    elif model_type == "gemma2":
        used_model = gemma2_model
        used_tokenizer = gemma2_tokenizer
        used_intervene_classifier_dict = gemma2_intervene_classifier_dict
        from_idx = 20
        to_idx = 30
    else:
        used_model = model
        used_tokenizer = tokenizer
        used_intervene_classifier_dict = intervene_classifier_dict
        from_idx = 25
        to_idx = 35
    
    # Merge extra control probes for sensitivity analysis (uses control/intervene probes)
    used_intervene_classifier_dict = merge_extra_probes(used_intervene_classifier_dict, model_type, probe_type='control')
    print(f"Merged control probes for response attribution ({model_type}), total attributes: {list(used_intervene_classifier_dict.keys())}")
    
    # Find attribute using helper function (handles both default and extra probes)
    attribute = find_attribute_for_subattribute(subattribute, cate_labels)
    
    if attribute is None:
        status = "failed"
        error_msg = f"Could not find attribute for subattribute: {subattribute}"
        response = jsonify({'msg': error_msg,
                            'id': user_id,
                            'verbose_status': status})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,PATCH,OPTIONS')
        return response, 400
    
    chat_files = os.listdir("chat_history")
    chat_files = [filename for filename in chat_files if filename.endswith(".json")]

    responded = False
    parsed = False
    if user_id is None or not (user_id + ".json" in chat_files):
        status_code = 404
        status = "failed"
        error_msg = "User id does not exist."
        response = jsonify({'msg': error_msg,
                            'id': user_id,
                            'verbose_status': status})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,PATCH,OPTIONS')
        return response, status_code
    else:
        if os.path.getsize(os.path.join("chat_history", user_id + ".json")):
            with open(os.path.join("chat_history", user_id + ".json"), "r") as infile:
                messages = json.load(infile)
                for msg in messages:
                    msg["content"] = msg["content"].replace("\\n", "\n")
            infile.close()
        else:
            status = "failed"
            error_msg = "No chat history"
            status_code = 404
            response = jsonify({'msg': error_msg,
                                'id': user_id,
                                'verbose_status': status})
            response.headers.add('Access-Control-Allow-Origin', '*')
            response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,PATCH,OPTIONS')
            return response, status_code
    # try:
    print(subattribute)
    if subattribute == "Uncertainty":
        chat_files = os.listdir("chat_history")
        token_files = [filename for filename in chat_files if filename.endswith(".pkl")]
        if user_id is None or not ("last_token_act_" + user_id + ".pkl" in token_files):
            response = jsonify({'data': "Chat history file not found",
                                'verbose_status': "failed"})
            response.headers.add('Access-Control-Allow-Origin', '*')
            response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,PATCH,OPTIONS')
            return response, 404
        else:
            if os.path.getsize(os.path.join("chat_history", "last_token_act_" + user_id + ".pkl")):
                with open(os.path.join("chat_history", "last_token_act_" + user_id + ".pkl"), "rb") as infile:
                    last_token_act = pickle.load(infile)
                infile.close()
            else:
                response = jsonify({'data': "Corrupted last token activation",
                                    'verbose_status': "failed"})
                response.headers.add('Access-Control-Allow-Origin', '*')
                response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,PATCH,OPTIONS')
                return response, 500
        responses = output_uncertainty(messages, 
                                       used_tokenizer,
                                       last_token_act["uncertainty"].cpu().detach().tolist())
    else:
        # Verify classifier exists for this attribute
        if attribute not in used_intervene_classifier_dict:
            status = "failed"
            error_msg = f"Classifier for attribute '{attribute}' not found. Available: {list(used_intervene_classifier_dict.keys())}"
            response = jsonify({'msg': error_msg,
                                'id': user_id,
                                'verbose_status': status})
            response.headers.add('Access-Control-Allow-Origin', '*')
            response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,PATCH,OPTIONS')
            return response, 500
        
        # Get special prompt, use attribute name as fallback for extra probes
        special_prompt = attribute_to_prompt.get(attribute, attribute)
        
        responses = output_sensitivity(messages, 
                                       used_model, 
                                       used_tokenizer, 
                                       classifiers=used_intervene_classifier_dict,
                                       attribute=attribute,
                                       subattribute=subattribute,
                                       special_prompt=special_prompt,
                                       from_idx=from_idx,
                                       to_idx=to_idx,
                                       attribute_targets=attribute_targets)
    responded = True
    parsed = True
    status = "success"
    response = jsonify({"message": responses,
                        'id': user_id,
                        'verbose_status': status})
    
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,PATCH,OPTIONS')
    return response, 201


def load_extra_probes_for_model(model_type):
    """
    Load extra probes from the extra directories for a specific model.
    Handles both newly added probes and deleted probes:
    - Loads new probes that haven't been loaded before
    - Detects and removes deleted probes from cache
    Also loads metadata from JSON stats files and extends cate_labels, attribute_to_prompt, and translate_keys.
    
    Returns: Dict with loaded probe info including metadata
    """
    global extra_probes_cache, cate_labels, attribute_to_prompt, translate_keys
    
    if model_type not in EXTRA_PROBE_DIRS:
        return {"error": f"Unknown model type: {model_type}"}
    
    result = {
        "model": model_type,
        "newly_loaded": [],
        "deleted": [],
        "available_probes": [],
        "probe_metadata": {}
    }
    
    # Get the model config for hidden neurons
    hidden_neurons = 4096 if model_type == "llama3" else 3584 if model_type == "gemma2" else 4096
    
    # Track all files that currently exist on disk
    # Store as "probe_type/attribute_name/filename" to track files in subdirectories
    existing_files = set()
    
    for probe_type in ["control", "read"]:
        probe_dir = EXTRA_PROBE_DIRS[model_type][probe_type]
        stats_dir = probe_dir.replace("_probes", "_probes_stats")
        
        if not os.path.exists(probe_dir):
            print(f"Extra probe directory not found: {probe_dir}")
            continue
        
        # Look for subdirectories (each represents an attribute)
        try:
            subdirs = [d for d in os.listdir(probe_dir) if os.path.isdir(os.path.join(probe_dir, d))]
        except Exception as e:
            print(f"Error listing subdirectories in {probe_dir}: {e}")
            subdirs = []
        
        for attribute_subdir in subdirs:
            attribute_probe_dir = os.path.join(probe_dir, attribute_subdir)
            
            # Get all .pth files in this attribute subdirectory
            try:
                checkpoint_files = [f for f in os.listdir(attribute_probe_dir) if f.endswith('.pth')]
            except Exception as e:
                print(f"Error listing files in {attribute_probe_dir}: {e}")
                continue
            
            # Add files with probe_type and attribute subdirectory to distinguish files
            existing_files.update([f"{probe_type}/{attribute_subdir}/{f}" for f in checkpoint_files])
            
            for checkpoint_file in checkpoint_files:
                # Check if this file has already been loaded (with full path)
                file_key = f"{probe_type}/{attribute_subdir}/{checkpoint_file}"
                if file_key in extra_probes_cache[model_type]["loaded_files"]:
                    continue
                
                # Parse the filename to extract base name (should be meta_attribute)
                # Expected format: meta_attribute_at_layer_N.pth
                try:
                    filename_base = checkpoint_file[:checkpoint_file.find("_at_layer_")]
                    layer_num = int(checkpoint_file[checkpoint_file.rfind("_") + 1:checkpoint_file.rfind(".pth")])
                    
                    weight_path = os.path.join(attribute_probe_dir, checkpoint_file)
                    
                    # Parse folder name to extract target: {meta_attribute}_{target}
                    # e.g., "ConfidenceANDUncertain_user" or "positive_negative_chatbot"
                    # Extract meta_attribute and target from folder name
                    folder_target = "user"  # Default
                    folder_meta_attribute = attribute_subdir  # Default to full folder name
                    
                    if "_" in attribute_subdir:
                        # Find the last underscore to split meta_attribute from target
                        last_underscore_idx = attribute_subdir.rfind("_")
                        potential_target = attribute_subdir[last_underscore_idx + 1:]
                        if potential_target in ["user", "chatbot"]:
                            folder_target = potential_target
                            folder_meta_attribute = attribute_subdir[:last_underscore_idx]
                    
                    # Load metadata from JSON stats file in attribute subdirectory
                    # Filename should use meta_attribute (same as filename_base)
                    attribute_stats_dir = os.path.join(stats_dir, attribute_subdir)
                    stats_file = os.path.join(attribute_stats_dir, f"{filename_base}_at_layer.json")
                    metadata = None
                    attribute1 = filename_base  # Fallback to filename
                    attribute2 = f"non-{filename_base}"
                    # Use folder meta_attribute as default, or filename_base if folder parsing failed
                    meta_attribute = folder_meta_attribute if folder_meta_attribute != attribute_subdir else filename_base
                    target = folder_target  # Use folder-based target as default
                    
                    if os.path.exists(stats_file):
                        try:
                            with open(stats_file, 'r') as f:
                                metadata = json.load(f)
                                attribute1 = metadata.get("attribute1", filename_base)
                                attribute2 = metadata.get("attribute2", f"non-{filename_base}")
                                meta_attribute = metadata.get("meta_attribute", filename_base)
                                target = metadata.get("target", "user")  # Load target from metadata
                                
                                # Store metadata for frontend (keyed by meta_attribute)
                                if meta_attribute not in result["probe_metadata"]:
                                    # Handle icon as array or single value for backward compatibility
                                    icon_value = metadata.get("icon", ["FaQuestion", "FaQuestion"])
                                    if not isinstance(icon_value, list):
                                        # Old format: single icon string, convert to array
                                        icon_value = [icon_value, icon_value]
                                    
                                    result["probe_metadata"][meta_attribute] = {
                                        "attribute1": attribute1,
                                        "attribute2": attribute2,
                                        "meta_attribute": meta_attribute,
                                        "icon": icon_value,
                                        "target": target,
                                        "probe_type": metadata.get("probe_type", probe_type),
                                        "best_layer": metadata.get("best_layer"),
                                        "best_accuracy": metadata.get("best_accuracy"),
                                        "average_accuracy": metadata.get("average_accuracy"),
                                        "filename_attribute": filename_base  # Store for reference
                                    }
                            
                            print(f"Loaded metadata for {filename_base}: {attribute1} vs {attribute2} (meta: {meta_attribute}, target: {target})")
                        except Exception as e:
                            print(f"Failed to load stats for {filename_base}: {e}")
                    
                    # Store target in global attribute_targets dict
                    attribute_targets[meta_attribute] = target
                    print(f"Set target for {meta_attribute}: {target}")
                    
                    # IMPORTANT: Use meta_attribute as the key, NOT attribute_name
                    # Extend cate_labels with the meta_attribute as key
                    if meta_attribute not in cate_labels:
                        cate_labels[meta_attribute] = [attribute1, attribute2]
                        print(f"Extended cate_labels with {meta_attribute}: {[attribute1, attribute2]}")
                    
                    # Extend attribute_to_prompt with meta_attribute as both key and value
                    if meta_attribute not in attribute_to_prompt:
                        attribute_to_prompt[meta_attribute] = meta_attribute
                        print(f"Extended attribute_to_prompt with {meta_attribute}: {meta_attribute}")
                    
                    # Extend translate_keys with 1:1 mapping (meta_attribute -> meta_attribute)
                    if meta_attribute not in translate_keys:
                        translate_keys[meta_attribute] = meta_attribute
                        print(f"Extended translate_keys with {meta_attribute}: {meta_attribute}")
                    
                    # Extend special_prompts for activation querying
                    if meta_attribute not in special_prompts:
                        special_prompts.append(meta_attribute)
                        print(f"Extended special_prompts with {meta_attribute}")
                    
                    # Add default intervention range for extra probes
                    if meta_attribute not in default_NminNmax:
                        default_NminNmax[meta_attribute] = [-1, 1]  # Default range for extra probes
                        print(f"Extended default_NminNmax with {meta_attribute}: [-1, 1]")
                    
                    # Add default scale factor for extra probes based on model type
                    if meta_attribute not in extra_probe_scale_factors:
                        if model_type == "llama3":
                            default_scale = 0.15
                        elif model_type == "gemma2":
                            default_scale = 1.0
                        elif model_type == "mistral":
                            default_scale = 0.60  # Use mistral's typical scale
                        else:
                            default_scale = 0.80  # Fallback for other models
                        extra_probe_scale_factors[meta_attribute] = default_scale
                        print(f"Extended extra_probe_scale_factors with {meta_attribute}: {default_scale}")
                    
                    # Determine number of classes (binary for custom probes)
                    num_class = 2
                    
                    # Load the probe using meta_attribute as the key
                    if meta_attribute not in extra_probes_cache[model_type][probe_type]:
                        extra_probes_cache[model_type][probe_type][meta_attribute] = {}
                    
                    extra_probes_cache[model_type][probe_type][meta_attribute][layer_num] = load_probe_classifier(
                        LinearProbeClassification,
                        hidden_neurons,
                        num_classes=num_class,
                        weight_path=weight_path,
                        logistic=True
                    )
                    
                    # Mark as loaded (with full path including subdirectory)
                    extra_probes_cache[model_type]["loaded_files"].add(file_key)
                    result["newly_loaded"].append({
                        "meta_attribute": meta_attribute,
                        "attribute1": attribute1,
                        "attribute2": attribute2,
                        "layer": layer_num,
                        "type": probe_type,
                        "file": file_key
                    })
                    
                    print(f"Loaded extra probe: {meta_attribute} (layer {layer_num}) for {model_type} {probe_type} [file: {attribute_name}]")
                    
                except Exception as e:
                    print(f"Failed to load probe {checkpoint_file}: {e}")
                    traceback.print_exc()
                    continue
    
    # Check for deleted probes
    deleted_files = extra_probes_cache[model_type]["loaded_files"] - existing_files
    
    if deleted_files:
        print(f"\nDetected {len(deleted_files)} deleted probe file(s) for {model_type}")
        
        # Track which meta_attributes need to be removed
        meta_attributes_to_remove = set()
        
        for deleted_file_key in deleted_files:
            try:
                # Parse the file key: "probe_type/attribute_subdir/filename"
                if "/" in deleted_file_key:
                    parts = deleted_file_key.split("/")
                    if len(parts) == 3:
                        # New format: probe_type/attribute_subdir/filename
                        probe_type, attribute_subdir, deleted_file = parts
                    elif len(parts) == 2:
                        # Old format: probe_type/filename
                        probe_type, deleted_file = parts
                        attribute_subdir = None
                    else:
                        print(f"Unexpected file key format: {deleted_file_key}")
                        continue
                else:
                    # Very old format without any prefix
                    deleted_file = deleted_file_key
                    probe_type = None
                    attribute_subdir = None
                
                # Parse the filename to extract base name (should be meta_attribute)
                filename_base = deleted_file[:deleted_file.find("_at_layer_")]
                layer_num = int(deleted_file[deleted_file.rfind("_") + 1:deleted_file.rfind(".pth")])
                
                # Load metadata to get meta_attribute
                # Parse folder name to extract meta_attribute and target
                folder_target = "user"  # Default target
                folder_meta_attribute = attribute_subdir if attribute_subdir else filename_base  # Default
                
                # Parse folder name to extract target if available
                if attribute_subdir and "_" in attribute_subdir:
                    last_underscore_idx = attribute_subdir.rfind("_")
                    potential_target = attribute_subdir[last_underscore_idx + 1:]
                    if potential_target in ["user", "chatbot"]:
                        folder_target = potential_target
                        folder_meta_attribute = attribute_subdir[:last_underscore_idx]
                
                # Use folder meta_attribute as initial default
                meta_attribute = folder_meta_attribute
                
                if probe_type:
                    probe_dir = EXTRA_PROBE_DIRS[model_type][probe_type]
                    stats_dir = probe_dir.replace("_probes", "_probes_stats")
                    
                    # If we have attribute_subdir, use it to find stats file
                    # Filename should use meta_attribute (same as filename_base)
                    if attribute_subdir:
                        stats_file = os.path.join(stats_dir, attribute_subdir, f"{filename_base}_at_layer.json")
                    else:
                        # Old format: stats at root level
                        stats_file = os.path.join(stats_dir, f"{filename_base}_at_layer.json")
                    
                    if os.path.exists(stats_file):
                        try:
                            with open(stats_file, 'r') as f:
                                metadata = json.load(f)
                                meta_attribute = metadata.get("meta_attribute", attribute_name)
                        except Exception as e:
                            print(f"Failed to load metadata for deleted file {attribute_name}: {e}")
                
                # If probe_type is known from the file key, use it directly
                if probe_type and probe_type in ["control", "read"]:
                    if meta_attribute in extra_probes_cache[model_type][probe_type]:
                        if layer_num in extra_probes_cache[model_type][probe_type][meta_attribute]:
                            # Remove this specific layer
                            del extra_probes_cache[model_type][probe_type][meta_attribute][layer_num]
                            print(f"Removed {probe_type} probe: {meta_attribute} (layer {layer_num})")
                            
                            # If no layers remain for this meta_attribute in this probe type, remove it
                            if not extra_probes_cache[model_type][probe_type][meta_attribute]:
                                del extra_probes_cache[model_type][probe_type][meta_attribute]
                                print(f"Removed {meta_attribute} from {probe_type} cache (no layers remaining)")
                else:
                    # Fallback: search both probe types (for old format compatibility)
                    for probe_type in ["control", "read"]:
                        if meta_attribute in extra_probes_cache[model_type][probe_type]:
                            if layer_num in extra_probes_cache[model_type][probe_type][meta_attribute]:
                                # Remove this specific layer
                                del extra_probes_cache[model_type][probe_type][meta_attribute][layer_num]
                                print(f"Removed {probe_type} probe: {meta_attribute} (layer {layer_num})")
                                
                                # If no layers remain for this meta_attribute in this probe type, remove it
                                if not extra_probes_cache[model_type][probe_type][meta_attribute]:
                                    del extra_probes_cache[model_type][probe_type][meta_attribute]
                                    print(f"Removed {meta_attribute} from {probe_type} cache (no layers remaining)")
                
                # Check if this meta_attribute is completely gone from both control and read
                meta_attribute_still_exists = (
                    meta_attribute in extra_probes_cache[model_type]["control"] or
                    meta_attribute in extra_probes_cache[model_type]["read"]
                )
                
                if not meta_attribute_still_exists:
                    meta_attributes_to_remove.add(meta_attribute)
                
                # Remove from loaded_files
                extra_probes_cache[model_type]["loaded_files"].discard(deleted_file_key)
                
                result["deleted"].append({
                    "file": deleted_file_key,
                    "meta_attribute": meta_attribute,
                    "layer": layer_num
                })
                
            except Exception as e:
                print(f"Error processing deleted file {deleted_file_key}: {e}")
                # Still remove it from loaded_files even if parsing fails
                extra_probes_cache[model_type]["loaded_files"].discard(deleted_file_key)
        
        # Remove meta_attributes from global dictionaries if completely gone
        for meta_attribute in meta_attributes_to_remove:
            if meta_attribute in cate_labels:
                del cate_labels[meta_attribute]
                print(f"Removed {meta_attribute} from cate_labels")
            
            if meta_attribute in attribute_to_prompt:
                del attribute_to_prompt[meta_attribute]
                print(f"Removed {meta_attribute} from attribute_to_prompt")
            
            if meta_attribute in translate_keys:
                del translate_keys[meta_attribute]
                print(f"Removed {meta_attribute} from translate_keys")
            
            # Remove from special_prompts
            if meta_attribute in special_prompts:
                special_prompts.remove(meta_attribute)
                print(f"Removed {meta_attribute} from special_prompts")
            
            # Remove from default_NminNmax
            if meta_attribute in default_NminNmax:
                del default_NminNmax[meta_attribute]
                print(f"Removed {meta_attribute} from default_NminNmax")
            
            # Remove from attribute_targets
            if meta_attribute in attribute_targets:
                del attribute_targets[meta_attribute]
                print(f"Removed {meta_attribute} from attribute_targets")
            
            # Remove from extra_probe_scale_factors
            if meta_attribute in extra_probe_scale_factors:
                del extra_probe_scale_factors[meta_attribute]
                print(f"Removed {meta_attribute} from extra_probe_scale_factors")
        
        print(f"Cleanup complete: {len(meta_attributes_to_remove)} meta_attribute(s) fully removed\n")
    
    # Get list of all available probe attributes
    available_attrs = set()
    for probe_type in ["control", "read"]:
        available_attrs.update(extra_probes_cache[model_type][probe_type].keys())
    
    result["available_probes"] = list(available_attrs)
    
    # IMPORTANT: Ensure probe_metadata is populated for ALL available probes, not just newly loaded ones
    # This fixes the issue where already-cached probes don't have metadata returned to the frontend
    for meta_attribute in available_attrs:
        if meta_attribute not in result["probe_metadata"]:
            # Probe was already cached, need to reconstruct metadata
            # Try to load from stats file for the most accurate metadata
            stats_loaded = False
            
            for probe_type in ["control", "read"]:
                if meta_attribute in extra_probes_cache[model_type][probe_type]:
                    # Try to find the stats file
                    probe_dir = EXTRA_PROBE_DIRS[model_type][probe_type]
                    stats_dir = probe_dir.replace("_probes", "_probes_stats")
                    
                    # Look for any checkpoint file matching this meta_attribute in subdirectories
                    if os.path.exists(probe_dir):
                        # Look through subdirectories
                        try:
                            subdirs = [d for d in os.listdir(probe_dir) if os.path.isdir(os.path.join(probe_dir, d))]
                        except Exception as e:
                            print(f"Error listing subdirectories for metadata reconstruction: {e}")
                            subdirs = []
                        
                        for attribute_subdir in subdirs:
                            attribute_probe_dir = os.path.join(probe_dir, attribute_subdir)
                            attribute_stats_dir = os.path.join(stats_dir, attribute_subdir)
                            
                            # Parse folder name to extract meta_attribute and target
                            # Format: {meta_attribute}_{target}
                            folder_target = "user"  # Default
                            folder_meta_attribute = attribute_subdir  # Default to full folder name
                            
                            if "_" in attribute_subdir:
                                last_underscore_idx = attribute_subdir.rfind("_")
                                potential_target = attribute_subdir[last_underscore_idx + 1:]
                                if potential_target in ["user", "chatbot"]:
                                    folder_target = potential_target
                                    folder_meta_attribute = attribute_subdir[:last_underscore_idx]
                            
                            try:
                                checkpoint_files = [f for f in os.listdir(attribute_probe_dir) if f.endswith('.pth')]
                            except Exception as e:
                                print(f"Error listing files in {attribute_probe_dir}: {e}")
                                continue
                            
                            # Check if this folder's meta_attribute matches what we're looking for
                            # Since folder is named {meta_attribute}_{target}, folder_meta_attribute should match
                            if folder_meta_attribute == meta_attribute:
                                # Found the right folder, load metadata from any checkpoint file
                                try:
                                    checkpoint_files_list = list(checkpoint_files)
                                    if checkpoint_files_list:
                                        # Use first checkpoint file to get filename base (should be meta_attribute)
                                        first_file = checkpoint_files_list[0]
                                        filename_base = first_file[:first_file.find("_at_layer_")]
                                        stats_file = os.path.join(attribute_stats_dir, f"{filename_base}_at_layer.json")
                                        
                                        if os.path.exists(stats_file):
                                            with open(stats_file, 'r') as f:
                                                metadata = json.load(f)
                                                # Verify meta_attribute matches (should always be true given folder name)
                                                found_meta_attribute = metadata.get("meta_attribute", folder_meta_attribute)
                                                attribute1 = metadata.get("attribute1", filename_base)
                                                attribute2 = metadata.get("attribute2", f"non-{filename_base}")
                                                # Use folder target if metadata doesn't specify, otherwise use metadata target
                                                target_value = metadata.get("target", folder_target)
                                                
                                                # Handle icon as array or single value
                                                icon_value = metadata.get("icon", ["FaQuestion", "FaQuestion"])
                                                if not isinstance(icon_value, list):
                                                    icon_value = [icon_value, icon_value]
                                                
                                                result["probe_metadata"][meta_attribute] = {
                                                    "attribute1": attribute1,
                                                    "attribute2": attribute2,
                                                    "meta_attribute": meta_attribute,
                                                    "icon": icon_value,
                                                    "target": target_value,
                                                    "probe_type": metadata.get("probe_type", probe_type),
                                                    "best_layer": metadata.get("best_layer"),
                                                    "best_accuracy": metadata.get("best_accuracy"),
                                                    "average_accuracy": metadata.get("average_accuracy"),
                                                    "filename_attribute": filename_base
                                                }
                                                # Update global attribute_targets with the loaded target
                                                attribute_targets[meta_attribute] = target_value
                                                stats_loaded = True
                                except Exception as e:
                                    print(f"Failed to load metadata: {e}")
                                
                                # Break out of subdirectory loop if we found and loaded metadata
                                if stats_loaded:
                                    break
                            
                            if stats_loaded:
                                break
                        
                        if stats_loaded:
                            break
            
            # Fallback: if no stats file found, use cate_labels as source
            if not stats_loaded and meta_attribute in cate_labels:
                labels = cate_labels[meta_attribute]
                result["probe_metadata"][meta_attribute] = {
                    "attribute1": labels[0] if len(labels) > 0 else meta_attribute,
                    "attribute2": labels[1] if len(labels) > 1 else f"non-{meta_attribute}",
                    "meta_attribute": meta_attribute,
                    "icon": ["FaQuestion", "FaQuestion"],
                    "probe_type": "unknown",
                    "filename_attribute": meta_attribute
                }
                print(f"Used cate_labels fallback for metadata of {meta_attribute}")
    
    return result


@app.route('/available_extra_probes', methods=['GET'])
def get_available_extra_probes():
    """
    Endpoint to list and load extra probes for all models.
    Returns information about available extra probes including metadata.
    Also detects and removes deleted probes from cache.
    """
    try:
        model_type = request.args.get('model', 'llama3')
        
        # Load any new probes and detect deleted ones
        load_result = load_extra_probes_for_model(model_type)
        
        response = jsonify({
            'status': 'success',
            'model': model_type,
            'available_probes': load_result.get('available_probes', []),
            'probe_metadata': load_result.get('probe_metadata', {}),
            'newly_loaded': load_result.get('newly_loaded', []),
            'deleted': load_result.get('deleted', []),
            'cache_info': {
                'total_loaded_files': len(extra_probes_cache[model_type]['loaded_files']),
                'control_probes': list(extra_probes_cache[model_type]['control'].keys()),
                'read_probes': list(extra_probes_cache[model_type]['read'].keys())
            }
        })
        
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Methods', 'GET,OPTIONS')
        return response, 200
        
    except Exception as e:
        error_trace = traceback.format_exc()
        print(f"Error in /available_extra_probes: {e}")
        print(error_trace)
        
        response = jsonify({
            'status': 'error',
            'message': str(e),
            'traceback': error_trace
        })
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response, 500


if __name__ == '__main__':
    pass
