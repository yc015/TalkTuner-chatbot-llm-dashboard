import copy
import torch
from collections import OrderedDict
from baukit import TraceDict
from torch import nn
from app.chat.interv_utils import optimize_one_inter_rep
import numpy as np


from IPython.core.display import display, HTML

from baukit import TraceDict
from torch import nn
from app.chat.utils import llama_v2_prompt, mistral_v2_prompt
import matplotlib
import re

torch_device = "cuda"

cmap = matplotlib.cm.get_cmap('PiYG')


def interpolate_color(color1, color2, factor):
    """Interpolates between two colors by a given factor and returns 'rgb(r, g, b)'."""
    r1, g1, b1 = int(color1[1:3], 16), int(color1[3:5], 16), int(color1[5:7], 16)
    r2, g2, b2 = int(color2[1:3], 16), int(color2[3:5], 16), int(color2[5:7], 16)
    r = int(r1 + (r2 - r1) * factor)
    g = int(g1 + (g2 - g1) * factor)
    b = int(b1 + (b2 - b1) * factor)
    return f'rgb({r}, {g}, {b})'


def get_color(number):
    """Returns a color based on the number within the range [-1, 1] using a diverging colormap in 'rgb(r, g, b)' format."""
    start_color = '#900252'
    middle_color = '#F9F9F9'
    stop_color = '#296719'
    
    if number < -1:
        return interpolate_color(start_color, start_color, 0)  # Ensure the input is within bounds
    elif number > 1:
        return interpolate_color(stop_color, stop_color, 0)  # Ensure the input is within bounds
    elif number < 0:
        # Map number from [-1, 0] to [0, 1] for interpolation between start and middle
        return interpolate_color(start_color, middle_color, (number + 1))
    else:
        # Map number from [0, 1] to [0, 1] for interpolation between middle and stop
        return interpolate_color(middle_color, stop_color, number)
    


def replace_words_with_unk(input_string):
    # words = input_string.split()
    words = re.findall(r"\S+|\n", input_string)
    replaced_strings = [' '.join(words[:i] + ["<unk>"] + words[i+1:]) for i in range(len(words))]
    return replaced_strings, words


def attribution(chat_history, model, tokenizer, classifier_dict, subattribute, attribute, special_prompt, from_idx=25, to_idx=35, mistral=False, attribute_targets=None):
    copychat_history = copy.deepcopy(chat_history)
    category_labels = {"gender": ["Male", "Female", "Other"],
                       "age": ["Child", "Adolescent", "Adult", "Older Adult", "Unknown"],
                       "education": ["Some Education", "High School", "College & More", "Unknown"],
                       "ethnics": ["Asian", "African", "White", 
                               "Hispanic", "Native Americans", "Arabs", "Jewish", "Unknown"],
                       "socioeco": ["Lower", "Middle", "Upper", "Unknown"],
                       "marital": ["Single", "Married", "Divorced", "Widowed"],
                       "language": ["Chinese", "Japanese", "English", "German", "Spanish", "Portuguese", "Arabic", "Russian"],
                       "religion": ["Christian", "Islam", "Buddhism", "Hinduism", "Judaism", "Atheism", "Unknown"],
                       "political": ["Left", "Right", "Moderate", "Unknown"],
                       "sycophancy": ["Sycophancy", "nonSycophant"],
                       "hallucination": ["Hallucinated", "Factual"],
                       }
    subattribute = subattribute.title()
    
    if attribute is None:
        for attr in list(category_labels.keys()):
            if subattribute in category_labels[attr]:
                attribute = attr
                break
    if not (attribute in category_labels.keys()):
        return ["Attribute does not exist"]
    elif not (subattribute in category_labels[attribute]):
        return ["Subattribute does not exist"]
    
    if copychat_history[-1]["role"] == "assistant":
        copychat_history = copychat_history[:-1]
    
    user_msg = copychat_history[-1]["content"]
    copychat_history = copychat_history[:-1]
    strings, words = replace_words_with_unk(user_msg)
    copychat_history.append({"role": "user", 
                              "content": user_msg},
                            )
    
    feature_idx = category_labels[attribute].index(subattribute)
    
    if mistral:
        prompt = mistral_v2_prompt(copychat_history)
    else:
        prompt = llama_v2_prompt(copychat_history)
    
    # Get the appropriate reading prompt based on target (user or chatbot)
    from app.chat.utils import get_reading_prompt_for_attribute
    reading_prompt_suffix = get_reading_prompt_for_attribute(special_prompt, attribute_targets)
    prompt += f"\n\nBased on the context of this conversation, I think the {special_prompt} {reading_prompt_suffix}"
    
    torch_device = "cuda"

    with torch.no_grad():
        inputs = tokenizer(prompt, return_tensors='pt',return_token_type_ids=False).to('cuda')
        tokens = model(**inputs, output_hidden_states=True)

    preds = []
    for layer_num in range(from_idx, to_idx):
        logits = classifier_dict[attribute][layer_num](tokens.hidden_states[layer_num][:, -1].to(torch.float))[0]
        preds.append(logits.cpu().detach().numpy())

    preds = np.array(preds)
    final_preds = np.mean(preds.reshape(-1, preds.shape[-1]), axis=0)[feature_idx]
    pred_diffs = []
    for msg in strings:
        copychat_history = copy.deepcopy(chat_history)
        copychat_history = copychat_history[:-1]
        copychat_history.append({"role": "user", 
                                  "content": msg},
                                )
        if mistral:
            prompt = mistral_v2_prompt(copychat_history)
        else:
            prompt = llama_v2_prompt(copychat_history)
        
        # Get the appropriate reading prompt based on target (user or chatbot)
        from app.chat.utils import get_reading_prompt_for_attribute
        reading_prompt_suffix = get_reading_prompt_for_attribute(special_prompt, attribute_targets)
        prompt +=  f"\n\nBased on the context of this conversation, I think the {special_prompt} {reading_prompt_suffix}"
        torch_device = "cuda"
        with torch.no_grad():
            inputs = tokenizer(prompt, return_tensors='pt',return_token_type_ids=False).to('cuda')
            tokens = model(**inputs, output_hidden_states=True)

        preds = []
        for layer_num in range(from_idx, to_idx):
            logits = classifier_dict[attribute][layer_num](tokens.hidden_states[layer_num][:, -1].to(torch.float))[0]
            preds.append(logits.cpu().detach().numpy())
        preds = np.array(preds)
        pred_diffs.append(final_preds - np.mean(preds.reshape(-1, preds.shape[-1]), axis=0)[feature_idx])

    values = pred_diffs
    values = np.array(values)
    # print(values)
    min_val = np.percentile(abs(values), 50)

    values[abs(values) < min_val] = 0
    # values /= 0.5
    values /= 1
    values = np.clip(values, -1, 1)
    # normalized_values = [val * 229 for val in values]
    normalized_values = [val * 1 for val in values]

    # Create a color mapping from values to background colors
    # colors = [f"rgb(249, {249 - int(val)}, {249 - int(val)})" if val > 0 else f"rgb({249 + int(val)}, {249 + int(val)}, 249)" for val in normalized_values]
    colors = [get_color(val) for val in normalized_values]

    # Generate the HTML string with reduced space for word pieces
    html_pieces = []
    words = [word.replace("\n", "<br>") for word in words]
    for i, (word, color) in enumerate(zip(words, colors)):
        if word != "<br>":
            span = f"<span id='attribution' style='background-color:{color}; padding: 0px 0px;'>{word.strip()}</span>"
        else:
            span = word
        html_pieces.append(span)

    html_str = " ".join(html_pieces)

    return html_str


def attribution_multi_msg(chat_history, model, tokenizer, classifier_dict, subattribute, attribute, special_prompt, from_idx=25, to_idx=35, msg_ids=[], mistral=False, attribute_targets=None):
    copychat_history = copy.deepcopy(chat_history)
    category_labels = {"gender": ["Male", "Female", "Other"],
                       "age": ["Child", "Adolescent", "Adult", "Older Adult", "Unknown"],
                       "education": ["Some Education", "High School", "College & More", "Unknown"],
                       "ethnics": ["Asian", "African", "White", 
                               "Hispanic", "Native Americans", "Arabs", "Jewish", "Unknown"],
                       "socioeco": ["Lower", "Middle", "Upper", "Unknown"],
                       "marital": ["Single", "Married", "Divorced", "Widowed"],
                       "language": ["Chinese", "Japanese", "English", "German", "Spanish", "Portuguese", "Arabic", "Russian"],
                       "religion": ["Christian", "Islam", "Buddhism", "Hinduism", "Judaism", "Atheism", "Unknown"],
                       "political": ["Left", "Right", "Moderate", "Unknown"],
                       "sycophancy": ["Sycophancy", "nonSycophant"],
                       "hallucination": ["Hallucinated", "Factual"],
                      }
    subattribute = subattribute.title()
    
    if attribute is None:
        for attr in list(category_labels.keys()):
            if subattribute in category_labels[attr]:
                attribute = attr
                break
    if not (attribute in category_labels.keys()):
        return ["Attribute does not exist"]
    elif not (subattribute in category_labels[attribute]):
        return ["Subattribute does not exist"]
    
    if copychat_history[-1]["role"] == "assistant":
        copychat_history = copychat_history[:-1]
    
    html_strs = []
    # msg_ids.sort(reverse=True)
    for msg_id in msg_ids:
        copychat_history = copy.deepcopy(chat_history)
        user_msg = copychat_history[msg_id]["content"]
        # copychat_history = copychat_history[:-1]
        strings, words = replace_words_with_unk(user_msg)

        feature_idx = category_labels[attribute].index(subattribute)
        if mistral:
            prompt = mistral_v2_prompt(copychat_history) 
        else:
            prompt = llama_v2_prompt(copychat_history)
        
        # Get the appropriate reading prompt based on target (user or chatbot)
        from app.chat.utils import get_reading_prompt_for_attribute
        reading_prompt_suffix = get_reading_prompt_for_attribute(special_prompt, attribute_targets)
        prompt += f"\n\nBased on the context of this conversation, I think the {special_prompt} {reading_prompt_suffix}"

        with torch.no_grad():
            inputs = tokenizer(prompt, return_tensors='pt',return_token_type_ids=False).to('cuda')
            tokens = model(**inputs, output_hidden_states=True)

        preds = []
        for layer_num in range(from_idx, to_idx):
            logits = classifier_dict[attribute][layer_num](tokens.hidden_states[layer_num][:, -1].to(torch.float))[0]
            preds.append(logits.cpu().detach().numpy())

        preds = np.array(preds)
        final_preds = np.mean(preds.reshape(-1, preds.shape[-1]), axis=0)[feature_idx]
        pred_diffs = []
        for msg in strings:
            copychat_history = copy.deepcopy(chat_history)
            role = "user" if msg_id % 2 == 0 else "assistant"
            copychat_history[msg_id] = {"role": role, "content": msg}
            
            if mistral:
                prompt = mistral_v2_prompt(copychat_history)
            else:
                prompt = llama_v2_prompt(copychat_history)
            
            if prompt.endswith("</s>"):
                prompt = prompt[:prompt.rfind("</s>")]
            
            # Get the appropriate reading prompt based on target (user or chatbot)
            from app.chat.utils import get_reading_prompt_for_attribute
            reading_prompt_suffix = get_reading_prompt_for_attribute(special_prompt, attribute_targets)
            prompt += f"\n\nBased on the context of this conversation, I think the {special_prompt} {reading_prompt_suffix}"
            
            torch_device = "cuda"
            with torch.no_grad():
                inputs = tokenizer(prompt, return_tensors='pt',return_token_type_ids=False).to('cuda')
                tokens = model(**inputs, output_hidden_states=True)

            preds = []
            for layer_num in range(from_idx, to_idx):
                logits = classifier_dict[attribute][layer_num](tokens.hidden_states[layer_num][:, -1].to(torch.float))[0]
                preds.append(logits.cpu().detach().numpy())
            preds = np.array(preds)
            pred_diffs.append(final_preds - np.mean(preds.reshape(-1, preds.shape[-1]), axis=0)[feature_idx])

        values = pred_diffs
        values = np.array(values)
        # print(values)
        min_val = np.percentile(abs(values), 50)

        values[abs(values) < min_val] = 0
        # values /= 0.5
        values /= 1
        values = np.clip(values, -1, 1)
        # normalized_values = [val * 229 for val in values]
        normalized_values = [val * 1 for val in values]

        # Create a color mapping from values to background colors
        # colors = [f"rgb(249, {249 - int(val)}, {249 - int(val)})" if val > 0 else f"rgb({249 + int(val)}, {249 + int(val)}, 249)" for val in normalized_values]
        colors = [get_color(val) for val in normalized_values]

        # Generate the HTML string with reduced space for word pieces
        html_pieces = []
        words = [word.replace("\n", "<br>") for word in words]
        for i, (word, color) in enumerate(zip(words, colors)):
            if word != "<br>":
                span = f"<span id='attribution' style='background-color:{color}; padding: 0px 0px;'>{word.strip()}</span>"
            else:
                span = word
            html_pieces.append(span)

        html_str = " ".join(html_pieces)
        html_strs.append(html_str)

    return html_strs


def total_variation_distance(P, Q):
    """
    Calculate the total variation distance between two discrete probability distributions
    using numpy for faster computation.
    
    Args:
    - P (numpy array): First probability distribution.
    - Q (numpy array): Second probability distribution.
    
    Returns:
    - float: The total variation distance.
    """
    
    # Convert lists to numpy arrays if they are not already
    P = np.asarray(P)
    Q = np.asarray(Q)

    # Check if both distributions have the same length
    if P.shape != Q.shape:
        raise ValueError("The two distributions must have the same length.")
    
    # Calculate the total variation distance using numpy operations
    tv_distance = 0.5 * np.sum(np.abs(P - Q))
    
    return tv_distance


def output_sensitivity(chat_history, model, tokenizer, classifiers, subattribute, attribute, special_prompt, from_idx=20, to_idx=40, residual=True, mistral=False, attribute_targets=None):
    copychat_history = copy.deepcopy(chat_history)
    category_labels = {"gender": ["Male", "Female", "Other"],
                       "age": ["Child", "Adolescent", "Adult", "Older Adult", "Unknown"],
                       "education": ["Some Education", "High School", "College & More", "Unknown"],
                       "ethnics": ["Asian", "African", "White", 
                               "Hispanic", "Native Americans", "Arabs", "Jewish", "Unknown"],
                       "socioeco": ["Lower", "Middle", "Upper", "Unknown"],
                       "marital": ["Single", "Married", "Divorced", "Widowed"],
                       "language": ["Chinese", "Japanese", "English", "German", "Spanish", "Portuguese", "Arabic", "Russian"],
                       "religion": ["Christian", "Islam", "Buddhism", "Hinduism", "Judaism", "Atheism", "Unknown"],
                       "political": ["Left", "Right", "Moderate", "Unknown"],
                       "sycophancy": ["Sycophancy", "nonSycohant"],
                       "hallucination": ["Hallucinated", "Factual"], 
                      }
    subattribute = subattribute.title()
    
    if attribute is None:
        for attr in list(category_labels.keys()):
            if subattribute in category_labels[attr]:
                attribute = attr
                break
    if not (attribute in category_labels.keys()):
        return ["Attribute does not exist"]
    elif not (subattribute in category_labels[attribute]):
        return ["Subattribute does not exist"]
    
    if mistral:
        prompt = mistral_v2_prompt(copychat_history[:-1])
    else:
        prompt = llama_v2_prompt(copychat_history[:-1])
    inputs = tokenizer(prompt, return_tensors='pt').to('cuda')
    start = inputs.input_ids.shape[1]
    if mistral:
        prompt = mistral_v2_prompt(copychat_history)
    else:
        prompt = llama_v2_prompt(copychat_history)
    with torch.no_grad():
        new_inputs = tokenizer(prompt, return_tensors='pt').to('cuda')
        original_tokens = model(**new_inputs,)

    cf_target = torch.nn.functional.one_hot(torch.Tensor([category_labels[attribute].index(subattribute)]).to(torch.long), 
                                                classifiers[attribute][0].proj[0].weight.shape[0]
                                               ).to(torch_device).to(torch.float)
    
    which_layers = []
    residual = True
    for name, module in model.named_modules():
        if residual and name!= "" and name[-1].isdigit():
            layer_num = name[name.rfind("model.layers.") + len("model.layers."):]
            if from_idx <= int(layer_num) < to_idx:
                which_layers.append(name)
        elif (not residual) and name.endswith(".mlp"):
            layer_num = name[name.rfind("model.layers.") + len("model.layers."):name.rfind(".mlp")]
            if from_idx <= int(layer_num) < to_idx:
                which_layers.append(name)
    N = 6.5

    modified_layer_names = which_layers
    def interv_attr(output, layer_name):
        if len(output) < 3:
            return output
        residual = True
        if residual:
            layer_num = layer_name[layer_name.rfind("model.layers.") + len("model.layers."):]
        else:
            layer_num = layer_name[layer_name.rfind("model.layers.") + len("model.layers."):layer_name.rfind(".mlp")]
        layer_num = int(layer_num) + 1
        probe = classifiers[attribute][layer_num]
        cloned_inter_rep = output[0][0][start - 1:].detach().clone().to(torch.float)
        with torch.enable_grad():
            cloned_inter_rep = optimize_one_inter_rep(cloned_inter_rep, layer_name, 
                                                      cf_target, probe,
                                                      lr=100, max_epoch=100, 
                                                      loss_func=None,
                                                      simplified=True,
                                                      N=N,
                                                      normalized=False)
        output[0][0][start - 1:] = cloned_inter_rep[:].to(torch.float16)
        return output
    with TraceDict(model, modified_layer_names, edit_output=interv_attr) as ret:
        # output = model(inputs)
        with torch.no_grad():
            new_inputs = tokenizer(prompt, return_tensors='pt',return_token_type_ids=False).to('cuda')
            intervened_tokens = model(**new_inputs, output_hidden_states=True, output_attentions=True,
                           )
    
    idx_map = ((intervened_tokens.logits.argmax(dim=-1) == torch.concat([new_inputs.input_ids[:, 1:], original_tokens.logits.argmax(dim=-1)[:, -1:]], dim=-1)))
    outputs = []
    for start_id in (idx_map == False).nonzero():
        if start_id[1] < start - 1:
            continue
        def edit_inter_rep_multi_layers(output, layer_name):
            if len(output) < 3:
                return output
            if residual:
                layer_num = layer_name[layer_name.rfind("model.layers.") + len("model.layers."):]
            else:
                layer_num = layer_name[layer_name.rfind("model.layers.") + len("model.layers."):layer_name.rfind(".mlp")]
            layer_num = int(layer_num) + 1
            if isinstance(attribute, list):
                probe = []
                for attr in attribute:
                    probe.append(classifiers[attr][layer_num])
            else:
                probe = classifiers[attribute][layer_num]
            if len(output[0][0]) > 1:
                cloned_inter_rep = output[0][0][start:].detach().clone().to(torch.float) 
            else:
                cloned_inter_rep = output[0][0][-1].unsqueeze(0).detach().clone().to(torch.float)
            with torch.enable_grad():
                cloned_inter_rep = optimize_one_inter_rep(cloned_inter_rep, layer_name, 
                                                      cf_target, probe,
                                                      lr=100, max_epoch=100, 
                                                      loss_func=None,
                                                      simplified=True,
                                                      N=N,
                                                      normalized=False)
            if len(output[0][0]) > 1:
                output[0][0][start:] = cloned_inter_rep[:].to(torch.float16)
            else:
                output[0][0][-1] = cloned_inter_rep[0].to(torch.float16)
            return output
        with TraceDict(model, modified_layer_names, edit_output=edit_inter_rep_multi_layers) as ret:
            # output = model(inputs)
            with torch.no_grad():
                i_tokens = model.generate(input_ids=torch.concat([inputs.input_ids, 
                                                                  original_tokens.logits.argmax(dim=-1)[:, inputs.input_ids.shape[1]:]],
                                                                 dim=1)[:, :start_id[1]],
                                          attention_mask=torch.ones_like(original_tokens.logits.argmax(dim=-1))[:, :start_id[1]],
                                          max_new_tokens=14,
                                          do_sample=False,
                                          temperature=0.001,
                                          top_p=1,
                                          output_hidden_states=True, 
                                          output_attentions=True,
                                       )
        output = tokenizer.decode(i_tokens[0][start_id[1]:], skip_special_tokens=True)
        output = output.strip("\n").strip()
        if len(output) > 0 and (not (output[-1] in [".", "?", "!", '"', "'"])):
            output += "......"
        outputs.append(output)

    np_idx_map = idx_map.cpu().detach().numpy()

    distance = []
    original_probs = torch.nn.functional.softmax(original_tokens.logits, dim=-1)[0].cpu().detach().numpy()
    intervened_probs = torch.nn.functional.softmax(intervened_tokens.logits, dim=-1)[0].cpu().detach().numpy()
    for index in range(start - 1, len(original_tokens[0][0])):
        dist = total_variation_distance(original_probs[index], intervened_probs[index])
        if np.isinf(dist):
            dist = -np.inf
        distance.append(dist)

    distance = [max(distance) if np.isinf(dist) else dist for dist in distance ]
    distance = [dist if dist > 0.001 else 0 for dist in distance]
    distance = np.array(distance)
    distance[1:][np_idx_map[0][start:]] = 0

    # Sample data
    words = tokenizer.convert_ids_to_tokens(new_inputs.input_ids[0])[start:]
    values = distance
    words = [word.replace("▁", " ") for word in words]
    words = [word.replace("<0x0A>", "<br>") for word in words]

    min_val = 0
    max_val = 1
    # normalized_values = [(val - min_val) / (max_val - min_val) * 255 for val in values]
    normalized_values = [(val - min_val) / (max_val - min_val) * 1 for val in values]

    # Create a color mapping from values to background colors
    # colors = [f"rgb(249, {249 - int(val)}, {249 - int(val)})" for val in normalized_values]
    colors = [get_color(val) for val in normalized_values]

    # Generate the HTML string with reduced space for word pieces
    html_pieces = []
    cur_color = colors[0]
    num_red = 229 # green now, because we change colormap
    counter = 0
    same_word = False
    for i, (word, color) in enumerate(zip(words, colors)):
        cur_num_red = int(color[color.find(",") + 2:color.rfind(",")])
        if word.startswith(" ") or word == "<br>":
            same_word = False
            num_red = cur_num_red
            cur_color = color
        elif cur_num_red < num_red:
            num_red = cur_num_red
            cur_color = color
        if same_word and not np_idx_map[:, start - 1 + i][0]:
            span = f"""<span style='background-color:{cur_color}; padding: 0px 0px;'  data-tooltip="{cur_output}" data-tooltip-position='right'>{word.strip()}</span>"""
            counter += 1
            same_word = True
        elif not np_idx_map[:, start - 1 + i][0]:
            cur_output = outputs[counter]
            span = f"""<span style='background-color:{cur_color}; padding: 0px 0px;'  data-tooltip="{cur_output}" data-tooltip-position='right'>{word.strip()}</span>"""
            counter += 1
            same_word = True
        elif same_word:
            span = f"""<span style='background-color:{cur_color}; padding: 0px 0px;'  data-tooltip="{cur_output}" data-tooltip-position='right'>{word.strip()}</span>"""
        else:
            span = f"<span style='background-color:{cur_color}; padding: 0px 0px;'>{word.strip()}</span>"
        if not word.startswith(" "):
            # Directly append the span for word pieces without any space in between
            html_pieces[-1] += span
        else:
            html_pieces.append(span)

    html_str = " ".join(html_pieces)
        
    return html_str


def output_uncertainty(chat_history, tokenizer, uncertainty, mistral=False):
    copychat_history = copy.deepcopy(chat_history)
    
    values = uncertainty
    # Sample data
    
    if mistral:
        prompt = mistral_v2_prompt(copychat_history[:-1])
    else:
        prompt = llama_v2_prompt(copychat_history[:-1])
    inputs = tokenizer(prompt, return_tensors='pt').to('cuda')
    start = inputs.input_ids.shape[1]
    if mistral:
        prompt = mistral_v2_prompt(copychat_history)
    else:
        prompt = llama_v2_prompt(copychat_history)
    new_inputs = tokenizer(prompt, return_tensors='pt',return_token_type_ids=False).to('cuda')
    words = tokenizer.convert_ids_to_tokens(new_inputs.input_ids[0])[start:]
    words = [word.replace("▁", " ") for word in words]
    words = [word.replace("<0x0A>", "<br>") for word in words]

    # normalized_values = [(val - min_val) / (max_val - min_val) * 255 for val in values]
    
    normalized_values = [val if val < 3 else 3 for val in values]
    normalized_values = [val / 3 for val in values]

    # Create a color mapping from values to background colors
    # colors = [f"rgb(249, {249 - int(val)}, {249 - int(val)})" for val in normalized_values]
    colors = [get_color(-val) for val in normalized_values]

    # Generate the HTML string with reduced space for word pieces
    html_pieces = []
    # print(len(colors), len(words))
    for i, (word, color) in enumerate(zip(words, colors)):
        span = f"<span style='background-color:{color}; padding: 0px 0px;'>{word.strip()}</span>"
        if "<0x" in word:
            continue
        if not word.startswith(" "):
            # Directly append the span for word pieces without any space in between
            html_pieces[-1] += span
        else:
            html_pieces.append(span)

    html_str = " ".join(html_pieces)
    # print(html_str)
        
    return html_str
