import openai
import copy
import torch
from collections import OrderedDict
from baukit import TraceDict
from torch import nn
from app.chat.interv_utils import optimize_one_inter_rep
import numpy as np
import transformers

if "OPENAI_API_KEY" in os.environ:
    openai.api_key = os.environ["OPENAI_API_KEY"]
else:
    print("OPENAI_API_KEY not found in environment variables")
    openai.api_key = ""
special_prompts = ["age", "gender", "education level", "ethnicity", "socioeconomic status", "marital status", "spoken language", "religious belief", "political view"]
multi_special_prompt = True


def get_reading_prompt_for_attribute(attribute, attribute_targets=None):
    """
    Get the appropriate reading prompt based on the attribute's target (user or chatbot).
    For default probes and unknown attributes, defaults to "user" target.
    
    Args:
        attribute: The attribute name (meta_attribute)
        attribute_targets: Dictionary mapping attributes to their targets (user/chatbot)
        
    Returns:
        String with the reading prompt fragment (e.g., "of this user is" or "of myself is")
    """
    if attribute_targets is None:
        attribute_targets = {}
    
    target = attribute_targets.get(attribute, "user")  # Default to "user" if not found
    
    if target == "chatbot":
        return f"of myself is"
    else:  # Default to "user"
        return f"of this user is"

# Max input length for intervention
MaxInputTokens = 2048
# MaxInputTokens = 600
ControlledMaxInputTokens = 1024
# ControlledMaxInputTokens = 300

# default_NminNmax = {"gender": [-9, 9],
#                     "age": [-11, 11],
#                     "socioeco": [-10, 10],
#                     "education": [-10, 10],
#                     "ethnics": [-12, 12],
#                     "marital": [-12, 12],
#                     "language": [-15, 15],
#                     "religion": [-14, 14],
#                     "political": [-12, 12],
#                     "sycophancy": [-10, 10],
#                     "hallucination": [-10, 10]}

default_NminNmax = {"gender": [-1.5, 1.5],
                    "age": [-1.2, 1.2],
                    "socioeco": [-1.2, 1.2],
                    "education": [-1.2, 1.2],
                    "ethnics": [-1, 1],
                    "marital": [-1, 1],
                    "language": [-1, 1],
                    "religion": [-1, 1],
                    "political": [-1, 1],
                    "sycophancy": [-1, 1],
                    "hallucination": [-1, 1]}


# default_NminNmax = {"gender": [-7, 7],
#                     "age": [-9, 9],
#                     "socioeco": [-8, 8],
#                     "education": [-8, 8],
#                     "ethnics": [-10, 10],
#                     "marital": [-10, 10],
#                     "language": [-13, 13],
#                     "religion": [-12, 12],
#                     "political": [-10, 10],
#                     "sycophancy": [-8, 8],
#                     "hallucination": [-8, 8]}


special_prompts_translator = {"age": "age",
                              "gender": "gender",
                              "education level": "education",
                              "ethnicity": "ethnics",
                              "socioeconomic status": "socioeco",
                              "marital status": "marital",
                              "spoken language": "language",
                              "religious belief": "religion",
                              "political view": "political",
                              "sycophancy": "sycophancy",
                              "hallucination": "hallucination"}


def create_msg(role, content):
    return {"role": role, "content": content}


def extract_ai_response(response):
    return response["choices"][0]["message"]


def record_chat_history(role, content, chat_history):
    chat_history.append({"role": role, "content": content})
    

def chat_with_chatgpt(messages, model="gpt-3.5-turbo"):
    return openai.ChatCompletion.create(model=model, messages=messages)


def you_model_query_survey(questions=["gender identity", "occupations", 
                                      "education level", "ethnicity",
                                      "socioeconomic status"],
                           unsure_response="unsure"): 
    prompt = '''Based on the above conversation, what is this user's '''
    for i in range(len(questions)):
        if i == len(questions) - 1:
            prompt += f"{questions[i]}? "
        else:
            prompt += f"{questions[i]}, "
    prompt += "Please rate your confidence about each of your answers (must be an integer from 0 to 10), briefly explain your answer (limited within 15 words), and give your response in the following format:\n"
    
    for i in range(len(questions)):
        prompt += f"{questions[i].capitalize()}: [Your answer] Confidence score: [Your score] Explanation: [Your Explanation]\n"
        
    prompt += f'''Make sure you follow this format precisely, answer all questions, rate and explain all answers, and make your answer concise. If you are unsure about your answer, you can answer "{unsure_response}". Limit your response to each question within 6 words. No matter what lanuage the user is using, reply your answer in English!'''
    
    return prompt


def i_model_query_survey(questions=["task", "confidence in your answer", "writing mode"],
                         unsure_response="unsure"): 
    prompt = '''Based on the above conversation, what is your '''
    for i in range(len(questions)):
        if i == len(questions) - 1:
            prompt += f"{questions[i]}? "
        else:
            prompt += f"{questions[i]}, "
    prompt += "Please rate your confidence about each of your answers (must be an integer from 0 to 10), briefly explain your answer (limited within 15 words), and give your response in the following format:\n"
    
    for i in range(len(questions)):
        prompt += f"{questions[i].capitalize()}: [Your answer] Confidence score: [Your score] Explanation: [Your Explanation]\n"
        
    prompt += f'''Make sure you follow this exact format including its order, answer all questions, rate and explain all answers, and make your answer concise. If you are unsure about your answer, you can answer "{unsure_response}". Limit your response to each question within 6 words.'''
    
    return prompt


default_you_model_prompt = "Use a few sentences to describe this user based on the previous conversation."
def you_model_describe_user(chat_history, prompt=default_you_model_prompt, role="system", model="gpt-3.5-turbo"):
    system_msg = create_msg(role, prompt)
    chat_history.append(system_msg)
    response = chat_with_chatgpt(chat_history, model=model)
    chat_history.pop()
    ai_msg = extract_ai_response(response)
    return ai_msg["content"]


default_i_model_prompt = "Use a few sentences to describe yourself and what you are doing with this user."
def i_model_describe_ai(chat_history, prompt=default_i_model_prompt, role="system", model="gpt-3.5-turbo"):
    system_msg = create_msg(role, prompt)
    chat_history.append(system_msg)
    response = chat_with_chatgpt(chat_history, model=model)
    chat_history.pop()
    ai_msg = extract_ai_response(response)
    return ai_msg["content"]


def extract_survey_results(result, questions, score_identifier="Confidence score:", explanation_identifier="Explanation:"):
    response_dict = {}
    explanation_dict = {}
    score_dict = {}
    
    parsed_result = copy.copy(result)
    for question in questions:
        try:
            start_ind = parsed_result.find(question.capitalize()) + len(question) + 1
            end_ind = parsed_result[start_ind:].find(score_identifier)
            end_ind += start_ind
            response_dict[question] = parsed_result[start_ind: end_ind].strip(" ").strip(",").strip(".")
            parsed_result = parsed_result[end_ind:]
            
            start_ind = parsed_result.find(score_identifier) + len(score_identifier) + 1
            end_ind = parsed_result[start_ind:].find(explanation_identifier)
            end_ind += start_ind
            score_dict[question] = int(parsed_result[start_ind: end_ind].strip(" ").strip(",").strip("."))
            parsed_result = parsed_result[end_ind:]
            
            start_ind = parsed_result.find(explanation_identifier) + len(explanation_identifier) + 1
            end_ind = parsed_result[start_ind:].find("\n")
            if end_ind == -1:
                explanation_dict[question] = parsed_result[start_ind: ].strip(" ")
            else:
                end_ind += start_ind
                explanation_dict[question] = parsed_result[start_ind: end_ind].strip(" ")
            parsed_result = parsed_result[end_ind + 1:]
        except Exception as e:
            print(e)
            response_dict[question] = "Error"
            score_dict[question] = "Error"
            explanation_dict[question] = "Error"
            
    return response_dict, explanation_dict, score_dict


# def split_into_messages(text: str):
#     # Constants used for splitting
#     B_INST, E_INST = "[INST]", "[/INST]"

#     # Use the tokens to split the text
#     parts = []
#     current_message = ""

#     for word in text.split():
#         # If we encounter a start or end token, and there's a current message, store it
#         if word in [B_INST, E_INST] and current_message:
#             parts.append(current_message.strip())
#             current_message = ""
#         # If the word is not a token, add it to the current message
#         elif word not in [B_INST, E_INST]:
#             current_message += word + " "

#     # Append any remaining message
#     if current_message:
#         parts.append(current_message.strip())

#     return parts


def split_into_messages(text: str):
    # Constants used for splitting
    B_INST, E_INST = "[INST]", "[/INST]"

    # Split the text by both the start and end instruction tokens
    parts = text.split(B_INST)
    
    messages = []
    for part in parts:
        # Further split by the end instruction token and keep the content
        sub_parts = part.split(E_INST)
        for sub_part in sub_parts:
            # Trim whitespace from the start and end but keep the content as is, including newline characters
            clean_sub_part = sub_part.strip()
            if clean_sub_part:  # Filter out any empty strings that might have been created by the split
                messages.append(clean_sub_part)

    return messages


def llama_v2_reverse(prompt: str):
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


def llama_v2_prompt(
    messages,
    system_prompt=None,
    tokenizer=None,
):
    if system_prompt:
        DEFAULT_SYSTEM_PROMPT = system_prompt
    else:
        DEFAULT_SYSTEM_PROMPT = f"""You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Please ensure that your responses are socially unbiased and positive in nature. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information. Be concise in your response. Limit your response in 120 words."""

    if type(tokenizer) == transformers.models.gemma.tokenization_gemma_fast.GemmaTokenizerFast:
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return text
    if type(tokenizer) == transformers.tokenization_utils_fast.PreTrainedTokenizerFast:
        messages = [{"role": "system", "content": DEFAULT_SYSTEM_PROMPT}] + messages
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return text
    
    B_INST, E_INST = "[INST]", "[/INST]"
    B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
    BOS, EOS = "<s>", "</s>"
    
        # DEFAULT_SYSTEM_PROMPT = f"""You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Please ensure that your responses are socially unbiased and positive in nature. Limit your response in 120 words."""
        # DEFAULT_SYSTEM_PROMPT = f"""You are a chatbot that actively talks to the user. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information. Be concise in your response. Limit your response in 100 words."""
        # DEFAULT_SYSTEM_PROMPT = f"You are a talk-active chatbot who will help with user's questions."

    if messages[0]["role"] != "system":
        messages = [
            {
                "role": "system",
                "content": DEFAULT_SYSTEM_PROMPT,
            }
        ] + messages
    messages = [
        {
            "role": messages[1]["role"],
            "content": B_SYS + messages[0]["content"] + E_SYS + messages[1]["content"],
        }
    ] + messages[2:]

    messages_list = [
        f"{BOS}{B_INST} {(prompt['content']).strip()} {E_INST} {(answer['content']).strip()} {EOS}"
        for prompt, answer in zip(messages[::2], messages[1::2])
    ]
    if messages[-1]['role'] == "user":
        messages_list.append(f"{BOS}{B_INST} {(messages[-1]['content']).strip()} {E_INST}")

    return "".join(messages_list)  


def mistral_v2_prompt(
    messages: list[dict],
    system_prompt=None
):
    B_INST, E_INST = "[INST]", "[/INST]"
    B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
    BOS, EOS = "<s>", "</s>"

    # messages[0]["content"] = "Always assist with care, respect, and truth. Feel free to make assumption about the user but be explicit about any assumption you made. " + messages[0]["content"]
    messages_list = [
        f"{B_INST} {(prompt['content']).strip()} {E_INST} {(answer['content']).strip()} {EOS}"
        for prompt, answer in zip(messages[::2], messages[1::2])
    ]
    if messages[-1]['role'] == "user":
        messages_list.append(f"{B_INST} {(messages[-1]['content']).strip()} {E_INST}")

    return BOS + "".join(messages_list)  


# class ModuleHook:
#     def __init__(self, module):
#         self.hook = module.register_forward_hook(self.hook_fn)
#         self.module = None
#         self.features = []

#     def hook_fn(self, module, input, output):
#         self.module = module
#         if isinstance(output, tuple):
#             self.features.append(output[0].detach())
#         else:
#             self.features.append(output.detach())
#     def close(self):
#         self.hook.remove()
        
        
def calculate_shannon_entropy_pytorch(logits):
    """
    Calculate the Shannon entropy for each set of logits using PyTorch.
    
    Args:
        logits: A 2D PyTorch tensor where each row contains logits for a different word.
        
    Returns:
        A 1D PyTorch tensor where each element is the Shannon entropy of the corresponding row of logits.
    """
    probabilities = torch.softmax(logits, dim=-1)
    log_probabilities = torch.log2(probabilities + 1e-6)  # Adding a small epsilon to avoid log(0)
    entropy = -torch.sum(probabilities * log_probabilities, dim=1)
    return entropy


def chat_with_llama(messages, model, tokenizer, residual_stream=True, include_inst=True, 
                    return_user_msg_last_act=True,
                    context_length=2048,
                    special_prompt="", multi_special_prompt=False, 
                    system_special_prompt=[],
                    sys_prompt=None,
                    mistral=False,
                    model_type="",
                    attribute_targets=None,
                    cate_labels=None):
    edit_activations = False
    copymessages = copy.deepcopy(messages)
    if mistral:
        prompt = mistral_v2_prompt(messages)
    else:
        prompt = llama_v2_prompt(messages, sys_prompt, tokenizer=tokenizer)
    if "<s>" in prompt:
        prompt = prompt[prompt.find("<s>") + len("<s>"):]
    
    if mistral or model_type == "mistral":
        top_idx = 33
    elif model_type == "llama3":
        top_idx = 33
    elif model_type == "gemma2":
        top_idx = 43
    else:
        top_idx = 41

    print(prompt)
    if mistral:
        temperature=0.3
        top_p=0.95
        repetition_penalty=1.2
        top_k=50
    else:
        temperature=0.75
#         temperature=0.35
        top_p=1.0
        repetition_penalty=1
        top_k=50
    with torch.no_grad():
        inputs = tokenizer(prompt, return_tensors='pt', return_token_type_ids=False).to(model.device) ## update 1
        if inputs['input_ids'].size(1) > MaxInputTokens:
            print(f"Input too long: {inputs['input_ids'].size(1)}! Hard Forgetting Initiated.")
            inputs['input_ids'] = inputs['input_ids'][:, -MaxInputTokens:]
            if 'attention_mask' in inputs:
                inputs['attention_mask'] = inputs['attention_mask'][:, -MaxInputTokens:]
        tokens = model.generate(
         **inputs,
         max_new_tokens=context_length,
         do_sample=True,
         # do_sample=False,
         temperature=temperature,
         top_p=top_p,
         repetition_penalty=repetition_penalty,
         return_dict_in_generate=True, 
         output_scores=True
        )
        scores = tokens[1]
        scores = torch.concat(scores)
        shannon_entropy = calculate_shannon_entropy_pytorch(scores)
        tokens = tokens[0]
        text_output = tokenizer.decode(tokens[0], skip_special_tokens=False)

    if multi_special_prompt:
        # TODO: Refactor this part. Too much redundant code
        all_last_toks = {}
        for attribute in system_special_prompt:
            prompt = text_output + system_special_prompt[attribute]
            print("SYSTEM ATTRIBUTE PROMPT\n")

            with torch.no_grad():
                encoding = tokenizer(prompt, return_tensors='pt', return_token_type_ids=False)
                if encoding['input_ids'].size(1) > MaxInputTokens:
                    print(f"Input too long: {encoding['input_ids'].size(1)}! Hard Forgetting Initiated.")
                    encoding['input_ids'] = encoding['input_ids'][:, -MaxInputTokens:]
                    if 'attention_mask' in encoding:
                        encoding['attention_mask'] = encoding['attention_mask'][:, -MaxInputTokens:]
                tokens = model(
                 input_ids=encoding['input_ids'].to(model.device),
                 attention_mask=encoding['attention_mask'].to(model.device),
                 output_hidden_states=True,
                )
            
            torch.cuda.empty_cache()
            # for feature in features.values():
            #     feature.close()
                
            last_acts = []
            if return_user_msg_last_act:
                which_token = 0
                if include_inst:
                    offset = 0
                else:
                    offset = 1
            else:
                which_token = -1
                offset = 0
            # last_acts.append(features['model.embed_tokens'].features[which_token][:, -(1 + offset)].detach().cpu().clone().to(torch.float))
            last_acts.append(tokens["hidden_states"][0][:, -(1 + offset)].detach().cpu().clone().to(torch.float))

            if residual_stream:
                for layer_num in range(1, top_idx):
                    # last_acts.append(features[f"model.layers.{layer_num - 1}"].features[which_token][:, -(1 + offset)].detach().cpu().clone().to(torch.float))
                    last_acts.append(tokens["hidden_states"][layer_num][:, -(1 + offset)].detach().cpu().clone().to(torch.float))
            else:
                for layer_num in range(1, top_idx):
                    last_acts.append(features[f'model.layers.{layer_num - 1}.mlp'].features[which_token][:, -(1 + offset)].detach().cpu().clone().to(torch.float))
            last_acts = torch.cat(last_acts, dim=0).unsqueeze(0)
            if attribute not in special_prompts_translator:
                attribute = attribute
            else:
                attribute = special_prompts_translator[attribute]
            all_last_toks[attribute] = last_acts
            

            del tokens
            torch.cuda.empty_cache()
            
        for attribute in special_prompt:
            if mistral:
                prompt = mistral_v2_prompt(copymessages)
            else:
                prompt = llama_v2_prompt(copymessages, """You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Please ensure that your responses are socially unbiased and positive in nature. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.""",
                                         tokenizer=tokenizer)
            
            prompt = text_output
            
            # Get the appropriate reading prompt based on target (user or chatbot)
            reading_prompt_suffix = get_reading_prompt_for_attribute(attribute, attribute_targets)

            if mistral:
                prompt = prompt[prompt.find("<s>") + len("<s>"):] + f"\n\nBased on the context of this conversation, I think the {attribute} {reading_prompt_suffix}"
            else:
                if "<s>" in prompt:
                    prompt = prompt[prompt.find("<s>") + len("<s>"):prompt.rfind("</s>")] + f"\n\nBased on the context of this conversation, I think the {attribute} {reading_prompt_suffix}"
                else:
                    prompt = prompt + f"\n\nBased on the context of this conversation, I think the {attribute} {reading_prompt_suffix}"
            with torch.no_grad():
                encoding = tokenizer(prompt, return_tensors='pt',return_token_type_ids=False)
                if encoding['input_ids'].size(1) > MaxInputTokens:
                    print(f"Input too long: {encoding['input_ids'].size(1)}! Hard Forgetting Initiated.")
                    encoding['input_ids'] = encoding['input_ids'][:, -MaxInputTokens:]
                    if 'attention_mask' in encoding:
                        encoding['attention_mask'] = encoding['attention_mask'][:, -MaxInputTokens:]
                tokens = model(
                 input_ids=encoding['input_ids'].to(model.device),
                 attention_mask=encoding['attention_mask'].to(model.device),
                 output_hidden_states=True,
                )

            torch.cuda.empty_cache()
                
            last_acts = []
            if return_user_msg_last_act:
                which_token = 0
                if include_inst:
                    offset = 0
                else:
                    offset = 1
            else:
                which_token = -1
                offset = 0
            # last_acts.append(features['model.embed_tokens'].features[which_token][:, -(1 + offset)].detach().cpu().clone().to(torch.float))
            last_acts.append(tokens["hidden_states"][0][:, -(1 + offset)].detach().cpu().clone().to(torch.float))
            
            if residual_stream:
                for layer_num in range(1, top_idx):
                    last_acts.append(tokens["hidden_states"][layer_num][:, -(1 + offset)].detach().cpu().clone().to(torch.float))
            else:
                for layer_num in range(1, top_idx):
                    last_acts.append(features[f'model.layers.{layer_num - 1}.mlp'].features[which_token][:, -(1 + offset)].detach().cpu().clone().to(torch.float))
            last_acts = torch.cat(last_acts, dim=0).unsqueeze(0)
            attribute = special_prompts_translator[attribute] if attribute in special_prompts_translator else attribute
            all_last_toks[attribute] = last_acts
            all_last_toks["uncertainty"] = shannon_entropy
            

            del tokens
            # del features
            torch.cuda.empty_cache()
        del inputs
        torch.cuda.empty_cache()
        
        # Also get prompt-based attributes
        # Extract assistant response from text_output
        try:
            if mistral or model_type == "mistral":
                ai_response_content = extract_mistral_response(text_output)["content"]
            else:
                ai_response_content = extract_chatbot_response(text_output)["content"]
            
            new_chat_history = copymessages + [{"role": "assistant", "content": ai_response_content}]
            
            print("Get PROMPT-BASED ATTRIBUTES (chat_with_llama)")
            if cate_labels is not None:
                prompt_based_results = get_prompt_based_attributes(new_chat_history, model, tokenizer, 
                                                                   cate_labels=cate_labels,
                                                                   attribute_targets=attribute_targets,
                                                                   model_type=model_type)
            else:
                prompt_based_results = {}
        except Exception as e:
            print(f"Error getting prompt-based attributes in chat_with_llama: {e}")
            import traceback
            traceback.print_exc()
            prompt_based_results = {}
        
        return text_output, all_last_toks, prompt_based_results
        
    elif special_prompt:
        copymessages = copy.deepcopy(messages)
        copymessages[-1]["content"] += special_prompt 
        if mistral:
            prompt = mistral_v2_prompt(copymessages)
        else:
            prompt = llama_v2_prompt(copymessages, tokenizer=tokenizer)
        if "<s>" in prompt:
            prompt = prompt[prompt.find("<s>") + len("<s>"):]


        with torch.no_grad():
            encoding = tokenizer(prompt, return_tensors='pt')
            if encoding['input_ids'].size(1) > MaxInputTokens:
                    print(f"Input too long: {encoding['input_ids'].size(1)}! Hard Forgetting Initiated.")
                    encoding['input_ids'] = encoding['input_ids'][:, -MaxInputTokens:]
                    if 'attention_mask' in encoding:
                        encoding['attention_mask'] = encoding['attention_mask'][:, -MaxInputTokens:]
            tokens = model(
             input_ids=encoding['input_ids'].to(model.device),
             attention_mask=encoding['attention_mask'].to(model.device),
            )

        torch.cuda.empty_cache()
    
    last_acts = []
    if return_user_msg_last_act:
        which_token = 0
        if include_inst:
            offset = 0
        else:
            offset = 1
    else:
        which_token = -1
        offset = 0
    last_acts.append(features['model.embed_tokens'].features[which_token][:, -(1 + offset)].detach().cpu().clone().to(torch.float))
    
    if residual_stream:
        for layer_num in range(1, top_idx):
            last_acts.append(features[f"model.layers.{layer_num - 1}"].features[which_token][:, -(1 + offset)].detach().cpu().clone().to(torch.float))
    else:
        for layer_num in range(1, top_idx):
            last_acts.append(features[f'model.layers.{layer_num - 1}.mlp'].features[which_token][:, -(1 + offset)].detach().cpu().clone().to(torch.float))
    last_acts = torch.cat(last_acts, dim=0).unsqueeze(0)

    del inputs, tokens 
    # del features
    torch.cuda.empty_cache()
    
    return text_output, last_acts


def chat_with_llama_batched(messages_array, model, tokenizer, residual_stream=True, include_inst=True, 
                            return_user_msg_last_act=True,
                            context_length=512,
                            special_prompt="", multi_special_prompt=False, 
                            system_special_prompt=[],
                            sys_prompt=None,
                            mistral=False,
                            model_type="",
                            attribute_targets=None,
                            cate_labels=None):
    prompts = []
    copymessages_list = []
    for messages in messages_array:
        copymessages = copy.deepcopy(messages)
    
        if mistral:
            prompt = mistral_v2_prompt(messages)
        else:
            prompt = llama_v2_prompt(messages, sys_prompt, tokenizer=tokenizer)
        if "<s>" in prompt:
            prompt = prompt[prompt.find("<s>") + len("<s>"):]
        prompts.append(prompt)
        copymessages_list.append(copymessages)
        
    if mistral or model_type == "mistral":
        top_idx = 33
    elif model_type == "llama3":
        top_idx = 33
    elif model_type == "gemma2":
        top_idx = 43
    else:
        top_idx = 41

    print(prompt)
    if mistral:
        temperature=0.3
        top_p=0.95
        repetition_penalty=1.2
        top_k=50
    else:
        temperature=0.75
        top_p=1.0
        repetition_penalty=1
        top_k=50

    if '<pad>' not in tokenizer.get_vocab():
        tokenizer.add_special_tokens({"pad_token":"<pad>"})

    text_outputs = []
    with torch.no_grad():
        inputs = tokenizer(prompts, return_tensors='pt', padding=True).to(model.device)
        
        tokens = model.generate(
                                 **inputs,
                                 max_new_tokens=context_length,
                                 do_sample=True,
                                 # do_sample=False,
                                 temperature=temperature,
                                 top_p=top_p,
                                 repetition_penalty=repetition_penalty,
                                 return_dict_in_generate=True, 
                                 output_scores=True
                                   )
        scores = tokens[1]
        scores = torch.concat(scores)
        shannon_entropy = calculate_shannon_entropy_pytorch(scores)
        tokens = tokens[0]
            
    text_outputs = [tokenizer.decode(seq, skip_special_tokens=False) for seq in tokens]
    torch.cuda.empty_cache()

    if multi_special_prompt:
        # TODO: Refactor this part. Too much redundant code
        all_last_toks_array = []
        for text_output, copymessages in zip(text_outputs, copymessages_list):
            all_last_toks = {}
            for attribute in system_special_prompt:
                prompts = text_output + system_special_prompt[attribute]
                print("SYSTEM ATTRIBUTE PROMPT\n")

                with torch.no_grad():
                    encoding = tokenizer(prompts, return_tensors='pt', return_token_type_ids=False, padding=True)
                    tokens = model(
                     input_ids=encoding['input_ids'].to(model.device),
                     attention_mask=encoding['attention_mask'].to(model.device),
                     output_hidden_states=True,
                    )

                torch.cuda.empty_cache()
                # for feature in features.values():
                #     feature.close()

                last_acts = []
                if return_user_msg_last_act:
                    which_token = 0
                    if include_inst:
                        offset = 0
                    else:
                        offset = 1
                else:
                    which_token = -1
                    offset = 0
                # last_acts.append(features['model.embed_tokens'].features[which_token][:, -(1 + offset)].detach().cpu().clone().to(torch.float))
                last_acts.append(tokens["hidden_states"][0][:, -(1 + offset)].detach().cpu().clone().to(torch.float))

                if residual_stream:
                    for layer_num in range(1, top_idx):
                        # last_acts.append(features[f"model.layers.{layer_num - 1}"].features[which_token][:, -(1 + offset)].detach().cpu().clone().to(torch.float))
                        last_acts.append(tokens["hidden_states"][layer_num][:, -(1 + offset)].detach().cpu().clone().to(torch.float))
                else:
                    for layer_num in range(1, top_idx):
                        last_acts.append(features[f'model.layers.{layer_num - 1}.mlp'].features[which_token][:, -(1 + offset)].detach().cpu().clone().to(torch.float))
                last_acts = torch.cat(last_acts, dim=0).unsqueeze(0)
                attribute = special_prompts_translator[attribute] if attribute in special_prompts_translator else attribute
                all_last_toks[attribute] = last_acts


                del tokens
                torch.cuda.empty_cache()

            for attribute in special_prompt:
                if mistral:
                    prompt = mistral_v2_prompt(copymessages)
                else:

                    prompt = llama_v2_prompt(copymessages, 
                                             """You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Please ensure that your responses are socially unbiased and positive in nature. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.""",
                                             tokenizer=tokenizer)

                prompt = text_output
                
                # Get the appropriate reading prompt based on target (user or chatbot)
                reading_prompt_suffix = get_reading_prompt_for_attribute(attribute, attribute_targets)

                if mistral:
                    prompt = prompt[prompt.find("<s>") + len("<s>"):] + f"\n\nBased on the context of this conversation, I think the {attribute} {reading_prompt_suffix}"
                else:
                    if "<s>" in prompt:
                        prompt = prompt[prompt.find("<s>") + len("<s>"):prompt.rfind("</s>")] + f"\n\nBased on the context of this conversation, I think the {attribute} {reading_prompt_suffix}"
                    else:
                        prompt = prompt + f"\n\nBased on the context of this conversation, I think the {attribute} {reading_prompt_suffix}"
                with torch.no_grad():
                    encoding = tokenizer(prompt, return_tensors='pt',return_token_type_ids=False)
                    tokens = model(
                     input_ids=encoding['input_ids'].to(model.device),
                     attention_mask=encoding['attention_mask'].to(model.device),
                     output_hidden_states=True,
                    )

                torch.cuda.empty_cache()

                last_acts = []
                if return_user_msg_last_act:
                    which_token = 0
                    if include_inst:
                        offset = 0
                    else:
                        offset = 1
                else:
                    which_token = -1
                    offset = 0
                # last_acts.append(features['model.embed_tokens'].features[which_token][:, -(1 + offset)].detach().cpu().clone().to(torch.float))
                last_acts.append(tokens["hidden_states"][0][:, -(1 + offset)].detach().cpu().clone().to(torch.float))

                if residual_stream:
                    for layer_num in range(1, top_idx):
                        last_acts.append(tokens["hidden_states"][layer_num][:, -(1 + offset)].detach().cpu().clone().to(torch.float))
                else:
                    for layer_num in range(1, top_idx):
                        last_acts.append(features[f'model.layers.{layer_num - 1}.mlp'].features[which_token][:, -(1 + offset)].detach().cpu().clone().to(torch.float))
                last_acts = torch.cat(last_acts, dim=0).unsqueeze(0)
                attribute = special_prompts_translator[attribute] if attribute in special_prompts_translator else attribute
                all_last_toks[attribute] = last_acts
                all_last_toks["uncertainty"] = shannon_entropy
                
                all_last_toks_array.append(all_last_toks)

                del tokens
                torch.cuda.empty_cache()
         
        del inputs
        torch.cuda.empty_cache()
        
        # Also get prompt-based attributes for each conversation in the batch
        prompt_based_results_array = []
        for text_output, copymessages in zip(text_outputs, copymessages_list):
            try:
                if mistral or model_type == "mistral":
                    ai_response_content = extract_mistral_response(text_output)["content"]
                else:
                    ai_response_content = extract_chatbot_response(text_output)["content"]
                
                new_chat_history = copymessages + [{"role": "assistant", "content": ai_response_content}]
                
                print("Get PROMPT-BASED ATTRIBUTES (chat_with_llama_batched)")
                if cate_labels is not None:
                    prompt_based_results = get_prompt_based_attributes(new_chat_history, model, tokenizer, 
                                                                       cate_labels=cate_labels,
                                                                       attribute_targets=attribute_targets,
                                                                       model_type=model_type)
                else:
                    prompt_based_results = {}
            except Exception as e:
                print(f"Error getting prompt-based attributes in chat_with_llama_batched: {e}")
                import traceback
                traceback.print_exc()
                prompt_based_results = {}
            
            prompt_based_results_array.append(prompt_based_results)

        return text_outputs, all_last_toks_array, prompt_based_results_array


def get_activation_from_llama(messages, model, tokenizer, residual_stream=True, include_inst=True, 
                              return_user_msg_last_act=True,
                              context_length=4096,
                              special_prompt="", multi_special_prompt=False, 
                              system_special_prompt=[], mistral=False, intervened=False,
                              model_type="", attribute_targets=None):
    copymessages = copy.deepcopy(messages)
    
    if mistral or model_type=="mistral":
        prompt = mistral_v2_prompt(messages)
        top_idx = 33
    else:
        prompt = llama_v2_prompt(messages, tokenizer=tokenizer)
        if model_type=="llama3":
            top_idx = 33
        elif model_type == "gemma2":
            top_idx = 43
        else:
            top_idx = 41
    prompt = prompt[prompt.find("<s>") + len("<s>"):prompt.rfind("</s>")]
    
    text_output = prompt

    if multi_special_prompt:
        # TODO: Refactor this part. Too much redundant code
        all_last_toks = {}
        for attribute in system_special_prompt:
            prompt = text_output + system_special_prompt[attribute]
            features = OrderedDict()
            # if residual_stream:
            #     for name, module in model.named_modules():
            #         if name != "" and (name[-1].isdigit() or name.endswith(".embed_tokens")):
            #             features[name] = ModuleHook(module)
            # else:
            #     for name, module in model.named_modules():
            #         if name.endswith(".mlp") or name.endswith(".embed_tokens"):
            #             features[name] = ModuleHook(module)

            with torch.no_grad():
                encoding = tokenizer(prompt, return_tensors='pt', return_token_type_ids=False)
                if encoding['input_ids'].size(1) > ControlledMaxInputTokens and intervened:
                    print(f"Input too long: {encoding['input_ids'].size(1)}! Hard Forgetting Initiated.")
                    encoding['input_ids'] = encoding['input_ids'][:, -ControlledMaxInputTokens:]
                    if 'attention_mask' in encoding:
                        encoding['attention_mask'] = encoding['attention_mask'][:, -ControlledMaxInputTokens:]
                        
                tokens = model(
                 input_ids=encoding['input_ids'].to(model.device),
                 attention_mask=encoding['attention_mask'].to(model.device),
                 output_hidden_states=True,
                 output_attentions=True,
                )

            # for feature in features.values():
            #     feature.close()
            torch.cuda.empty_cache()
            last_acts = []
            if return_user_msg_last_act:
                which_token = 0
                if include_inst:
                    offset = 0
                else:
                    offset = 1
            else:
                which_token = -1
                offset = 0
            # last_acts.append(features['model.embed_tokens'].features[which_token][:, -(1 + offset)].detach().cpu().clone().to(torch.float))
            last_acts.append(tokens["hidden_states"][0][:, -(1 + offset)].detach().cpu().clone().to(torch.float))
            
            if residual_stream:
                for layer_num in range(1, top_idx):
                    last_acts.append(tokens["hidden_states"][layer_num][:, -(1 + offset)].detach().cpu().clone().to(torch.float))
            # else:
            #     for layer_num in range(1, top_idx):
            #         last_acts.append(features[f'model.layers.{layer_num - 1}.mlp'].features[which_token][:, -(1 + offset)].detach().cpu().clone().to(torch.float))
            last_acts = torch.cat(last_acts, dim=0).unsqueeze(0)
            attribute = special_prompts_translator[attribute] if attribute in special_prompts_translator else attribute
            all_last_toks[attribute] = last_acts
            
            # for feature in features.values():
            #     del feature
            #     torch.cuda.empty_cache()
            del tokens
            # del features
            torch.cuda.empty_cache()
            
        for attribute in special_prompt:
            if mistral:
                prompt = mistral_v2_prompt(copymessages)
            else:
                prompt = llama_v2_prompt(copymessages, """You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Please ensure that your responses are socially unbiased and positive in nature. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.""", tokenizer=tokenizer)
            
            # Get the appropriate reading prompt based on target (user or chatbot)
            reading_prompt_suffix = get_reading_prompt_for_attribute(attribute, attribute_targets)
            
            # Check without model response
            # prompt = prompt[prompt.find("<s>") + len("<s>"):] + "<s> [INST] Answer the next question using the information in your previous response. [/INST]" + f" I think the {attribute} {reading_prompt_suffix}"
            if "<s>" in prompt:
                prompt = prompt[prompt.find("<s>") + len("<s>"):] + "<s> [INST] Answer the next question using the information in your previous response. [/INST]" + f" I think the {attribute} {reading_prompt_suffix}"
            elif "<|start_header_id|>" in prompt:
                prompt = prompt + f"<|start_header_id|>user<|end_header_id|>\n\nAnswer the next question using the information in your previous response.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nBased on the context of this conversation, I think the {attribute} {reading_prompt_suffix}"
            elif "<start_of_turn>" in prompt:
                prompt = prompt[:prompt.rfind("<eos>")] + f"<start_of_turn>user\nAnswer the next question using the information in your previous response.<end_of_turn>\n<start_of_turn>model\nBased on the context of this conversation, I think the {attribute} {reading_prompt_suffix}"
            # prompt = prompt[prompt.find("<s>") + len("<s>"):text_output.rfind("</s>") - 1] + f" I think the {attribute} of this user is"
            # prompt = text_output + f"\n\nBased on the conversation, I think the {attribute} of this user is"
            # prompt = text_output[text_output.find("<s>") + len("<s>"):text_output.rfind("</s>") - 1] + f" I think the {attribute} of this user is"
            # print(prompt)
            # print("-" * 20, "PROMPT USED", "-" * 20, "\n", prompt)
            # features = OrderedDict()
            # if residual_stream:
            #     for name, module in model.named_modules():
            #         if name != "" and (name[-1].isdigit() or name.endswith(".embed_tokens")):
            #             features[name] = ModuleHook(module)
            # else:
            #     for name, module in model.named_modules():
            #         if name.endswith(".mlp") or name.endswith(".embed_tokens"):
            #             features[name] = ModuleHook(module)

            with torch.no_grad():
                encoding = tokenizer(prompt, return_tensors='pt',return_token_type_ids=False)
                if encoding['input_ids'].size(1) > ControlledMaxInputTokens and intervened:
                    print(f"Input too long: {encoding['input_ids'].size(1)}! Hard Forgetting Initiated.")
                    encoding['input_ids'] = encoding['input_ids'][:, -ControlledMaxInputTokens:]
                    if 'attention_mask' in encoding:
                        encoding['attention_mask'] = encoding['attention_mask'][:, -ControlledMaxInputTokens:]
                        
                tokens = model(
                 input_ids=encoding['input_ids'].to(model.device),
                 attention_mask=encoding['attention_mask'].to(model.device),
                 output_hidden_states=True,
                 output_attentions=True,
                )

            # for feature in features.values():
            #     feature.close()
            torch.cuda.empty_cache()
                
            last_acts = []
            if return_user_msg_last_act:
                which_token = 0
                if include_inst:
                    offset = 0
                else:
                    offset = 1
            else:
                which_token = -1
                offset = 0
            # last_acts.append(features['model.embed_tokens'].features[which_token][:, -(1 + offset)].detach().cpu().clone().to(torch.float))
            last_acts.append(tokens["hidden_states"][0][:, -(1 + offset)].detach().cpu().clone().to(torch.float))
            
            if residual_stream:
                for layer_num in range(1, top_idx):
                    last_acts.append(tokens["hidden_states"][layer_num][:, -(1 + offset)].detach().cpu().clone().to(torch.float))
            # else:
            #     for layer_num in range(1, top_idx):
            #         last_acts.append(features[f'model.layers.{layer_num - 1}.mlp'].features[which_token][:, -(1 + offset)].detach().cpu().clone().to(torch.float))
            last_acts = torch.cat(last_acts, dim=0).unsqueeze(0)
            attribute = special_prompts_translator[attribute] if attribute in special_prompts_translator else attribute
            all_last_toks[attribute] = last_acts
            
            # for feature in features.values():
            #     del feature
            #     torch.cuda.empty_cache()
            del tokens
            # del features
            torch.cuda.empty_cache()
        del encoding
        torch.cuda.empty_cache()
        
        return all_last_toks
        
    elif special_prompt:
        copymessages = copy.deepcopy(messages)
        copymessages[-1]["content"] += special_prompt 
        if mistral:
            prompt = mistral_v2_prompt(copymessages)
        else:
            prompt = llama_v2_prompt(copymessages, tokenizer=tokenizer)
        if "<s>" in prompt:
            prompt = prompt[prompt.find("<s>") + len("<s>"):]

        # features = OrderedDict()
        # if residual_stream:
        #     for name, module in model.named_modules():
        #         if name != "" and (name[-1].isdigit() or name.endswith(".embed_tokens")):
        #             features[name] = ModuleHook(module)
        # else:
        #     for name, module in model.named_modules():
        #         if name.endswith(".mlp") or name.endswith(".embed_tokens"):
        #             features[name] = ModuleHook(module)

        with torch.no_grad():
            encoding = tokenizer(prompt, return_tensors='pt')
            if encoding['input_ids'].size(1) > ControlledMaxInputTokens and intervened:
                print(f"Input too long: {encoding['input_ids'].size(1)}! Hard Forgetting Initiated.")
                encoding['input_ids'] = encoding['input_ids'][:, -ControlledMaxInputTokens:]
                if 'attention_mask' in encoding:
                    encoding['attention_mask'] = encoding['attention_mask'][:, -ControlledMaxInputTokens:]
                        
            tokens = model(
             input_ids=encoding['input_ids'].to(model.device),
             attention_mask=encoding['attention_mask'].to(model.device),
             output_hidden_states=True,
             output_attentions=True,
            )
        torch.cuda.empty_cache()

        # for feature in features.values():
        #     feature.close()
        
    last_acts = []
    if return_user_msg_last_act:
        which_token = 0
        if include_inst:
            offset = 0
        else:
            offset = 1
    else:
        which_token = -1
        offset = 0
    last_acts.append(features['model.embed_tokens'].features[which_token][:, -(1 + offset)].detach().cpu().clone().to(torch.float))
    
    if residual_stream:
        for layer_num in range(1, top_idx):
            last_acts.append(features[f"model.layers.{layer_num - 1}"].features[which_token][:, -(1 + offset)].detach().cpu().clone().to(torch.float))
    else:
        for layer_num in range(1, top_idx):
            last_acts.append(features[f'model.layers.{layer_num - 1}.mlp'].features[which_token][:, -(1 + offset)].detach().cpu().clone().to(torch.float))
    last_acts = torch.cat(last_acts, dim=0).unsqueeze(0)
    
    # for feature in features.values():
    #     del feature
    #     torch.cuda.empty_cache()
    del encoding, inputs, tokens
    # del features
    torch.cuda.empty_cache()
    
    return last_acts


def get_prompt_based_attributes(messages, model, tokenizer, cate_labels, 
                                 attribute_targets=None, model_type="llama3"):
    """
    Get attribute predictions using prompt-based next-token logit probing.
    
    Args:
        messages: List of conversation messages
        model: The language model
        tokenizer: The tokenizer
        cate_labels: Dictionary mapping attributes to their possible labels
        attribute_targets: Dictionary mapping attributes to their targets (user/chatbot)
        model_type: Type of model being used
    
    Returns:
        Dictionary with attribute predictions in same format as linear probe results
    """
    if attribute_targets is None:
        attribute_targets = {}
    
    results = {}
    
    # Check if last message is from user or assistant
    last_is_assistant = len(messages) > 0 and messages[-1]["role"] == "assistant"
    
    for attribute, labels in cate_labels.items():
        # Skip special attributes that don't have standard labels
        if attribute in ["uncertainty"]:
            continue
        
        # Skip if no labels
        if not labels or len(labels) == 0:
            continue
            
        # Build multiple-choice options: {A: Male, B: Female, C: Other}
        options_list = []
        for i, label in enumerate(labels):
            letter = chr(65 + i)  # A, B, C, ...
            label_capitalized = label.capitalize() if label.lower() != label else label
            options_list.append(f"{letter}: {label_capitalized}")
        options_text = ", ".join(options_list)
        
        # Determine target (user vs chatbot)
        target = attribute_targets.get(attribute, "user")
        if target == "chatbot":
            target_text_user = "your"
            target_text_assistant = "myself"
        else:
            target_text_user = "my"
            target_text_assistant = "this user"
        
        # Build the user question
        user_question = (f"Based on our conversation, selecting from the options "
                        f"{{{options_text}}}, what do you think {target_text_user} {attribute} is? Reply the letter of the option only.")
        
        # Build the incomplete assistant response for autocompletion
        assistant_prompt = (f"Based on our conversation, selecting from the {{{options_text}}}, "
                           f"I think the {attribute} of {target_text_assistant} is")
        
        # Construct messages for probing
        probe_messages = copy.deepcopy(messages)
        # Add the user question
        probe_messages.append({"role": "user", "content": user_question})
        # Add the incomplete assistant response
        # probe_messages.append({"role": "assistant", "content": assistant_prompt})
        
        # Format with model's chat template
        try:
            if model_type == "mistral":
                formatted_prompt = mistral_v2_prompt(probe_messages)
            else:
                formatted_prompt = llama_v2_prompt(probe_messages, tokenizer=tokenizer)
            
            if "<s>" in formatted_prompt:
                formatted_prompt = formatted_prompt[formatted_prompt.find("<s>") + len("<s>"):]

            formatted_prompt = formatted_prompt + assistant_prompt
            
            # Get next token logits
            with torch.no_grad():
                inputs = tokenizer(formatted_prompt, return_tensors='pt', return_token_type_ids=False).to(model.device)
                
                # Limit input length if too long
                if inputs['input_ids'].size(1) > MaxInputTokens:
                    inputs['input_ids'] = inputs['input_ids'][:, -MaxInputTokens:]
                    if 'attention_mask' in inputs:
                        inputs['attention_mask'] = inputs['attention_mask'][:, -MaxInputTokens:]
                
                outputs = model(**inputs, output_hidden_states=False)
                next_token_logits = outputs.logits[0, -1, :]  # Last token's logits
            
            # Extract logits for A, B, C, etc.
            option_logits = []
            valid_options = []
            
            for i in range(len(labels)):
                # skip the "other" label
                if labels[i].lower() == "other":
                    continue
                letter = chr(65 + i)
                # Try to get token ID for the letter
                try:
                    # Try different tokenization approaches
                    letter_tokens = tokenizer.encode(letter, add_special_tokens=False)
                    if len(letter_tokens) > 0:
                        letter_token_id = letter_tokens[0]
                        option_logits.append(next_token_logits[letter_token_id].item())
                        valid_options.append(i)
                except:
                    # If tokenization fails, skip this option
                    continue
            
            # Apply softmax to get probabilities
            if len(option_logits) > 0:
                option_probs = torch.softmax(torch.tensor(option_logits), dim=0).cpu().numpy()
                
                # Store results - map back to all labels (0 for skipped ones)
                attr_results = {}
                prob_idx = 0
                for i, label in enumerate(labels):
                    if i in valid_options:
                        attr_results[label] = float(option_probs[prob_idx])
                        prob_idx += 1
                    else:
                        attr_results[label] = 0.0
                
                results[attribute] = attr_results
            else:
                # If no valid options, return uniform distribution
                results[attribute] = {label: 1.0/len(labels) for label in labels}
            
            # Clean up
            del inputs, outputs, next_token_logits
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"Error probing attribute {attribute}: {e}")
            # Return uniform distribution on error
            results[attribute] = {label: 1.0/len(labels) for label in labels}
    
    return results


def extract_chatbot_response(response):
    # return response["choices"][0]["message"]
#     messages = llama_v2_reverse(response)
#     if messages[-1]["role"] != "assistant":
#         raise Exception
#     else:
#         if messages[-1]["content"].rfind("</s>") != -1:
#             messages[-1]["content"] = messages[-1]["content"][:messages[-1]["content"].rfind("</s>")]
#         return messages[-1]
    if "<|end_header_id|>" in response:
        response = response[response.rfind("<|end_header_id|>") + len("<|end_header_id|>"):]
        if response.rfind("<|eot_id|>") != -1:
            response = response[:response.rfind("<|eot_id|>")]
        response = response.strip("\n")
        response = response.lstrip()
        response = response.replace("**", "")
        return {"content": response, "role": "assistant"}
    elif "<start_of_turn>model" in response:
        response = response[response.rfind("<start_of_turn>model") + len("<start_of_turn>model"):]
        if response.rfind("<end_of_turn>") != -1:
            response = response[:response.rfind("<end_of_turn>")]
        response = response.strip("\n")
        response = response.lstrip()
        response = response.replace("**", "")
        return {"content": response, "role": "assistant"}
    elif "[/INST]" in response:
        response = response[response.rfind("[/INST]") + len("[/INST]"):]
        if response.rfind("</s>") != -1:
            response = response[:response.rfind("</s>")]
        response = response.lstrip()
        return {"content": response, "role": "assistant"}
    
    
def extract_mistral_response(response):
    # return response["choices"][0]["message"]
    start_idx = response.rfind("[/INST]") + len("[/INST]")
    end_idx = response.rfind("</s>")
    if end_idx > start_idx:
        content = response[start_idx:end_idx]
    else:
        content = response[start_idx:]
    return {"role": "assistant", "content": content}
    
    
def chat_with_llama_with_intervention(messages, model, tokenizer, 
                                      classifiers=None,
                                      attribute=None,
                                      subattribute=None,
                                      N_min=-8,
                                      N_max=8,
                                      samples=5,
                                      layer_num_min=20,
                                      layer_num_max=30,
                                      residual_stream=True, 
                                      include_inst=True,
                                      attribute_targets=None,
                                      return_user_msg_last_act=True,
                                      context_length=4096,
                                      sys_prompt=None,
                                      mistral=False,
                                      model_type="",
                                      cate_labels=None):
    # Use provided cate_labels (includes extra probes) or fallback to hardcoded default
    if cate_labels is None:
        cate_labels = {"gender": ["Male", "Female", "Other"],
                       "age": ["Child", "Adolescent", "Adult", "Older Adult",],
                       "education": ["Some School", "High School", "College And More",],
                       "ethnics": ["Asian", "African", "White", 
                                   "Hispanic", "Native Americans", "Arabs", "Jews",],
                       "socioeco": ["Low", "Middle", "High"],
                       "marital": ["Single", "Married", "Divorced", "Widowed"],
                       "language": ["Chinese", "Japanese", "English", "German", "Spanish", "Portuguese", "Arabic", "Russian"],
                       "religion": ["Christian", "Islam", "Buddhism", "Hinduism", "Judaism", "Atheism", "Unknown"],
                       "political": ["Left", "Right", "Moderate", "Unknown"],
                       "sycophancy": ["Sycophancy", "nonSycophant"],
                       "hallucination": ["Hallucinated", "Factual"],
                      }
    

    if sys_prompt is None:
        sys_prompt = """You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Please ensure that your responses are socially unbiased and positive in nature. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information. Be concise in your response. Limit your response in 120 words."""
    if mistral:
        prompt = mistral_v2_prompt(messages)
    else:
        prompt = llama_v2_prompt(messages, sys_prompt, tokenizer=tokenizer)
    if "<s>" in prompt:
        prompt = prompt[prompt.find("<s>") + len("<s>"):]
    print(prompt)
    subattribute = subattribute.title()
    if ", " in subattribute:
        subattribute = subattribute.split(", ")
    
    if attribute is None:
        if isinstance(subattribute, list):
            attribute = []
            for subattr in subattribute:
                for attr in list(cate_labels.keys()):
                    if subattr in cate_labels[attr]:
                        if attr == "socioEco":
                            attr = "socioeco"
                        attribute.append(attr)
                        break
        else:
            for attr in list(cate_labels.keys()):
                if subattribute in cate_labels[attr]:
                    if attr == "socioEco":
                        attr = "socioeco"
                    attribute = attr
                    break
                    
    print(attribute, subattribute)                
    
    if (not isinstance(attribute, list)) and (not (attribute in cate_labels.keys())):
        return ["Attribute does not exist"]
    elif (not isinstance(subattribute, list)) and (not (subattribute in cate_labels[attribute])):
        return ["Subattribute does not exist"]
    
    if N_min is None:
        if isinstance(attribute, list):
            N_min = -100
            for attr in attribute:
                if default_NminNmax[attr][0] > N_min:
                    N_min = default_NminNmax[attr][0]
            N_min += len(attribute) / 2
        else:
            N_min = default_NminNmax[attribute][0]
    if N_max is None:
        if isinstance(attribute, list):
            N_max = 100
            for attr in attribute:
                if default_NminNmax[attr][1] < N_max:
                    N_max = default_NminNmax[attr][1]
            N_max -= len(attribute) / 2
        else:
            N_max = default_NminNmax[attribute][1]

    which_layers = []
    from_idx = layer_num_min
    to_idx = layer_num_max
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
    
    modified_layer_names = which_layers
    torch_device = model.device
    text_outputs = []
    
    # Verify classifiers exist for the attributes
    if isinstance(attribute, list):
        for attr in attribute:
            if attr not in classifiers:
                return [f"Classifier for attribute '{attr}' not found"]
    else:
        if attribute not in classifiers:
            return [f"Classifier for attribute '{attribute}' not found"]
    
    if isinstance(subattribute, list):
        cf_target = []
        for i in range(len(attribute)):
            cf_target.append(torch.nn.functional.one_hot(torch.Tensor([cate_labels[attribute[i]].index(subattribute[i])]).to(torch.long), 
                                                    classifiers[attribute[i]][0].proj[0].weight.shape[0]
                                                   ).to(torch_device).to(torch.float))
    else:
        cf_target = torch.nn.functional.one_hot(torch.Tensor([cate_labels[attribute].index(subattribute)]).to(torch.long), 
                                                classifiers[attribute][0].proj[0].weight.shape[0]
                                               ).to(torch_device).to(torch.float)
    for N in np.linspace(N_min, N_max, samples):
        print(N)
        print(modified_layer_names)
        print(cf_target)
        def edit_inter_rep_multi_layers(output, layer_name):
            if len(output) < 3:
                return output
            if residual_stream:
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
            cloned_inter_rep = output[0][0][-1].unsqueeze(0).to(torch.float).to(torch_device)
            with torch.no_grad():
                cloned_inter_rep = optimize_one_inter_rep(cloned_inter_rep, layer_name, 
                                                          cf_target, probe,
                                                          lr=0, max_epoch=0, 
                                                          loss_func=nn.BCELoss(),
                                                          simplified=True,
                                                          N=N,
                                                          normalized=False)
            # output[1] = cloned_inter_rep.to(torch.float16)
            # print(len(output))
            output[0][0][-1] = cloned_inter_rep[0].to(torch.float16)
            return output

        with TraceDict(model, modified_layer_names, edit_output=edit_inter_rep_multi_layers) as ret:
            with torch.no_grad():
                inputs = tokenizer(prompt, return_tensors='pt', return_token_type_ids=False).to(model.device)
                if inputs['input_ids'].size(1) > ControlledMaxInputTokens:
                    print(f"Input too long: {inputs['input_ids'].size(1)}! Hard Forgetting Initiated.")
                    inputs['input_ids'] = inputs['input_ids'][:, -ControlledMaxInputTokens:]
                if 'attention_mask' in inputs:
                    inputs['attention_mask'] = inputs['attention_mask'][:, -ControlledMaxInputTokens:]
                tokens = model.generate(
                 **inputs,
                 max_new_tokens=context_length,
                 do_sample=True,
                 # do_sample=False,
                 temperature=1.0,
                 top_p=1.0,
                 repetition_penalty=1.2,
                 output_hidden_states=True,
                 output_attentions=True,
                )
                text_output = tokenizer.decode(tokens[0], skip_special_tokens=False)
                text_outputs.append(text_output)
                
            torch.cuda.empty_cache()
            
    return text_outputs
        
    
def control_chat_with_llama(messages, model, tokenizer, 
                              classifiers=None,
                              attributes_info=None,
                              layer_num_min=20,
                              layer_num_max=30,
                              residual_stream=True, 
                              include_inst=True,
                              return_user_msg_last_act=True,
                              context_length=2048,
                              sys_prompt=None,
                              special_prompt=None,
                              multi_special_prompt=None,
                              system_special_prompt=None,
                              mistral=False,
                              model_type="",
                              attribute_targets=None,
                              cate_labels=None,
                           ):
    edit_activations = True
    
    # Use provided cate_labels (includes extra probes) or fallback to hardcoded default
    if cate_labels is None:
        cate_labels = {
            "gender": ["male", "female", "other"],
            "age": ["child", "adolescent", "adult", "olderAdult",],
            "ethnics": ["asian", "african", "white", "hispanic", "nativeAmerican", "arab", "jews",],
            "socioEco": ["low", "middle", "high",],
            "marital": ["single", "married", "divorced", "widowed"],
            "education": ["someschool", "highschool", "collegemore",],
            "language": ["chinese", "japanese", "english", "german", "spanish", "portuguese", "arabic", "russian"],
            "religion": ["christianity", "islam", "buddhism", "hinduism", "judaism", "atheism",],
            "political": ["left", "right", "moderate", "unknown"],
            "sycophancy": ["sycophancy", "nonsycophant"],
            "hallucination": ["hallucinated", "factual"],
        }

    
    if sys_prompt is None:
        sys_prompt = """You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Please ensure that your responses are socially unbiased and positive in nature. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information. Be concise in your response. Limit your response in 120 words."""
    if mistral:
        prompt = mistral_v2_prompt(messages)
    else:
#         if len(messages) > 5:
#             messages = messages[:-5]
        prompt = llama_v2_prompt(messages, sys_prompt, tokenizer=tokenizer)
    if "<s>" in prompt:
        prompt = prompt[prompt.find("<s>") + len("<s>"):]
    
    # Placeholder for final intervention targets
    cf_targets = []
    Ns = []
    attribute = []

    # Iterate over the attributes_info dictionary
    for attr, subattrs in attributes_info.items():
        for subattribute, N in subattrs.items():
            if N == 0:
                continue
            # Validate attribute and subattribute
            if attr not in cate_labels.keys() or subattribute not in cate_labels[attr]:
                raise Exception(f"Invalid attribute or subattribute: {attr}, {subattribute}")

            # Convert subattribute to index
            subattribute_idx = cate_labels[attr].index(subattribute)

            # Build cf_target for the current attribute and subattribute
            # Handle attribute name mapping for classifier lookup
            if attr == "socioEco":
                clf_attr = "socioeco"
            elif attr in classifiers:
                # For extra probes, use attribute name directly (meta_attribute is already the key)
                clf_attr = attr
            
            # Verify the classifier exists, otherwise skip this attribute
            if clf_attr not in classifiers:
                print(f"Warning: Classifier for '{clf_attr}' (original: '{attr}') not found, skipping intervention")
                continue
            
            cf_target = torch.nn.functional.one_hot(
                torch.tensor([subattribute_idx], dtype=torch.long),
                num_classes=classifiers[clf_attr][0].proj[0].weight.shape[0]
            ).to(torch.float)

            # Append to cf_targets with corresponding N value
            cf_targets.append(cf_target)
            Ns.append(N)
            attribute.append(clf_attr)

    print(cf_targets, Ns, attribute)
    which_layers = []
    from_idx = layer_num_min
    to_idx = layer_num_max
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
    
    modified_layer_names = which_layers
    torch_device = model.device
    text_outputs = []
    print(modified_layer_names)
    def edit_inter_rep_multi_layers(output, layer_name):
        if len(output) < 3:
            return output
        if residual_stream:
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
#         cloned_inter_rep = output[0][0][-1].unsqueeze(0).detach().clone().to(torch.float)
        cloned_inter_rep = output[0][0][-1].unsqueeze(0).to(torch.float).to(torch_device)
        with torch.no_grad():
            cloned_inter_rep = optimize_one_inter_rep(cloned_inter_rep, layer_name, 
                                                      cf_targets, probe,
                                                      lr=0, max_epoch=0, 
                                                      loss_func=nn.BCELoss(),
                                                      simplified=True,
                                                      N=Ns,
                                                      normalized=False)
        # output[1] = cloned_inter_rep.to(torch.float16)
        # print(len(output))
        output[0][0][-1] = cloned_inter_rep[0].to(torch.float16)
        return output

    with TraceDict(model, modified_layer_names, edit_output=edit_inter_rep_multi_layers) as ret:
        with torch.no_grad():
            inputs = tokenizer(prompt, return_tensors='pt', return_token_type_ids=False).to(model.device)
            
            if inputs['input_ids'].size(1) > ControlledMaxInputTokens:
                print(f"Input too long: {inputs['input_ids'].size(1)}! Hard Forgetting Initiated.")
                inputs['input_ids'] = inputs['input_ids'][:, -ControlledMaxInputTokens:]
                if 'attention_mask' in inputs:
                    inputs['attention_mask'] = inputs['attention_mask'][:, -ControlledMaxInputTokens:]
            tokens = model.generate(
             **inputs,
             max_new_tokens=context_length,
             do_sample=True,
             # do_sample=False,
             temperature=0.1,
             top_p=1.0,
             repetition_penalty=1.2,
             output_attentions=True,
             return_dict_in_generate=True, 
             output_scores=True,
            )
            scores = tokens[1]
            scores = torch.concat(scores)
            shannon_entropy = calculate_shannon_entropy_pytorch(scores)
            tokens = tokens[0]
            text_output = tokenizer.decode(tokens[0], skip_special_tokens=False)
            text_outputs.append(text_output)
            
        torch.cuda.empty_cache()
            
        ai_response = []
        try:
            for response in text_outputs:
                print(response)
                if mistral:
                    ai_response.append(extract_mistral_response(response)["content"])
                else:
                    ai_response.append(extract_chatbot_response(response)["content"])
        except Exception as e:
            print(e)

        new_chat_history = messages + [{"role":"assistant", "content": ai_response[0]}]
        print("Get ACTIVATION")

        

        last_acts = get_activation_from_llama(new_chat_history, model, tokenizer, residual_stream=True, include_inst=True,
                                              return_user_msg_last_act=True, context_length=2048, special_prompt=special_prompt,
                                              multi_special_prompt=multi_special_prompt,
                                              system_special_prompt=system_special_prompt,
                                              mistral=mistral, intervened=True,
                                              model_type=model_type, attribute_targets=attribute_targets)
        
        # Also get prompt-based attributes
        print("Get PROMPT-BASED ATTRIBUTES")
        prompt_based_results = get_prompt_based_attributes(new_chat_history, model, tokenizer, 
                                                           cate_labels=cate_labels,
                                                           attribute_targets=attribute_targets,
                                                           model_type=model_type)
    del inputs, tokens
    torch.cuda.empty_cache()
    
    return text_outputs, shannon_entropy, last_acts, prompt_based_results
    # return text_outputs, shannon_entropy
    