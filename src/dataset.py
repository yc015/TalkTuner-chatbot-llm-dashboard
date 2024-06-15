import os
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
import torch.nn.functional as F

import torch
from tqdm.auto import tqdm
from collections import OrderedDict


class ModuleHook:
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)
        self.module = None
        self.features = []

    def hook_fn(self, module, input, output):
        self.module = module
        self.features.append(output.detach())

    def close(self):
        self.hook.remove()
        
        
def remove_last_k_words(s, k):
    """
    Remove the last k words from the string s.
    Any words that appear before the last occurrence of "[INST]" will not be removed.
    """
    
    # Split string into words
    words = s.split()
    
    # Find the last occurrence of "[INST]"
    if "[/INST]" in words:
        last_inst_index = max([i for i, word in enumerate(words) if word == "[/INST]"])
    else:
        last_inst_index = -1
    
    # If k words to be removed are less than words after last INST, remove those words.
    # Otherwise, keep the words up to and including INST and remove words after that.
    if len(words) - last_inst_index - 1 > k:
        return ' '.join(words[:-k])
    else:
        return ' '.join(words[:last_inst_index+1])
        

def split_conversation(text, user_identifier="HUMAN:", ai_identifier="ASSISTANT:"):
    user_messages = []
    assistant_messages = []

    lines = text.split("\n")

    current_user_message = ""
    current_assistant_message = ""

    for line in lines:
        line = line.lstrip(" ")
        if line.startswith(user_identifier):
            if current_assistant_message:
                assistant_messages.append(current_assistant_message.strip())
                current_assistant_message = ""
            current_user_message += line.replace(user_identifier, "").strip() + " "
        elif line.startswith(ai_identifier):
            if current_user_message:
                user_messages.append(current_user_message.strip())
                current_user_message = ""
            current_assistant_message += line.replace(ai_identifier, "").strip() + " "

    if current_user_message:
        user_messages.append(current_user_message.strip())
    if current_assistant_message:
        assistant_messages.append(current_assistant_message.strip())

    return user_messages, assistant_messages


def llama_v2_prompt(
    messages: list[dict],
    system_prompt=None
):
    B_INST, E_INST = "[INST]", "[/INST]"
    B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
    BOS, EOS = "<s>", "</s>"
    if system_prompt:
        DEFAULT_SYSTEM_PROMPT = system_prompt
    else:
        DEFAULT_SYSTEM_PROMPT = f"""You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Please ensure that your responses are socially unbiased and positive in nature. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""

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
    if messages[-1]["role"] == "user":
        messages_list.append(f"{BOS}{B_INST} {(messages[-1]['content']).strip()} {E_INST}")

    return "".join(messages_list)  


prompt_translator = {"_age_": "age",
                     "_gender_": "gender",
                     "_socioeco_": "socioeconomic status",
                     "_education_": "education level",}


class TextDataset(Dataset):
    def __init__(self, directory, tokenizer, model, label_idf="_age_", label_to_id=None, 
                 convert_to_llama2_format=False, user_identifier="HUMAN:", ai_identifier="ASSISTANT:", control_probe=False,
                 additional_datas=None, residual_stream=False, new_format=False, if_augmented=False, k=20,
                 remove_last_ai_response=False, include_inst=False, one_hot=False, last_tok_pos=-1):
        self.file_paths = [os.path.join(directory, f) for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f)) and f.endswith('.txt')]
        self.tokenizer = tokenizer
        self.labels = []
        self.acts = []
        self.texts = []
        self.label_idf = label_idf
        self.label_to_id = label_to_id
        self.model = model
        self.convert_to_llama2_format = convert_to_llama2_format
        self.user_identifier = user_identifier
        self.ai_identifier = ai_identifier
        self.additional_datas = additional_datas
        self.residual_stream = residual_stream
        self.new_format = new_format
        self.if_augmented = if_augmented
        self.k = k
        self.if_remove_last_ai_response = remove_last_ai_response
        self.include_inst = include_inst
        self.one_hot = one_hot
        self.last_tok_pos = last_tok_pos
        self.control_probe = control_probe
        if self.additional_datas:
            for directory in self.additional_datas:
                self.file_paths += [os.path.join(directory, f) for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f)) and f.endswith('.txt')]
        self._load_in_data()

    def __len__(self):
        return len(self.texts)
    
    def _load_in_data(self):
        for idx in tqdm(range(len(self.file_paths))):
            file_path = self.file_paths[idx]
            corrupted_file_paths = []

            int_idx = file_path[file_path.find("conversation_")+len("conversation_"):]
            int_idx = int(int_idx[:int_idx.find("_")])
            
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
                
            if self.convert_to_llama2_format:
                if "### Human:" in text:
                    user_msgs, ai_msgs = split_conversation(text, "### Human:", "### Assistant:")
                elif "### User:" in text:
                    user_msgs, ai_msgs = split_conversation(text, "### User:", "### Assistant:")
                else:
                    user_msgs, ai_msgs = split_conversation(text, self.user_identifier, self.ai_identifier)
                messages_dict = []

                for user_msg, ai_msg in zip(user_msgs, ai_msgs):
                    messages_dict.append({'content': user_msg, 'role': 'user'})
                    messages_dict.append({'content': ai_msg, 'role': 'assistant'})
                    
                if len(messages_dict) < 1:
                    corrupted_file_paths.append(file_path)
                    print(f"Corrupted file at {file_path}")
                    continue
                    
                if self.if_remove_last_ai_response and messages_dict[-1]["role"] == "assistant":
                    messages_dict = messages_dict[:-1]
                try:
                    text = llama_v2_prompt(messages_dict) 
                except:
                    corrupted_file_paths.append(file_path)
                    print(f"Corrupted file at {file_path}")
                    continue
              
            if self.new_format and self.if_remove_last_ai_response and self.include_inst:
                text = text[text.find("<s>") + len("<s>"):]
            elif self.new_format and self.include_inst:
                text = text[text.find("<s>") + len("<s>"):]
            elif self.new_format:
                text = text[text.find("<s>") + len("<s>"): text.rfind("[/INST]") - 1]
            
            label = file_path[file_path.rfind(self.label_idf) + len(self.label_idf):file_path.rfind(".txt")]
            
            if label not in self.label_to_id.keys():
                continue
                
            if self.label_to_id:
                label = self.label_to_id[label]
                
            if self.one_hot:
                label = F.one_hot(torch.Tensor([label]).to(torch.long), len(self.label_to_id.keys()))
                
            
            if not self.control_probe:
                text += f" I think the {prompt_translator[self.label_idf]} of this user is"
            with torch.no_grad():
                encoding = self.tokenizer(
                  text,
                  truncation=True,
                  max_length=2048,
                  return_attention_mask=True,
                  return_tensors='pt'
                )
                
                features = OrderedDict()
                for name, module in self.model.named_modules():
                    if name.endswith(".mlp") or name.endswith(".embed_tokens"):
                        features[name] = ModuleHook(module)
                        
                output = self.model(input_ids=encoding['input_ids'].to("cuda"),
                                    attention_mask=encoding['attention_mask'].to("cuda"),
                                    output_hidden_states=True,
                                    return_dict=True)
                for feature in features.values():
                    feature.close()
            
            last_acts = []
            if self.if_augmented:
                if self.residual_stream:
                    for layer_num in range(41):
                        last_acts.append(output["hidden_states"][layer_num][:, -self.k:].detach().cpu().clone().to(torch.float))
                    last_acts = torch.cat(last_acts, dim=0)
                else:
                    last_acts.append(features['model.embed_tokens'].features[0][:, -self.k:].detach().cpu().clone().to(torch.float))
                    for layer_num in range(1, 41):
                        last_acts.append(features[f'model.layers.{layer_num - 1}.mlp'].features[0][:, -self.k:].detach().cpu().clone().to(torch.float))
                    last_acts = torch.cat(last_acts, dim=0)
            else:
                if self.residual_stream:
                    for layer_num in range(41):
                        last_acts.append(output["hidden_states"][layer_num][:, -1].detach().cpu().clone().to(torch.float))
                    last_acts = torch.cat(last_acts)
                else:
                    last_acts.append(features['model.embed_tokens'].features[0][:, -1].detach().cpu().clone().to(torch.float))
                    for layer_num in range(1, 41):
                        last_acts.append(features[f'model.layers.{layer_num - 1}.mlp'].features[0][:, -1].detach().cpu().clone().to(torch.float))
                    last_acts = torch.cat(last_acts)
            
            self.texts.append(text)
            self.labels.append(label)
            self.acts.append(last_acts)
            
        for path in corrupted_file_paths:
            self.file_paths.remove(path)
    def __getitem__(self, idx):
        label = self.labels[idx]
        text = self.texts[idx]
 
        if self.if_augmented:
            random_k = torch.randint(0, self.k, [1])[0].item()
            hidden_states = self.acts[idx][:, -random_k]
        else:
            hidden_states = self.acts[idx]
        
        return {
            'hidden_states': hidden_states,
            'file_path': self.file_paths[idx],
            'age': label,
            'text': text,
        }
    