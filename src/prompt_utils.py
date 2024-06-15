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
    # DEFAULT_SYSTEM_PROMPT = """You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Please ensure that your responses are socially unbiased and positive in nature. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""

    # Split by the message separators
    # prompt = promp
    # segments = [s.strip() for s in prompt.split(E_INST) if s.strip()]

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