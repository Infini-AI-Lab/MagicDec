import torch
from eval_utils import create_prompt

def truncate_input(input: list, max_length: int, manner="middle"):
    if len(input) <= max_length:
        return input
    if manner == "middle":
        split = max_length // 2
        return input[0:split] + input[-split:]
    else:
        return None


def truncate_by_tokens(input, tok, max_tokens, manner: str = "middle"):
    tokens = tok.encode(input)
    len_before = len(tokens)
    print(f"# tokens before: {len_before}")
    tokens = truncate_input(tokens, max_length=max_tokens, manner=manner)
    len_after = len(tokens)  # type: ignore
    print(f"# tokens after: {len_after}")
    assert len_after <= len_before
    assert len_after <= max_tokens
    return tok.decode(tokens, skip_special_tokens=True)

# for mathcalc
def prepare_data(examples, tokenizer, data_name, model_name, prefix_len, data_dir, start_idx, stop_idx):
    tokenized_prompts = []  
    for i in range(start_idx, stop_idx):
        eg = examples[i]
        input_text = create_prompt(eg, data_name, model_name, data_dir)
        # input_text = input_text.split("\n")[0]
        input_text = truncate_by_tokens(input_text, tokenizer, prefix_len, manner="middle")
        tokenized_prompt = tokenizer.encode(input_text, return_tensors="pt")[:,:prefix_len]
        tokenized_prompts.append(tokenized_prompt)
    return tokenized_prompts