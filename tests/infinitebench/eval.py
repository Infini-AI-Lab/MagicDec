import time
import torch
import sys
sys.path.append("..")
from pathlib import Path
import torch.distributed as dist
from MagicDec.Engine.utils import setup_seed, cuda_graph_for_sampling_argmax_batch, sampling_argmax_batch
from MagicDec.Data.data_converter import convert_pg19_dataset
from transformers import AutoTokenizer
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
import argparse
from MagicDec.Engine.SnapKV.backend import LMBackend

import json
from pathlib import Path
import time
from typing import List, Tuple, Any

from torch import Tensor
from transformers.modeling_outputs import BaseModelOutputWithPast

from eval_utils import (
    dump_jsonl,
    create_prompt,
    load_data,
    get_answer,
    DATA_NAME_TO_MAX_NEW_TOKENS,
)
from prepare_data import prepare_data

parser = argparse.ArgumentParser(description='Process model configuration and partitions.')
parser.add_argument('--model', type=Path, default=Path("/scratch/models/meta-llama/Meta-Llama-3.1-8B/model.pth"), help='model')
parser.add_argument('--model_name', type=str, default="meta-llama/Meta-Llama-3.1-8B", help='model name')
parser.add_argument('--dataset', type=str, default="pg19", help='Dataset name.')
parser.add_argument('--draft_budget', type=int, default=4097, help='Dataset end index.')
parser.add_argument('--rank_group', nargs='+', type=int, help='Target group of ranks')
parser.add_argument('--compile', action='store_true', help='Whether to compile the model.')

parser.add_argument('--gamma', type=int, default=7, help='start')

parser.add_argument('--B', type=int, default=45, help='Batch size.')
parser.add_argument('--prefix_len', type=int, default=100000, help='Prefix length')
parser.add_argument('--max_len', type=int, default=100096, help='Generate length')
parser.add_argument('--window_size', type=int, default=32, help='Generate length')

parser.add_argument('--seed', type=int, default=123, help='Random seed.')

parser.add_argument('--printoutput', action='store_true', help='Whether to compile the model.')
parser.add_argument('--benchmark', action='store_true', help='Whether to compile the model.')

parser.add_argument(
        "--task",
        type=str,
        required=True,
        help="Which task to use. Note that \"all\" can only be used in `compute_scores.py`.",  # noqa
)
parser.add_argument('--data_dir', type=str, default="../data", help='Data directory.')
parser.add_argument('--output_dir', type=str, default="../results", help='Output directory.')
parser.add_argument('--start_idx', type=int, default=0, help='Dataset start index.')
parser.add_argument('--stop_idx', type=int, help='Dataset end index.')
parser.add_argument('--verbose', action='store_true')



args = parser.parse_args()

MAX_POSITION_ID = 200000  # Determined by the model
TRUNCATE_LEN = 200000

# sampling_params = SamplingParams(temperature=0.8, top_p=0.95)


def get_pred(
    model,
    tok: AutoTokenizer,
    input_text: str,
    max_tokens: int,
    verbose: bool = False,
) -> str:
    """
    Truncate down to 128k then make inference.
    """
    print("Truncating...")
    input_text = truncate_by_tokens(input_text, tok, TRUNCATE_LEN)
    if verbose:
        print("# chars:", len(input_text))
        print("=============== Input ===============")
        print(input_text[:200])
        print("...")
        print(input_text[-200:])
        print("=====================================")
    outputs = model.generate([input_text], sampling_params)

    output = outputs[0].outputs[0].text
    print("Chunked generation:", output)
    return output


def load_model(
    model_name: str = "../../../yarn-mistral-7b-128k",
    ngpu=8,
):
    print("Loading tokenizer")
    tok = AutoTokenizer.from_pretrained(model_name)
    tok.pad_token = tok.eos_token
    print("Loading model")
    start_time = time.time()
    llm = LLM(model=model_name, tensor_parallel_size=ngpu)
    print("Time taken:", round(time.time() - start_time))
    return llm, tok  # type: ignore

########## magicdec ##########
args = parser.parse_args()
assert args.prefix_len < args.max_len
assert (args.prefix_len - args.window_size) % 128 == 0
# assert args.max_len % 128 == 0
assert (args.max_len + 127) // 128 == args.prefix_len // 128 + 1
assert (args.draft_budget - 1) % 128 == 0

# Init model parallelism
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
global print
from MagicDec.Engine.tp import init_dist
use_tp = len(args.rank_group) > 1
global_group = None
rank = 0
if use_tp:
    rank, global_group = init_dist()
    if rank != args.rank_group[0]:
        print = lambda *args, **kwargs: None

if rank == 0:
    with open("result.txt", "a") as file:
        file.write(f"SnapKV-Selfspec: Prefix:{args.prefix_len}; Bsz:{args.B}; Gamma:{args.gamma}; Draft budget:{args.draft_budget}\n")

setup_seed(args.seed)
print(f"Using device={DEVICE}")

MAX_LEN_TARGET = args.max_len
DTYPE = torch.bfloat16
BATCH_SIZE = args.B
benchmark = args.benchmark
checkpoint_path = args.model

target_dec_len = args.gamma + 1
draft_dec_len = 1

# Load target model
engine = LMBackend(dtype=DTYPE, device=DEVICE, dec_len=target_dec_len, draft_dec_len=draft_dec_len)
engine.load_model(checkpoint_path, use_tp=use_tp, rank_group = args.rank_group, group=global_group)
vocab_size = engine.model.config.vocab_size
if args.compile:
    engine.compile()
engine.setup_caches(max_batch_size=BATCH_SIZE, max_seq_length=MAX_LEN_TARGET, draft_budget=args.draft_budget, window_size=args.window_size)

# Load dataset
tokenizer = AutoTokenizer.from_pretrained(args.model_name)
tokenizer.pad_token = tokenizer.eos_token
eot_1 = tokenizer.eos_token_id
if tokenizer.unk_token_id is not None:
    eot_2 = tokenizer.unk_token_id
else:
    eot_2 = tokenizer.encode("<|eot_id|>")[-1]
print(f"eot_1: {eot_1}, eot_2: {eot_2}")

# print(json.dumps(vars(args), indent=4))
data_name = args.task

# sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
# Data
result_dir = Path(args.output_dir, args.model_name)
result_dir.mkdir(exist_ok=True, parents=True)
examples = load_data(data_name, data_dir=args.data_dir)

if args.stop_idx is None:
    args.stop_idx = len(examples)
    output_path = (
        result_dir / f"preds_{data_name}.jsonl"
    )
else:
    output_path = (
        result_dir / f"preds_{data_name}_{args.start_idx}-{args.stop_idx}.jsonl"  # noqa
    )

preds = []
print("==== Evaluation ====")
print(f"# examples: {len(examples)}")
print(f"Start index: {args.start_idx}")
print(f"Stop index: {args.stop_idx}")
print(f"Verbose: {args.verbose}")
print(f"Max tokens: {MAX_LEN_TARGET}")
# for i in range(args.start_idx, args.stop_idx):
#     eg = examples[i]
#     input_text = create_prompt(eg, data_name, model_name, args.data_dir)
#     print(f"====== Example {i} ======")
#     pred = get_pred(
#         model, tok, input_text, max_tokens=max_tokens, verbose=args.verbose
#     )
#     if args.verbose:
#         print(pred)
#     preds.append(
#         {
#             "id": i,
#             "prediction": pred,
#             "ground_truth": get_answer(eg, data_name),
#         }
#     )
#     dump_jsonl(preds, output_path)

# prepare dataset
dataset = prepare_data(examples, tokenizer, data_name, args.model_name, args.prefix_len, args.data_dir, args.start_idx, args.stop_idx)

dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)
num_eval_steps = min(10, len(dataloader))

total_time = 0.0
num_gen_tokens = 0
target_steps = 0
if benchmark:
    draft_time = 0.0
    target_time = 0.0
    verify_loop = 0.0

for step, batch in tqdm(enumerate(dataloader), total=num_eval_steps):
    if step >= num_eval_steps:
        break
    input_ids = batch[0].to(DEVICE)
    terminal = False
    tokens_buffer= torch.zeros((BATCH_SIZE, args.gamma+1), device=DEVICE).long()
    output = torch.zeros(BATCH_SIZE, MAX_LEN_TARGET+1, device=DEVICE).long()
    output[:, :input_ids.shape[1]] = input_ids
    num_nodes = torch.zeros(BATCH_SIZE,device=DEVICE).long()
    num_nodes += input_ids.shape[1]

    tokens_buffer[:, :1] = engine.encode(input_ids=input_ids)[:,-1:]

    torch.cuda.synchronize()
    start = time.perf_counter()
    while terminal == False:

        # Draft speculation
        if benchmark:
            torch.cuda.synchronize()
            t1 = time.time()

        for i in range(args.gamma):
            tokens_buffer[:,i+1:i+2] = engine.speculate(tokens_buffer[:, i].view(-1,1))

        if benchmark:
            torch.cuda.synchronize()
            t2 = time.time()
            draft_time+=t2-t1

        # Target Verification
        target_tokens = engine.verify(tokens_buffer)

        if benchmark:
            torch.cuda.synchronize()
            t3 = time.time()
            target_time+=t3-t2

        target_steps+=1

    # Verification
        # Vectorized Verify Loop
        draft_tokens = tokens_buffer[:, 1:args.gamma+1]
        flag_accept_matrix = (target_tokens[:, :args.gamma] == draft_tokens)  # shape: (BATCH_SIZE, gamma)
        eot_condition = ((draft_tokens == eot_1) | (draft_tokens == eot_2))  # shape: (BATCH_SIZE, gamma)

        # Compute accept_flags by considering both the acceptance condition and EOT tokens
        accept_flags_int = (flag_accept_matrix & (~eot_condition)).int()
        accept_flags_cumprod = torch.cumprod(accept_flags_int, dim=1)
        accept_flags_matrix = accept_flags_cumprod.bool()

        # Compute the number of accepted tokens
        accept_nums = accept_flags_matrix.sum(dim=1, keepdim=True) + 1  # shape: (BATCH_SIZE, 1)

        # Check for termination conditions
        condition = (eot_condition & accept_flags_matrix).any(dim=1, keepdim=True)
        if condition.any():
            terminal = True
        
        # Rollback the memory length
        engine.cachelens = engine.cachelens - args.gamma - 1
        engine.paged_kv_last_page_len = engine.paged_kv_last_page_len - args.gamma - 1
        engine.draft_cachelens = engine.draft_cachelens - args.gamma -1
        engine.draft_paged_kv_last_page_len = engine.draft_paged_kv_last_page_len - args.gamma -1

        # Put the accepted tokens to output
        positions = torch.arange(output.shape[1], device=DEVICE).view(1, -1).repeat(BATCH_SIZE, 1)
        mask = (positions < (engine.cachelens.view(-1,1) + accept_nums)) & (positions >= engine.cachelens.view(-1, 1))
        positions_buffer = torch.arange(args.gamma+1, device=DEVICE).view(1, -1).repeat(BATCH_SIZE, 1)
        mask_buffer = positions_buffer<accept_nums.view(-1,1)
        output[mask] = tokens_buffer[mask_buffer]

        # Set the cache length to the accepted length
        engine.cachelens += accept_nums.flatten().to(torch.int32)
        engine.paged_kv_last_page_len += accept_nums.flatten().to(torch.int32)
        engine.draft_cachelens += accept_nums.flatten().to(torch.int32)
        engine.draft_paged_kv_last_page_len += accept_nums.flatten().to(torch.int32)
        
        # Get the bonus tokens
        indices = accept_nums - 1
        bonus_tokens = target_tokens.gather(1, indices)
        if (bonus_tokens == eot_1).any() or (bonus_tokens == eot_2).any():
            terminal = True
        num_nodes += accept_nums.flatten()

        # Check Number of Nodes + Bonus Token <= max_target_token
        # if num_nodes.max() + 1 >= args.prefix_len + gen_len:
        # if num_nodes.max() + 1 + args.gamma > MAX_LEN_TARGET:
        if num_nodes.max() - args.prefix_len >= 80:
            terminal = True
        # Put Bonus tokens to the tokens buffer, and prepare the variables for next itr
        if not terminal:
            tokens_buffer[:, :1] = bonus_tokens
        
        if not terminal:
            if benchmark:
                torch.cuda.synchronize()
                t4 = time.time()
                verify_loop += t4-t3
        else:
            for i in range(BATCH_SIZE):
                output[i, num_nodes[i]] = bonus_tokens[i]
            num_nodes += 1
            if benchmark:
                torch.cuda.synchronize()
                t4 = time.time()
                verify_loop += t4-t3

    torch.cuda.synchronize()
    end=time.perf_counter()
    total_time += end-start
    num_gen_tokens += (num_nodes.sum() - (input_ids.shape[1] + 1) * BATCH_SIZE)
    if args.printoutput:
        for i in range(BATCH_SIZE):
            print("Sequence ", i)
            print(tokenizer.decode(output[i, args.prefix_len:num_nodes[i]]))
    print("total time :{:.5f}s, time per iter :{:.5f}s, decoding step: {}, large model step: {}".format(total_time, total_time / target_steps, num_gen_tokens, target_steps))
    if benchmark:
        print("target time :{:.5f}s, draft time :{:.5f}s, verify loop : {}, avg generate len per sentence: {}".format(target_time/target_steps, draft_time / target_steps, verify_loop/target_steps, num_gen_tokens/target_steps/BATCH_SIZE))
    if step < 5:   # TODO: revert to 10?
        total_time = 0.0
        num_gen_tokens = 0
        target_steps = 0
        if benchmark:
            draft_time = 0.0
            target_time = 0.0
            verify_loop = 0.0
    if use_tp:
        dist.barrier()

if rank == 0:
    with open("result.txt", "a") as file:
        file.write("total time :{:.5f}s, time per iter :{:.5f}s, decoding step: {}, large model step: {}, avg latency: {} \n".format(total_time, total_time / target_steps, num_gen_tokens, target_steps, total_time / num_gen_tokens * BATCH_SIZE))
        file.write("target time :{:.5f}s, draft time :{:.5f}s, verify loop : {}, avg generate len per sentence: {} \n".format(target_time/target_steps, draft_time / target_steps, verify_loop/target_steps, num_gen_tokens/target_steps/BATCH_SIZE))