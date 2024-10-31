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
from MagicDec.Engine.backend_spec import LMBackend, LMBackendDraft


parser = argparse.ArgumentParser(description='Process model configuration and partitions.')
parser.add_argument('--model', type=Path, default=Path("/home/rsadhukh/opensource/specdec/gpt-fast/checkpoints/meta-llama/Meta-Llama-3.1-8B/model.pth"), help='model')
parser.add_argument('--draft', type=Path, default=Path("/home/rsadhukh/opensource/specdec/gpt-fast/checkpoints/meta-llama/Meta-Llama-3.1-8B/model.pth"), help='draft model')
parser.add_argument('--model_name', type=str, default="meta-llama/Meta-Llama-3.1-8B", help='model name')
parser.add_argument('--dataset', type=str, default="pg19", help='Dataset name.')
parser.add_argument('--draft_budget', type=int, default=-1, help='Dataset end index.')
parser.add_argument('--rank_group', nargs='+', type=int, help='Target group of ranks')
parser.add_argument('--compile', action='store_true', help='Whether to compile the model.')

parser.add_argument('--gamma', type=int, default=7, help='start')

parser.add_argument('--B', type=int, default=4, help='Batch size.')
parser.add_argument('--prefix_len', type=int, default=4128, help='Prefix length')
parser.add_argument('--max_len', type=int, default=4224, help='Generate length')
parser.add_argument('--window_size', type=int, default=32, help='Generate length')

parser.add_argument('--seed', type=int, default=123, help='Random seed.')

parser.add_argument('--printoutput', action='store_true', help='Whether to compile the model.')
parser.add_argument('--benchmark', action='store_true', help='Whether to compile the model.')

args = parser.parse_args()
assert args.prefix_len < args.max_len
assert (args.prefix_len - args.window_size) % 128 == 0
assert args.max_len % 128 == 0
if args.draft_budget > 0:
    assert (args.draft_budget - 1) % 128 == 0

# Init model parallelism
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
global print
from MagicDec.Engine.tp import init_dist
use_tp = len(args.rank_group) > 1
global_group = None
if use_tp:
    rank, global_group = init_dist()
    if rank != args.rank_group[0]:
        print = lambda *args, **kwargs: None

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
engine = LMBackend(dtype=DTYPE, device=DEVICE, dec_len=target_dec_len)
engine.load_model(checkpoint_path, use_tp=use_tp, rank_group = args.rank_group, group=global_group)
vocab_size = engine.model.config.vocab_size

# load draft model
draft_engine = LMBackendDraft(dtype=DTYPE, device=DEVICE, dec_len=draft_dec_len)
draft_engine.load_model(args.draft, use_tp=use_tp, rank_group = args.rank_group, group=global_group)

if args.compile:
    engine.compile()
    draft_engine.compile()

# Setup caches
engine.setup_caches(max_batch_size=BATCH_SIZE, max_seq_length=MAX_LEN_TARGET)
draft_engine.setup_caches(max_batch_size=BATCH_SIZE, max_seq_length=MAX_LEN_TARGET) # no sparse attention as of now

# Load dataset
tokenizer = AutoTokenizer.from_pretrained(args.model_name)
tokenizer.pad_token = tokenizer.eos_token
eot_1 = tokenizer.eos_token_id
if tokenizer.unk_token_id is not None:
    eot_2 = tokenizer.unk_token_id
else:
    eot_2 = tokenizer.encode("<|eot_id|>")[-1]
print(f"eot_1: {eot_1}, eot_2: {eot_2}")

if args.dataset == "pg19":
    dataset = convert_pg19_dataset(tokenizer=tokenizer, seq_len=args.prefix_len)
elif args.dataset.startswith("ruler"):
    dataset = convert_ruler_dataset(tokenizer=tokenizer, task=args.dataset.split(":")[1], model_name=args.model_name, seq_len=args.prefix_len)
else:
    raise ValueError(f"Unknown dataset {args.dataset}")
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)
num_eval_steps = min(50, len(dataloader))

total_time = 0.0
num_gen_tokens = 0
target_steps = 0
if benchmark:
    draft_time = 0.0
    target_time = 0.0
    verify_loop = 0.0

for step, batch in tqdm(enumerate(dataloader), total=num_eval_steps):
    # if step < 26:
    #     continue
    if step >= num_eval_steps:
        break
    input_ids = batch[0].to(DEVICE)
    terminal = False
    tokens_buffer= torch.zeros((BATCH_SIZE, args.gamma+1), device=DEVICE).long()
    output = torch.zeros(BATCH_SIZE, MAX_LEN_TARGET+1, device=DEVICE).long()
    output[:, :input_ids.shape[1]] = input_ids
    num_nodes = torch.zeros(BATCH_SIZE,device=DEVICE).long()
    num_nodes += input_ids.shape[1]

    tokens_buffer[:, :1] = engine.encode(input_ids)[:, -1:]
    # draft prefill
    draft_engine.encode(input_ids=input_ids)

    # TODO: simplify code
    next_double = False
    double_buffer = None

    torch.cuda.synchronize()
    start = time.perf_counter()
    while terminal == False:

        # Draft speculation
        if benchmark:
            torch.cuda.synchronize()
            t1 = time.time()

        for i in range(args.gamma):
            if i == 0 and next_double:
                next_tokens = draft_engine.inference(double_buffer)
                tokens_buffer[:, i+1:i+2] = next_tokens.gather(1, cachelens_update.view(-1,1) - 1)
                draft_engine.cachelens = draft_engine.cachelens - (2 - cachelens_update)
                next_double = False
            else:
                tokens_buffer[:, i+1:i+2] = draft_engine.inference(tokens_buffer[:, i:i+1])

        if benchmark:
            torch.cuda.synchronize()
            t2 = time.time()
            draft_time += t2-t1

        # Target verification
        target_tokens = engine.inference(tokens_buffer)
        # target_tokens = []
        # for i in range(args.gamma+1):
        #     target_tokens.append(engine.inference(tokens_buffer[:, i:i+1]))
        # target_tokens = torch.cat(target_tokens, dim=1)

        if benchmark:
            torch.cuda.synchronize()
            t3 = time.time()
            target_time+=t3-t2

        target_steps+=1

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

        # if (accept_nums != args.gamma + 1).any():
        #     import pdb; pdb.set_trace()

        # Check for termination conditions
        condition = (eot_condition & accept_flags_matrix).any(dim=1, keepdim=True)
        if condition.any():
            terminal = True

        # Rollback the memory length
        engine.cachelens = engine.cachelens - args.gamma - 1
        engine.paged_kv_last_page_len = engine.paged_kv_last_page_len - args.gamma - 1
        draft_engine.cachelens = draft_engine.cachelens - args.gamma -1
        draft_engine.paged_kv_last_page_len = draft_engine.paged_kv_last_page_len - args.gamma -1

        # Put the accepted tokens to output
        positions = torch.arange(output.shape[1], device=DEVICE).view(1, -1).repeat(BATCH_SIZE, 1)
        mask = (positions < (engine.cachelens.view(-1,1) + accept_nums)) & (positions >= engine.cachelens.view(-1, 1))
        positions_buffer = torch.arange(args.gamma+1, device=DEVICE).view(1, -1).repeat(BATCH_SIZE, 1)
        mask_buffer = positions_buffer<accept_nums.view(-1,1)
        output[mask] = tokens_buffer[mask_buffer]

        # Set the cache length to the accepted length
        engine.cachelens += accept_nums.flatten().to(torch.int32)
        engine.paged_kv_last_page_len += accept_nums.flatten().to(torch.int32)
        draft_engine.cachelens += accept_nums.flatten().to(torch.int32)
        draft_engine.paged_kv_last_page_len += accept_nums.flatten().to(torch.int32)
        
        # Get the bonus tokens
        indices = accept_nums - 1
        bonus_tokens = target_tokens.gather(1, indices)
        if (bonus_tokens == eot_1).any() or (bonus_tokens == eot_2).any():
            terminal = True
        num_nodes += accept_nums.flatten()

        # Check Number of Nodes + Bonus Token <= max_target_token
        if num_nodes.max() + 1 + args.gamma > MAX_LEN_TARGET:
            terminal = True
        # Put Bonus tokens to the tokens buffer, and prepare the variables for next itr
        if not terminal:
            tokens_buffer[:, :1] = bonus_tokens
            if accept_nums.max() == args.gamma + 1:
                next_double = True
                double_buffer = torch.zeros((BATCH_SIZE, 2), device=DEVICE).long()
                mask = (accept_nums == args.gamma + 1).squeeze()
                double_buffer[:, 0] = torch.where(mask, tokens_buffer[:, -1], bonus_tokens[:, 0])
                double_buffer[:, 1] = torch.where(mask, bonus_tokens[:, 0], torch.zeros_like(bonus_tokens[:, 0]))
                # non_zero_mask = double_buffer != 0
                non_zero_mask = torch.ones_like(double_buffer, dtype=torch.bool)
                non_zero_mask[:, 1] = mask
                cachelens_update = non_zero_mask.sum(dim=1).flatten()

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
    num_gen_tokens += (num_nodes.sum() - (input_ids.shape[1]+1)*BATCH_SIZE)
    if args.printoutput:
        for i in range(BATCH_SIZE):
            print(tokenizer.decode(output[i, args.prefix_len:num_nodes[i]]))
    print("total time :{:.5f}s, time per iter :{:.5f}s, decoding step: {}, large model step: {}".format(total_time, total_time / target_steps, num_gen_tokens, target_steps))
    if benchmark:
        print("target time :{:.5f}s, draft time :{:.5f}s, verify loop : {}, avg generate len per sentence: {}".format(target_time/target_steps, draft_time / target_steps, verify_loop/target_steps, num_gen_tokens/target_steps/BATCH_SIZE))
    if step < 3:   # TODO: revert to 10?
        total_time = 0.0
        num_gen_tokens = 0
        target_steps = 0
        if benchmark:
            draft_time = 0.0
            target_time = 0.0
            verify_loop = 0.0
    if use_tp:
        dist.barrier()