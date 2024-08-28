import time
import torch
import sys
sys.path.append("..")
from pathlib import Path
import torch.distributed as dist
from FlashSpec.Engine.utils import setup_seed, cuda_graph_for_sampling_argmax_batch, sampling_argmax_batch
from FlashSpec.Data.data_converter import convert_pg19_dataset
from transformers import AutoTokenizer
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
import argparse
import contextlib
from FlashSpec.Engine.backend_selfspec import LMBackend

parser = argparse.ArgumentParser(description='Process model configuration and partitions.')
parser.add_argument('--model', type=Path, default=Path("checkpoints/meta-llama/Llama-2-7b-hf/model.pth"), help='model')
parser.add_argument('--model_name', type=str, default="meta-llama/Llama-2-7b-hf", help='model name')
parser.add_argument('--streamingllm_budget', type=int, default=256, help='Dataset end index.')
parser.add_argument('--rank_group', nargs='+', type=int, help='Target group of ranks')
parser.add_argument('--compile', action='store_true', help='Whether to compile the model.')

parser.add_argument('--gamma', type=int, default=5, help='start')

parser.add_argument('--B', type=int, default=1, help='Batch size.')
parser.add_argument('--prefix_len', type=int, default=4000, help='Prefix length')
parser.add_argument('--gen_len', type=int, default=64, help='Generate length')

parser.add_argument('--seed', type=int, default=123, help='Random seed.')

parser.add_argument('--printoutput', action='store_true', help='Whether to compile the model.')
parser.add_argument('--benchmark', action='store_true', help='Whether to compile the model.')

# Assert max length <= max context length
args = parser.parse_args()
# assert args.prefix_len + args.gen_len + args.gamma + 1 <= 4096

# Init model parallelism
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
global print
from FlashSpec.Engine.tp import init_dist
use_tp = len(args.rank_group) > 1
global_group = None
if use_tp:
    rank, global_group = init_dist()
    if rank != args.rank_group[0]:
        print = lambda *args, **kwargs: None

setup_seed(args.seed)
print(f"Using device={DEVICE}")
MAX_LEN_TARGET = args.prefix_len + args.gen_len + args.gamma
DTYPE = torch.bfloat16
BATCH_SIZE = args.B
benchmark = args.benchmark
checkpoint_path = args.model

target_dec_list = [args.gamma + 1]
draft_dec_list = [1,2]

# Load target model
engine = LMBackend(dtype=DTYPE, device=DEVICE, dec_list=target_dec_list, draft_dec_list=draft_dec_list)
engine.load_model(checkpoint_path, use_tp=use_tp, rank_group = args.rank_group, group=global_group)
if args.compile:
    engine.compile()
engine.setup_caches(max_batch_size=BATCH_SIZE, max_seq_length=MAX_LEN_TARGET, streamingllm_budget=args.streamingllm_budget, buffer=args.gen_len+args.gamma)
target_sample = cuda_graph_for_sampling_argmax_batch(device=DEVICE, dtype=DTYPE, batch_size=BATCH_SIZE, idx_len=args.gamma+1)
draft_sample = {}
for i in [1, 2]:
    draft_sample[i] = cuda_graph_for_sampling_argmax_batch(device=DEVICE, dtype=DTYPE, batch_size=BATCH_SIZE, idx_len=i)

# Load dataset
tokenizer = AutoTokenizer.from_pretrained(args.model_name)
tokenizer.pad_token = tokenizer.eos_token
eot_1 = tokenizer.eos_token_id
if tokenizer.unk_token_id is not None:
    eot_2 = tokenizer.unk_token_id
else:
    eot_2 = tokenizer.encode("<|eot_id|>")[-1]
print(f"eot_1: {eot_1}, eot_2: {eot_2}")
repeats = 20
no_runs = int(BATCH_SIZE*repeats)
dataset = convert_pg19_dataset(tokenizer=tokenizer, seq_len=args.prefix_len) #, end=no_runs)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)
num_eval_steps = min(10, len(dataloader))

total_time = 0.0
num_gen_tokens = 0
target_steps = 0
if benchmark:
    draft_time = 0.0
    target_time = 0.0
    verify_loop = 0.0

prof = contextlib.nullcontext()

for step, batch in tqdm(enumerate(dataloader), total=num_eval_steps):
    if step >= num_eval_steps:
        break
    input_ids = batch[0].to(DEVICE)
    terminal = False
    tokens_buffer= torch.zeros((BATCH_SIZE, args.gamma+1), device=DEVICE).long()
    output = torch.zeros(BATCH_SIZE, args.prefix_len + args.gen_len + args.gamma + 1, device=DEVICE).long()
    output[:, :input_ids.shape[1]] = input_ids
    num_nodes = torch.zeros(BATCH_SIZE,device=DEVICE).long()
    num_nodes += input_ids.shape[1]

    logits = engine.encode(input_ids=input_ids)[:,-1]
    
    tokens_buffer[:,:1] = sampling_argmax_batch(logits=logits)
    
    next_double = False
    double_buffer = None
    cachelens_update = None

    torch.cuda.synchronize()
    start = time.perf_counter()
    while terminal == False:

        # Draft speculation
        if (step == num_eval_steps - 1) and (rank == 0):
            torch.profiler._utils._init_for_cuda_graphs()
            prof = torch.profiler.profile()

        if benchmark:
            torch.cuda.synchronize()
            t1 = time.time()

        with prof:    
            for i in range(args.gamma):
                if i == 0:
                    if next_double:
                        # The cachelens should increase 1 or 2
                        next_tokens = draft_sample[2](engine.draft_inference(double_buffer, cachelen_update=cachelens_update))
                        tokens_buffer[:,i+1:i+2] = next_tokens.gather(1, cachelens_update.view(-1,1) - 1)
                        next_double = False
                    else:
                        tokens_buffer[:,i+1:i+2] = draft_sample[1](engine.draft_inference(tokens_buffer[:, i].view(-1,1)))
                    continue
                tokens_buffer[:,i+1:i+2] = draft_sample[1](engine.draft_inference(tokens_buffer[:, i].view(-1,1)))

        if benchmark:
            torch.cuda.synchronize()
            t2 = time.time()
            draft_time+=t2-t1

        # Target Verification
        target_logits = engine.inference(tokens_buffer)

        if benchmark:
            torch.cuda.synchronize()
            t3 = time.time()
            target_time+=t3-t2

        target_tokens = target_sample(target_logits)
        target_steps+=1

    # Verify loop
        bonus_tokens = torch.full((BATCH_SIZE, 1), 0, device=DEVICE).long()
        accept_nums = torch.full((BATCH_SIZE, 1), 1, device=DEVICE).long()
        accept_flags = torch.full((BATCH_SIZE, 1), True, device=DEVICE)
        for pos in range(args.gamma):
            target_token = target_tokens[:, pos]
            draft_token = tokens_buffer[:, pos+1]
            flag_accept = (target_token == draft_token).unsqueeze(1)
            # Ensure flags remain False once they have been set to False
            accept_flags = accept_flags & flag_accept
            # Only increase accept_nums where accept_flags are still True
            accept_nums += accept_flags.int()
            # Wether or not terminate
            condition = ((draft_token.unsqueeze(1) == eot_1) | (draft_token.unsqueeze(1) == eot_2)) & accept_flags
            if condition.any():
                terminal = True
            accept_flags = accept_flags & ~condition
        
        # Rollback the memory length
        engine.cachelens = engine.cachelens - args.gamma - 1

        # Put the accepted tokens to output
        positions = torch.arange(output.shape[1], device=DEVICE).view(1, -1).repeat(BATCH_SIZE, 1)
        mask = (positions < (engine.cachelens.view(-1,1) + accept_nums)) & (positions >= engine.cachelens.view(-1, 1))
        positions_buffer = torch.arange(args.gamma+1, device=DEVICE).view(1, -1).repeat(BATCH_SIZE, 1)
        mask_buffer = positions_buffer<accept_nums.view(-1,1)
        output[mask] = tokens_buffer[mask_buffer]

        # Set the cache length to the accepted length
        engine.cachelens += accept_nums.flatten()
        max_limit = torch.full_like(accept_nums, args.gamma, device = DEVICE)
        limited_accept_nums = torch.min(accept_nums, max_limit)
        engine.draft_cachelens = engine.draft_cachelens - args.gamma
        # engine.draft_cachelens += accept_nums.flatten()
        engine.draft_cachelens += limited_accept_nums.flatten()
        
        # Get the bonus tokens
        indices = accept_nums - 1
        bonus_tokens = target_tokens.gather(1, indices)
        if (bonus_tokens == 2).any() or (bonus_tokens == 0).any():
            terminal = True
        num_nodes += accept_nums.flatten()

        # Check Number of Nodes + Bonus Token <= max_target_token
        if num_nodes.max() + 1 >= args.prefix_len + args.gen_len:
            terminal = True
        # Put Bonus tokens to the tokens buffer, and prepare the variables for next itr
        if not terminal:
            tokens_buffer[:, :1] = bonus_tokens
            if accept_nums.max() == args.gamma + 1:
                next_double = True
                double_buffer = torch.zeros((BATCH_SIZE, 2), device=DEVICE).long()
                mask = (accept_nums == (args.gamma + 1)).squeeze()
                double_buffer[:, 0] = torch.where(mask, tokens_buffer[:, -1], bonus_tokens[:, 0])
                double_buffer[:, 1] = torch.where(mask, bonus_tokens[:, 0], torch.zeros_like(bonus_tokens[:, 0]))
                non_zero_mask = double_buffer != 0
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

if hasattr(prof, "export_chrome_trace"):
    prof.export_chrome_trace(f"prof_selfspec.json")