import time
import torch
import sys
sys.path.append("..")
from pathlib import Path
import torch.distributed as dist
from MagicDec.Engine.utils import setup_seed, sampling_argmax_batch, cuda_graph_for_sampling_argmax_batch
from MagicDec.Data.data_converter import convert_pg19_dataset
from transformers import AutoTokenizer
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
import argparse
from MagicDec.Engine.backend import LMBackend

parser = argparse.ArgumentParser(description='Process model configuration and partitions.')
parser.add_argument('--model', type=Path, default=Path("checkpoints/meta-llama/Meta-Llama-3.1-8B/model.pth"), help='model')
parser.add_argument('--model_name', type=str, default="meta-llama/Meta-Llama-3.1-8B", help='model name')

parser.add_argument('--B', type=int, default=8, help='Batch size.')
parser.add_argument('--prefix_len', type=int, default=4000, help='Prefix length')

parser.add_argument('--gamma', type=int, default=1, help='Gamma')

parser.add_argument('--seed', type=int, default=123, help='Random seed.')

parser.add_argument('--compile', action='store_true', help='Whether to compile the model.')
parser.add_argument('--rank_group', nargs='+', type=int, help='Target group of ranks')

args = parser.parse_args()

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
global print
from MagicDec.Engine.tp import init_dist
use_tp = len(args.rank_group) > 1
global_group = None
if use_tp:
    rank, global_group = init_dist()
    if rank != args.rank_group[0]:
        # only print on rank 0
        print = lambda *args, **kwargs: None
setup_seed(args.seed)
print(f"Using device={DEVICE}")
# MAX_LEN = args.prefix_len + args.gen_len - 1
prefix_len = args.prefix_len
num_gen_tokens = 96
DTYPE = torch.bfloat16
BATCH_SIZE = args.B
checkpoint_path = args.model


tokenizer = AutoTokenizer.from_pretrained(args.model_name)
tokenizer.pad_token = tokenizer.eos_token
eot_1 = tokenizer.eos_token_id
if tokenizer.unk_token_id is not None:
    eot_2 = tokenizer.unk_token_id
else:
    eot_2 = tokenizer.encode("<|eot_id|>")[-1]
print(f"eot_1: {eot_1}, eot_2: {eot_2}")

dataset = convert_pg19_dataset(tokenizer=tokenizer, seq_len=args.prefix_len+num_gen_tokens)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)
num_eval_steps = min(5, len(dataloader))

gamma = args.gamma + 1

engine = LMBackend(dtype=DTYPE, device=DEVICE, dec_len=gamma)
engine.load_model(checkpoint_path, use_tp=use_tp, rank_group = args.rank_group, group=global_group)
if args.compile:
    engine.compile()

engine.setup_caches(max_batch_size=BATCH_SIZE, max_seq_length=args.prefix_len+num_gen_tokens)

total_time = 0.0
model_steps = 0
pbar = tqdm(enumerate(dataloader), total=num_eval_steps)
for step, batch in pbar:
    if step >= num_eval_steps:
        break
    input_ids = batch[0].to(DEVICE)
    tokens = input_ids.clone()
    input_ids = input_ids[:, :prefix_len]
    terminate = False

    engine.encode(input_ids=input_ids)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    for i in range(0, num_gen_tokens, gamma):
        cur_input_ids = tokens[:, prefix_len+i:prefix_len+i+gamma]
        engine.inference(input_ids=cur_input_ids)
        model_steps += 1

    torch.cuda.synchronize()
    t2 = time.perf_counter()
    total_time += t2 - t1

    avg_verification_time = total_time / model_steps
    pbar.set_description(f"Verification time: {avg_verification_time:.4f}, model steps: {model_steps}")
    if step < 3:
        total_time = 0.0
        model_steps = 0
    
    if use_tp:
        dist.barrier()

print(f"gamma: {gamma-1}, Verification time: {avg_verification_time:.4f}")
