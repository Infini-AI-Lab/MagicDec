import time
import torch
import sys
sys.path.append("..")
from pathlib import Path
import torch.distributed as dist
from MagicDec.Engine.utils import setup_seed, sampling_argmax_batch
from MagicDec.Data.data_converter import convert_pg19_dataset
from transformers import AutoTokenizer
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
import argparse
from MagicDec.Engine.backend import LMBackend

parser = argparse.ArgumentParser(description='Process model configuration and partitions.')
parser.add_argument('--model', type=Path, default=Path("../FlashSpec/checkpoints/meta-llama/Meta-Llama-3.1-8B/model.pth"), help='model')
parser.add_argument('--model_name', type=str, default="meta-llama/Meta-Llama-3.1-8B", help='model name')

parser.add_argument('--B', type=int, default=128, help='Batch size.')
parser.add_argument('--prefix_len', type=int, default=1025, help='Prefix length')
parser.add_argument('--gen_len', type=int, default=128, help='Generate length')

parser.add_argument('--seed', type=int, default=123, help='Random seed.')

parser.add_argument('--compile', action='store_true', help='Whether to compile the model.')
parser.add_argument('--rank_group', nargs='+', type=int, help='Target group of ranks')
parser.add_argument('--printoutput', action='store_true', help='Whether to compile the model.')

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
MAX_LEN = args.prefix_len + args.gen_len - 1
DTYPE = torch.bfloat16
BATCH_SIZE = args.B
checkpoint_path = args.model
engine = LMBackend(dtype=DTYPE, device=DEVICE)
engine.load_model(checkpoint_path, use_tp=use_tp, rank_group = args.rank_group, group=global_group)
if args.compile:
    engine.compile()
engine.setup_caches(max_batch_size=BATCH_SIZE, max_seq_length=MAX_LEN)

tokenizer = AutoTokenizer.from_pretrained(args.model_name)
tokenizer.pad_token = tokenizer.eos_token
eot_1 = tokenizer.eos_token_id
if tokenizer.unk_token_id is not None:
    eot_2 = tokenizer.unk_token_id
else:
    eot_2 = tokenizer.encode("<|eot_id|>")[-1]
print(f"eot_1: {eot_1}, eot_2: {eot_2}")

dataset = convert_pg19_dataset(tokenizer=tokenizer, seq_len=args.prefix_len)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)
num_eval_steps = len(dataloader)

total_time = 0.0
model_steps = 0
for step, batch in tqdm(enumerate(dataloader), total=num_eval_steps):
    input_ids = batch[0].to(DEVICE)
    terminate = False
    output = input_ids.clone()
    logits = engine.encode(input_ids=input_ids)[:,-1]
    next_tokens = sampling_argmax_batch(logits=logits)
    output = torch.cat((output, next_tokens),dim=-1)
    while output.size(1)<args.prefix_len + args.gen_len and terminate == False:
        input_ids=next_tokens.clone()
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        logits = engine.inference(input_ids=input_ids)[:, -1]
        torch.cuda.synchronize()
        t2=time.perf_counter()
        next_tokens = sampling_argmax_batch(logits=logits)
        output = torch.cat((output, next_tokens),dim=-1)
        model_steps += 1
        total_time += t2 - t1
        if (next_tokens[:,-1] == eot_1)._is_any_true() or (next_tokens[:,-1] == eot_2)._is_any_true(): terminate = True

    if args.printoutput:
        for i in range(BATCH_SIZE):
            print(tokenizer.decode(output[i, args.prefix_len:]))

    print(f"Tokens per second :{total_time/model_steps}")
    if step == 0:
        total_time = 0.0
        model_steps = 0
    if use_tp:
        dist.barrier()