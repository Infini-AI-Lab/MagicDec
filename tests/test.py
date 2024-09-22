import sys
import time
from pathlib import Path
sys.path.append("..")
import torch
import torch._dynamo.config
import torch._inductor.config
import argparse
from MagicDec.Engine.backend import LMBackend
import contextlib

parser = argparse.ArgumentParser(description='Your CLI description.')
parser.add_argument('--checkpoint_path', type=Path, default=Path("../FlashSpec/checkpoints/meta-llama/Meta-Llama-3.1-8B/model.pth"), help='Model checkpoint path.')
parser.add_argument('--compile', action='store_true', help='Whether to compile the model.')
parser.add_argument('--B', type=int, default=1, help='batch size')
parser.add_argument('--M', type=int, default=256, help='max len')
parser.add_argument('--P', type=int, default=128, help='prefix len')
parser.add_argument('--T', type=int, default=1000, help='repeat times')
parser.add_argument('--declen', type=int, default= 1, help='dec len')
parser.add_argument('--rank_group', nargs='+', type=int, help='Group of ranks')
parser.add_argument('--profile', type=Path, default=Path("tests/profile.txt"), help='Profile path.')

args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
global print
from MagicDec.Engine.tp import init_dist
use_tp = len(args.rank_group)>1
global_group = None
if use_tp:
    rank, global_group = init_dist()
    if rank != 0:
        # only print on rank 0
        print = lambda *args, **kwargs: None
print(f"Using device={device}")

checkpoint_path = args.checkpoint_path
precision = torch.bfloat16
max_seq_length = args.M
max_batch_size = args.B
prefix_len = args.P
declen = args.declen

warm_up = 10
T = args.T

llm = LMBackend(dtype=precision, device=device, dec_len=declen)
llm.load_model(checkpoint_path, use_tp=use_tp, rank_group=args.rank_group, group = global_group)
if args.compile:
    llm.compile()
llm.setup_caches(max_batch_size=max_batch_size, max_seq_length=max_seq_length)

prompt = torch.randint(low=3, high=30000, size=(max_batch_size, prefix_len), device=device)
llm.encode(input_ids=prompt, benchmark=True)

profile = args.profile

total_time = 0.0
if (not profile) or (use_tp and rank != 0):
    prof = contextlib.nullcontext()
else:
    torch.profiler._utils._init_for_cuda_graphs()
    prof = torch.profiler.profile()

with torch.inference_mode():
        for _ in range(warm_up):
            dec = torch.randint(low=3, high=30000, size=(max_batch_size, declen), device=device)
            logits = llm.inference(input_ids=dec, benchmark=True)
        for _ in range(T):
            dec = torch.randint(low=3, high=30000, size=(max_batch_size, declen), device=device)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            logits = llm.inference(input_ids=dec, benchmark=True)
            torch.cuda.synchronize()
            t2 = time.perf_counter()
            total_time += t2 - t1
print("Batch Size:{}, Max Length :{}, Decode Length :{}, Prefix Length :{}, inference time:{}s".format(max_batch_size, max_seq_length, declen, prefix_len, total_time / T))
torch.cuda.synchronize()
with prof:
    llm.inference(input_ids=dec, benchmark=True)
    llm.inference(input_ids=dec, benchmark=True)
    llm.inference(input_ids=dec, benchmark=True)
if hasattr(prof, "export_chrome_trace"):
        if use_tp:
            prof.export_chrome_trace(f"{profile}_rank_{rank}.json")
        else:
            prof.export_chrome_trace(f"{profile}.json")
