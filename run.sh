# ENABLE_INTRA_NODE_COMM=1 torchrun --standalone --nproc_per_node=8 tests/baseline_benchmark.py --compile --rank_group 0 1 2 3 4 5 6 7 --B 1 --prefix_len 257 --max_len 384
# ENABLE_INTRA_NODE_COMM=1 torchrun --standalone --nproc_per_node=8 tests/baseline_benchmark.py --compile --rank_group 0 1 2 3 4 5 6 7 --B 16 --prefix_len 257 --max_len 384
# ENABLE_INTRA_NODE_COMM=1 torchrun --standalone --nproc_per_node=8 tests/baseline_benchmark.py --compile --rank_group 0 1 2 3 4 5 6 7 --B 32 --prefix_len 257 --max_len 384
# ENABLE_INTRA_NODE_COMM=1 torchrun --standalone --nproc_per_node=8 tests/baseline_benchmark.py --compile --rank_group 0 1 2 3 4 5 6 7 --B 64 --prefix_len 257 --max_len 384
# ENABLE_INTRA_NODE_COMM=1 torchrun --standalone --nproc_per_node=8 tests/baseline_benchmark.py --compile --rank_group 0 1 2 3 4 5 6 7 --B 128 --prefix_len 257 --max_len 384
# ENABLE_INTRA_NODE_COMM=1 torchrun --standalone --nproc_per_node=8 tests/baseline_benchmark.py --compile --rank_group 0 1 2 3 4 5 6 7 --B 256 --prefix_len 257 --max_len 384
# ENABLE_INTRA_NODE_COMM=1 torchrun --standalone --nproc_per_node=8 tests/baseline_benchmark.py --compile --rank_group 0 1 2 3 4 5 6 7 --B 512 --prefix_len 257 --max_len 384

# ENABLE_INTRA_NODE_COMM=1 torchrun --standalone --nproc_per_node=8 tests/baseline_benchmark.py --compile --rank_group 0 1 2 3 4 5 6 7 --B 1 --prefix_len 8193 --max_len 8320
# ENABLE_INTRA_NODE_COMM=1 torchrun --standalone --nproc_per_node=8 tests/baseline_benchmark.py --compile --rank_group 0 1 2 3 4 5 6 7 --B 16 --prefix_len 8193 --max_len 8320
# ENABLE_INTRA_NODE_COMM=1 torchrun --standalone --nproc_per_node=8 tests/baseline_benchmark.py --compile --rank_group 0 1 2 3 4 5 6 7 --B 32 --prefix_len 8193 --max_len 8320
# ENABLE_INTRA_NODE_COMM=1 torchrun --standalone --nproc_per_node=8 tests/baseline_benchmark.py --compile --rank_group 0 1 2 3 4 5 6 7 --B 64 --prefix_len 8193 --max_len 8320
# ENABLE_INTRA_NODE_COMM=1 torchrun --standalone --nproc_per_node=8 tests/baseline_benchmark.py --compile --rank_group 0 1 2 3 4 5 6 7 --B 128 --prefix_len 8193 --max_len 8320
# ENABLE_INTRA_NODE_COMM=1 torchrun --standalone --nproc_per_node=8 tests/baseline_benchmark.py --compile --rank_group 0 1 2 3 4 5 6 7 --B 256 --prefix_len 8193 --max_len 8320
# ENABLE_INTRA_NODE_COMM=1 torchrun --standalone --nproc_per_node=8 tests/baseline_benchmark.py --compile --rank_group 0 1 2 3 4 5 6 7 --B 480 --prefix_len 8193 --max_len 8320

# ENABLE_INTRA_NODE_COMM=1 torchrun --standalone --nproc_per_node=8 tests/SnapKV/longspec_benchmark.py --compile --rank_group 0 1 2 3 4 5 6 7 --draft_rank_group 0 1 2 3 4 5 6 7 --B 128 --prefix_len 257 --max_len 384 --benchmark --gamma 2
# ENABLE_INTRA_NODE_COMM=1 torchrun --standalone --nproc_per_node=8 tests/SnapKV/longspec_benchmark.py --compile --rank_group 0 1 2 3 4 5 6 7 --draft_rank_group 0 1 2 3 4 5 6 7 --B 256 --prefix_len 257 --max_len 384 --benchmark --gamma 1
# ENABLE_INTRA_NODE_COMM=1 torchrun --standalone --nproc_per_node=8 tests/SnapKV/longspec_benchmark.py --compile --rank_group 0 1 2 3 4 5 6 7 --draft_rank_group 0 1 2 3 4 5 6 7 --B 256 --prefix_len 257 --max_len 384 --benchmark --gamma 2
# ENABLE_INTRA_NODE_COMM=1 torchrun --standalone --nproc_per_node=8 tests/SnapKV/longspec_benchmark.py --compile --rank_group 0 1 2 3 4 5 6 7 --draft_rank_group 0 1 2 3 4 5 6 7 --B 512 --prefix_len 257 --max_len 384 --benchmark --gamma 1
# ENABLE_INTRA_NODE_COMM=1 torchrun --standalone --nproc_per_node=8 tests/SnapKV/longspec_benchmark.py --compile --rank_group 0 1 2 3 4 5 6 7 --draft_rank_group 0 1 2 3 4 5 6 7 --B 512 --prefix_len 257 --max_len 384 --benchmark --gamma 2

# ENABLE_INTRA_NODE_COMM=1 torchrun --standalone --nproc_per_node=8 tests/SnapKV/longspec_benchmark.py --compile --rank_group 0 1 2 3 4 5 6 7 --draft_rank_group 0 1 2 3 4 5 6 7 --B 16 --prefix_len 8193 --max_len 8320 --benchmark --gamma 2
# ENABLE_INTRA_NODE_COMM=1 torchrun --standalone --nproc_per_node=8 tests/SnapKV/longspec_benchmark.py --compile --rank_group 0 1 2 3 4 5 6 7 --draft_rank_group 0 1 2 3 4 5 6 7 --B 16 --prefix_len 8193 --max_len 8320 --benchmark --gamma 3
# ENABLE_INTRA_NODE_COMM=1 torchrun --standalone --nproc_per_node=8 tests/SnapKV/longspec_benchmark.py --compile --rank_group 0 1 2 3 4 5 6 7 --draft_rank_group 0 1 2 3 4 5 6 7 --B 64 --prefix_len 8193 --max_len 8320 --benchmark --gamma 2
# ENABLE_INTRA_NODE_COMM=1 torchrun --standalone --nproc_per_node=8 tests/SnapKV/longspec_benchmark.py --compile --rank_group 0 1 2 3 4 5 6 7 --draft_rank_group 0 1 2 3 4 5 6 7 --B 64 --prefix_len 8193 --max_len 8320 --benchmark --gamma 3
# ENABLE_INTRA_NODE_COMM=1 torchrun --standalone --nproc_per_node=8 tests/SnapKV/longspec_benchmark.py --compile --rank_group 0 1 2 3 4 5 6 7 --draft_rank_group 0 1 2 3 4 5 6 7 --B 128 --prefix_len 8193 --max_len 8320 --benchmark --gamma 2
# ENABLE_INTRA_NODE_COMM=1 torchrun --standalone --nproc_per_node=8 tests/SnapKV/longspec_benchmark.py --compile --rank_group 0 1 2 3 4 5 6 7 --draft_rank_group 0 1 2 3 4 5 6 7 --B 128 --prefix_len 8193 --max_len 8320 --benchmark --gamma 3
# ENABLE_INTRA_NODE_COMM=1 torchrun --standalone --nproc_per_node=8 tests/SnapKV/longspec_benchmark.py --compile --rank_group 0 1 2 3 4 5 6 7 --draft_rank_group 0 1 2 3 4 5 6 7 --B 256 --prefix_len 8193 --max_len 8320 --benchmark --gamma 1
# ENABLE_INTRA_NODE_COMM=1 torchrun --standalone --nproc_per_node=8 tests/SnapKV/longspec_benchmark.py --compile --rank_group 0 1 2 3 4 5 6 7 --draft_rank_group 0 1 2 3 4 5 6 7 --B 256 --prefix_len 8193 --max_len 8320 --benchmark --gamma 2
# ENABLE_INTRA_NODE_COMM=1 torchrun --standalone --nproc_per_node=8 tests/SnapKV/longspec_benchmark.py --compile --rank_group 0 1 2 3 4 5 6 7 --draft_rank_group 0 1 2 3 4 5 6 7 --B 256 --prefix_len 8193 --max_len 8320 --benchmark --gamma 3
# ENABLE_INTRA_NODE_COMM=1 torchrun --standalone --nproc_per_node=8 tests/SnapKV/longspec_benchmark.py --compile --rank_group 0 1 2 3 4 5 6 7 --draft_rank_group 0 1 2 3 4 5 6 7 --B 480 --prefix_len 8193 --max_len 8320 --benchmark --gamma 1
# ENABLE_INTRA_NODE_COMM=1 torchrun --standalone --nproc_per_node=8 tests/SnapKV/longspec_benchmark.py --compile --rank_group 0 1 2 3 4 5 6 7 --draft_rank_group 0 1 2 3 4 5 6 7 --B 480 --prefix_len 8193 --max_len 8320 --benchmark --gamma 2

# ENABLE_INTRA_NODE_COMM=1 torchrun --standalone --nproc_per_node=8 tests/StreamingLLM/longspec_benchmark.py --compile --rank_group 0 1 2 3 4 5 6 7 --draft_rank_group 0 1 2 3 4 5 6 7 --B 16 --prefix_len 8193 --max_len 8320 --benchmark --gamma 2 --draft_budget 513
# ENABLE_INTRA_NODE_COMM=1 torchrun --standalone --nproc_per_node=8 tests/StreamingLLM/longspec_benchmark.py --compile --rank_group 0 1 2 3 4 5 6 7 --draft_rank_group 0 1 2 3 4 5 6 7 --B 16 --prefix_len 8193 --max_len 8320 --benchmark --gamma 3 --draft_budget 513
# ENABLE_INTRA_NODE_COMM=1 torchrun --standalone --nproc_per_node=8 tests/StreamingLLM/longspec_benchmark.py --compile --rank_group 0 1 2 3 4 5 6 7 --draft_rank_group 0 1 2 3 4 5 6 7 --B 64 --prefix_len 8193 --max_len 8320 --benchmark --gamma 2 --draft_budget 513
# ENABLE_INTRA_NODE_COMM=1 torchrun --standalone --nproc_per_node=8 tests/StreamingLLM/longspec_benchmark.py --compile --rank_group 0 1 2 3 4 5 6 7 --draft_rank_group 0 1 2 3 4 5 6 7 --B 64 --prefix_len 8193 --max_len 8320 --benchmark --gamma 3 --draft_budget 513
# ENABLE_INTRA_NODE_COMM=1 torchrun --standalone --nproc_per_node=8 tests/StreamingLLM/longspec_benchmark.py --compile --rank_group 0 1 2 3 4 5 6 7 --draft_rank_group 0 1 2 3 4 5 6 7 --B 128 --prefix_len 8193 --max_len 8320 --benchmark --gamma 2 --draft_budget 513
# ENABLE_INTRA_NODE_COMM=1 torchrun --standalone --nproc_per_node=8 tests/StreamingLLM/longspec_benchmark.py --compile --rank_group 0 1 2 3 4 5 6 7 --draft_rank_group 0 1 2 3 4 5 6 7 --B 128 --prefix_len 8193 --max_len 8320 --benchmark --gamma 3 --draft_budget 513
# ENABLE_INTRA_NODE_COMM=1 torchrun --standalone --nproc_per_node=8 tests/StreamingLLM/longspec_benchmark.py --compile --rank_group 0 1 2 3 4 5 6 7 --draft_rank_group 0 1 2 3 4 5 6 7 --B 256 --prefix_len 8193 --max_len 8320 --benchmark --gamma 2 --draft_budget 513
# ENABLE_INTRA_NODE_COMM=1 torchrun --standalone --nproc_per_node=8 tests/StreamingLLM/longspec_benchmark.py --compile --rank_group 0 1 2 3 4 5 6 7 --draft_rank_group 0 1 2 3 4 5 6 7 --B 256 --prefix_len 8193 --max_len 8320 --benchmark --gamma 3 --draft_budget 513
# ENABLE_INTRA_NODE_COMM=1 torchrun --standalone --nproc_per_node=8 tests/StreamingLLM/longspec_benchmark.py --compile --rank_group 0 1 2 3 4 5 6 7 --draft_rank_group 0 1 2 3 4 5 6 7 --B 256 --prefix_len 8193 --max_len 8320 --benchmark --gamma 4 --draft_budget 513

# ENABLE_INTRA_NODE_COMM=1 torchrun --standalone --nproc_per_node=8 tests/SnapKV/selfspec_benchmark.py --compile --rank_group 0 1 2 3 4 5 6 7 --B 16 --prefix_len 8193 --max_len 8320 --benchmark --gamma 2 --draft_budget 513
# ENABLE_INTRA_NODE_COMM=1 torchrun --standalone --nproc_per_node=8 tests/SnapKV/selfspec_benchmark.py --compile --rank_group 0 1 2 3 4 5 6 7 --B 64 --prefix_len 8193 --max_len 8320 --benchmark --gamma 2 --draft_budget 513
# ENABLE_INTRA_NODE_COMM=1 torchrun --standalone --nproc_per_node=8 tests/SnapKV/selfspec_benchmark.py --compile --rank_group 0 1 2 3 4 5 6 7 --B 128 --prefix_len 8193 --max_len 8320 --benchmark --gamma 2 --draft_budget 513
# ENABLE_INTRA_NODE_COMM=1 torchrun --standalone --nproc_per_node=8 tests/SnapKV/selfspec_benchmark.py --compile --rank_group 0 1 2 3 4 5 6 7 --B 256 --prefix_len 8193 --max_len 8320 --benchmark --gamma 2 --draft_budget 513
# ENABLE_INTRA_NODE_COMM=1 torchrun --standalone --nproc_per_node=8 tests/SnapKV/selfspec_benchmark.py --compile --rank_group 0 1 2 3 4 5 6 7 --B 256 --prefix_len 8193 --max_len 8320 --benchmark --gamma 3 --draft_budget 513

# ENABLE_INTRA_NODE_COMM=1 torchrun --standalone --nproc_per_node=8 tests/baseline_benchmark.py --compile --rank_group 0 1 2 3 4 5 6 7 --B 16 --prefix_len 32769 --max_len 32896
# ENABLE_INTRA_NODE_COMM=1 torchrun --standalone --nproc_per_node=8 tests/baseline_benchmark.py --compile --rank_group 0 1 2 3 4 5 6 7 --B 32 --prefix_len 32769 --max_len 32896
# ENABLE_INTRA_NODE_COMM=1 torchrun --standalone --nproc_per_node=8 tests/baseline_benchmark.py --compile --rank_group 0 1 2 3 4 5 6 7 --B 64 --prefix_len 32769 --max_len 32896
# ENABLE_INTRA_NODE_COMM=1 torchrun --standalone --nproc_per_node=8 tests/baseline_benchmark.py --compile --rank_group 0 1 2 3 4 5 6 7 --B 128 --prefix_len 32769 --max_len 32896


# ENABLE_INTRA_NODE_COMM=1 torchrun --standalone --nproc_per_node=8 tests/StreamingLLM/longspec_benchmark.py --compile --rank_group 0 1 2 3 4 5 6 7 --draft_rank_group 0 1 2 3 4 5 6 7 --B 16 --prefix_len 32769 --max_len 32896 --benchmark --gamma 2 --draft_budget 513
# ENABLE_INTRA_NODE_COMM=1 torchrun --standalone --nproc_per_node=8 tests/StreamingLLM/longspec_benchmark.py --compile --rank_group 0 1 2 3 4 5 6 7 --draft_rank_group 0 1 2 3 4 5 6 7 --B 16 --prefix_len 32769 --max_len 32896 --benchmark --gamma 3 --draft_budget 513
# ENABLE_INTRA_NODE_COMM=1 torchrun --standalone --nproc_per_node=8 tests/StreamingLLM/longspec_benchmark.py --compile --rank_group 0 1 2 3 4 5 6 7 --draft_rank_group 0 1 2 3 4 5 6 7 --B 32 --prefix_len 32769 --max_len 32896 --benchmark --gamma 2 --draft_budget 513
# ENABLE_INTRA_NODE_COMM=1 torchrun --standalone --nproc_per_node=8 tests/StreamingLLM/longspec_benchmark.py --compile --rank_group 0 1 2 3 4 5 6 7 --draft_rank_group 0 1 2 3 4 5 6 7 --B 32 --prefix_len 32769 --max_len 32896 --benchmark --gamma 3 --draft_budget 513
# ENABLE_INTRA_NODE_COMM=1 torchrun --standalone --nproc_per_node=8 tests/StreamingLLM/longspec_benchmark.py --compile --rank_group 0 1 2 3 4 5 6 7 --draft_rank_group 0 1 2 3 4 5 6 7 --B 64 --prefix_len 32769 --max_len 32896 --benchmark --gamma 2 --draft_budget 513
# ENABLE_INTRA_NODE_COMM=1 torchrun --standalone --nproc_per_node=8 tests/StreamingLLM/longspec_benchmark.py --compile --rank_group 0 1 2 3 4 5 6 7 --draft_rank_group 0 1 2 3 4 5 6 7 --B 64 --prefix_len 32769 --max_len 32896 --benchmark --gamma 3 --draft_budget 513
# ENABLE_INTRA_NODE_COMM=1 torchrun --standalone --nproc_per_node=8 tests/StreamingLLM/longspec_benchmark.py --compile --rank_group 0 1 2 3 4 5 6 7 --draft_rank_group 0 1 2 3 4 5 6 7 --B 128 --prefix_len 32769 --max_len 32896 --benchmark --gamma 2 --draft_budget 513
# ENABLE_INTRA_NODE_COMM=1 torchrun --standalone --nproc_per_node=8 tests/StreamingLLM/longspec_benchmark.py --compile --rank_group 0 1 2 3 4 5 6 7 --draft_rank_group 0 1 2 3 4 5 6 7 --B 128 --prefix_len 32769 --max_len 32896 --benchmark --gamma 3 --draft_budget 513
# ENABLE_INTRA_NODE_COMM=1 torchrun --standalone --nproc_per_node=8 tests/StreamingLLM/longspec_benchmark.py --compile --rank_group 0 1 2 3 4 5 6 7 --draft_rank_group 0 1 2 3 4 5 6 7 --B 128 --prefix_len 32769 --max_len 32896 --benchmark --gamma 4 --draft_budget 513

# ENABLE_INTRA_NODE_COMM=1 torchrun --standalone --nproc_per_node=8 tests/SnapKV/selfspec_benchmark.py --compile --rank_group 0 1 2 3 4 5 6 7 --B 16 --prefix_len 32800 --max_len 32896 --benchmark --gamma 3 --draft_budget 1025
# ENABLE_INTRA_NODE_COMM=1 torchrun --standalone --nproc_per_node=8 tests/SnapKV/selfspec_benchmark.py --compile --rank_group 0 1 2 3 4 5 6 7 --B 16 --prefix_len 32800 --max_len 32896 --benchmark --gamma 3 --draft_budget 513
# ENABLE_INTRA_NODE_COMM=1 torchrun --standalone --nproc_per_node=8 tests/SnapKV/selfspec_benchmark.py --compile --rank_group 0 1 2 3 4 5 6 7 --B 16 --prefix_len 32800 --max_len 32896 --benchmark --gamma 4 --draft_budget 513
# ENABLE_INTRA_NODE_COMM=1 torchrun --standalone --nproc_per_node=8 tests/SnapKV/selfspec_benchmark.py --compile --rank_group 0 1 2 3 4 5 6 7 --B 32 --prefix_len 32800 --max_len 32896 --benchmark --gamma 3 --draft_budget 513
# ENABLE_INTRA_NODE_COMM=1 torchrun --standalone --nproc_per_node=8 tests/SnapKV/selfspec_benchmark.py --compile --rank_group 0 1 2 3 4 5 6 7 --B 32 --prefix_len 32800 --max_len 32896 --benchmark --gamma 4 --draft_budget 1025
# ENABLE_INTRA_NODE_COMM=1 torchrun --standalone --nproc_per_node=8 tests/SnapKV/selfspec_benchmark.py --compile --rank_group 0 1 2 3 4 5 6 7 --B 64 --prefix_len 32800 --max_len 32896 --benchmark --gamma 3 --draft_budget 513
# ENABLE_INTRA_NODE_COMM=1 torchrun --standalone --nproc_per_node=8 tests/SnapKV/selfspec_benchmark.py --compile --rank_group 0 1 2 3 4 5 6 7 --B 64 --prefix_len 32800 --max_len 32896 --benchmark --gamma 5 --draft_budget 1025
# ENABLE_INTRA_NODE_COMM=1 torchrun --standalone --nproc_per_node=8 tests/SnapKV/selfspec_benchmark.py --compile --rank_group 0 1 2 3 4 5 6 7 --B 128 --prefix_len 32800 --max_len 32896 --benchmark --gamma 2 --draft_budget 513
# ENABLE_INTRA_NODE_COMM=1 torchrun --standalone --nproc_per_node=8 tests/SnapKV/selfspec_benchmark.py --compile --rank_group 0 1 2 3 4 5 6 7 --B 128 --prefix_len 32800 --max_len 32896 --benchmark --gamma 3 --draft_budget 513
# ENABLE_INTRA_NODE_COMM=1 torchrun --standalone --nproc_per_node=8 tests/SnapKV/selfspec_benchmark.py --compile --rank_group 0 1 2 3 4 5 6 7 --B 128 --prefix_len 32800 --max_len 32896 --benchmark --gamma 6 --draft_budget 1025

# ENABLE_INTRA_NODE_COMM=1 torchrun --standalone --nproc_per_node=8 tests/SnapKV/selfspec_benchmark.py --compile --rank_group 0 1 2 3 4 5 6 7 --B 16 --prefix_len 32800 --max_len 32896 --benchmark --gamma 3 --draft_budget 2049
# ENABLE_INTRA_NODE_COMM=1 torchrun --standalone --nproc_per_node=8 tests/SnapKV/selfspec_benchmark.py --compile --rank_group 0 1 2 3 4 5 6 7 --B 16 --prefix_len 32800 --max_len 32896 --benchmark --gamma 3 --draft_budget 513
# ENABLE_INTRA_NODE_COMM=1 torchrun --standalone --nproc_per_node=8 tests/SnapKV/selfspec_benchmark.py --compile --rank_group 0 1 2 3 4 5 6 7 --B 16 --prefix_len 32800 --max_len 32896 --benchmark --gamma 4 --draft_budget 513
# ENABLE_INTRA_NODE_COMM=1 torchrun --standalone --nproc_per_node=8 tests/SnapKV/selfspec_benchmark.py --compile --rank_group 0 1 2 3 4 5 6 7 --B 32 --prefix_len 32800 --max_len 32896 --benchmark --gamma 3 --draft_budget 513
# ENABLE_INTRA_NODE_COMM=1 torchrun --standalone --nproc_per_node=8 tests/SnapKV/selfspec_benchmark.py --compile --rank_group 0 1 2 3 4 5 6 7 --B 32 --prefix_len 32800 --max_len 32896 --benchmark --gamma 4 --draft_budget 2049
# ENABLE_INTRA_NODE_COMM=1 torchrun --standalone --nproc_per_node=8 tests/SnapKV/selfspec_benchmark.py --compile --rank_group 0 1 2 3 4 5 6 7 --B 64 --prefix_len 32800 --max_len 32896 --benchmark --gamma 3 --draft_budget 513
# ENABLE_INTRA_NODE_COMM=1 torchrun --standalone --nproc_per_node=8 tests/SnapKV/selfspec_benchmark.py --compile --rank_group 0 1 2 3 4 5 6 7 --B 64 --prefix_len 32800 --max_len 32896 --benchmark --gamma 5 --draft_budget 2049
# ENABLE_INTRA_NODE_COMM=1 torchrun --standalone --nproc_per_node=8 tests/SnapKV/selfspec_benchmark.py --compile --rank_group 0 1 2 3 4 5 6 7 --B 128 --prefix_len 32800 --max_len 32896 --benchmark --gamma 2 --draft_budget 513
# ENABLE_INTRA_NODE_COMM=1 torchrun --standalone --nproc_per_node=8 tests/SnapKV/selfspec_benchmark.py --compile --rank_group 0 1 2 3 4 5 6 7 --B 128 --prefix_len 32800 --max_len 32896 --benchmark --gamma 3 --draft_budget 513
ENABLE_INTRA_NODE_COMM=1 torchrun --standalone --nproc_per_node=8 tests/SnapKV/selfspec_benchmark.py --compile --rank_group 0 1 2 3 4 5 6 7 --B 128 --prefix_len 32032 --max_len 32128 --benchmark --gamma 6 --draft_budget 2049


# ENABLE_INTRA_NODE_COMM=1 torchrun --standalone --nproc_per_node=8 tests/SnapKV/longspec_benchmark.py --compile --rank_group 0 1 2 3 4 5 6 7 --draft_rank_group 0 1 2 3 4 5 6 7 --B 16 --prefix_len 257 --max_len 384 --benchmark
# ENABLE_INTRA_NODE_COMM=1 torchrun --standalone --nproc_per_node=8 tests/SnapKV/longspec_benchmark.py --compile --rank_group 0 1 2 3 4 5 6 7 --draft_rank_group 0 1 2 3 4 5 6 7 --B 32 --prefix_len 257 --max_len 384 --benchmark
# ENABLE_INTRA_NODE_COMM=1 torchrun --standalone --nproc_per_node=8 tests/SnapKV/longspec_benchmark.py --compile --rank_group 0 1 2 3 4 5 6 7 --draft_rank_group 0 1 2 3 4 5 6 7 --B 64 --prefix_len 257 --max_len 384 --benchmark
# ENABLE_INTRA_NODE_COMM=1 torchrun --standalone --nproc_per_node=8 tests/SnapKV/longspec_benchmark.py --compile --rank_group 0 1 2 3 4 5 6 7 --draft_rank_group 0 1 2 3 4 5 6 7 --B 128 --prefix_len 257 --max_len 384 --benchmark
# ENABLE_INTRA_NODE_COMM=1 torchrun --standalone --nproc_per_node=8 tests/SnapKV/longspec_benchmark.py --compile --rank_group 0 1 2 3 4 5 6 7 --draft_rank_group 0 1 2 3 4 5 6 7 --B 256 --prefix_len 257 --max_len 384 --benchmark
# ENABLE_INTRA_NODE_COMM=1 torchrun --standalone --nproc_per_node=8 tests/SnapKV/longspec_benchmark.py --compile --rank_group 0 1 2 3 4 5 6 7 --draft_rank_group 0 1 2 3 4 5 6 7 --B 512 --prefix_len 257 --max_len 384 --benchmark




# torchrun --standalone --nproc_per_node=8 tests/selfspec_benchmark.py --rank_group 0 1 2 3 4 5 6 7 --benchmark --compile --draft_budget 257 --B 32 --prefix_len 4128 --max_len 4224 --gamma 2
# torchrun --standalone --nproc_per_node=8 tests/selfspec_benchmark.py --rank_group 0 1 2 3 4 5 6 7 --benchmark --compile --draft_budget 257 --B 32 --prefix_len 4128 --max_len 4224 --gamma 3
# torchrun --standalone --nproc_per_node=8 tests/selfspec_benchmark.py --rank_group 0 1 2 3 4 5 6 7 --benchmark --compile --draft_budget 257 --B 32 --prefix_len 4128 --max_len 4224 --gamma 4
# torchrun --standalone --nproc_per_node=8 tests/selfspec_benchmark.py --rank_group 0 1 2 3 4 5 6 7 --benchmark --compile --draft_budget 257 --B 32 --prefix_len 4128 --max_len 4224 --gamma 5
# torchrun --standalone --nproc_per_node=8 tests/selfspec_benchmark.py --rank_group 0 1 2 3 4 5 6 7 --benchmark --compile --draft_budget 257 --B 32 --prefix_len 4128 --max_len 4224 --gamma 6

# torchrun --standalone --nproc_per_node=8 tests/selfspec_benchmark.py --rank_group 0 1 2 3 4 5 6 7 --benchmark --compile --draft_budget 257 --B 64 --prefix_len 4128 --max_len 4224 --gamma 2
# torchrun --standalone --nproc_per_node=8 tests/selfspec_benchmark.py --rank_group 0 1 2 3 4 5 6 7 --benchmark --compile --draft_budget 257 --B 64 --prefix_len 4128 --max_len 4224 --gamma 3
# torchrun --standalone --nproc_per_node=8 tests/selfspec_benchmark.py --rank_group 0 1 2 3 4 5 6 7 --benchmark --compile --draft_budget 257 --B 64 --prefix_len 4128 --max_len 4224 --gamma 4
# torchrun --standalone --nproc_per_node=8 tests/selfspec_benchmark.py --rank_group 0 1 2 3 4 5 6 7 --benchmark --compile --draft_budget 257 --B 64 --prefix_len 4128 --max_len 4224 --gamma 5
# torchrun --standalone --nproc_per_node=8 tests/selfspec_benchmark.py --rank_group 0 1 2 3 4 5 6 7 --benchmark --compile --draft_budget 257 --B 64 --prefix_len 4128 --max_len 4224 --gamma 6

# torchrun --standalone --nproc_per_node=8 tests/selfspec_benchmark.py --rank_group 0 1 2 3 4 5 6 7 --benchmark --compile --draft_budget 257 --B 128 --prefix_len 4128 --max_len 4224 --gamma 2
# torchrun --standalone --nproc_per_node=8 tests/selfspec_benchmark.py --rank_group 0 1 2 3 4 5 6 7 --benchmark --compile --draft_budget 257 --B 128 --prefix_len 4128 --max_len 4224 --gamma 3
# torchrun --standalone --nproc_per_node=8 tests/selfspec_benchmark.py --rank_group 0 1 2 3 4 5 6 7 --benchmark --compile --draft_budget 257 --B 128 --prefix_len 4128 --max_len 4224 --gamma 4
# torchrun --standalone --nproc_per_node=8 tests/selfspec_benchmark.py --rank_group 0 1 2 3 4 5 6 7 --benchmark --compile --draft_budget 257 --B 128 --prefix_len 4128 --max_len 4224 --gamma 5
# torchrun --standalone --nproc_per_node=8 tests/selfspec_benchmark.py --rank_group 0 1 2 3 4 5 6 7 --benchmark --compile --draft_budget 257 --B 128 --prefix_len 4128 --max_len 4224 --gamma 6

# torchrun --standalone --nproc_per_node=8 tests/selfspec_benchmark.py --rank_group 0 1 2 3 4 5 6 7 --benchmark --compile --draft_budget 257 --B 256 --prefix_len 4128 --max_len 4224 --gamma 2
# torchrun --standalone --nproc_per_node=8 tests/selfspec_benchmark.py --rank_group 0 1 2 3 4 5 6 7 --benchmark --compile --draft_budget 257 --B 256 --prefix_len 4128 --max_len 4224 --gamma 3
# torchrun --standalone --nproc_per_node=8 tests/selfspec_benchmark.py --rank_group 0 1 2 3 4 5 6 7 --benchmark --compile --draft_budget 257 --B 256 --prefix_len 4128 --max_len 4224 --gamma 4
# torchrun --standalone --nproc_per_node=8 tests/selfspec_benchmark.py --rank_group 0 1 2 3 4 5 6 7 --benchmark --compile --draft_budget 257 --B 256 --prefix_len 4128 --max_len 4224 --gamma 5
# torchrun --standalone --nproc_per_node=8 tests/selfspec_benchmark.py --rank_group 0 1 2 3 4 5 6 7 --benchmark --compile --draft_budget 257 --B 256 --prefix_len 4128 --max_len 4224 --gamma 6




# torchrun --standalone --nproc_per_node=8 tests/baseline_benchmark.py --compile --rank_group 0 1 2 3 4 5 6 7 --B 32 --prefix_len 16032 --max_len 16128
# torchrun --standalone --nproc_per_node=8 tests/baseline_benchmark.py --compile --rank_group 0 1 2 3 4 5 6 7 --B 64 --prefix_len 16032 --max_len 16128
# torchrun --standalone --nproc_per_node=8 tests/baseline_benchmark.py --compile --rank_group 0 1 2 3 4 5 6 7 --B 128 --prefix_len 16032 --max_len 16128

# torchrun --standalone --nproc_per_node=8 tests/selfspec_benchmark.py --rank_group 0 1 2 3 4 5 6 7 --benchmark --compile --draft_budget 1025 --B 32 --prefix_len 16032 --max_len 16128 --gamma 2
# torchrun --standalone --nproc_per_node=8 tests/selfspec_benchmark.py --rank_group 0 1 2 3 4 5 6 7 --benchmark --compile --draft_budget 1025 --B 32 --prefix_len 16032 --max_len 16128 --gamma 3
# torchrun --standalone --nproc_per_node=8 tests/selfspec_benchmark.py --rank_group 0 1 2 3 4 5 6 7 --benchmark --compile --draft_budget 1025 --B 32 --prefix_len 16032 --max_len 16128 --gamma 4
# torchrun --standalone --nproc_per_node=8 tests/selfspec_benchmark.py --rank_group 0 1 2 3 4 5 6 7 --benchmark --compile --draft_budget 1025 --B 32 --prefix_len 16032 --max_len 16128 --gamma 5
# torchrun --standalone --nproc_per_node=8 tests/selfspec_benchmark.py --rank_group 0 1 2 3 4 5 6 7 --benchmark --compile --draft_budget 1025 --B 32 --prefix_len 16032 --max_len 16128 --gamma 6

# torchrun --standalone --nproc_per_node=8 tests/selfspec_benchmark.py --rank_group 0 1 2 3 4 5 6 7 --benchmark --compile --draft_budget 1025 --B 64 --prefix_len 16032 --max_len 16128 --gamma 2
# torchrun --standalone --nproc_per_node=8 tests/selfspec_benchmark.py --rank_group 0 1 2 3 4 5 6 7 --benchmark --compile --draft_budget 1025 --B 64 --prefix_len 16032 --max_len 16128 --gamma 3
# torchrun --standalone --nproc_per_node=8 tests/selfspec_benchmark.py --rank_group 0 1 2 3 4 5 6 7 --benchmark --compile --draft_budget 1025 --B 64 --prefix_len 16032 --max_len 16128 --gamma 4
# torchrun --standalone --nproc_per_node=8 tests/selfspec_benchmark.py --rank_group 0 1 2 3 4 5 6 7 --benchmark --compile --draft_budget 1025 --B 64 --prefix_len 16032 --max_len 16128 --gamma 5
# torchrun --standalone --nproc_per_node=8 tests/selfspec_benchmark.py --rank_group 0 1 2 3 4 5 6 7 --benchmark --compile --draft_budget 1025 --B 64 --prefix_len 16032 --max_len 16128 --gamma 6

# torchrun --standalone --nproc_per_node=8 tests/selfspec_benchmark.py --rank_group 0 1 2 3 4 5 6 7 --benchmark --compile --draft_budget 1025 --B 128 --prefix_len 16032 --max_len 16128 --gamma 2
# torchrun --standalone --nproc_per_node=8 tests/selfspec_benchmark.py --rank_group 0 1 2 3 4 5 6 7 --benchmark --compile --draft_budget 1025 --B 128 --prefix_len 16032 --max_len 16128 --gamma 3
# torchrun --standalone --nproc_per_node=8 tests/selfspec_benchmark.py --rank_group 0 1 2 3 4 5 6 7 --benchmark --compile --draft_budget 1025 --B 128 --prefix_len 16032 --max_len 16128 --gamma 4
# torchrun --standalone --nproc_per_node=8 tests/selfspec_benchmark.py --rank_group 0 1 2 3 4 5 6 7 --benchmark --compile --draft_budget 1025 --B 128 --prefix_len 16032 --max_len 16128 --gamma 5
# torchrun --standalone --nproc_per_node=8 tests/selfspec_benchmark.py --rank_group 0 1 2 3 4 5 6 7 --benchmark --compile --draft_budget 1025 --B 128 --prefix_len 16032 --max_len 16128 --gamma 6


# ENABLE_INTRA_NODE_COMM=1 torchrun --standalone --nproc_per_node=4 tests/baseline_benchmark.py --compile --rank_group 0 1 2 3 --B 32 --prefix_len 32032 --max_len 32128
# ENABLE_INTRA_NODE_COMM=1 torchrun --standalone --nproc_per_node=4 tests/selfspec_benchmark.py --rank_group 0 1 2 3 --benchmark --compile --draft_budget 2049 --B 32 --prefix_len 32032 --max_len 32128 --gamma 5

# ENABLE_INTRA_NODE_COMM=1 torchrun --standalone --nproc_per_node=8 tests/baseline_benchmark.py --compile --rank_group 0 1 2 3 4 5 6 7 --B 16 --prefix_len 32032 --max_len 32128
# ENABLE_INTRA_NODE_COMM=1 torchrun --standalone --nproc_per_node=8 tests/baseline_benchmark.py --compile --rank_group 0 1 2 3 4 5 6 7 --B 32 --prefix_len 32032 --max_len 32128
# ENABLE_INTRA_NODE_COMM=1 torchrun --standalone --nproc_per_node=8 tests/baseline_benchmark.py --compile --rank_group 0 1 2 3 4 5 6 7 --B 64 --prefix_len 32032 --max_len 32128
# ENABLE_INTRA_NODE_COMM=1 torchrun --standalone --nproc_per_node=8 tests/baseline_benchmark.py --compile --rank_group 0 1 2 3 4 5 6 7 --B 128 --prefix_len 32032 --max_len 32128
# # torchrun --standalone --nproc_per_node=8 tests/baseline_benchmark.py --compile --rank_group 0 1 2 3 4 5 6 7 --B 128 --prefix_len 32032 --max_len 32128

# torchrun --standalone --nproc_per_node=8 tests/selfspec_benchmark.py --rank_group 0 1 2 3 4 5 6 7 --benchmark --compile --draft_budget 2049 --B 16 --prefix_len 32032 --max_len 32128 --gamma 3
# ENABLE_INTRA_NODE_COMM=1 torchrun --standalone --nproc_per_node=8 tests/selfspec_benchmark.py --rank_group 0 1 2 3 4 5 6 7 --benchmark --compile --draft_budget 2049 --B 16 --prefix_len 32032 --max_len 32128 --gamma 4
# ENABLE_INTRA_NODE_COMM=1 torchrun --standalone --nproc_per_node=8 tests/selfspec_benchmark.py --rank_group 0 1 2 3 4 5 6 7 --benchmark --compile --draft_budget 2049 --B 16 --prefix_len 32032 --max_len 32128 --gamma 5
# torchrun --standalone --nproc_per_node=8 tests/selfspec_benchmark.py --rank_group 0 1 2 3 4 5 6 7 --benchmark --compile --draft_budget 2049 --B 16 --prefix_len 32032 --max_len 32128 --gamma 6
# torchrun --standalone --nproc_per_node=8 tests/selfspec_benchmark.py --rank_group 0 1 2 3 4 5 6 7 --benchmark --compile --draft_budget 2049 --B 16 --prefix_len 32032 --max_len 32128 --gamma 7

# torchrun --standalone --nproc_per_node=8 tests/selfspec_benchmark.py --rank_group 0 1 2 3 4 5 6 7 --benchmark --compile --draft_budget 2049 --B 32 --prefix_len 32032 --max_len 32128 --gamma 3
# ENABLE_INTRA_NODE_COMM=1 torchrun --standalone --nproc_per_node=8 tests/selfspec_benchmark.py --rank_group 0 1 2 3 4 5 6 7 --benchmark --compile --draft_budget 2049 --B 32 --prefix_len 32032 --max_len 32128 --gamma 4
# torchrun --standalone --nproc_per_node=8 tests/selfspec_benchmark.py --rank_group 0 1 2 3 4 5 6 7 --benchmark --compile --draft_budget 2049 --B 32 --prefix_len 32032 --max_len 32128 --gamma 5
# torchrun --standalone --nproc_per_node=8 tests/selfspec_benchmark.py --rank_group 0 1 2 3 4 5 6 7 --benchmark --compile --draft_budget 2049 --B 32 --prefix_len 32032 --max_len 32128 --gamma 6
# torchrun --standalone --nproc_per_node=8 tests/selfspec_benchmark.py --rank_group 0 1 2 3 4 5 6 7 --benchmark --compile --draft_budget 2049 --B 32 --prefix_len 32032 --max_len 32128 --gamma 7

# torchrun --standalone --nproc_per_node=8 tests/selfspec_benchmark.py --rank_group 0 1 2 3 4 5 6 7 --benchmark --compile --draft_budget 2049 --B 64 --prefix_len 32032 --max_len 32128 --gamma 3
# torchrun --standalone --nproc_per_node=8 tests/selfspec_benchmark.py --rank_group 0 1 2 3 4 5 6 7 --benchmark --compile --draft_budget 2049 --B 64 --prefix_len 32032 --max_len 32128 --gamma 4
# ENABLE_INTRA_NODE_COMM=1 torchrun --standalone --nproc_per_node=8 tests/selfspec_benchmark.py --rank_group 0 1 2 3 4 5 6 7 --benchmark --compile --draft_budget 2049 --B 64 --prefix_len 32032 --max_len 32128 --gamma 5
# torchrun --standalone --nproc_per_node=8 tests/selfspec_benchmark.py --rank_group 0 1 2 3 4 5 6 7 --benchmark --compile --draft_budget 2049 --B 64 --prefix_len 32032 --max_len 32128 --gamma 6
# torchrun --standalone --nproc_per_node=8 tests/selfspec_benchmark.py --rank_group 0 1 2 3 4 5 6 7 --benchmark --compile --draft_budget 2049 --B 64 --prefix_len 32032 --max_len 32128 --gamma 7

# torchrun --standalone --nproc_per_node=8 tests/selfspec_benchmark.py --rank_group 0 1 2 3 4 5 6 7 --benchmark --compile --draft_budget 2049 --B 128 --prefix_len 32032 --max_len 32128 --gamma 3
# torchrun --standalone --nproc_per_node=8 tests/selfspec_benchmark.py --rank_group 0 1 2 3 4 5 6 7 --benchmark --compile --draft_budget 2049 --B 128 --prefix_len 32032 --max_len 32128 --gamma 4
# torchrun --standalone --nproc_per_node=8 tests/selfspec_benchmark.py --rank_group 0 1 2 3 4 5 6 7 --benchmark --compile --draft_budget 2049 --B 128 --prefix_len 32032 --max_len 32128 --gamma 5
# ENABLE_INTRA_NODE_COMM=1 torchrun --standalone --nproc_per_node=8 tests/selfspec_benchmark.py --rank_group 0 1 2 3 4 5 6 7 --benchmark --compile --draft_budget 2049 --B 128 --prefix_len 32032 --max_len 32128 --gamma 6
# torchrun --standalone --nproc_per_node=8 tests/selfspec_benchmark.py --rank_group 0 1 2 3 4 5 6 7 --benchmark --compile --draft_budget 2049 --B 128 --prefix_len 32032 --max_len 32128 --gamma 7

# torchrun --standalone --nproc_per_node=1 tests/StreamingLLM/longspec_benchmark.py --rank_group 0 --draft_rank_group 0 --benchmark --draft_budget 2049 --B 16 --prefix_len 32001 --max_len 32128 --gamma 1

# srun --overlap --pty --jobid 1834212 torchrun --standalone --nproc_per_node=1 tests/SnapKV/longspec_benchmark.py --rank_group 0 --draft_rank_group 0 --benchmark --draft_budget 513 --B 16 --prefix_len 8065 --max_len 8192 --gamma 1


# torchrun --standalone --nproc_per_node=8 tests/baseline_benchmark.py --compile --rank_group 0 1 2 3 4 5 6 7 --B 32 --prefix_len 64032 --max_len 64128
# ENABLE_INTRA_NODE_COMM=1 torchrun --standalone --nproc_per_node=8 tests/baseline_benchmark.py --compile --rank_group 0 1 2 3 4 5 6 7 --B 64 --prefix_len 64032 --max_len 64128

# torchrun --standalone --nproc_per_node=8 tests/selfspec_benchmark.py --rank_group 0 1 2 3 4 5 6 7 --benchmark --compile --draft_budget 4097 --B 32 --prefix_len 64032 --max_len 64128 --gamma 4
# torchrun --standalone --nproc_per_node=8 tests/selfspec_benchmark.py --rank_group 0 1 2 3 4 5 6 7 --benchmark --compile --draft_budget 4097 --B 32 --prefix_len 64032 --max_len 64128 --gamma 5
# ENABLE_INTRA_NODE_COMM=1 torchrun --standalone --nproc_per_node=8 tests/selfspec_benchmark.py --rank_group 0 1 2 3 4 5 6 7 --benchmark --compile --draft_budget 4097 --B 32 --prefix_len 64032 --max_len 64128 --gamma 6
# torchrun --standalone --nproc_per_node=8 tests/selfspec_benchmark.py --rank_group 0 1 2 3 4 5 6 7 --benchmark --compile --draft_budget 4097 --B 32 --prefix_len 64032 --max_len 64128 --gamma 7
# torchrun --standalone --nproc_per_node=8 tests/selfspec_benchmark.py --rank_group 0 1 2 3 4 5 6 7 --benchmark --compile --draft_budget 4097 --B 32 --prefix_len 64032 --max_len 64128 --gamma 8

# torchrun --standalone --nproc_per_node=8 tests/selfspec_benchmark.py --rank_group 0 1 2 3 4 5 6 7 --benchmark --compile --draft_budget 4097 --B 64 --prefix_len 64032 --max_len 64128 --gamma 4
# torchrun --standalone --nproc_per_node=8 tests/selfspec_benchmark.py --rank_group 0 1 2 3 4 5 6 7 --benchmark --compile --draft_budget 4097 --B 64 --prefix_len 64032 --max_len 64128 --gamma 5
# ENABLE_INTRA_NODE_COMM=1 torchrun --standalone --nproc_per_node=8 tests/selfspec_benchmark.py --rank_group 0 1 2 3 4 5 6 7 --benchmark --compile --draft_budget 4097 --B 64 --prefix_len 64032 --max_len 64128 --gamma 6
# torchrun --standalone --nproc_per_node=8 tests/selfspec_benchmark.py --rank_group 0 1 2 3 4 5 6 7 --benchmark --compile --draft_budget 4097 --B 64 --prefix_len 64032 --max_len 64128 --gamma 7
# torchrun --standalone --nproc_per_node=8 tests/selfspec_benchmark.py --rank_group 0 1 2 3 4 5 6 7 --benchmark --compile --draft_budget 4097 --B 64 --prefix_len 64032 --max_len 64128 --gamma 8




# torchrun --standalone --nproc_per_node=8 tests/baseline_benchmark.py --compile --rank_group 0 1 2 3 4 5 6 7 --B 16 --prefix_len 100000 --max_len 100096
# ENABLE_INTRA_NODE_COMM=1 torchrun --standalone --nproc_per_node=8 tests/baseline_benchmark.py --compile --rank_group 0 1 2 3 4 5 6 7 --B 41 --prefix_len 100000 --max_len 100096
# ENABLE_INTRA_NODE_COMM=1 torchrun --standalone --nproc_per_node=8 tests/selfspec_benchmark.py --rank_group 0 1 2 3 4 5 6 7 --benchmark --compile --draft_budget 4097 --B 41 --prefix_len 100000 --max_len 100096 --gamma 7

# torchrun --standalone --nproc_per_node=8 tests/selfspec_benchmark.py --rank_group 0 1 2 3 4 5 6 7 --benchmark --compile --draft_budget 2049 --B 16 --prefix_len 100000 --max_len 100096 --gamma 4
# torchrun --standalone --nproc_per_node=8 tests/selfspec_benchmark.py --rank_group 0 1 2 3 4 5 6 7 --benchmark --compile --draft_budget 2049 --B 16 --prefix_len 100000 --max_len 100096 --gamma 5
# torchrun --standalone --nproc_per_node=8 tests/selfspec_benchmark.py --rank_group 0 1 2 3 4 5 6 7 --benchmark --compile --draft_budget 2049 --B 16 --prefix_len 100000 --max_len 100096 --gamma 6
# torchrun --standalone --nproc_per_node=8 tests/selfspec_benchmark.py --rank_group 0 1 2 3 4 5 6 7 --benchmark --compile --draft_budget 2049 --B 16 --prefix_len 100000 --max_len 100096 --gamma 7
# torchrun --standalone --nproc_per_node=8 tests/selfspec_benchmark.py --rank_group 0 1 2 3 4 5 6 7 --benchmark --compile --draft_budget 2049 --B 16 --prefix_len 100000 --max_len 100096 --gamma 8

# torchrun --standalone --nproc_per_node=8 tests/selfspec_benchmark.py --rank_group 0 1 2 3 4 5 6 7 --benchmark --compile --draft_budget 2049 --B 32 --prefix_len 100000 --max_len 100096 --gamma 4
# torchrun --standalone --nproc_per_node=8 tests/selfspec_benchmark.py --rank_group 0 1 2 3 4 5 6 7 --benchmark --compile --draft_budget 2049 --B 32 --prefix_len 100000 --max_len 100096 --gamma 5
# torchrun --standalone --nproc_per_node=8 tests/selfspec_benchmark.py --rank_group 0 1 2 3 4 5 6 7 --benchmark --compile --draft_budget 2049 --B 32 --prefix_len 100000 --max_len 100096 --gamma 6
# torchrun --standalone --nproc_per_node=8 tests/selfspec_benchmark.py --rank_group 0 1 2 3 4 5 6 7 --benchmark --compile --draft_budget 2049 --B 32 --prefix_len 100000 --max_len 100096 --gamma 7
# torchrun --standalone --nproc_per_node=8 tests/selfspec_benchmark.py --rank_group 0 1 2 3 4 5 6 7 --benchmark --compile --draft_budget 2049 --B 32 --prefix_len 100000 --max_len 100096 --gamma 8

# torchrun --standalone --nproc_per_node=8 tests/selfspec_benchmark.py --rank_group 0 1 2 3 4 5 6 7 --benchmark --compile --draft_budget 4097 --B 16 --prefix_len 100000 --max_len 100096 --gamma 4
# torchrun --standalone --nproc_per_node=8 tests/selfspec_benchmark.py --rank_group 0 1 2 3 4 5 6 7 --benchmark --compile --draft_budget 4097 --B 16 --prefix_len 100000 --max_len 100096 --gamma 5
# torchrun --standalone --nproc_per_node=8 tests/selfspec_benchmark.py --rank_group 0 1 2 3 4 5 6 7 --benchmark --compile --draft_budget 4097 --B 16 --prefix_len 100000 --max_len 100096 --gamma 6
# torchrun --standalone --nproc_per_node=8 tests/selfspec_benchmark.py --rank_group 0 1 2 3 4 5 6 7 --benchmark --compile --draft_budget 4097 --B 16 --prefix_len 100000 --max_len 100096 --gamma 7
# torchrun --standalone --nproc_per_node=8 tests/selfspec_benchmark.py --rank_group 0 1 2 3 4 5 6 7 --benchmark --compile --draft_budget 4097 --B 16 --prefix_len 100000 --max_len 100096 --gamma 8

# torchrun --standalone --nproc_per_node=8 tests/selfspec_benchmark.py --rank_group 0 1 2 3 4 5 6 7 --benchmark --compile --draft_budget 4097 --B 32 --prefix_len 100000 --max_len 100096 --gamma 4
# torchrun --standalone --nproc_per_node=8 tests/selfspec_benchmark.py --rank_group 0 1 2 3 4 5 6 7 --benchmark --compile --draft_budget 4097 --B 32 --prefix_len 100000 --max_len 100096 --gamma 5
# torchrun --standalone --nproc_per_node=8 tests/selfspec_benchmark.py --rank_group 0 1 2 3 4 5 6 7 --benchmark --compile --draft_budget 4097 --B 32 --prefix_len 100000 --max_len 100096 --gamma 6
# ENABLE_INTRA_NODE_COMM=1 torchrun --standalone --nproc_per_node=8 tests/selfspec_benchmark.py --rank_group 0 1 2 3 4 5 6 7 --benchmark --compile --draft_budget 4097 --B 41 --prefix_len 100000 --max_len 100096 --gamma 7
# torchrun --standalone --nproc_per_node=8 tests/selfspec_benchmark.py --rank_group 0 1 2 3 4 5 6 7 --benchmark --compile --draft_budget 4097 --B 32 --prefix_len 100000 --max_len 100096 --gamma 8

# torchrun --standalone --nproc_per_node=8 tests/selfspec_benchmark.py --rank_group 0 1 2 3 4 5 6 7 --benchmark --compile --draft_budget 8193 --B 16 --prefix_len 100000 --max_len 100096 --gamma 4
# torchrun --standalone --nproc_per_node=8 tests/selfspec_benchmark.py --rank_group 0 1 2 3 4 5 6 7 --benchmark --compile --draft_budget 8193 --B 16 --prefix_len 100000 --max_len 100096 --gamma 5
# torchrun --standalone --nproc_per_node=8 tests/selfspec_benchmark.py --rank_group 0 1 2 3 4 5 6 7 --benchmark --compile --draft_budget 8193 --B 16 --prefix_len 100000 --max_len 100096 --gamma 6
# torchrun --standalone --nproc_per_node=8 tests/selfspec_benchmark.py --rank_group 0 1 2 3 4 5 6 7 --benchmark --compile --draft_budget 8193 --B 16 --prefix_len 100000 --max_len 100096 --gamma 7
# torchrun --standalone --nproc_per_node=8 tests/selfspec_benchmark.py --rank_group 0 1 2 3 4 5 6 7 --benchmark --compile --draft_budget 8193 --B 16 --prefix_len 100000 --max_len 100096 --gamma 8

# torchrun --standalone --nproc_per_node=8 tests/selfspec_benchmark.py --rank_group 0 1 2 3 4 5 6 7 --benchmark --compile --draft_budget 8193 --B 32 --prefix_len 100000 --max_len 100096 --gamma 4
# torchrun --standalone --nproc_per_node=8 tests/selfspec_benchmark.py --rank_group 0 1 2 3 4 5 6 7 --benchmark --compile --draft_budget 8193 --B 32 --prefix_len 100000 --max_len 100096 --gamma 5
# torchrun --standalone --nproc_per_node=8 tests/selfspec_benchmark.py --rank_group 0 1 2 3 4 5 6 7 --benchmark --compile --draft_budget 8193 --B 32 --prefix_len 100000 --max_len 100096 --gamma 6
# torchrun --standalone --nproc_per_node=8 tests/selfspec_benchmark.py --rank_group 0 1 2 3 4 5 6 7 --benchmark --compile --draft_budget 8193 --B 32 --prefix_len 100000 --max_len 100096 --gamma 7
# torchrun --standalone --nproc_per_node=8 tests/selfspec_benchmark.py --rank_group 0 1 2 3 4 5 6 7 --benchmark --compile --draft_budget 8193 --B 32 --prefix_len 100000 --max_len 100096 --gamma 8


