export ENABLE_INTRA_NODE_COMM=1

# torchrun --standalone --nproc_per_node=8 tests/baseline_benchmark.py --compile --rank_group 0 1 2 3 4 5 6 7 --B 32 --prefix_len 4128 --max_len 4224
# torchrun --standalone --nproc_per_node=8 tests/baseline_benchmark.py --compile --rank_group 0 1 2 3 4 5 6 7 --B 64 --prefix_len 4128 --max_len 4224
# torchrun --standalone --nproc_per_node=8 tests/baseline_benchmark.py --compile --rank_group 0 1 2 3 4 5 6 7 --B 128 --prefix_len 4128 --max_len 4224
# torchrun --standalone --nproc_per_node=8 tests/baseline_benchmark.py --compile --rank_group 0 1 2 3 4 5 6 7 --B 256 --prefix_len 4128 --max_len 4224

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


ENABLE_INTRA_NODE_COMM=1 torchrun --standalone --nproc_per_node=4 tests/baseline_benchmark.py --compile --rank_group 0 1 2 3 --B 32 --prefix_len 32032 --max_len 32128
ENABLE_INTRA_NODE_COMM=1 torchrun --standalone --nproc_per_node=4 tests/selfspec_benchmark.py --rank_group 0 1 2 3 --benchmark --compile --draft_budget 2049 --B 32 --prefix_len 32032 --max_len 32128 --gamma 5

ENABLE_INTRA_NODE_COMM=1 torchrun --standalone --nproc_per_node=8 tests/baseline_benchmark.py --compile --rank_group 0 1 2 3 4 5 6 7 --B 16 --prefix_len 32032 --max_len 32128
ENABLE_INTRA_NODE_COMM=1 torchrun --standalone --nproc_per_node=8 tests/baseline_benchmark.py --compile --rank_group 0 1 2 3 4 5 6 7 --B 32 --prefix_len 32032 --max_len 32128
ENABLE_INTRA_NODE_COMM=1 torchrun --standalone --nproc_per_node=8 tests/baseline_benchmark.py --compile --rank_group 0 1 2 3 4 5 6 7 --B 64 --prefix_len 32032 --max_len 32128
ENABLE_INTRA_NODE_COMM=1 torchrun --standalone --nproc_per_node=8 tests/baseline_benchmark.py --compile --rank_group 0 1 2 3 4 5 6 7 --B 128 --prefix_len 32032 --max_len 32128
# torchrun --standalone --nproc_per_node=8 tests/baseline_benchmark.py --compile --rank_group 0 1 2 3 4 5 6 7 --B 128 --prefix_len 32032 --max_len 32128

torchrun --standalone --nproc_per_node=8 tests/selfspec_benchmark.py --rank_group 0 1 2 3 4 5 6 7 --benchmark --compile --draft_budget 2049 --B 16 --prefix_len 32032 --max_len 32128 --gamma 3
ENABLE_INTRA_NODE_COMM=1 torchrun --standalone --nproc_per_node=8 tests/selfspec_benchmark.py --rank_group 0 1 2 3 4 5 6 7 --benchmark --compile --draft_budget 2049 --B 16 --prefix_len 32032 --max_len 32128 --gamma 4
ENABLE_INTRA_NODE_COMM=1 torchrun --standalone --nproc_per_node=8 tests/selfspec_benchmark.py --rank_group 0 1 2 3 4 5 6 7 --benchmark --compile --draft_budget 2049 --B 16 --prefix_len 32032 --max_len 32128 --gamma 5
torchrun --standalone --nproc_per_node=8 tests/selfspec_benchmark.py --rank_group 0 1 2 3 4 5 6 7 --benchmark --compile --draft_budget 2049 --B 16 --prefix_len 32032 --max_len 32128 --gamma 6
torchrun --standalone --nproc_per_node=8 tests/selfspec_benchmark.py --rank_group 0 1 2 3 4 5 6 7 --benchmark --compile --draft_budget 2049 --B 16 --prefix_len 32032 --max_len 32128 --gamma 7

torchrun --standalone --nproc_per_node=8 tests/selfspec_benchmark.py --rank_group 0 1 2 3 4 5 6 7 --benchmark --compile --draft_budget 2049 --B 32 --prefix_len 32032 --max_len 32128 --gamma 3
ENABLE_INTRA_NODE_COMM=1 torchrun --standalone --nproc_per_node=8 tests/selfspec_benchmark.py --rank_group 0 1 2 3 4 5 6 7 --benchmark --compile --draft_budget 2049 --B 32 --prefix_len 32032 --max_len 32128 --gamma 4
torchrun --standalone --nproc_per_node=8 tests/selfspec_benchmark.py --rank_group 0 1 2 3 4 5 6 7 --benchmark --compile --draft_budget 2049 --B 32 --prefix_len 32032 --max_len 32128 --gamma 5
torchrun --standalone --nproc_per_node=8 tests/selfspec_benchmark.py --rank_group 0 1 2 3 4 5 6 7 --benchmark --compile --draft_budget 2049 --B 32 --prefix_len 32032 --max_len 32128 --gamma 6
torchrun --standalone --nproc_per_node=8 tests/selfspec_benchmark.py --rank_group 0 1 2 3 4 5 6 7 --benchmark --compile --draft_budget 2049 --B 32 --prefix_len 32032 --max_len 32128 --gamma 7

torchrun --standalone --nproc_per_node=8 tests/selfspec_benchmark.py --rank_group 0 1 2 3 4 5 6 7 --benchmark --compile --draft_budget 2049 --B 64 --prefix_len 32032 --max_len 32128 --gamma 3
torchrun --standalone --nproc_per_node=8 tests/selfspec_benchmark.py --rank_group 0 1 2 3 4 5 6 7 --benchmark --compile --draft_budget 2049 --B 64 --prefix_len 32032 --max_len 32128 --gamma 4
ENABLE_INTRA_NODE_COMM=1 torchrun --standalone --nproc_per_node=8 tests/selfspec_benchmark.py --rank_group 0 1 2 3 4 5 6 7 --benchmark --compile --draft_budget 2049 --B 64 --prefix_len 32032 --max_len 32128 --gamma 5
torchrun --standalone --nproc_per_node=8 tests/selfspec_benchmark.py --rank_group 0 1 2 3 4 5 6 7 --benchmark --compile --draft_budget 2049 --B 64 --prefix_len 32032 --max_len 32128 --gamma 6
torchrun --standalone --nproc_per_node=8 tests/selfspec_benchmark.py --rank_group 0 1 2 3 4 5 6 7 --benchmark --compile --draft_budget 2049 --B 64 --prefix_len 32032 --max_len 32128 --gamma 7

torchrun --standalone --nproc_per_node=8 tests/selfspec_benchmark.py --rank_group 0 1 2 3 4 5 6 7 --benchmark --compile --draft_budget 2049 --B 128 --prefix_len 32032 --max_len 32128 --gamma 3
torchrun --standalone --nproc_per_node=8 tests/selfspec_benchmark.py --rank_group 0 1 2 3 4 5 6 7 --benchmark --compile --draft_budget 2049 --B 128 --prefix_len 32032 --max_len 32128 --gamma 4
torchrun --standalone --nproc_per_node=8 tests/selfspec_benchmark.py --rank_group 0 1 2 3 4 5 6 7 --benchmark --compile --draft_budget 2049 --B 128 --prefix_len 32032 --max_len 32128 --gamma 5
ENABLE_INTRA_NODE_COMM=1 torchrun --standalone --nproc_per_node=8 tests/selfspec_benchmark.py --rank_group 0 1 2 3 4 5 6 7 --benchmark --compile --draft_budget 2049 --B 128 --prefix_len 32032 --max_len 32128 --gamma 6
torchrun --standalone --nproc_per_node=8 tests/selfspec_benchmark.py --rank_group 0 1 2 3 4 5 6 7 --benchmark --compile --draft_budget 2049 --B 128 --prefix_len 32032 --max_len 32128 --gamma 7




torchrun --standalone --nproc_per_node=8 tests/baseline_benchmark.py --compile --rank_group 0 1 2 3 4 5 6 7 --B 32 --prefix_len 64032 --max_len 64128
ENABLE_INTRA_NODE_COMM=1 torchrun --standalone --nproc_per_node=8 tests/baseline_benchmark.py --compile --rank_group 0 1 2 3 4 5 6 7 --B 64 --prefix_len 64032 --max_len 64128

torchrun --standalone --nproc_per_node=8 tests/selfspec_benchmark.py --rank_group 0 1 2 3 4 5 6 7 --benchmark --compile --draft_budget 4097 --B 32 --prefix_len 64032 --max_len 64128 --gamma 4
torchrun --standalone --nproc_per_node=8 tests/selfspec_benchmark.py --rank_group 0 1 2 3 4 5 6 7 --benchmark --compile --draft_budget 4097 --B 32 --prefix_len 64032 --max_len 64128 --gamma 5
ENABLE_INTRA_NODE_COMM=1 torchrun --standalone --nproc_per_node=8 tests/selfspec_benchmark.py --rank_group 0 1 2 3 4 5 6 7 --benchmark --compile --draft_budget 4097 --B 32 --prefix_len 64032 --max_len 64128 --gamma 6
torchrun --standalone --nproc_per_node=8 tests/selfspec_benchmark.py --rank_group 0 1 2 3 4 5 6 7 --benchmark --compile --draft_budget 4097 --B 32 --prefix_len 64032 --max_len 64128 --gamma 7
torchrun --standalone --nproc_per_node=8 tests/selfspec_benchmark.py --rank_group 0 1 2 3 4 5 6 7 --benchmark --compile --draft_budget 4097 --B 32 --prefix_len 64032 --max_len 64128 --gamma 8

torchrun --standalone --nproc_per_node=8 tests/selfspec_benchmark.py --rank_group 0 1 2 3 4 5 6 7 --benchmark --compile --draft_budget 4097 --B 64 --prefix_len 64032 --max_len 64128 --gamma 4
torchrun --standalone --nproc_per_node=8 tests/selfspec_benchmark.py --rank_group 0 1 2 3 4 5 6 7 --benchmark --compile --draft_budget 4097 --B 64 --prefix_len 64032 --max_len 64128 --gamma 5
ENABLE_INTRA_NODE_COMM=1 torchrun --standalone --nproc_per_node=8 tests/selfspec_benchmark.py --rank_group 0 1 2 3 4 5 6 7 --benchmark --compile --draft_budget 4097 --B 64 --prefix_len 64032 --max_len 64128 --gamma 6
torchrun --standalone --nproc_per_node=8 tests/selfspec_benchmark.py --rank_group 0 1 2 3 4 5 6 7 --benchmark --compile --draft_budget 4097 --B 64 --prefix_len 64032 --max_len 64128 --gamma 7
torchrun --standalone --nproc_per_node=8 tests/selfspec_benchmark.py --rank_group 0 1 2 3 4 5 6 7 --benchmark --compile --draft_budget 4097 --B 64 --prefix_len 64032 --max_len 64128 --gamma 8




torchrun --standalone --nproc_per_node=8 tests/baseline_benchmark.py --compile --rank_group 0 1 2 3 4 5 6 7 --B 16 --prefix_len 100000 --max_len 100096
ENABLE_INTRA_NODE_COMM=1 torchrun --standalone --nproc_per_node=8 tests/baseline_benchmark.py --compile --rank_group 0 1 2 3 4 5 6 7 --B 41 --prefix_len 100000 --max_len 100096
ENABLE_INTRA_NODE_COMM=1 torchrun --standalone --nproc_per_node=8 tests/selfspec_benchmark.py --rank_group 0 1 2 3 4 5 6 7 --benchmark --compile --draft_budget 4097 --B 41 --prefix_len 100000 --max_len 100096 --gamma 7

torchrun --standalone --nproc_per_node=8 tests/selfspec_benchmark.py --rank_group 0 1 2 3 4 5 6 7 --benchmark --compile --draft_budget 2049 --B 16 --prefix_len 100000 --max_len 100096 --gamma 4
torchrun --standalone --nproc_per_node=8 tests/selfspec_benchmark.py --rank_group 0 1 2 3 4 5 6 7 --benchmark --compile --draft_budget 2049 --B 16 --prefix_len 100000 --max_len 100096 --gamma 5
torchrun --standalone --nproc_per_node=8 tests/selfspec_benchmark.py --rank_group 0 1 2 3 4 5 6 7 --benchmark --compile --draft_budget 2049 --B 16 --prefix_len 100000 --max_len 100096 --gamma 6
torchrun --standalone --nproc_per_node=8 tests/selfspec_benchmark.py --rank_group 0 1 2 3 4 5 6 7 --benchmark --compile --draft_budget 2049 --B 16 --prefix_len 100000 --max_len 100096 --gamma 7
torchrun --standalone --nproc_per_node=8 tests/selfspec_benchmark.py --rank_group 0 1 2 3 4 5 6 7 --benchmark --compile --draft_budget 2049 --B 16 --prefix_len 100000 --max_len 100096 --gamma 8

torchrun --standalone --nproc_per_node=8 tests/selfspec_benchmark.py --rank_group 0 1 2 3 4 5 6 7 --benchmark --compile --draft_budget 2049 --B 32 --prefix_len 100000 --max_len 100096 --gamma 4
torchrun --standalone --nproc_per_node=8 tests/selfspec_benchmark.py --rank_group 0 1 2 3 4 5 6 7 --benchmark --compile --draft_budget 2049 --B 32 --prefix_len 100000 --max_len 100096 --gamma 5
torchrun --standalone --nproc_per_node=8 tests/selfspec_benchmark.py --rank_group 0 1 2 3 4 5 6 7 --benchmark --compile --draft_budget 2049 --B 32 --prefix_len 100000 --max_len 100096 --gamma 6
torchrun --standalone --nproc_per_node=8 tests/selfspec_benchmark.py --rank_group 0 1 2 3 4 5 6 7 --benchmark --compile --draft_budget 2049 --B 32 --prefix_len 100000 --max_len 100096 --gamma 7
torchrun --standalone --nproc_per_node=8 tests/selfspec_benchmark.py --rank_group 0 1 2 3 4 5 6 7 --benchmark --compile --draft_budget 2049 --B 32 --prefix_len 100000 --max_len 100096 --gamma 8

torchrun --standalone --nproc_per_node=8 tests/selfspec_benchmark.py --rank_group 0 1 2 3 4 5 6 7 --benchmark --compile --draft_budget 4097 --B 16 --prefix_len 100000 --max_len 100096 --gamma 4
torchrun --standalone --nproc_per_node=8 tests/selfspec_benchmark.py --rank_group 0 1 2 3 4 5 6 7 --benchmark --compile --draft_budget 4097 --B 16 --prefix_len 100000 --max_len 100096 --gamma 5
torchrun --standalone --nproc_per_node=8 tests/selfspec_benchmark.py --rank_group 0 1 2 3 4 5 6 7 --benchmark --compile --draft_budget 4097 --B 16 --prefix_len 100000 --max_len 100096 --gamma 6
torchrun --standalone --nproc_per_node=8 tests/selfspec_benchmark.py --rank_group 0 1 2 3 4 5 6 7 --benchmark --compile --draft_budget 4097 --B 16 --prefix_len 100000 --max_len 100096 --gamma 7
torchrun --standalone --nproc_per_node=8 tests/selfspec_benchmark.py --rank_group 0 1 2 3 4 5 6 7 --benchmark --compile --draft_budget 4097 --B 16 --prefix_len 100000 --max_len 100096 --gamma 8

torchrun --standalone --nproc_per_node=8 tests/selfspec_benchmark.py --rank_group 0 1 2 3 4 5 6 7 --benchmark --compile --draft_budget 4097 --B 32 --prefix_len 100000 --max_len 100096 --gamma 4
torchrun --standalone --nproc_per_node=8 tests/selfspec_benchmark.py --rank_group 0 1 2 3 4 5 6 7 --benchmark --compile --draft_budget 4097 --B 32 --prefix_len 100000 --max_len 100096 --gamma 5
torchrun --standalone --nproc_per_node=8 tests/selfspec_benchmark.py --rank_group 0 1 2 3 4 5 6 7 --benchmark --compile --draft_budget 4097 --B 32 --prefix_len 100000 --max_len 100096 --gamma 6
ENABLE_INTRA_NODE_COMM=1 torchrun --standalone --nproc_per_node=8 tests/selfspec_benchmark.py --rank_group 0 1 2 3 4 5 6 7 --benchmark --compile --draft_budget 4097 --B 41 --prefix_len 100000 --max_len 100096 --gamma 7
torchrun --standalone --nproc_per_node=8 tests/selfspec_benchmark.py --rank_group 0 1 2 3 4 5 6 7 --benchmark --compile --draft_budget 4097 --B 32 --prefix_len 100000 --max_len 100096 --gamma 8

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


