export ENABLE_INTRA_NODE_COMM=1

torchrun --standalone --nproc_per_node=8 tests/baseline_benchmark.py --compile --rank_group 0 1 2 3 4 5 6 7 --B 45 --prefix_len 100000 --max_len 100096

torchrun --standalone --nproc_per_node=8 tests/selfspec_benchmark.py --rank_group 0 1 2 3 4 5 6 7 --benchmark --compile --draft_budget 4097 --B 45 --prefix_len 100000 --max_len 100096 --gamma 6
torchrun --standalone --nproc_per_node=8 tests/selfspec_benchmark.py --rank_group 0 1 2 3 4 5 6 7 --benchmark --compile --draft_budget 4097 --B 45 --prefix_len 100000 --max_len 100096 --gamma 7
torchrun --standalone --nproc_per_node=8 tests/selfspec_benchmark.py --rank_group 0 1 2 3 4 5 6 7 --benchmark --compile --draft_budget 4097 --B 45 --prefix_len 100000 --max_len 100096 --gamma 8
torchrun --standalone --nproc_per_node=8 tests/selfspec_benchmark.py --rank_group 0 1 2 3 4 5 6 7 --benchmark --compile --draft_budget 4097 --B 45 --prefix_len 100000 --max_len 100096 --gamma 9