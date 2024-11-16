<div align="center">
<h1><img src="static/images/icons/MagicDec.png" height="40px" align="top"/> MagicDec: Breaking Throughput-Latency Trade-off for Long Context Generation <br> with Speculative Decoding
</h1>

</div>
<div align="center">
<b><a href="https://github.com/jianc99">Jian Chen*</a></b><sup>1</sup>,
<b><a href="https://github.com/Vashistht">Vashisth Tiwari*</a></b><sup>1</sup>,
<b><a href="https://github.com/ranonrkm">Ranajoy Sadhukhan*</a></b><sup>1</sup>,
<b><a href="https://github.com/dreaming-panda">Zhuoming Chen</a></b><sup>1</sup>,
<b><a >Jinyuan Shi</a></b><sup>2</sup>,
<b><a >Ian En-Hsu Yen</a></b><sup>2</sup>,
<b><a href="https://github.com/keroro824">Beidi Chen</a></b><sup>1,3</sup>
</div>

<div align="center">
<sup>1</sup>Carnegie Mellon University
<sup>2</sup>Moffett AI
<sup>3</sup>Meta AI (FAIR)
</div>

<div align="center">
[<a href="https://arxiv.org/abs/2408.11049">Paper</a>] | [<a href="https://infini-ai-lab.github.io/MagicDec">Blog</a>]
</div>
<br>

## Update
Happy to share the latest update of MagicDec. Now MagicDec integerates flashinfer and paged attention to further accelerate inference. We add support of SnapKV-based drafting for higher speculation quality. Please make sure PyTorch version greater than 2.5 to use the new features like custom all-reduce can be used.

## Installation

### Environment Set Up
``` bash
conda create -n magicdec python=3.11
conda activate magicdec
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install flashinfer -i https://flashinfer.ai/whl/cu121/torch2.4/
```

### Prepare Checkpoints
Currently, we support [Llama-2-7b](https://huggingface.co/meta-llama/Llama-2-7b-hf) and its long context variant [Llama-2-7b-32k](https://huggingface.co/togethercomputer/LLaMA-2-7B-32K), [Llama-2-13b](https://huggingface.co/meta-llama/Llama-2-13b-hf), [Llama-2-70b](https://huggingface.co/meta-llama/Llama-2-70b-hf), [Llama-3-8b](https://huggingface.co/meta-llama/Meta-Llama-3-8B), [Llama-3-70b](https://huggingface.co/meta-llama/Meta-Llama-3-70B), [llama-68m](https://huggingface.co/JackFram/llama-68m), [TinyLlama](https://huggingface.co/TinyLlama/TinyLlama_v1.1), [Llama-3.1-8b](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B), [Llama-3.1-70b](https://huggingface.co/meta-llama/Llama-3.1-70B) and [Llama-3.2-1b](https://huggingface.co/meta-llama/Llama-3.2-1B).

We can first download the checkpoints we need through `download.py`. The `--repo_id` should be set to the repository ID to download from. The `--hf_token` should be your HuggingFace API token. The `--out_dir` should be the directory you want to save the checkpoint.
```bash
python download.py --repo_id meta-llama/Meta-Llama-3.1-8B --hf_token "YOUR HUGGINGFACE API TOKEN" --out_dir checkpoints/meta-llama/Meta-Llama-3.1-8B
```
Then we need to convert the downloaded checkpoint. `--checkpoint_dir` should be set to the directory we just saved the checkpoint. This script will generate a new `model.pth` file in the configured directory.
```bash
python convert_hf_checkpoint.py --checkpoint_dir checkpoints/meta-llama/Meta-Llama-3.1-8B
```

## Evaluations
We conducted all the experiments in the paper on 8xA100, 8xH100 and 8xL40. We used PG-19 as the dataset for all the experiments.
### Baseline
We used the new one-shot and two-shot all-reduce of PyTorch 2.5 by setting `ENABLE_INTRA_NODE_COMM=1`. `--nproc_per_node` should be set to the number of GPUs you want to do tensor parallelism. `--model` should be set to the directory of the `model.pth`, which is the checkpoint we want to serve. `--model_name` should be set to the repo id of the checkpoint, which is used to load tokenizer. `--rank_group` should be set to the list of GPU id in tensor parallelism. `--B` is the batch size, `--prefix_len` is the prefill length, `--max_len` is the max number of tokens we want to generate for each sentence. `--printoutput` is the flag which decides whether or not to print the output after generation finishes. `--compile` is the flag to decide whether or not use torch.compile to accelerate the generation.
```bash
ENABLE_INTRA_NODE_COMM=1 torchrun --standalone --nproc_per_node=8 tests/baseline_benchmark.py --model checkpoints/meta-llama/Meta-Llama-3.1-8B/model.pth --model_name meta-llama/Meta-Llama-3.1-8B --rank_group 0 1 2 3 4 5 6 7 --B 1 --prefix_len 3969 --max_len 4096 --printoutput --compile
```

### Standalone Draft
For standalone draft experiment, we use `--target` and `--model` to set the target and draft checkpoint. `--model_name` should be set to the repo id of target model, which will used to load the corresponding tokenizer. `--rank_group` should be set to the GPU id we want to do tensor parallelism for the target model, and `--draft_rank_group` should be set to the GPU id we want to do TP for the draft model. `--draft_budget` should be set to the KV budget for the draft model. Set `--draft_budget` to -1 to disable KV compression of draft (Use full KV).

#### SnapKV-based Drafting
```bash
ENABLE_INTRA_NODE_COMM=1 torchrun --standalone --nproc_per_node=8 tests/SnapKV/longspec_benchmark.py --target checkpoints/meta-llama/Meta-Llama-3.1-8B/model.pth --model checkpoints/meta-llama/Llama-3.2-1B/model.pth --model_name meta-llama/Meta-Llama-3.1-8B --rank_group 0 1 2 3 4 5 6 7 --draft_rank_group 0 1 2 3 --gamma 3 --B 64 --prefix_len 16032 --max_len 16128 --draft_budget 257 --benchmark --compile
```

#### StreamingLLM-based Drafting
```bash
ENABLE_INTRA_NODE_COMM=1 torchrun --standalone --nproc_per_node=8 tests/StreamingLLM/longspec_benchmark.py --target checkpoints/meta-llama/Meta-Llama-3.1-8B/model.pth --model checkpoints/meta-llama/Llama-3.2-1B/model.pth --model_name meta-llama/Meta-Llama-3.1-8B --rank_group 0 1 2 3 4 5 6 7 --draft_rank_group 0 1 2 3 --gamma 3 --B 64 --prefix_len 16032 --max_len 16128 --draft_budget 257 --benchmark --compile
```

### Self-Speculation
Similar to the standalone draft, but here we do not need to configure the draft model as it is the target model itself.

#### SnapKV-based Drafting
```bash
ENABLE_INTRA_NODE_COMM=1 torchrun --standalone --nproc_per_node=8 tests/SnapKV/selfspec_benchmark.py --model checkpoints/meta-llama/Meta-Llama-3.1-8B/model.pth --model_name meta-llama/Meta-Llama-3.1-8B --rank_group 0 1 2 3 4 5 6 7 --gamma 3 --B 64 --prefix_len 16032 --max_len 16128 --draft_budget 257 --benchmark --compile
```

#### StreamingLLM-based Drafting
```bash
ENABLE_INTRA_NODE_COMM=1 torchrun --standalone --nproc_per_node=8 tests/SnapKV/selfspec_benchmark.py --model checkpoints/meta-llama/Meta-Llama-3.1-8B/model.pth --model_name meta-llama/Meta-Llama-3.1-8B --rank_group 0 1 2 3 4 5 6 7 --gamma 3 --B 64 --prefix_len 16032 --gen_len 16128 --draft_budget 257 --benchmark --compile
```

## Citation
If you find MagicDec useful or relevant to your project and research, please kindly cite our paper:

```bibtex
@article{chen2024magicdec,
  title={MagicDec: Breaking the Latency-Throughput Tradeoff for Long Context Generation with Speculative Decoding},
  author={Chen, Jian and Tiwari, Vashisth and Sadhukhan, Ranajoy and Chen, Zhuoming and Shi, Jinyuan and Yen, Ian En-Hsu and Chen, Beidi},
  journal={arXiv preprint arXiv:2408.11049},
  year={2024}
}
```

