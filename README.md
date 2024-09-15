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

## Installation

### Environment Set Up
``` bash
conda create -n magicdec python=3.11
conda activate magicdec
pip install torch==2.5.0.dev20240813+cu121 --index-url https://download.pytorch.org/whl/nightly/cu121/
pip install -r requirements.txt
pip install flashinfer -i https://flashinfer.ai/whl/cu121/torch2.4/
```

### Prepare Checkpoints
Currently, we support [Llama-2-7b](https://huggingface.co/meta-llama/Llama-2-7b-hf) and its long context variant [Llama-2-7b-32k](https://huggingface.co/togethercomputer/LLaMA-2-7B-32K), [Llama-2-13b](https://huggingface.co/meta-llama/Llama-2-13b-hf), [Llama-2-70b](https://huggingface.co/meta-llama/Llama-2-70b-hf), [Llama-3-8b](https://huggingface.co/meta-llama/Meta-Llama-3-8B), [Llama-3-70b](https://huggingface.co/meta-llama/Meta-Llama-3-70B), [Llama-3.1-8b](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B), [llama-68m](https://huggingface.co/JackFram/llama-68m) and [TinyLlama](https://huggingface.co/TinyLlama/TinyLlama_v1.1).

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
We used the new one-shot and two-shot all reduce of Pytorch nightly by setting `ENABLE_INTRA_NODE_COMM=1`. `--nproc_per_node` should be set to the number of GPUs you want to do tensor parallelism. `--model` should be set to the directory of the `model.pth` file of the checkpoint we want to serve. `--model_name` should be set to the repo id of the checkpoint. `--rank_group` should be set to the list of GPU id in tensor parallelism. `--B` is the batch size, `--prefix_len` is the prefill length, `--gen_len` is the number of tokens we want to generate for each sentence. `--printoutput` is the flag which decides whether or not print the output after generation finishes. `--compile` is the flag to decide whether or not use torch.compile to accelerate the generation.
```bash
ENABLE_INTRA_NODE_COMM=1 torchrun --standalone --nproc_per_node=8 tests/baseline_benchmark.py --model checkpoints/meta-llama/Meta-Llama-3.1-8B/model.pth --model_name meta-llama/Meta-Llama-3.1-8B --rank_group 0 1 2 3 4 5 6 7 --B 1 --prefix_len 4000 --gen_len 64 --printoutput --compile
```

### Standalone Draft
For standalone draft experiment, we use `--target` and `--model` to set the target and draft checkpoint. `--model_name` should be set to the repo id of target model, which will used to load the corresponding tokenizer. `--rank_group` should be set to the GPU id we want to do tensor parallelism for the target model, and `--draft_group` should be set to the GPU id we want to do TP for the draft model. Here as Tinyllama 1.1b model only has 4 KV heads, so we can only use 4 GPUs to do TP for it. `--streamingllm_budget` should be set to the KV budget for the draft model.
```bash
ENABLE_INTRA_NODE_COMM=1 torchrun --standalone --nproc_per_node=8 tests/longspec_benchmark.py --target checkpoints/togethercomputer/LLaMA-2-7B-32K/model.pth --model checkpoints/TinyLlama/TinyLlama_v1.1/model.pth --model_name togethercomputer/LLaMA-2-7B-32K --rank_group 0 1 2 3 4 5 6 7 --draft_ranks 0 1 2 3 --gamma 3 --B 64 --prefix_len 16000 --gen_len 64 --streamingllm_budget 256 --benchmark --compile
```

### Self-Speculation
Similar to the standalone draft, but here we do not need to configure the draft model as it is the target model itself.
```bash
ENABLE_INTRA_NODE_COMM=1 torchrun --standalone --nproc_per_node=8 tests/selfspec_benchmark.py --model checkpoints/meta-llama/Meta-Llama-3.1-8B/model.pth --model_name meta-llama/Meta-Llama-3.1-8B --rank_group 0 1 2 3 4 5 6 7 --gamma 3 --B 64 --prefix_len 16000 --gen_len 64 --streamingllm_budget 256 --benchmark --compile
```

## Environment Issue
We discovered that installing Flash-Attention directly with the PyTorch nightly build causes performance issues. However, these issues are resolved if we first install PyTorch 2.4.0 along with Flash-Attention, and then upgrade to the nightly version of PyTorch. We have adopted this approach. We anticipate that these problems will be addressed with the release of PyTorch 2.5.0 and the officially supported version of Flash-Attention.

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

