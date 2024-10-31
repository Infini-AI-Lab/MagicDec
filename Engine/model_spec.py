from dataclasses import dataclass
from typing import Optional

from einops import rearrange
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F
import torch.distributed as dist
import math 

def find_multiple(n: int, k: int) -> int:
    if n % k == 0:
        return n
    return n + k - (n % k)

@dataclass
class ModelArgs:
    block_size: int = 2048
    vocab_size: int = 32000
    n_layer: int = 32
    n_head: int = 32
    dim: int = 4096
    intermediate_size: int = None
    n_local_heads: int = -1
    head_dim: int = 64
    rope_base: float = 10000
    norm_eps: float = 1e-5
    scaling_factor:float = 1.0
    # llama 3.1 with high_freq_factor and low_freq_factor
    low_freq_factor: int = None # added new
    high_freq_factor: int = None  # added new
    original_max_position_embeddings: int = None   # added new

    def __post_init__(self):
        if self.n_local_heads == -1:
            self.n_local_heads = self.n_head
        if self.intermediate_size is None:
            hidden_dim = 4 * self.dim
            n_hidden = int(2 * hidden_dim / 3)
            self.intermediate_size = find_multiple(n_hidden, 256)
        self.head_dim = self.dim // self.n_head

    @classmethod
    def from_name(cls, name: str):
        if name in transformer_configs:
            return cls(**transformer_configs[name])
        # fuzzy search
        config = [config for config in transformer_configs if config.lower() in str(name).lower()]
        # We may have two or more configs matched (e.g. "7B" and "Mistral-7B"). Find the best config match,
        # take longer name (as it have more symbols matched)
        if len(config) > 1:
            config.sort(key=len, reverse=True)
            assert len(config[0]) != len(config[1]), name # make sure only one 'best' match
        print(config)
        return cls(**transformer_configs[config[0]])


transformer_configs = {
    "llama-2-7b": dict(block_size=4096, n_layer=32, n_head=32, dim=4096),
    'llama-2-7b-32k': dict(block_size=32768, n_layer=32, dim= 4096, vocab_size=32000, scaling_factor=8),
    "llama-2-13b": dict(block_size=4096, n_layer=40, n_head=40, dim=5120),
    "llama-2-70b": dict(block_size=4096, n_layer=80, n_head=64, dim=8192, n_local_heads=8, intermediate_size=28672),
    "llama-3-8b": dict(block_size=8192, n_layer=32, n_head=32, n_local_heads=8, dim=4096, intermediate_size=14336, vocab_size=128256, rope_base=500000),
    "llama-3-70b": dict(block_size=8192, n_layer=80, n_head=64, n_local_heads=8, dim=8192, intermediate_size=28672, vocab_size=128256, rope_base=500000),
    "68m": dict(block_size=2048, n_layer=2, n_head=12, n_local_heads=12, dim=768, intermediate_size=3072, vocab_size=32000),
    "tinyllama": dict(block_size =2048, n_layer=22, n_head=32, n_local_heads=4, dim=2048, intermediate_size=5632, vocab_size=32000),
    "llama-3.1-8b": dict(block_size=131072, n_layer=32, n_head=32, n_local_heads=8, dim=4096, intermediate_size=14336, vocab_size=128256, rope_base=500000.0, scaling_factor=8, high_freq_factor=4, low_freq_factor=1, original_max_position_embeddings=8192),
    "llama-3.1-70b": dict(block_size=131072, n_layer=80, n_head=64, n_local_heads=8, dim=8192, intermediate_size=28672, vocab_size=128256, rope_base=500000.0, scaling_factor=8, high_freq_factor=4, low_freq_factor=1, original_max_position_embeddings=8192),
}

class KVCache(nn.Module):
    def __init__(self, max_num_pages, page_size, n_heads, head_dim, dtype=torch.bfloat16):
        super().__init__()
        cache_shape = (max_num_pages, 2, page_size, n_heads, head_dim)
        self.register_buffer('kv_cache', torch.zeros(cache_shape, dtype=dtype))

    def update(self, k, v, kv_append_indptr, kv_page_indices, kv_page_indptr, kv_page_lastlen):
        torch.ops.mylib.update_kv(
            k,
            v,
            kv_append_indptr,
            self.kv_cache,
            kv_page_indices,
            kv_page_indptr,
            kv_page_lastlen,
        )
        return self.kv_cache


class Transformer(nn.Module):
    def __init__(self, config: ModelArgs) -> None:
        super().__init__()
        self.config = config
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.dim)
        self.layers = nn.ModuleList(TransformerBlock(config) for _ in range(config.n_layer))
        self.norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.output = nn.Linear(config.dim, config.vocab_size, bias=False)
        self.world_size = None
        self.rank = None
        self.process_group = None

    def setup_caches(self, num_pages, page_size, is_draft=False, budget = -1, window_size = 32, pooling="avgpool", kernel_size=5):

        head_dim = self.config.dim // self.config.n_head
        # dtype = self.output.weight.dtype
        dtype = self.output.weight.dtype if self.output.weight.dtype == torch.float16 else torch.bfloat16

        for b in self.layers:
            b.attention.kv_cache = KVCache(num_pages, page_size, self.config.n_local_heads, head_dim, dtype)
            b.attention.attn_decode = torch.ops.mylib.decode if not is_draft else torch.ops.mylib.speculate
            b.attention.attn_prefill = torch.ops.mylib.prefill if not is_draft else torch.ops.mylib.draft_prefill
            b.attention.rope = torch.ops.mylib.llama31rope
            # if is_draft and budget > 0:
            #     b.attention.is_sparse = True
            #     b.attention.budget = budget
            #     b.attention.window_size = window_size
            #     b.attention.pooling = pooling
            #     b.attention.kernel_size = kernel_size

    def forward(self, idx: Tensor, input_pos: Tensor, kv_append_indptr: Tensor, kv_page_indices: Tensor, kv_page_indptr: Tensor, kv_page_lastlen: Tensor) -> Tensor:
        x = self.tok_embeddings(idx)
        for i, layer in enumerate(self.layers):
            x = layer(x, input_pos, kv_append_indptr, kv_page_indices, kv_page_indptr, kv_page_lastlen)
        x = self.norm(x)
        logits = self.output(x)
        if self.process_group != None:
            all_max_value = torch.zeros((x.shape[0], x.shape[1], self.world_size), dtype=logits.dtype, device=logits.device)
            all_max_indices = torch.zeros((x.shape[0], x.shape[1], self.world_size), dtype=torch.long, device=logits.device)
            all_max_value[:, :, self.rank], all_max_indices[:, :, self.rank] = torch.max(logits, dim=-1)
            all_max_indices[:, :, self.rank] += self.rank * logits.shape[-1]
            dist.all_reduce(all_max_value)
            dist.all_reduce(all_max_indices)
            global_select_indices = torch.argmax(all_max_value, dim=-1)
            global_indices = torch.gather(all_max_indices, dim=-1, index=global_select_indices.unsqueeze(-1))
            return global_indices.squeeze(-1)
        return torch.argmax(logits, dim=-1)

    def prefill(self, idx: Tensor, input_pos: Tensor, kv_append_indptr: Tensor, kv_page_indices: Tensor, kv_page_indptr: Tensor, kv_page_lastlen: Tensor, is_last = False):
        x = self.tok_embeddings(idx)
        for i, layer in enumerate(self.layers):
            x = layer.prefill(x, input_pos, kv_append_indptr, kv_page_indices, kv_page_indptr, kv_page_lastlen, is_last)
        x = self.norm(x)
        logits = self.output(x)
        if self.process_group != None:
            all_max_value = torch.zeros((x.shape[0], x.shape[1], self.world_size), dtype=logits.dtype, device=logits.device)
            all_max_indices = torch.zeros((x.shape[0], x.shape[1], self.world_size), dtype=torch.long, device=logits.device)
            all_max_value[:, :, self.rank], all_max_indices[:, :, self.rank] = torch.max(logits, dim=-1)
            all_max_indices[:, :, self.rank] += self.rank * logits.shape[-1]
            dist.all_reduce(all_max_value)
            dist.all_reduce(all_max_indices)
            global_select_indices = torch.argmax(all_max_value, dim=-1)
            global_indices = torch.gather(all_max_indices, dim=-1, index=global_select_indices.unsqueeze(-1))
            return global_indices.squeeze(-1)
        return torch.argmax(logits, dim=-1)

    @classmethod
    def from_name(cls, name: str):
        return cls(ModelArgs.from_name(name))
    

class TransformerBlock(nn.Module):
    def __init__(self, config: ModelArgs) -> None:
        super().__init__()
        self.attention = Attention(config)
        self.feed_forward = FeedForward(config)
        self.ffn_norm = RMSNorm(config.dim, config.norm_eps)
        self.attention_norm = RMSNorm(config.dim, config.norm_eps)

    def forward(self, x: Tensor, offsets: Tensor, kv_append_indptr: Tensor, kv_page_indices: Tensor, kv_page_indptr: Tensor, kv_page_lastlen: Tensor) -> Tensor:
        h = x + self.attention(self.attention_norm(x), offsets, kv_append_indptr, kv_page_indices, kv_page_indptr, kv_page_lastlen)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out

    def prefill(self, x: Tensor, offsets: Tensor, kv_append_indptr: Tensor, kv_page_indices: Tensor, kv_page_indptr: Tensor, kv_page_lastlen: Tensor, is_last = False):
        h = x + self.attention.prefill(self.attention_norm(x), offsets, kv_append_indptr, kv_page_indices, kv_page_indptr, kv_page_lastlen, is_last)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out        

class Attention(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        assert config.dim % config.n_head == 0

        total_head_dim = (config.n_head + 2 * config.n_local_heads) * config.head_dim
        # key, query, value projections for all heads, but in a batch
        self.wqkv = nn.Linear(config.dim, total_head_dim, bias=False)
        self.wo = nn.Linear(config.dim, config.dim, bias=False)
        self.kv_cache = None
        self.process_group = None
        self.attn_decode = None
        self.attn_prefill = None
        self.attn_draft = None
        self.rope = None
        self.is_sparse = False

        self.window_size = None
        self.pooling = None
        self.kernel_size = None
        self.draft_budget = None

        self.n_head = config.n_head
        self.head_dim = config.head_dim
        self.n_local_heads = config.n_local_heads
        self.dim = config.dim
        self._register_load_state_dict_pre_hook(self.load_hook)

    def load_hook(self, state_dict, prefix, *args):
        if prefix + "wq.weight" in state_dict:
            wq = state_dict.pop(prefix + "wq.weight")
            wk = state_dict.pop(prefix + "wk.weight")
            wv = state_dict.pop(prefix + "wv.weight")
            state_dict[prefix + "wqkv.weight"] = torch.cat([wq, wk, wv])

    def forward(self, x: Tensor, offsets: Tensor, kv_append_indptr: Tensor, kv_page_indices: Tensor, kv_page_indptr: Tensor, kv_page_lastlen: Tensor) -> Tensor:
        bsz, seqlen, _ = x.shape
        kv_size = self.n_local_heads * self.head_dim
        q, k, v = self.wqkv(x).split([self.dim, kv_size, kv_size], dim=-1)
        q = q.view(bsz * seqlen, self.n_head, self.head_dim)
        k = k.view(bsz * seqlen, self.n_local_heads, self.head_dim)
        v = v.contiguous().view(bsz * seqlen, self.n_local_heads, self.head_dim)
        q, k = self.rope(q, k, kv_append_indptr, offsets)
        kv_cache = self.kv_cache.update(k, v, kv_append_indptr, kv_page_indices, kv_page_indptr, kv_page_lastlen)
        y = self.attn_decode(q, kv_cache)
        y = y.contiguous().view(bsz, seqlen, self.dim)
        y = self.wo(y)
        if self.process_group != None:
            dist.all_reduce(y)
        return y

    def prefill(self, x: Tensor, offsets: Tensor, kv_append_indptr: Tensor, kv_page_indices: Tensor, kv_page_indptr, kv_page_lastlen: Tensor, is_last = False):
        bsz, seqlen, _ = x.shape
        kv_size = self.n_local_heads * self.head_dim
        q, k, v = self.wqkv(x).split([self.dim, kv_size, kv_size], dim=-1)
        q = q.view(bsz * seqlen, self.n_head, self.head_dim)
        k = k.view(bsz * seqlen, self.n_local_heads, self.head_dim)
        v = v.contiguous().view(bsz * seqlen, self.n_local_heads, self.head_dim)
        q, k = self.rope(q, k, kv_append_indptr, offsets)
        kv_cache = self.kv_cache.update(k, v, kv_append_indptr, kv_page_indices, kv_page_indptr, kv_page_lastlen)
        y = self.attn_prefill(q, kv_cache)
        
        if is_last and self.is_sparse:
            self.gen_sparse_kv(q, kv_cache[:, 0], kv_cache[:, 1], bsz, seqlen, offsets[0]+seqlen,
                                (kv_append_indptr/seqlen*self.budget).to(torch.int32), kv_page_indptr, kv_page_indices, kv_page_lastlen)
        
        y = y.contiguous().view(bsz, seqlen, self.dim)
        y = self.wo(y)
        if self.process_group != None:
            dist.all_reduce(y)
        return y

    def gen_draft_kv(self, q, k, v, bsz, seqlen, context_len, kv_append_indptr):
        raise NotImplementedError

class FeedForward(nn.Module):
    def __init__(self, config: ModelArgs) -> None:
        super().__init__()
        self.w1 = nn.Linear(config.dim, config.intermediate_size, bias=False)
        self.w3 = nn.Linear(config.dim, config.intermediate_size, bias=False)
        self.w2 = nn.Linear(config.intermediate_size, config.dim, bias=False)
        self.process_group = None

    def forward(self, x: Tensor) -> Tensor:
        y = self.w2(F.silu(self.w1(x)) * self.w3(x))
        if self.process_group != None:
            dist.all_reduce(y)
        return y


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)

    def forward(self, x: Tensor) -> Tensor:
        output = self._norm(x.float()).type_as(x)
        return output * self.weight