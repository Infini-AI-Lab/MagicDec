from dataclasses import dataclass
from typing import Optional

from einops import rearrange
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F
import torch.distributed as dist
import flashinfer

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
    qkv_bias: bool = False

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
    "llama-3.2-1b": dict(block_size=131072, n_layer=16, n_head=32, n_local_heads=8, dim=2048, intermediate_size=8192, vocab_size=128256, rope_base=500000.0, scaling_factor=32, high_freq_factor=4, low_freq_factor=1, original_max_position_embeddings=8192),
    "Qwen2.5-7b": dict(block_size=131072, n_layer=28, n_head=28, n_local_heads=4, dim=3584, intermediate_size=18944, vocab_size=152064, rope_base=1000000.0, qkv_bias=True, norm_eps=1e-6),
    "Qwen2.5-14b": dict(block_size=131072, n_layer=48, n_head=40, n_local_heads=8, dim=5120, intermediate_size=13824, vocab_size=152064, rope_base=1000000.0, qkv_bias=True, norm_eps=1e-6),
    "Qwen2.5-32b": dict(block_size=131072, n_layer=64, n_head=40, n_local_heads=8, dim=5120, intermediate_size=27648, vocab_size=152064, rope_base=1000000.0, qkv_bias=True, norm_eps=1e-6),
    "Yi-1.5-6b": dict(block_size=4096, n_layer=32, n_head=32, n_local_heads=4, dim=4096, intermediate_size=11008, vocab_size=64000, rope_base=500000.0),
    "Yi-1.5-34b-32k": dict(block_size=32768, n_layer=60, n_head=56, n_local_heads=8, dim=7168, intermediate_size=20480, vocab_size=64000, rope_base=500000.0),
    "Mistral-7B-v0.1": dict(n_layer=32, n_head=32, n_local_heads=8, dim=4096, intermediate_size=14336, vocab_size=32000),
    "Mistral-7B-v0.3": dict(n_layer=32, n_head=32, n_local_heads=8, dim=4096, intermediate_size=14336, vocab_size=32768, rope_base=1000000.0),
}

class KVCache(nn.Module):
    def __init__(self, max_num_pages, page_size, n_heads, head_dim, dtype=torch.bfloat16, draft_max_num_pages=0, kv_len=512):
        super().__init__()
        cache_shape = (max_num_pages, 2, page_size, n_heads, head_dim)
        self.register_buffer('kv_cache', torch.zeros(cache_shape, dtype=dtype))
        draft_shape = (draft_max_num_pages, 2, page_size, n_heads, head_dim)
        self.register_buffer('draft_cache', torch.zeros(draft_shape, dtype=dtype))
        self.kv_len = kv_len
        self.max_num_pages = draft_max_num_pages
        self.page_size = page_size
        
    def update_target(self, k, v, kv_append_indptr, kv_page_indices, kv_page_indptr, kv_page_lastlen):
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
    
    def update_draft(self, k, v, kv_append_indptr, kv_page_indices, kv_page_indptr, kv_page_lastlen):
        torch.ops.mylib.update_kv(
            k,
            v,
            kv_append_indptr,
            self.draft_cache,
            kv_page_indices,
            kv_page_indptr,
            kv_page_lastlen,
        )
        return self.draft_cache
    
    def prefill_draft(self, k, v, kv_append_indptr, kv_page_indices, kv_page_indptr, kv_page_lastlen, bsz, context_len, seq_len, n_local_heads, head_dim, rope, is_last):
        if context_len+seq_len <= self.kv_len:
            torch.ops.mylib.update_kv(
                k,
                v,
                kv_append_indptr,
                self.draft_cache,
                kv_page_indices,
                kv_page_indptr,
                kv_page_lastlen,
            )
            key_states = self.draft_cache[:, 0].clone().reshape(bsz, -1, n_local_heads, head_dim)
            value_states = self.draft_cache[:, 1].clone()
            key_to_rotate = key_states[:, :context_len+seq_len].reshape(-1, n_local_heads, head_dim)
            key_states[:, :context_len+seq_len] = rope(key_to_rotate, key_to_rotate, kv_append_indptr//seq_len*(context_len+seq_len), torch.zeros(bsz, dtype=torch.int32, device=kv_append_indptr.device))[1].reshape(bsz, -1, n_local_heads, head_dim)
            key_states = key_states.reshape(self.max_num_pages, self.page_size, n_local_heads, head_dim)
            return torch.cat((key_states.unsqueeze(1), value_states.unsqueeze(1)),dim = 1)
        else:
            k_out = self.draft_cache[:,0].reshape(bsz, -1, n_local_heads, head_dim)
            v_out = self.draft_cache[:,1].reshape(bsz, -1, n_local_heads, head_dim)
            k_val = k.reshape(bsz, seq_len, n_local_heads, head_dim)
            v_val = v.reshape(bsz, seq_len, n_local_heads, head_dim)
            new_k = torch.cat((k_out[:, 16:self.kv_len], k_val), dim=1)[:, -self.kv_len+16:].reshape(-1, n_local_heads, head_dim).contiguous()
            new_v = torch.cat((v_out[:, 16:self.kv_len], v_val), dim=1)[:, -self.kv_len+16:].reshape(-1, n_local_heads, head_dim).contiguous()
            torch.ops.mylib.update_kv(
                new_k,
                new_v,
                kv_append_indptr//seq_len*(self.kv_len-16),
                self.draft_cache,
                kv_page_indices,
                kv_page_indptr,
                kv_page_lastlen,
            )
            key_states = self.draft_cache[:, 0].clone().reshape(bsz, -1, n_local_heads, head_dim)
            value_states = self.draft_cache[:, 1].clone()
            key_to_rotate = key_states[:, :self.kv_len].reshape(-1, n_local_heads, head_dim)
            key_states[:, :self.kv_len] = rope(key_to_rotate, key_to_rotate, kv_append_indptr//seq_len*self.kv_len, torch.zeros(bsz, dtype=torch.int32, device=kv_append_indptr.device))[1].reshape(bsz, -1, n_local_heads, head_dim)
            key_states = key_states.reshape(self.max_num_pages, self.page_size, n_local_heads, head_dim)
            rotated_kv = torch.cat((key_states.unsqueeze(1), value_states.unsqueeze(1)),dim = 1)
            if is_last:
                self.draft_cache.copy_(rotated_kv)
            return rotated_kv

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

    def setup_caches(self, num_pages, page_size, draft_num_pages = 0, draft_budget = 0):

        head_dim = self.config.dim // self.config.n_head
        # dtype = self.output.weight.dtype
        dtype = self.output.weight.dtype if self.output.weight.dtype == torch.float16 else torch.bfloat16

        if (self.config.high_freq_factor is not None) and (self.config.low_freq_factor is not None):
            torch.library.define(
                "mylib::rope",
                "(Tensor q, Tensor k, Tensor indptr, Tensor offsets) -> (Tensor ropeq, Tensor ropek)",
            )
            @torch.library.impl("mylib::rope", "cuda")
            def rope(q, k, indptr, offsets):
                return flashinfer.rope.apply_llama31_rope(q, k, indptr, offsets, interleave=True, rope_scale=self.config.scaling_factor, rope_theta=self.config.rope_base, low_freq_factor=self.config.low_freq_factor, high_freq_factor=self.config.high_freq_factor, old_context_len=self.config.original_max_position_embeddings)

            @torch.library.register_fake("mylib::rope")
            def rope_abstract(q, k, indptr, offsets):
                return torch.empty_like(q), torch.empty_like(k)
        else:
            torch.library.define(
                "mylib::rope",
                "(Tensor q, Tensor k, Tensor indptr, Tensor offsets) -> (Tensor ropeq, Tensor ropek)",
            )
            @torch.library.impl("mylib::rope", "cuda")
            def rope(q, k, indptr, offsets):
                return flashinfer.rope.apply_rope(q, k, indptr, offsets, interleave=True, rope_scale=self.config.scaling_factor, rope_theta=self.config.rope_base)

            @torch.library.register_fake("mylib::rope")
            def rope_abstract(q, k, indptr, offsets):
                return torch.empty_like(q), torch.empty_like(k)

        for b in self.layers:
            b.attention.kv_cache = KVCache(num_pages, page_size, self.config.n_local_heads, head_dim, dtype, draft_num_pages, draft_budget)
            b.attention.attn_decode = torch.ops.mylib.target_decode
            b.attention.attn_prefill = torch.ops.mylib.target_prefill
            b.attention.rope = torch.ops.mylib.rope
            b.attention.attn_draft = torch.ops.mylib.draft_decode
    
    def verify(self, idx: Tensor, input_pos: Tensor, kv_append_indptr: Tensor, kv_page_indices: Tensor, kv_page_indptr: Tensor, kv_page_lastlen: Tensor) -> Tensor:
        x = self.tok_embeddings(idx)
        for i, layer in enumerate(self.layers):
            x = layer.verify(x, input_pos, kv_append_indptr, kv_page_indices, kv_page_indptr, kv_page_lastlen)
        x = self.norm(x)
        logits = self.output(x)
        # return logits
        if self.process_group != None:
            all_max_value = torch.zeros((x.shape[0], x.shape[1], self.world_size), dtype=logits.dtype, device=logits.device)
            all_max_indices = torch.zeros((x.shape[0], x.shape[1], self.world_size), dtype=torch.long, device=logits.device)
            all_max_value[:, :, self.rank], all_max_indices[:, :, self.rank] = torch.max(logits, dim=-1)
            all_max_indices[:, :, self.rank] += self.rank * logits.shape[-1]
            dist.all_reduce(all_max_value, group = self.process_group)
            dist.all_reduce(all_max_indices, group = self.process_group)
            global_select_indices = torch.argmax(all_max_value, dim=-1)
            global_indices = torch.gather(all_max_indices, dim=-1, index=global_select_indices.unsqueeze(-1))
            return global_indices.squeeze(-1)
        return torch.argmax(logits, dim=-1)
    
    def draft_forward(self, idx: Tensor, input_pos: Tensor, kv_append_indptr: Tensor, kv_page_indices: Tensor, kv_page_indptr: Tensor, kv_page_lastlen: Tensor) -> Tensor:
        x = self.tok_embeddings(idx)
        for i, layer in enumerate(self.layers):
            x = layer.draft_forward(x, input_pos, kv_append_indptr, kv_page_indices, kv_page_indptr, kv_page_lastlen)
        x = self.norm(x)
        logits = self.output(x)
        # return logits
        if self.process_group != None:
            all_max_value = torch.zeros((x.shape[0], x.shape[1], self.world_size), dtype=logits.dtype, device=logits.device)
            all_max_indices = torch.zeros((x.shape[0], x.shape[1], self.world_size), dtype=torch.long, device=logits.device)
            all_max_value[:, :, self.rank], all_max_indices[:, :, self.rank] = torch.max(logits, dim=-1)
            all_max_indices[:, :, self.rank] += self.rank * logits.shape[-1]
            dist.all_reduce(all_max_value, group = self.process_group)
            dist.all_reduce(all_max_indices, group = self.process_group)
            global_select_indices = torch.argmax(all_max_value, dim=-1)
            global_indices = torch.gather(all_max_indices, dim=-1, index=global_select_indices.unsqueeze(-1))
            return global_indices.squeeze(-1)
        return torch.argmax(logits, dim=-1)

    def prefill(self, idx: Tensor, input_pos: Tensor, kv_append_indptr: Tensor, kv_page_indices: Tensor, kv_page_indptr: Tensor, kv_page_lastlen: Tensor) -> Tensor:
        x = self.tok_embeddings(idx)
        for i, layer in enumerate(self.layers):
            x = layer.prefill(x, input_pos, kv_append_indptr, kv_page_indices, kv_page_indptr, kv_page_lastlen)
        x = self.norm(x)
        logits = self.output(x)
        # return logits
        if self.process_group != None:
            all_max_value = torch.zeros((x.shape[0], x.shape[1], self.world_size), dtype=logits.dtype, device=logits.device)
            all_max_indices = torch.zeros((x.shape[0], x.shape[1], self.world_size), dtype=torch.long, device=logits.device)
            all_max_value[:, :, self.rank], all_max_indices[:, :, self.rank] = torch.max(logits, dim=-1)
            all_max_indices[:, :, self.rank] += self.rank * logits.shape[-1]
            dist.all_reduce(all_max_value, group = self.process_group)
            dist.all_reduce(all_max_indices, group = self.process_group)
            global_select_indices = torch.argmax(all_max_value, dim=-1)
            global_indices = torch.gather(all_max_indices, dim=-1, index=global_select_indices.unsqueeze(-1))
            return global_indices.squeeze(-1)
        return torch.argmax(logits, dim=-1)
    
    def draft_prefill(self, idx: Tensor, input_pos: Tensor, kv_append_indptr: Tensor, kv_page_indices: Tensor, kv_page_indptr: Tensor, kv_page_lastlen: Tensor, is_last: bool) -> Tensor:
        x = self.tok_embeddings(idx)
        for i, layer in enumerate(self.layers):
            x = layer.draft_prefill(x, input_pos, kv_append_indptr, kv_page_indices, kv_page_indptr, kv_page_lastlen, is_last)
        x = self.norm(x)
        logits = self.output(x)
        # return logits
        if self.process_group != None:
            all_max_value = torch.zeros((x.shape[0], x.shape[1], self.world_size), dtype=logits.dtype, device=logits.device)
            all_max_indices = torch.zeros((x.shape[0], x.shape[1], self.world_size), dtype=torch.long, device=logits.device)
            all_max_value[:, :, self.rank], all_max_indices[:, :, self.rank] = torch.max(logits, dim=-1)
            all_max_indices[:, :, self.rank] += self.rank * logits.shape[-1]
            dist.all_reduce(all_max_value, group = self.process_group)
            dist.all_reduce(all_max_indices, group = self.process_group)
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
    
    def verify(self, x: Tensor, offsets: Tensor, kv_append_indptr: Tensor, kv_page_indices: Tensor, kv_page_indptr: Tensor, kv_page_lastlen: Tensor) -> Tensor:
        h = x + self.attention.verify(self.attention_norm(x), offsets, kv_append_indptr, kv_page_indices, kv_page_indptr, kv_page_lastlen)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out
    
    def draft_forward(self, x: Tensor, offsets: Tensor, kv_append_indptr: Tensor, kv_page_indices: Tensor, kv_page_indptr: Tensor, kv_page_lastlen: Tensor) -> Tensor:
        h = x + self.attention.draft_forward(self.attention_norm(x), offsets, kv_append_indptr, kv_page_indices, kv_page_indptr, kv_page_lastlen)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out

    def prefill(self, x: Tensor, offsets: Tensor, kv_append_indptr: Tensor, kv_page_indices: Tensor, kv_page_indptr: Tensor, kv_page_lastlen: Tensor) -> Tensor:
        h = x + self.attention.prefill(self.attention_norm(x), offsets, kv_append_indptr, kv_page_indices, kv_page_indptr, kv_page_lastlen)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out
    
    def draft_prefill(self, x: Tensor, offsets: Tensor, kv_append_indptr: Tensor, kv_page_indices: Tensor, kv_page_indptr: Tensor, kv_page_lastlen: Tensor, is_last = False) -> Tensor:
        h = x + self.attention.draft_prefill(self.attention_norm(x), offsets, kv_append_indptr, kv_page_indices, kv_page_indptr, kv_page_lastlen, is_last)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


class Attention(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        assert config.dim % config.n_head == 0

        total_head_dim = (config.n_head + 2 * config.n_local_heads) * config.head_dim
        # key, query, value projections for all heads, but in a batch
        self.wqkv = nn.Linear(config.dim, total_head_dim, bias=config.qkv_bias)
        self.wo = nn.Linear(config.dim, config.dim, bias=False)
        self.kv_cache: KVCache = None
        self.process_group = None
        self.attn_decode = None
        self.attn_prefill = None
        self.attn_draft = None
        self.rope = None

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
        
        if prefix + "wq.bias" in state_dict:    
            bq = state_dict.pop(prefix + "wq.bias")
            bk = state_dict.pop(prefix + "wk.bias")
            bv = state_dict.pop(prefix + "wv.bias")
            state_dict[prefix + "wqkv.bias"] = torch.cat([bq, bk, bv])
    
    def verify(self, x: Tensor, offsets: Tensor, kv_append_indptr: Tensor, kv_page_indices: Tensor, kv_page_indptr: Tensor, kv_page_lastlen: Tensor) -> Tensor:
        bsz, seqlen, _ = x.shape
        kv_size = self.n_local_heads * self.head_dim
        q, k, v = self.wqkv(x).split([self.dim, kv_size, kv_size], dim=-1)
        q = q.view(bsz * seqlen, self.n_head, self.head_dim)
        k = k.view(bsz * seqlen, self.n_local_heads, self.head_dim)
        v = v.contiguous().view(bsz * seqlen, self.n_local_heads, self.head_dim)
        q, k = self.rope(q, k, kv_append_indptr, offsets)
        kv_cache = self.kv_cache.update_target(k, v, kv_append_indptr, kv_page_indices, kv_page_indptr, kv_page_lastlen)
        y = self.attn_decode(q, kv_cache)
        y = y.contiguous().view(bsz, seqlen, self.dim)
        y = self.wo(y)
        if self.process_group != None:
            dist.all_reduce(y, group = self.process_group)
        return y
    
    def draft_forward(self, x: Tensor, offsets: Tensor, kv_append_indptr: Tensor, kv_page_indices: Tensor, kv_page_indptr: Tensor, kv_page_lastlen: Tensor) -> Tensor:
        bsz, seqlen, _ = x.shape
        kv_size = self.n_local_heads * self.head_dim
        q, k, v = self.wqkv(x).split([self.dim, kv_size, kv_size], dim=-1)
        q = q.view(bsz * seqlen, self.n_head, self.head_dim)
        k = k.view(bsz * seqlen, self.n_local_heads, self.head_dim)
        v = v.contiguous().view(bsz * seqlen, self.n_local_heads, self.head_dim)
        q, k = self.rope(q, k, kv_append_indptr, offsets)
        kv_cache = self.kv_cache.update_draft(k, v, kv_append_indptr, kv_page_indices, kv_page_indptr, kv_page_lastlen)
        y = self.attn_draft(q, kv_cache)
        y = y.contiguous().view(bsz, seqlen, self.dim)
        y = self.wo(y)
        if self.process_group != None:
            dist.all_reduce(y, group = self.process_group)
        return y

    def prefill(self, x: Tensor, offsets: Tensor, kv_append_indptr: Tensor, kv_page_indices: Tensor, kv_page_indptr: Tensor, kv_page_lastlen: Tensor) -> Tensor:
        bsz, seqlen, _ = x.shape
        kv_size = self.n_local_heads * self.head_dim
        q, k, v = self.wqkv(x).split([self.dim, kv_size, kv_size], dim=-1)
        q = q.view(bsz * seqlen, self.n_head, self.head_dim)
        k = k.view(bsz * seqlen, self.n_local_heads, self.head_dim)
        v = v.contiguous().view(bsz * seqlen, self.n_local_heads, self.head_dim)
        q, k = self.rope(q, k, kv_append_indptr, offsets)
        kv_cache = self.kv_cache.update_target(k, v, kv_append_indptr, kv_page_indices, kv_page_indptr, kv_page_lastlen)
        y = self.attn_prefill(q, kv_cache)
        y = y.contiguous().view(bsz, seqlen, self.dim)
        y = self.wo(y)
        if self.process_group != None:
            dist.all_reduce(y, group = self.process_group)
        return y
    
    def draft_prefill(self, x: Tensor, offsets: Tensor, kv_append_indptr: Tensor, kv_page_indices: Tensor, kv_page_indptr: Tensor, kv_page_lastlen: Tensor, is_last) -> Tensor:
        bsz, seqlen, _ = x.shape
        kv_size = self.n_local_heads * self.head_dim
        q, k, v = self.wqkv(x).split([self.dim, kv_size, kv_size], dim=-1)
        q = q.view(bsz * seqlen, self.n_head, self.head_dim)
        k = k.view(bsz * seqlen, self.n_local_heads, self.head_dim).contiguous()
        v = v.contiguous().view(bsz * seqlen, self.n_local_heads, self.head_dim)
        if offsets[0] + seqlen <= self.kv_cache.kv_len:
            q, _ = self.rope(q, k, kv_append_indptr, offsets)
        else:
            q, _ = self.rope(q, k, kv_append_indptr, torch.full((bsz,), self.kv_cache.kv_len - seqlen, dtype=torch.int32, device=offsets.device))
        kv_cache = self.kv_cache.prefill_draft(k, v, kv_append_indptr, kv_page_indices, kv_page_indptr, kv_page_lastlen, bsz, offsets[0], seqlen, self.n_local_heads, self.head_dim, self.rope, is_last)
        y = self.attn_prefill(q, kv_cache)
        y = y.contiguous().view(bsz, seqlen, self.dim)
        y = self.wo(y)
        if self.process_group != None:
            dist.all_reduce(y, group = self.process_group)
        return y

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
            dist.all_reduce(y, group = self.process_group)
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