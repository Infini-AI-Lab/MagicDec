from dataclasses import dataclass
from typing import Optional

from einops import rearrange
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F
import torch.distributed as dist
import math 
from MagicDec.Engine.utils import repeat_kv, unrepeat_kv

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
}

class KVCache(nn.Module):
    def __init__(self, max_num_pages, page_size, n_heads, head_dim, dtype=torch.bfloat16, spec=False, draft_max_num_pages=0):
        super().__init__()
        cache_shape = (max_num_pages, 2, page_size, n_heads, head_dim)
        self.register_buffer('kv_cache', torch.zeros(cache_shape, dtype=dtype))
        if spec:
            draft_shape = (draft_max_num_pages, 2, page_size, n_heads, head_dim)
            self.register_buffer('draft_cache', torch.zeros(draft_shape, dtype=dtype))
        
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

class Transformer(nn.Module):
    def __init__(self, config: ModelArgs) -> None:
        super().__init__()
        self.config = config
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.dim)
        self.layers = nn.ModuleList(TransformerBlock(config) for _ in range(config.n_layer))
        self.norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.output = nn.Linear(config.dim, config.vocab_size, bias=False)
        # self.freqs_cis: Optional[Tensor] = None

    def setup_caches(self, num_pages, page_size, spec=False, draft_num_pages = 0, draft_budget = 0, window_size = 32, non_snap_layers=0):

        head_dim = self.config.dim // self.config.n_head
        dtype = self.output.weight.dtype

        for i, b in enumerate(self.layers):
            b.attention.kv_cache = KVCache(num_pages, page_size, self.config.n_local_heads, head_dim, dtype, spec, draft_num_pages)
            b.attention.attn_decode = torch.ops.mylib.target_decode
            b.attention.attn_prefill = torch.ops.mylib.target_prefill
            b.attention.rope = torch.ops.mylib.llama31rope
            if spec and i >= non_snap_layers:
                b.attention.attn_draft = torch.ops.mylib.draft_decode
                b.attention.is_spec = True
                b.attention.draft_budget = draft_budget
                b.attention.window_size = window_size
                b.attention.pooling = 'avgpool'
                b.attention.kernel_size = 5
        self.non_snap_layers = non_snap_layers

        # if (self.config.high_freq_factor is not None) and (self.config.low_freq_factor is not None):
        #     self.freqs_cis = precompute_freqs_cis(self.config.block_size, self.config.dim // self.config.n_head, self.config.rope_base,dtype,
        #                                           self.config.scaling_factor, self.config.low_freq_factor, self.config.high_freq_factor, self.config.original_max_position_embeddings)
        # else:
        #     self.freqs_cis = precompute_freqs_cis(self.config.block_size, self.config.dim // self.config.n_head, self.config.rope_base,dtype,
        #                                           self.config.scaling_factor)

    def forward(self, idx: Tensor, input_pos: Tensor, kv_append_indptr: Tensor, kv_page_indices: Tensor, kv_page_indptr: Tensor, kv_page_lastlen: Tensor) -> Tensor:
        x = self.tok_embeddings(idx)
        for i, layer in enumerate(self.layers):
            x = layer(x, input_pos, kv_append_indptr, kv_page_indices, kv_page_indptr, kv_page_lastlen)
        x = self.norm(x)
        logits = self.output(x)
        return logits
    
    def draft_forward(self, idx: Tensor, input_pos: Tensor, kv_append_indptr: Tensor, 
                      draft_kv_page_indices: Tensor, draft_kv_page_indptr: Tensor, draft_kv_page_lastlen: Tensor,
                      target_kv_page_indices: Tensor=None, target_kv_page_indptr: Tensor=None, target_kv_page_lastlen: Tensor=None) -> Tensor:
        x = self.tok_embeddings(idx)
        for i, layer in enumerate(self.layers):
            if i <= self.non_snap_layers:
                x = layer(x, input_pos, kv_append_indptr, target_kv_page_indices, target_kv_page_indptr, target_kv_page_lastlen)
            else:
                x = layer.draft_forward(x, input_pos, kv_append_indptr, draft_kv_page_indices, draft_kv_page_indptr, draft_kv_page_lastlen)
        x = self.norm(x)
        logits = self.output(x)
        return logits

    def prefill(self, idx: Tensor, input_pos: Tensor, kv_append_indptr: Tensor, kv_page_indices: Tensor, kv_page_indptr: Tensor, kv_page_lastlen: Tensor, is_last = False, draft_paged_kv_indptr=None, draft_paged_kv_indices=None, draft_paged_kv_last_page_len=None) -> Tensor:
        x = self.tok_embeddings(idx)
        for i, layer in enumerate(self.layers):
            x = layer.prefill(x, input_pos, kv_append_indptr, kv_page_indices, kv_page_indptr, kv_page_lastlen, is_last, draft_paged_kv_indptr, draft_paged_kv_indices, draft_paged_kv_last_page_len)
        x = self.norm(x)
        logits = self.output(x)
        return logits

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
    
    def draft_forward(self, x: Tensor, offsets: Tensor, kv_append_indptr: Tensor, kv_page_indices: Tensor, kv_page_indptr: Tensor, kv_page_lastlen: Tensor) -> Tensor:
        h = x + self.attention.draft_forward(self.attention_norm(x), offsets, kv_append_indptr, kv_page_indices, kv_page_indptr, kv_page_lastlen)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out

    def prefill(self, x: Tensor, offsets: Tensor, kv_append_indptr: Tensor, kv_page_indices: Tensor, kv_page_indptr: Tensor, kv_page_lastlen: Tensor, is_last = False, draft_paged_kv_indptr=None, draft_paged_kv_indices=None, draft_paged_kv_last_page_len=None) -> Tensor:
        h = x + self.attention.prefill(self.attention_norm(x), offsets, kv_append_indptr, kv_page_indices, kv_page_indptr, kv_page_lastlen, is_last, draft_paged_kv_indptr, draft_paged_kv_indices, draft_paged_kv_last_page_len)
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
        self.is_spec = False

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
        kv_cache = self.kv_cache.update_target(k, v, kv_append_indptr, kv_page_indices, kv_page_indptr, kv_page_lastlen)
        y = self.attn_decode(q, kv_cache)
        y = y.contiguous().view(bsz, seqlen, self.dim)
        y = self.wo(y)
        if self.process_group != None:
            dist.all_reduce(y)
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
            dist.all_reduce(y)
        return y

    def prefill(self, x: Tensor, offsets: Tensor, kv_append_indptr: Tensor, kv_page_indices: Tensor, kv_page_indptr: Tensor, kv_page_lastlen: Tensor, is_last = False ,draft_paged_kv_indptr=None, draft_paged_kv_indices=None, draft_paged_kv_last_page_len=None) -> Tensor:
        bsz, seqlen, _ = x.shape
        kv_size = self.n_local_heads * self.head_dim
        q, k, v = self.wqkv(x).split([self.dim, kv_size, kv_size], dim=-1)
        q = q.view(bsz * seqlen, self.n_head, self.head_dim)
        k = k.view(bsz * seqlen, self.n_local_heads, self.head_dim)
        v = v.contiguous().view(bsz * seqlen, self.n_local_heads, self.head_dim)
        q, k = self.rope(q, k, kv_append_indptr, offsets)
        kv_cache = self.kv_cache.update_target(k, v, kv_append_indptr, kv_page_indices, kv_page_indptr, kv_page_lastlen)
        y = self.attn_prefill(q, kv_cache)
        if is_last and self.is_spec:
            self.gen_draft_kv(q, kv_cache[:, 0], kv_cache[:, 1], bsz, seqlen, offsets[0]+seqlen, (kv_append_indptr/seqlen*self.draft_budget).to(torch.int32), draft_paged_kv_indptr, draft_paged_kv_indices, draft_paged_kv_last_page_len)
        y = y.contiguous().view(bsz, seqlen, self.dim)
        y = self.wo(y)
        if self.process_group != None:
            dist.all_reduce(y)
        return y
    
    def gen_draft_kv(self, q, k, v, bsz, seqlen, context_len, kv_append_indptr, draft_paged_kv_indptr, draft_paged_kv_indices, draft_paged_kv_last_page_len):
        # draft_key_states = torch.randn(bsz*self.draft_budget, k.shape[2], k.shape[3], dtype=torch.bfloat16, device=q.device)
        # draft_value_states = torch.randn(bsz*self.draft_budget, k.shape[2], k.shape[3], dtype=torch.bfloat16, device=q.device)
        # self.kv_cache.update_draft(draft_key_states, draft_value_states, kv_append_indptr, draft_paged_kv_indices, draft_paged_kv_indptr, draft_paged_kv_last_page_len)
        # return
        # query_states = q.reshape(bsz, seqlen, self.n_head, self.head_dim).transpose(1,2)
        # key_states = k.reshape(bsz, -1, self.n_local_heads, self.head_dim)[:,:context_len].transpose(1,2)
        # value_states = v.reshape(bsz, -1, self.n_local_heads, self.head_dim)[:, :context_len].transpose(1,2)

        # # TODO Can not repeat keys and values for memory crunch
        # nrepeat = self.n_head // self.n_local_heads
        # query_states = rearrange(query_states, 'b (h r) l d -> b h (r l) d', r=nrepeat).contiguous()
        
        # B, H, L, D = query_states.shape
        # topk = self.draft_budget - self.window_size
        # topk_indices = torch.zeros(B, H, L, topk, device=query_states.device, dtype=torch.int64)

        # mask = torch.full((self.window_size, self.window_size), torch.finfo(query_states.dtype).min, device=query_states.device)
        # mask_cond = torch.arange(mask.size(-1), device=query_states.device)
        # mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
        # mask = mask.to(query_states.device)

        # window_size = self.window_size
        # chunk_size = 32
        # assert chunk_size % nrepeat == 0
        # num_chunks = (L + chunk_size - 1) // chunk_size
        # key_states = []
        # value_states = []
        # for chunk_id in range(num_chunks):
        #     start_idx = chunk_id * chunk_size
        #     end_idx = min(start_idx + chunk_size, L)
        #     chunk_query_states = query_states[:, :, start_idx:end_idx, :]
        #     attn_weights = torch.einsum('b h i d, b h j d -> b h i j', chunk_query_states, key_states)
        #     attn_weights[:, :, -window_size:, -window_size:] += mask

        #     attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        #     attn_weights_sum = attn_weights[:, :, -window_size:, : -window_size].sum(dim = -2)
        #     if self.pooling == 'avgpool':
        #         attn_cache = F.avg_pool1d(attn_weights_sum, kernel_size = self.kernel_size, padding=self.kernel_size//2, stride=1)
        #     elif self.pooling == 'maxpool':
        #         attn_cache = F.max_pool1d(attn_weights_sum, kernel_size = self.kernel_size, padding=self.kernel_size//2, stride=1)
        #     else:
        #         raise ValueError('Pooling method not supported')
        
        #     # avg across each kv group
        #     attn_cache = rearrange(attn_cache, 'b h (r l) s -> b h r l s', r=nrepeat)
        #     attn_cache = attn_cache.sum(dim=2)

        #     indices = attn_cache.topk(topk, dim=-1).indices
        #     indices = indices.unsqueeze(-1).expand(-1, -1, -1, self.head_dim)
        #     k_past_compress = key_states[:, :, :-self.window_size, :].gather(dim = 2, index = indices)
        #     v_past_compress = value_states[:, :, :-self.window_size, :].gather(dim = 2, index = indices)
        #     k_cur = key_states[:, :, -window_size:, :]
        #     v_cur = value_states[:, :, -window_size:, :]
        #     chunk_key_states = torch.cat([k_past_compress, k_cur], dim = 2).transpose(1,2)
        #     chunk_value_states = torch.cat([v_past_compress, v_cur], dim = 2).transpose(1,2)
        #     key_states.append(chunk_key_states)
        #     value_states.append(chunk_value_states)
        # key_states = torch.cat(key_states, dim = 2)
        # value_states = torch.cat(value_states, dim = 2)
        # import time
        # self.kv_cache.update_draft(key_states, value_states, kv_append_indptr, draft_paged_kv_indptr, draft_paged_kv_indices, draft_paged_kv_last_page_len)
        # print(key_states.shape, value_states.shape)
        # time.sleep(1000)
        num_pages_per_request = k.shape[0] // bsz
        full_query_states = q.reshape(bsz, seqlen, self.n_head, self.head_dim).transpose(1,2)
        full_key_states = torch.empty((bsz, self.draft_budget, self.n_local_heads, self.head_dim), dtype=k.dtype, device=k.device)
        full_value_states = torch.empty((bsz, self.draft_budget, self.n_local_heads, self.head_dim), dtype=v.dtype, device=v.device)
        for i in range(bsz):
            key_states = k[i*num_pages_per_request: (i+1)*num_pages_per_request].reshape(1, -1, self.n_local_heads, self.head_dim)[:,:context_len].transpose(1,2)
            value_states = v[i*num_pages_per_request: (i+1)*num_pages_per_request].reshape(1, -1, self.n_local_heads, self.head_dim)[:,:context_len].transpose(1,2)
            query_states = full_query_states[i].unsqueeze(0)

            # TODO Can not repeat it
            key_states = repeat_kv(key_states, self.n_head//self.n_local_heads)
            value_states = repeat_kv(value_states, self.n_head//self.n_local_heads)

            attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
            mask = torch.full((self.window_size, self.window_size), torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
            mask_cond = torch.arange(mask.size(-1), device=attn_weights.device)
            mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
            mask = mask.to(attn_weights.device)
            attention_mask = mask[None, None, :, :]

            attn_weights[:, :, -self.window_size:, -self.window_size:] += attention_mask
            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            attn_weights_sum = attn_weights[:, :, -self.window_size:, : -self.window_size].sum(dim = -2)
            if self.pooling == 'avgpool':
                attn_cache = F.avg_pool1d(attn_weights_sum, kernel_size = self.kernel_size, padding=self.kernel_size//2, stride=1)
            elif self.pooling == 'maxpool':
                attn_cache = F.max_pool1d(attn_weights_sum, kernel_size = self.kernel_size, padding=self.kernel_size//2, stride=1)
            else:
                raise ValueError('Pooling method not supported')
            indices = attn_cache.topk(self.draft_budget - self.window_size, dim=-1).indices
            indices = indices.unsqueeze(-1).expand(-1, -1, -1, self.head_dim)
            k_past_compress = key_states[:, :, :-self.window_size, :].gather(dim = 2, index = indices)
            v_past_compress = value_states[:, :, :-self.window_size, :].gather(dim = 2, index = indices)
            k_cur = key_states[:, :, -self.window_size:, :]
            v_cur = value_states[:, :, -self.window_size:, :]
            key_states = torch.cat([k_past_compress, k_cur], dim = 2)
            value_states = torch.cat([v_past_compress, v_cur], dim = 2)

            full_key_states[i] = unrepeat_kv(key_states, self.n_head//self.n_local_heads).transpose(1,2).view(self.draft_budget, self.n_local_heads, self.head_dim)
            full_value_states[i] = unrepeat_kv(value_states, self.n_head//self.n_local_heads).transpose(1,2).view(self.draft_budget, self.n_local_heads, self.head_dim)

        full_key_states = full_key_states.view(-1, self.n_local_heads, self.head_dim)
        full_value_states = full_value_states.view(-1, self.n_local_heads, self.head_dim)

        # # TODO Need to pass in the draft kv page information tensor to update the draft KV cache
        # import time
        # print(kv_append_indptr, draft_paged_kv_indptr, draft_paged_kv_indices, draft_paged_kv_last_page_len)
        # time.sleep(1000)
        self.kv_cache.update_draft(full_key_states, full_value_states, kv_append_indptr, draft_paged_kv_indices, draft_paged_kv_indptr, draft_paged_kv_last_page_len)
        # print(key_states.shape, value_states.shape)
        # time.sleep(1000)


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