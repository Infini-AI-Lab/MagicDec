import torch
import numpy as np
import random
from torch.nn.functional import softmax
from flash_attn import flash_attn_with_kvcache

torch.library.define(
    "mylib::custom_func",
    "(Tensor q, Tensor(a!) k_cache, Tensor(b!) v_cache, Tensor k, Tensor v, Tensor cache_seqlens) -> Tensor",
)

@torch.library.impl("mylib::custom_func", "cuda")
def custom_func(q, k_cache, v_cache, k, v, cache_seqlens):
    return flash_attn_with_kvcache(
        q, k_cache, v_cache, k=k, v=v, cache_seqlens=cache_seqlens, causal=True
    )

@torch.library.impl_abstract("mylib::custom_func")
def custom_func_abstract(q, k_cache, v_cache, k, v, cache_seqlens):
    return torch.empty_like(q)

torch.library.define(
    "mylib::custom_func_2",
    "(Tensor q, Tensor(a!) k_cache, Tensor(a!) v_cache) -> Tensor",
)

@torch.library.impl("mylib::custom_func_2", "cuda")
def custom_func_2(q, k_cache, v_cache):
    return flash_attn_with_kvcache(
        q, k_cache, v_cache, causal=True
    )

@torch.library.impl_abstract("mylib::custom_func_2")
def custom_func_2_abstract(q, k_cache, v_cache):
    return torch.empty_like(q)

torch.library.define(
    "mylib::gqa_custom",
    "(Tensor q, Tensor(a!) k_cache, Tensor(b!) v_cache, Tensor k, Tensor v, Tensor cache_seqlens) -> Tensor",
)

@torch.library.impl_abstract("mylib::gqa_custom")
def gqa_custom_abstract(q, k_cache, v_cache, k, v, cache_seqlens):
    return torch.empty_like(q)

# @torch.library.impl("mylib::gqa_custom", "cuda")
# def gqa_custom(q, k_cache, v_cache, k, v, cache_seqlens):
#     B, T, H_q, D = q.size()
#     H_k = k.size(2)
#     rep = H_q // H_k
#     q_reshaped = q.view(B, T, H_k, rep, D).transpose(2, 3).contiguous().view(B, T*rep, H_k, D).contiguous()
#     y_past, lse_past = flash_attn_with_kvcache(q_reshaped, k_cache, v_cache, None, None, cache_seqlens=cache_seqlens, causal=True, return_softmax_lse=True)
#     y_new, lse_new = flash_attn_with_kvcache(q, k, v, None, None, None, causal=True, return_softmax_lse=True)     
#     y_past = y_past.view(B, T, rep, H_k, D).transpose(2, 3).contiguous().view(B, T, H_q, D)
#     lse_past = rearrange(lse_past, 'b h (t r) -> b t (h r) 1', r=rep).contiguous()
    
#     lse_past = lse_past.to(y_past.dtype)
#     lse_new = lse_new.unsqueeze(-1).transpose(1, 2).to(y_new.dtype)
    
#     sumexp_past = torch.exp(lse_past.float())
#     sumexp_new = torch.exp(lse_new.float())

#     sumexp_total = sumexp_past + sumexp_new
#     y = (y_past * sumexp_past + y_new * sumexp_new) / sumexp_total
    
#     # insert new k and v to k_cache and v_cache, starting from cache_seqlens position
#     insert_indices = cache_seqlens.unsqueeze(-1) + torch.arange(T, device=cache_seqlens.device).unsqueeze(0)
#     insert_indices = insert_indices[..., None, None].expand(-1, -1, H_k, D)
#     k_cache.scatter_(1, insert_indices, k)
#     v_cache.scatter_(1, insert_indices, v)   

#     return y.to(q.dtype)

@torch.library.impl("mylib::gqa_custom", "cuda")
def gqa_custom(q, k_cache, v_cache, k, v, cache_seqlens):
    B, T, H_q, D = q.size()
    H_k = k.size(2)
    rep = H_q // H_k
    q_reshaped = q.view(B, T, H_k, rep, D).transpose(2, 3).contiguous().view(B, T*rep, H_k, D).contiguous()
    v_new = torch.zeros(B, T*rep, H_k, D, device=q.device, dtype=q.dtype)
    k_new = torch.zeros_like(v_new)
    
    # the extra 1's added to the partition functions
    # they are of the pattern [0, 1, 2, ..., rep-1, rep-1, rep, rep+1, ..., 2*rep-1, 2*rep-1, 2*rep, ...]
    offset = torch.ones(rep, device=q.device, dtype=q.dtype)
    offset[0].zero_()
    extra = torch.cumsum(offset.repeat(T), dim=0)[None, None, :]
    insert_indices = torch.arange(0, T*rep, rep, device=q.device)[None, :, None, None].expand(B, -1, H_k, D)
    k_new.scatter_(1, insert_indices, k)
    v_new.scatter_(1, insert_indices, v)
    
    # print(q_reshaped.shape, k_cache.shape, k_new.shape)
    y, lse = flash_attn_with_kvcache(q_reshaped, k_cache, v_cache, k_new, v_new, cache_seqlens=cache_seqlens, causal=True, return_softmax_lse=True)
    
    extra = extra.expand_as(lse)
    correction = 1./ (1 - extra * torch.exp(-lse))
    correction = correction.transpose(1, 2).unsqueeze(-1)
    y = y * correction.to(y.dtype)
    y = y.view(B, T, rep, H_k, D).transpose(2, 3).contiguous().view(B, T, H_q, D)
    
    # insert new k and v to k_cache and v_cache, starting from cache_seqlens position
    insert_indices = cache_seqlens.unsqueeze(-1) + torch.arange(T, device=cache_seqlens.device).unsqueeze(0)
    insert_indices = insert_indices[..., None, None].expand(-1, -1, H_k, D)
    k_cache.scatter_(1, insert_indices, k)
    v_cache.scatter_(1, insert_indices, v)   

    return y.to(q.dtype)

def get_sampling_logits(logits :torch.Tensor, top_p:float, T: float, replicate = False):
    if replicate:
        logits = logits.clone()
    shape = logits.shape
    if top_p < 1.0:
        if len(shape)==3:
            batch_size, seq_len, voc_size = logits.size()
            logits = logits.reshape(-1, voc_size)
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(
        torch.nn.functional.softmax(sorted_logits / T, dim=-1), dim=-1)
        filter = cumulative_probs > top_p
        filter[..., 1:] = filter[..., :-1].clone()
        filter[..., 0] = 0
        indices_to_remove = filter.scatter(-1, sorted_indices, filter)
        logits[indices_to_remove] = float('-inf')
        if len(shape)==3:
            logits = logits.reshape(batch_size, seq_len, voc_size)
    return logits

def sample(logits, top_p, T):
    shape = logits.shape
    if len(shape)==3:
        batch_size, seq_len, _ = logits.size()
    else:
        batch_size, _ = logits.size()
        seq_len = 1
    logits = get_sampling_logits(logits=logits, top_p=top_p, T=T, replicate=True)
    logits = softmax(logits / T, dim=-1)
    next_tokens = logits.view(-1, 32000).multinomial(num_samples=1).view(batch_size, seq_len)
    return next_tokens

def cg_get_sampling_logits(logits :torch.Tensor, top_p:float, T: float):
    logits = logits.clone()
    batch_size, seq_len, voc_size = logits.size()
    logits = logits.reshape(-1, voc_size)
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(
    torch.nn.functional.softmax(sorted_logits / T, dim=-1), dim=-1)
    filter = cumulative_probs > top_p
    filter[..., 1:] = filter[..., :-1].clone()
    filter[..., 0] = 0
    indices_to_remove = filter.scatter(-1, sorted_indices, filter)
    logits[indices_to_remove] = float('-inf')
    logits = logits.reshape(batch_size, seq_len, voc_size)
    return logits

def cg_sample(logits, top_p, T):
    batch_size, seq_len, _ = logits.size()
    logits = get_sampling_logits(logits=logits, top_p=top_p, T=T, replicate=True)
    logits = softmax(logits / T, dim=-1)
    next_tokens = logits.view(-1, 32000).multinomial(num_samples=1).view(batch_size, seq_len)
    return next_tokens

def cuda_graph_for_target_sample(
                device="cuda:0", dtype=torch.bfloat16, 
                dim=32000, n_warmups=3, mempool=None,
                idx_len = 3, batch_size=1, top_p = 0.9, T = 0.6):
    
    static_sampling_logits = torch.full((batch_size, idx_len, dim), 1, dtype=dtype, device=device)
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(n_warmups):
            static_tokens = cg_sample(
                 static_sampling_logits,
                 top_p=top_p, T=T
            )
        s.synchronize()
    torch.cuda.current_stream().wait_stream(s)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, pool=mempool):
        static_tokens = cg_sample(
                 static_sampling_logits,
                 top_p=top_p, T=T
            )
    def run(target_logits, top_p=None, T=None):
        static_sampling_logits.copy_(target_logits)
        graph.replay()
        return static_tokens.clone()
    return run

def sampling_argmax_batch(logits: torch.Tensor):
    return logits.topk(k=1, dim=-1).indices.flatten(start_dim=1).long()

def cuda_graph_for_sampling_argmax_batch(
                device="cuda:0", dtype=torch.bfloat16, 
                dim=32000, n_warmups=3, mempool=None,
                idx_len = 1, batch_size=1):
    
    static_sampling_logits = torch.full((batch_size, idx_len, dim), 1, dtype=dtype, device=device)
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(n_warmups):
            static_position = sampling_argmax_batch(
                 static_sampling_logits,
            )
        s.synchronize()
    torch.cuda.current_stream().wait_stream(s)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, pool=mempool):
        static_position = sampling_argmax_batch(
                 static_sampling_logits,
            )
    def run(draft_logits):
        static_sampling_logits.copy_(draft_logits)
        graph.replay()
        return static_position.clone()
    return run

def device_sync(device):
    if "cuda" in device:
        torch.cuda.synchronize(device)
    elif ("cpu" in device) or ("mps" in device):
        pass
    else:
        print(f"device={device} is not yet suppported")

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

def load_model(checkpoint_path, device, precision, use_tp, rank_group=None, group=None):
    from MagicDec.Engine.model import Transformer
    with torch.device('meta'):
        model = Transformer.from_name(checkpoint_path.parent.name)
    checkpoint = torch.load(str(checkpoint_path), mmap=True, weights_only=True)
    if "model" in checkpoint and "stories" in str(checkpoint_path):
        checkpoint = checkpoint["model"]
    model.load_state_dict(checkpoint, assign=True)

    if use_tp:
        from MagicDec.Engine.tp import apply_tp
        print("Applying tensor parallel to model ...")
        apply_tp(model, rank_group, group=group)

    model = model.to(device=device, dtype=precision)
    return model.eval()


def load_model_draft(checkpoint_path, device, precision, use_tp, rank_group=None, group=None):
    import MagicDec.Engine.model_draft as draft
    with torch.device('meta'):
        model = draft.Transformer.from_name(checkpoint_path.parent.name)
    checkpoint = torch.load(str(checkpoint_path), mmap=True, weights_only=True)
    if "model" in checkpoint and "stories" in str(checkpoint_path):
        checkpoint = checkpoint["model"]
    model.load_state_dict(checkpoint, assign=True)

    if use_tp:
        from MagicDec.Engine.tp import apply_tp
        print("Applying tensor parallel to model ...")
        apply_tp(model, rank_group, group=group)

    model = model.to(device=device, dtype=precision)
    return model.eval()

def load_model_selfspec(checkpoint_path, device, precision, use_tp, rank_group=None, group=None):
    import MagicDec.Engine.model_selfspec as selfspec
    with torch.device('meta'):
        model = selfspec.Transformer.from_name(checkpoint_path.parent.name)
    checkpoint = torch.load(str(checkpoint_path), mmap=True, weights_only=True)
    if "model" in checkpoint and "stories" in str(checkpoint_path):
        checkpoint = checkpoint["model"]
    model.load_state_dict(checkpoint, assign=True)

    if use_tp:
        from MagicDec.Engine.tp import apply_tp
        print("Applying tensor parallel to model ...")
        apply_tp(model, rank_group, group=group)

    model = model.to(device=device, dtype=precision)
    return model.eval()