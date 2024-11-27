# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import os
from typing import List, Optional

import torch
import torch.distributed as dist
from torch import nn
if os.uname().sysname != "Darwin":
    from torch.distributed import _functional_collectives as funcol
else:
    # Distributed is not supported on MacOS
    funcol = None

from MagicDec.Engine.SnapKV.model import Attention, FeedForward, Transformer
from itertools import accumulate


def _get_global_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", "0"))

def is_local():
    return _get_global_rank() == 0

def local_break():
    if is_local():
        breakpoint()
    dist.barrier()

def _get_world_size() -> int:
    return int(os.environ.get("LOCAL_WORLD_SIZE", "1"))

def _select_kv_heads(num_kv_heads, rank_group:list):
    global_rank = _get_global_rank()
    rank = rank_group.index(global_rank)
    world_size = len(rank_group)
    base_heads = num_kv_heads // world_size
    remainder = num_kv_heads % world_size
    distribution = [base_heads] * world_size
    for i in range(remainder):
        distribution[i] += 1
    cumulative_distribution = list(accumulate(distribution))
    if rank == 0:
        start = 0
        end = cumulative_distribution[0]
    else:
        start = cumulative_distribution[rank-1]
        end = cumulative_distribution[rank]
    return start, end

def init_dist(draft_ranks=None):
    global_rank = _get_global_rank()
    world_size = _get_world_size()
    torch.cuda.set_device(global_rank)
    dist.init_process_group(backend="nccl", rank=global_rank, world_size=world_size, device_id=torch.device(f'cuda:{global_rank}'))
    global_group = dist.group.WORLD
    if draft_ranks != None:
        draft_group = dist.new_group(draft_ranks)
        return global_rank, global_group, draft_group
    else:
        return global_rank, global_group


def _apply_tp_linear(linear: nn.Linear, style: str, weight_splits: List[int] = [], rank_group=None, num_kv_heads = None, num_heads = None, head_dim = None) -> None:
    num_group = num_heads//num_kv_heads
    kv_start, kv_end = _select_kv_heads(num_kv_heads, rank_group)
    q_start = kv_start*num_group*head_dim
    q_end = kv_end*num_group*head_dim
    kv_start = kv_start*head_dim
    kv_end = kv_end*head_dim

    # Linear's weight matrix is transposed, and is of shape
    # (linear.out_features, linear.in_features)
    dim_lookup = {
        "colwise": (0, "out_features"),
        "rowwise": (1, "in_features")
    }
    assert style in dim_lookup
    shard_dim, size_attr = dim_lookup[style]

    # ensure we can shard evenly
    # assert getattr(linear, size_attr) % world_size == 0
    def shard(x, dim, start, end):
        # assert x.size(dim=dim) % world_size == 0
        if dim==0:
            return x[start:end]
        elif dim==1:
            return x[:,start:end]

    def shard_qkv(qkv, dim, weight_splits):
        q, k, v = qkv.split(weight_splits, dim=dim)
        q = shard(q, dim, q_start, q_end)
        k = shard(k, dim, kv_start, kv_end)
        v = shard(v, dim, kv_start, kv_end)
        return torch.cat((q,k,v), dim=dim)

    # shard
    if weight_splits:
        # attention
        assert len(weight_splits) == 3
        sharded_weight = shard_qkv(linear.weight, shard_dim, weight_splits)
        if hasattr(linear, "scales") and style == "colwise":
            linear.scales = shard_qkv(linear.scales, 0, weight_splits)
    else:
        sharded_weight = shard(linear.weight, shard_dim, q_start, q_end)
        if hasattr(linear, "scales") and style == "colwise":
            linear.scales = shard(linear.scales, 0, q_start, q_end)

    # local_break()
    linear.weight = nn.Parameter(sharded_weight, requires_grad=False)
    setattr(linear, size_attr, linear.weight.shape[shard_dim])

    # shape info should still be synced
    # assert linear.weight.shape == (linear.out_features, linear.in_features)

def _apply_tp_linear_mlp(linear: nn.Linear, style: str, weight_splits: List[int] = [], rank_group=None) -> None:
    global_rank = _get_global_rank()
    rank = rank_group.index(global_rank)
    world_size = len(rank_group)

    # Linear's weight matrix is transposed, and is of shape
    # (linear.out_features, linear.in_features)
    dim_lookup = {
        "colwise": (0, "out_features"),
        "rowwise": (1, "in_features")
    }
    assert style in dim_lookup
    shard_dim, size_attr = dim_lookup[style]

    # ensure we can shard evenly
    # assert getattr(linear, size_attr) % world_size == 0
    def shard(x, dim):
        # assert x.size(dim=dim) % world_size == 0
        return torch.chunk(x, world_size, dim=dim)[rank]

    # shard
    sharded_weight = shard(linear.weight, shard_dim)
    if hasattr(linear, "scales") and style == "colwise":
        linear.scales = shard(linear.scales, 0)

    # local_break()
    linear.weight = nn.Parameter(sharded_weight, requires_grad=False)
    setattr(linear, size_attr, linear.weight.shape[shard_dim])

    # shape info should still be synced
    # assert linear.weight.shape == (linear.out_features, linear.in_features)


def _apply_tp_ffn(mlp: FeedForward, rank_group, group) -> None:
    assert hasattr(mlp, "w1")
    assert hasattr(mlp, "w3")
    assert hasattr(mlp, "w2")

    _apply_tp_linear_mlp(mlp.w1, "colwise", rank_group=rank_group)
    _apply_tp_linear_mlp(mlp.w3, "colwise", rank_group=rank_group)
    _apply_tp_linear_mlp(mlp.w2, "rowwise", rank_group=rank_group)
    mlp.process_group = group

    # mlp.register_forward_hook(lambda _module, _input, output: funcol.all_reduce(
    #     output, "sum", group))


def _apply_tp_attn(attn: Attention, rank_group, config, group) -> None:
    assert hasattr(attn, "wqkv")
    assert hasattr(attn, "wo")

    kv_size = attn.n_local_heads * attn.head_dim
    _apply_tp_linear(attn.wqkv, "colwise", [attn.dim, kv_size, kv_size], rank_group=rank_group, num_kv_heads = attn.n_local_heads, num_heads = attn.n_head, head_dim=attn.head_dim)
    _apply_tp_linear(attn.wo, "rowwise", rank_group=rank_group, num_kv_heads = attn.n_local_heads, num_heads = attn.n_head, head_dim=attn.head_dim)

    # overwrite
    attn.n_head = config.n_head
    attn.dim = config.dim
    attn.head_dim = attn.dim // attn.n_head
    attn.n_local_heads = config.n_local_heads
    attn.process_group = group
    # attn.register_forward_hook(lambda _module, _input, output: funcol.all_reduce(
    #     output, "sum", group))


def _apply_tp_Transformer(Transformer: Transformer, rank_group, process_group) -> None:
    # overwrite config before Transformer.setup_cache is called
    num_heads = Transformer.config.n_head
    num_kv_heads = Transformer.config.n_local_heads
    num_group = num_heads // num_kv_heads
    start, end= _select_kv_heads(num_kv_heads, rank_group)
    local_num_kv_heads = end-start
    local_num_heads= local_num_kv_heads*num_group
    local_dim = Transformer.config.dim * local_num_kv_heads // num_kv_heads
    Transformer.config.n_head = local_num_heads
    Transformer.config.dim = local_dim
    Transformer.config.n_local_heads = local_num_kv_heads
    _apply_tp_linear_mlp(Transformer.output, "colwise", rank_group=rank_group)
    Transformer.process_group = process_group
    Transformer.world_size = dist.get_world_size(process_group)
    Transformer.rank = dist.get_rank(process_group)


def apply_tp(model: Transformer, rank_group, group) -> None:
    _apply_tp_Transformer(model, rank_group, group)
    for block in model.layers:
        # Apply to MLP
        _apply_tp_ffn(block.feed_forward, rank_group, group)
        _apply_tp_attn(block.attention, rank_group, model.config, group)
