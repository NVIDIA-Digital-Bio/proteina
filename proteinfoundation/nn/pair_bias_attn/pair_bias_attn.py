# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.


from typing import Optional

import torch
from einops import rearrange
from torch import Tensor, einsum, nn

from proteinfoundation.nn.alphafold3_pytorch_utils.utils import (
    rearrange_to_dense_trunk,
)
from proteinfoundation.nn.pair_bias_attn.rotary import RotaryEmbedding
def exists(val) -> bool:
    """returns whether val is not none"""
    return val is not None


def default(x, y):
    """returns x if it exists, otherwise y"""
    return x if exists(x) else y


max_neg_value = lambda x: torch.finfo(x.dtype).min


class PairBiasAttention(nn.Module):
    """
    Scalar Feature masked attention with pair bias and gating.
    Code modified from
    https://github.com/MattMcPartlon/protein-docking/blob/main/protein_learning/network/modules/node_block.py
    """

    def __init__(
        self,
        node_dim: int,
        dim_head: int,
        heads: int,
        bias: bool,
        dim_out: int,
        qkln: bool,
        pair_dim: Optional[int] = None,
        apply_rotary: bool = False,
        **kawrgs  # noqa
    ):
        super().__init__()
        inner_dim = dim_head * heads
        self.node_dim, self.pair_dim = node_dim, pair_dim
        self.dim_head = dim_head
        self.heads, self.scale = heads, dim_head**-0.5
        # self.to_qkv = nn.Linear(node_dim, inner_dim * 3, bias=bias)
        self.to_q = nn.Linear(node_dim, inner_dim, bias=bias)
        self.to_kv = nn.Linear(node_dim, inner_dim * 2, bias=bias)
        self.to_g = nn.Linear(node_dim, inner_dim)
        self.to_out_node = nn.Linear(inner_dim, default(dim_out, node_dim))
        self.node_norm = nn.LayerNorm(node_dim)
        self.q_layer_norm = nn.LayerNorm(inner_dim) if qkln else nn.Identity()
        self.k_layer_norm = nn.LayerNorm(inner_dim) if qkln else nn.Identity()
        if exists(pair_dim):
            self.to_bias = nn.Linear(pair_dim, heads, bias=False)
            self.pair_norm = nn.LayerNorm(pair_dim)
        else:
            self.to_bias, self.pair_norm = None, None
        self.rotary = RotaryEmbedding(dim=dim_head) if apply_rotary else None
            
    def _apply_rotary(self, q: torch.Tensor, k: torch.Tensor):
        q = q.unflatten(-1, (self.heads, self.dim_head))
        k = k.unflatten(-1, (self.heads, self.dim_head))
        q, k = self.rotary(q, k)
        q = q.flatten(-2, -1)
        k = k.flatten(-2, -1)
        return q, k

    def forward(
        self,
        node_feats: Tensor,
        pair_feats: Optional[Tensor],
        mask: Optional[Tensor],
        n_queries: int,
        n_keys: int,
    ) -> Tensor:
        """Multi-head scalar Attention Layer

        :param node_feats: scalar features of shape (b,n,d_s)
        :param pair_feats: pair features of shape (b,n,n,d_e)
        :param mask: boolean tensor of node adjacencies
        :return:
        """
        assert exists(self.to_bias) or not exists(pair_feats)
        node_feats, h = self.node_norm(node_feats), self.heads
        pair_feats = self.pair_norm(pair_feats) if exists(pair_feats) else None
        q = self.to_q(node_feats)
        k, v = self.to_kv(node_feats).chunk(2, dim=-1)
        q = self.q_layer_norm(q)
        k = self.k_layer_norm(k)
        # add rotary embedding
        if self.rotary is not None:
            q, k = self._apply_rotary(q, k)
        g = self.to_g(node_feats)
        b = (
            rearrange(self.to_bias(pair_feats), "b ... h -> b h ...")
            if exists(pair_feats)
            else 0
        )
        q, k, v, g = map(
            lambda t: rearrange(t, "b ... (h d) -> b h ... d", h=h), (q, k, v, g)
        )
        if n_queries and n_keys:
            attn_feats = self._local_attn(q, k, v, b, mask, n_queries, n_keys)
        else:
            attn_feats = self._attn(q, k, v, b, mask)
        attn_feats = rearrange(
            torch.sigmoid(g) * attn_feats, "b h n d -> b n (h d)", h=h
        )
        return self.to_out_node(attn_feats)

    def _attn(self, q, k, v, b, mask: Optional[Tensor]) -> Tensor:
        """Perform attention update"""
        sim = einsum("b h i d, b h j d -> b h i j", q, k) * self.scale
        if exists(mask):
            mask = rearrange(mask, "b i j -> b () i j")
            sim = sim.masked_fill(~mask, max_neg_value(sim))
        attn = torch.softmax(sim + b, dim=-1)
        return einsum("b h i j, b h j d -> b h i d", attn, v)
    
    def _local_attn(
        self,
        q,
        k,
        v,
        b,
        mask: Optional[Tensor],
        n_queries: int,
        n_keys: int,
        inf: float = 1e10
    ) -> Tensor:
        # Rerrange to dense trunks
        # q: [*, n, d] -> [*, n_trunks, n_queries, d]
        # attn_bias: [*, n, d] -> [*, n_trunks, n_queries, n_keys]
        q_trunked, k_trunked, v_trunked, _, q_pad_length = (
            rearrange_to_dense_trunk(
                q=q,
                k=k,
                v=v,
                n_queries=n_queries,
                n_keys=n_keys,
                attn_bias=None,
                inf=inf,
            )
        )
        
        sim = einsum("b h n i d, b h n j d -> b h n i j", q_trunked, k_trunked) * self.scale
        if exists(mask):
            mask = rearrange(mask, "b n i j -> b () n i j")
            sim = sim.masked_fill(~mask, max_neg_value(sim))
        attn = torch.softmax(sim + b, dim=-1)
        out = einsum("b h n i j, b h n j d -> b h n i d", attn, v_trunked)
        out = out.reshape(*out.shape[:-3], -1, out.shape[-1])
        if q_pad_length > 0:
            out = out[..., :-q_pad_length, :]
        return out
