# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.


from typing import Dict, Optional, Union

import einops
import einops.layers
import einops.layers.torch
import math
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from functools import partial
from openfold.model.msa import MSARowAttentionWithPairBias
from openfold.model.pair_transition import PairTransition
from openfold.model.structure_module import InvariantPointAttention
from openfold.model.triangular_multiplicative_update import (
    TriangleMultiplicationIncoming,
    TriangleMultiplicationOutgoing,
)
from openfold.utils.rigid_utils import Rigid
from proteinfoundation.nn.feature_factory import FeatureFactory
from proteinfoundation.nn.pair_bias_attn.pair_bias_attn import PairBiasAttention
from proteinfoundation.nn.alphafold3_pytorch_utils.modules import (
    AdaptiveLayerNorm,
    AdaptiveLayerNormOutputScale,
    Transition,
)
from proteinfoundation.nn.alphafold3_pytorch_utils.utils import (
    rearrange_qk_to_dense_trunk,
    broadcast_token_to_local_atom_pair,
    broadcast_token_to_atom,
    aggregate_atom_to_token,
)


class MultiHeadAttention(torch.nn.Module):
    """Typical multi-head self-attention attention using pytorch's module."""

    def __init__(self, dim_token, nheads, dropout=0.0):
        super().__init__()

        self.to_q = torch.nn.Linear(dim_token, dim_token)
        self.to_kv = torch.nn.Linear(dim_token, 2 * dim_token, bias=False)

        self.mha = torch.nn.MultiheadAttention(
            embed_dim=dim_token,
            num_heads=nheads,
            dropout=dropout,
            batch_first=True,
        )

    def forward(self, x, mask):
        """
        Args:
            x: Input sequence, shape [b, n, dim_token]
            mask: binary mask, shape [b, n]

        Returns:
            Updated sequence, shape [b, n, dim_token]
        """
        query = self.to_q(x)  # [b, n, dim_token]
        key, value = self.to_kv(x).chunk(2, dim=-1)  # Each [b, n, dim_token]
        return (
            self.mha(
                query=query,
                key=key,
                value=value,
                key_padding_mask=~mask,  # Indicated what should be ignores with True, that's why the ~
                need_weights=False,
                is_causal=False,
            )[0]
            * mask[..., None]
        )  # [b, n, dim_token]


class MultiHeadBiasedAttention(torch.nn.Module):
    """Multi-head self-attention with pair bias, based on openfold."""

    def __init__(self, dim_token, dim_pair, nheads, dropout=0.0):
        super().__init__()

        self.row_attn_pair_bias = MSARowAttentionWithPairBias(
            c_m=dim_token,
            c_z=dim_pair,
            c_hidden=int(dim_token // nheads),  # Per head dimension
            no_heads=nheads,
        )

    def forward(self, x, pair_rep, mask):
        """
        Args:
            x: Input sequence, shape [b, n, dim_token]
            pair_rep: Pair representation, shape [b, n, n, dim_pair]
            mask: Binary mask, shape [b, n]

        Returns:
            Updated sequence representation, shape [b, n, dim_token]
        """
        # Add extra dimension for MSA, unused here but required by openfold
        x = einops.rearrange(x, "b n d -> b () n d")  # [b, 1, n, dim_token]
        mask = einops.rearrange(mask, "b n -> b () n") * 1.0  # float [b, 1, n]
        x = self.row_attn_pair_bias(x, pair_rep, mask)  # [b, 1, n, dim_token]
        x = x * mask[..., None]
        x = einops.rearrange(
            x, "b () n c -> b n c"
        )  # Remove extra dimension [b, n, dim_token]
        return x


class MultiHeadAttentionADALN(torch.nn.Module):
    """Typical multi-head self-attention with adaptive layer norm applied to input
    and adaptive scaling applied to output."""

    def __init__(self, dim_token, nheads, dim_cond, dropout=0.0):
        super().__init__()
        self.adaln = AdaptiveLayerNorm(dim=dim_token, dim_cond=dim_cond)
        self.mha = MultiHeadAttention(
            dim_token=dim_token, nheads=nheads, dropout=dropout
        )
        self.scale_output = AdaptiveLayerNormOutputScale(
            dim=dim_token, dim_cond=dim_cond
        )

    def forward(self, x, cond, mask):
        """
        Args:
            x: Input sequence representation, shape [b, n, dim_token]
            cond: Conditioning variables, shape [b, n, dim_cond]
            mask: Binary mask, shape [b, n]

        Returns:
            Updated sequence representation, shape [b, n, dim_token].
        """
        x = self.adaln(x, cond, mask)
        x = self.mha(x, mask)
        x = self.scale_output(x, cond, mask)
        return x * mask[..., None]


class MultiHeadBiasedAttentionADALN(torch.nn.Module):
    """Pair biased multi-head self-attention with adaptive layer norm applied to input
    and adaptive scaling applied to output."""

    def __init__(self, dim_token, dim_pair, nheads, dim_cond, dropout=0.0):
        super().__init__()
        self.adaln = AdaptiveLayerNorm(dim=dim_token, dim_cond=dim_cond)
        self.mha = MultiHeadBiasedAttention(
            dim_token=dim_token, dim_pair=dim_pair, nheads=nheads, dropout=dropout
        )
        self.scale_output = AdaptiveLayerNormOutputScale(
            dim=dim_token, dim_cond=dim_cond
        )

    def forward(self, x, pair_rep, cond, mask):
        """
        Args:
            x: Input sequence representation, shape [b, n, dim_token]
            cond: Conditioning variables, shape [b, n, dim_cond]
            pair_rep: Pair represnetation, shape [b, n, n, dim_pair]
            mask: Binary mask, shape [b, n]

        Returns:
            Updated sequence representation, shape [b, n, dim_token].
        """
        x = self.adaln(x, cond, mask)
        x = self.mha(x, pair_rep, mask)
        x = self.scale_output(x, cond, mask)
        return x * mask[..., None]


class MultiHeadBiasedAttentionADALN_MM(torch.nn.Module):
    """Pair biased multi-head self-attention with adaptive layer norm applied to input
    and adaptive scaling applied to output."""

    def __init__(self, dim_token, dim_pair, nheads, dim_cond, use_qkln, apply_rotary):
        super().__init__()
        dim_head = int(dim_token // nheads)
        self.adaln = AdaptiveLayerNorm(dim=dim_token, dim_cond=dim_cond)
        self.mha = PairBiasAttention(
            node_dim=dim_token,
            dim_head=dim_head,
            heads=nheads,
            bias=True,
            dim_out=dim_token,
            qkln=use_qkln,
            pair_dim=dim_pair,
            apply_rotary=apply_rotary
        )
        self.scale_output = AdaptiveLayerNormOutputScale(
            dim=dim_token, dim_cond=dim_cond
        )

    def forward(
        self,
        x,
        pair_rep,
        cond,
        mask,
        pair_mask,
        n_queries,
        n_keys
    ):
        """
        Args:
            x: Input sequence representation, shape [b, n, dim_token]
            cond: Conditioning variables, shape [b, n, dim_cond]
            pair_rep: Pair represnetation, shape [b, n, n, dim_pair]
            mask: Binary mask, shape [b, n]

        Returns:
            Updated sequence representation, shape [b, n, dim_token].
        """
        x = self.adaln(x, cond, mask)
        x = self.mha(
            node_feats=x,
            pair_feats=pair_rep,
            mask=pair_mask,
            n_queries=n_queries,
            n_keys=n_keys
        )
        x = self.scale_output(x, cond, mask)
        return x * mask[..., None]


class TransitionADALN(torch.nn.Module):
    """Transition layer with adaptive layer norm applied to input and adaptive
    scaling aplied to output."""

    def __init__(self, *, dim, dim_cond, expansion_factor=4):
        super().__init__()
        self.adaln = AdaptiveLayerNorm(dim=dim, dim_cond=dim_cond)
        self.transition = Transition(
            dim=dim, expansion_factor=expansion_factor, layer_norm=False
        )
        self.scale_output = AdaptiveLayerNormOutputScale(dim=dim, dim_cond=dim_cond)

    def forward(self, x, cond, mask):
        """
        Args:
            x: Input sequence representation, shape [b, n, dim]
            cond: conditioning variables, shape [b, n, dim_cond]
            mask: binary mask, shape [b, n]

        Returns:
            Updated sequence representation, shape [b, n, dim]
        """
        x = self.adaln(x, cond, mask)  # [b, n, dim]
        x = self.transition(x, mask)  # [b, n, dim]
        x = self.scale_output(x, cond, mask)  # [b, n, dim]
        return x * mask[..., None]  # [b, n, dim]


class MultiheadAttnAndTransition(torch.nn.Module):
    """Layer that applies mha and transition to a sequence representation. Both layers are their adaptive versions
    which rely on conditining variables (see above).

    Args:
        dim_token: Token dimension in sequence representation.
        dim_pair: Dimension of pair representation.
        nheads: Number of attention heads.
        dim_cond: Dimension of conditioning variables.
        residual_mha: Whether to use a residual connection in the mha layer.
        residual_transition: Whether to use a residual connection in the transition layer.
        parallel_mha_transition: Whether to run mha and transition in parallel or sequentially.
        use_attn_pair_bias: Whether to use a pair represnetation to bias attention.
        use_qkln: Whether to use layer norm on keyus and queries for attention.
        dropout: droput use in the self-attention layer.
    """

    def __init__(
        self,
        dim_token,
        dim_pair,
        nheads,
        dim_cond,
        residual_mha,
        residual_transition,
        parallel_mha_transition,
        use_attn_pair_bias,
        use_qkln,
        apply_rotary,
        dropout=0.0,
        expansion_factor=4,
    ):
        super().__init__()
        self.parallel = parallel_mha_transition
        self.use_attn_pair_bias = use_attn_pair_bias

        # If parallel do not allow both layers to have a residual connection since it leads to adding x twice
        if self.parallel and residual_mha and residual_transition:
            residual_transition = False

        self.residual_mha = residual_mha
        self.residual_transition = residual_transition

        self.mhba = MultiHeadBiasedAttentionADALN_MM(
            dim_token=dim_token,
            dim_pair=dim_pair,
            nheads=nheads,
            dim_cond=dim_cond,
            use_qkln=use_qkln,
            apply_rotary=apply_rotary
        )

        self.transition = TransitionADALN(
            dim=dim_token, dim_cond=dim_cond, expansion_factor=expansion_factor
        )

    def _apply_mha(self, x, pair_rep, cond, mask, pair_mask, n_queries, n_keys):
        x_attn = self.mhba(x, pair_rep, cond, mask, pair_mask, n_queries, n_keys)
        if self.residual_mha:
            x_attn = x_attn + x
        return x_attn * mask[..., None]

    def _apply_transition(self, x, cond, mask):
        x_tr = self.transition(x, cond, mask)
        if self.residual_transition:
            x_tr = x_tr + x
        return x_tr * mask[..., None]

    def forward(self, x, pair_rep, cond, mask, pair_mask=None, n_queries=None, n_keys=None):
        """
        Args:
            x: Input sequence representation, shape [b, n, dim_token]
            cond: conditioning variables, shape [b, n, dim_cond]
            mask: binary mask, shape [b, n]
            pair_rep: Pair representation (if provided, if no bias will be ignored), shape [b, n, n, dim_pair] or None

        Returns:
            Updated sequence representation, shape [b, n, dim].
        """
        x = x * mask[..., None]
        if self.parallel:
            x = self._apply_mha(x, pair_rep, cond, mask, pair_mask, n_queries, n_keys) + self._apply_transition(
                x, cond, mask
            )
        else:
            x = self._apply_mha(x, pair_rep, cond, mask, pair_mask, n_queries, n_keys)
            x = self._apply_transition(x, cond, mask)
        return x * mask[..., None]


class PairReprUpdate(torch.nn.Module):
    """Layer to update the pair representation."""

    def __init__(
        self,
        token_dim,
        pair_dim,
        expansion_factor_transition=2,
        use_tri_mult=False,
        tri_mult_c=196,
    ):
        super().__init__()

        self.use_tri_mult = use_tri_mult
        self.layer_norm_in = torch.nn.LayerNorm(token_dim)
        self.linear_x = torch.nn.Linear(token_dim, int(2 * pair_dim), bias=False)

        if use_tri_mult:
            tri_mult_c = min(pair_dim, tri_mult_c)
            self.tri_mult_out = TriangleMultiplicationOutgoing(
                c_z=pair_dim, c_hidden=tri_mult_c
            )
            self.tri_mult_in = TriangleMultiplicationIncoming(
                c_z=pair_dim, c_hidden=tri_mult_c
            )
        self.transition_out = PairTransition(
            c_z=pair_dim, n=expansion_factor_transition
        )

    def _apply_mask(self, pair_rep, pair_mask):
        """
        pair_rep has shape [b, n, n, pair_dim]
        pair_mask has shape [b, n, n]
        """
        return pair_rep * pair_mask[..., None]

    def forward(self, x, pair_rep, mask):
        """
        Args:
            x: Input sequence, shape [b, n, token_dim]
            pair_rep: Input pair representation, shape [b, n, n, pair_dim]
            mask: binary mask, shape [b, n]

        Returns:
            Updated pair representation, shape [b, n, n, pair_dim].
        """
        pair_mask = mask[:, None, :] * mask[:, :, None]  # [b, n, n]
        x = x * mask[..., None]  # [b, n, token_dim]
        x_proj_1, x_proj_2 = self.linear_x(self.layer_norm_in(x)).chunk(
            2, dim=-1
        )  # [b, n, pair_dim] each
        pair_rep = (
            pair_rep + x_proj_1[:, None, :, :] + x_proj_2[:, :, None, :]
        )  # [b, n, n, pair_dim]
        pair_rep = self._apply_mask(pair_rep, pair_mask)  # [b, n, n, pair_dim]
        if self.use_tri_mult:
            pair_rep = pair_rep + checkpoint(
                self.tri_mult_out, *(pair_rep, pair_mask * 1.0)
            )
            pair_rep = self._apply_mask(pair_rep, pair_mask)  # [b, n, n, pair_dim]
            pair_rep = pair_rep + checkpoint(
                self.tri_mult_in, *(pair_rep, pair_mask * 1.0)
            )
            pair_rep = self._apply_mask(pair_rep, pair_mask)  # [b, n, n, pair_dim]
        pair_rep = pair_rep + checkpoint(
            self.transition_out, *(pair_rep, pair_mask * 1.0)
        )
        pair_rep = self._apply_mask(pair_rep, pair_mask)  # [b, n, n, pair_dim]
        return pair_rep


class PairReprBuilder(torch.nn.Module):
    """
    Builds initial pair representation. Essentially the pair feature factory, but potentially with
    an adaptive layer norm layer as well.
    """

    def __init__(self, feats_repr, feats_cond, dim_feats_out, dim_cond_pair, **kwargs):
        super().__init__()

        self.init_repr_factory = FeatureFactory(
            feats=feats_repr,
            dim_feats_out=dim_feats_out,
            use_ln_out=True,
            mode="pair",
            **kwargs,
        )

        self.cond_factory = None  # Build a pair feature for conditioning and use it for adaln the pair representation
        if feats_cond is not None:
            if len(feats_cond) > 0:
                self.cond_factory = FeatureFactory(
                    feats=feats_cond,
                    dim_feats_out=dim_cond_pair,
                    use_ln_out=True,
                    mode="pair",
                    **kwargs,
                )
                self.adaln = AdaptiveLayerNorm(
                    dim=dim_feats_out, dim_cond=dim_cond_pair
                )

    def forward(self, batch_nn):
        mask = batch_nn["mask"]  # [b, n]
        pair_mask = mask[:, :, None] * mask[:, None, :]  # [b, n, n]
        repr = self.init_repr_factory(batch_nn)  # [b, n, n, dim_feats_out]
        if self.cond_factory is not None:
            cond = self.cond_factory(batch_nn)  # [b, n, n, dim_cond]
            repr = self.adaln(repr, cond, pair_mask)
        return repr


class AtomAttentionEncoder(torch.nn.Module):
    """
    Encoder for atom attention.
    """
    def __init__(
        self,
        atom_dim: int,
        atom_dim_pair: int,
        token_dim: int,
        pair_repr_dim: int,
        dim_cond: int,
        n_queries: int = 32,
        n_keys: int = 128,
        n_layers: int = 2,
        nheads: int = 4,
        use_qkln: bool = True,
        apply_rotary: bool = False,
        ref_pos_augment: bool = False,
        **kwargs,
    ):
        super(AtomAttentionEncoder, self).__init__()
        
        self.atom_dim = atom_dim
        self.atom_dim_pair = atom_dim_pair
        self.token_dim = token_dim
        self.pair_repr_dim = pair_repr_dim
        self.dim_cond = dim_cond
        self.n_queries = n_queries
        self.n_keys = n_keys
        self.n_layers = n_layers
        self.nheads = nheads
        self.use_qkln = use_qkln
        
        self.ref_pos = torch.tensor(
            [
                [ 0.8053, -1.9707, -1.4140],
                [ 0.7207, -0.5147, -1.3373],
                [-0.0555, -0.0849, -0.1298],
                [ 0.5367,  0.1090,  0.9654],
            ],
            dtype=torch.float,
        )  # [N, CA, C, O]
        self.ref_element = F.one_hot(torch.tensor(
            [
                6, 5, 5, 7
            ],
            dtype=torch.long,
        ), num_classes=10).float()  # [N, CA, C, O]
        self.ref_charge = torch.tensor(
            [
                [0], [0], [0], [0]
            ],
            dtype=torch.float,
        )  # [N, CA, C, O]
        self.ref_mask = torch.tensor(
            [
                [1], [1], [1], [1]
            ],
            dtype=torch.float,
        )  # [N, CA, C, O]
        self.input_feature = {
            "ref_pos": 3,
            "ref_charge": 1,
            "ref_mask": 1,
            "ref_element": 10,
        }
        
        self.linear_no_bias_f = torch.nn.Linear(
            sum(self.input_feature.values()), atom_dim, bias=False
        )
        self.linear_no_bias_d = torch.nn.Linear(
            3, atom_dim_pair, bias=False
        )
        self.linear_no_bias_invd = torch.nn.Linear(
            1, atom_dim_pair, bias=False
        )
        
        self.linear_no_bias_v = torch.nn.Linear(
            1, atom_dim_pair, bias=False
        )
        self.layernorm_c = torch.nn.LayerNorm(dim_cond)
        self.linear_no_bias_c = torch.nn.Linear(
            dim_cond, atom_dim, bias=False
        )
        self.layernorm_z = torch.nn.LayerNorm(pair_repr_dim)  # memory bottleneck
        self.linear_no_bias_z = torch.nn.Linear(
            pair_repr_dim, atom_dim_pair, bias=False
        )
        self.linear_3d_embed = torch.nn.Linear(
            3, atom_dim, bias=False
        )
        self.linear_no_bias_cl = torch.nn.Linear(
            atom_dim, atom_dim_pair, bias=False
        )
        self.linear_no_bias_cm = torch.nn.Linear(
            atom_dim, atom_dim_pair, bias=False
        )
        self.small_mlp = torch.nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.Linear(atom_dim_pair, atom_dim_pair, bias=False),
            torch.nn.ReLU(),
            torch.nn.Linear(atom_dim_pair, atom_dim_pair, bias=False),
            torch.nn.ReLU(),
            torch.nn.Linear(atom_dim_pair, atom_dim_pair, bias=False),
        )
        self.linear_no_bias_q = torch.nn.Linear(
            atom_dim, token_dim, bias=False
        )
        self.transformer_layers = torch.nn.ModuleList([
            MultiheadAttnAndTransition(
                dim_token=atom_dim,
                dim_pair=atom_dim_pair,
                nheads=nheads,
                dim_cond=atom_dim,
                residual_mha=True,
                residual_transition=True,
                parallel_mha_transition=False,
                use_attn_pair_bias=True,
                use_qkln=use_qkln,
                apply_rotary=apply_rotary
            )
            for _ in range(n_layers)
        ])
    
    def forward(self, coors_3d, pair_repr, cond, coors_mask, mask):
        b, n_atoms = coors_3d.shape[:2]
        n_tokens = n_atoms // 4
        assert n_tokens == mask.shape[1]
        device = coors_3d.device
        
        # Expand ref features to match batch size
        expand_ref_feats = partial(einops.repeat, pattern="n c -> b (a n) c", b=b, a=n_tokens)
        (
            ref_pos,
            ref_charge,
            ref_mask,
            ref_element,
        ) = map(expand_ref_feats, (
            self.ref_pos,
            self.ref_charge,
            self.ref_mask,
            self.ref_element,
        ))
        ref_feats_to_device = lambda x: x.to(device)
        (
            ref_pos,
            ref_charge,
            ref_mask,
            ref_element,
        ) = map(ref_feats_to_device, (
            ref_pos,
            ref_charge,
            ref_mask,
            ref_element,
        ))
        ref_space_uid = einops.repeat(torch.arange(n_tokens, device=device), "n -> b (n c)", b=b, c=4)
        atom_to_token_idx = ref_space_uid.clone()
        
        # Embed input features
        c_l= torch.cat(
            [
                ref_pos,
                ref_charge,
                ref_mask,
                ref_element,
            ],
            dim=-1
        ).to(device) * coors_mask[..., None]
        c_l = self.linear_no_bias_f(c_l)
        
        # Prepare tensors in dense trunks for local operations
        q_trunked_list, k_trunked_list, pad_info = rearrange_qk_to_dense_trunk(
            q=[ref_pos, ref_space_uid, coors_mask],
            k=[ref_pos, ref_space_uid, coors_mask],
            dim_q=[-2, -1, -1],
            dim_k=[-2, -1, -1],
            n_queries=self.n_queries,
            n_keys=self.n_keys,
            compute_mask=True,
        )
        
        # Compute atom pair feature
        pair_mask = (
            q_trunked_list[2][..., None] * k_trunked_list[2][..., None, :]
        ) * pad_info["mask_trunked"] # [..., n_blocks, n_queries, n_keys]
        d_lm = (
            q_trunked_list[0][..., None, :] - k_trunked_list[0][..., None, :, :]
        ) * pair_mask[..., None]  # [..., n_blocks, n_queries, n_keys, 3]
        v_lm = (
            q_trunked_list[1][..., None].int() == k_trunked_list[1][..., None, :].int()
        ).unsqueeze(
            dim=-1
        ) * pair_mask[..., None]  # [..., n_blocks, n_queries, n_keys, 1]
        
        p_lm = (self.linear_no_bias_d(d_lm) * v_lm) * pad_info[
            "mask_trunked"
        ].unsqueeze(
            dim=-1
        ) * pair_mask[..., None]  # [..., n_blocks, n_queries, n_keys, C_atompair]
        
        p_lm = (
            p_lm
            + self.linear_no_bias_invd(
                1 / (1 + (d_lm**2).sum(dim=-1, keepdim=True))
            )
            * v_lm
        )
        p_lm = p_lm + self.linear_no_bias_v(v_lm.to(dtype=p_lm.dtype)) * v_lm
        
        q_l = c_l.clone()
        
        c_l = c_l + self.linear_no_bias_c(
            self.layernorm_c(
                broadcast_token_to_atom(
                    x_token=cond, atom_to_token_idx=atom_to_token_idx
                )
            )
        )  # [..., N_atom, c_atom]
        
        z_local_pairs, _ = broadcast_token_to_local_atom_pair(
            z_token=pair_repr,
            atom_to_token_idx=atom_to_token_idx,
            n_queries=self.n_queries,
            n_keys=self.n_keys,
            compute_mask=False,
        )  # [..., n_blocks, n_queries, n_keys, c_z]
        z_local_pairs = z_local_pairs * pair_mask[..., None]
        
        p_lm = p_lm + self.linear_no_bias_z(
            self.layernorm_z(z_local_pairs)
        )  # [..., n_blocks, n_queries, n_keys, c_atompair]

        # Add the noisy positions
        q_l = q_l + self.linear_3d_embed(
            coors_3d
        ) * coors_mask[..., None]  # [..., N_atom, c_atom]
        
        # Add the combined single conditioning to the pair representation
        c_l_q, c_l_k, _ = rearrange_qk_to_dense_trunk(
            q=c_l,
            k=c_l,
            dim_q=-2,
            dim_k=-2,
            n_queries=self.n_queries,
            n_keys=self.n_keys,
            compute_mask=False,
        )
        
        p_lm = (
            p_lm
            + self.linear_no_bias_cl(F.relu(c_l_q[..., None, :]))
            + self.linear_no_bias_cm(F.relu(c_l_k[..., None, :, :]))
        )  # [..., n_blocks, n_queries, n_keys, c_atompair]

        # Run a small MLP on the pair activations
        p_lm = p_lm + self.small_mlp(p_lm)
        
        for layer in self.transformer_layers:
            q_l = layer(q_l, p_lm, c_l, coors_mask, pair_mask, self.n_queries, self.n_keys)
        
        # Aggregate per-atom representation to per-token representation
        a = aggregate_atom_to_token(
            x_atom=F.relu(self.linear_no_bias_q(q_l)) * coors_mask[..., None],
            atom_to_token_idx=atom_to_token_idx,
            n_token=n_tokens,
            reduce="mean",
        )
        
        return a, q_l, c_l, p_lm, pair_mask
    
    
class AtomAttentionDecoder(torch.nn.Module):
    """
    Decoder for atom attention.
    """
    def __init__(
        self,
        atom_dim: int,
        atom_dim_pair: int,
        token_dim: int,
        dim_cond: int,
        n_queries: int = 32,
        n_keys: int = 128,
        n_layers: int = 2,
        nheads: int = 4,
        use_qkln: bool = True,
        apply_rotary: bool = False,
        **kwargs,
    ):
        super(AtomAttentionDecoder, self).__init__()
        
        self.atom_dim = atom_dim
        self.atom_dim_pair = atom_dim_pair
        self.token_dim = token_dim
        self.dim_cond = dim_cond
        self.n_queries = n_queries
        self.n_keys = n_keys
        self.n_layers = n_layers
        self.nheads = nheads
        self.use_qkln = use_qkln

        self.linear_no_bias_a = torch.nn.Linear(
            token_dim, atom_dim, bias=False
        )
        self.layernorm_q = torch.nn.LayerNorm(atom_dim)
        self.linear_no_bias_out = torch.nn.Linear(
            atom_dim, 3, bias=False
        )
        
        self.transformer_layers = torch.nn.ModuleList([
            MultiheadAttnAndTransition(
                dim_token=atom_dim,
                dim_pair=atom_dim_pair,
                nheads=nheads,
                dim_cond=atom_dim,
                residual_mha=True,
                residual_transition=True,
                parallel_mha_transition=False,
                use_attn_pair_bias=True,
                use_qkln=use_qkln,
                apply_rotary=apply_rotary,
            ) for _ in range(n_layers)
        ])
        
    def forward(self, a, q_skip, c_skip, p_skip, coors_mask, pair_mask):
        b, n_tokens = a.shape[:2]
        n_atoms = n_tokens * 4
        device = a.device
        atom_to_token_idx = einops.repeat(torch.arange(n_tokens, device=device), "n -> (n c)", c=4)
        q = (
            self.linear_no_bias_a(
                broadcast_token_to_atom(
                    x_token=a, atom_to_token_idx=atom_to_token_idx
                )  # [..., N_atom, c_token]
            )  # [..., N_atom, c_atom]
            + q_skip
        ) * coors_mask[..., None]
        
        for layer in self.transformer_layers:
            q = layer(q, p_skip, c_skip, coors_mask, pair_mask, self.n_queries, self.n_keys)

        r = self.linear_no_bias_out(self.layernorm_q(q)) * coors_mask[..., None]
        
        return r

class ProteinTransformerAF3(torch.nn.Module):
    """
    Final neural network mimicking the one used in AF3 diffusion. It consists of:

    (1) Input preparation
    (1.a) Initial sequence representation from features
    (1.b) Embed coordaintes and add to initial sequence representation
    (1.c) Conditioning variables from features

    (2) Main trunk
    (2.a) A sequence of layers similar to algorithm 23 of AF3 (multi head attn, transition) using adaptive layer norm
    and adaptive output scaling (also from adaptive layer norm paper)

    (3) Recovering 3D coordinates
    (3.a) A layer that takes as input tokens and produces coordinates
    """

    def __init__(self, **kwargs):
        """
        Initializes the NN. The seqs and pair representations used are just zero in case
        no features are required."""
        super(ProteinTransformerAF3, self).__init__()
        self.module_type = kwargs.get("module_type", "decoder")
        self.ca_only = kwargs.get("ca_only", True)
        self.use_attn_pair_bias = kwargs["use_attn_pair_bias"]
        self.nlayers = kwargs["nlayers"]
        self.token_dim = kwargs["token_dim"]
        self.pair_repr_dim = kwargs["pair_repr_dim"]
        self.update_coors_on_the_fly = kwargs.get(
            "update_coors_on_the_fly", False
        )
        self.update_seq_with_coors = None
        self.update_pair_repr = kwargs.get(
            "update_pair_repr", False
        )
        self.update_pair_repr_every_n = kwargs.get(
            "update_pair_repr_every_n", 2
        )
        self.use_tri_mult = kwargs.get("use_tri_mult", False)
        self.feats_pair_cond = kwargs.get("feats_pair_cond", [])
        self.use_qkln = kwargs.get("use_qkln", False)
        self.apply_rotary = kwargs.get("apply_rotary", False)
        self.num_buckets_predict_pair = kwargs.get(
            "num_buckets_predict_pair", None
        )
        self.sampling_ratio = kwargs.get("sampling_ratio", 1)
        self.dim_latent = kwargs.get("dim_latent", None)

        # Registers
        self.num_registers = kwargs.get("num_registers", None)
        if self.num_registers is None or self.num_registers <= 0:
            self.num_registers = 0
            self.registers = None
        else:
            self.num_registers = int(self.num_registers)
            self.registers = torch.nn.Parameter(
                torch.randn(self.num_registers, self.token_dim) / 20.0
            )

        # To encode corrupted 3d positions
        if not self.ca_only:
            self.atom_encoder = AtomAttentionEncoder(
                **kwargs.get("atom_encoder", {}),
                token_dim=kwargs["token_dim"],
                dim_cond=kwargs["dim_cond"],
                pair_repr_dim=kwargs["pair_repr_dim"],
            )
        else:
            self.linear_3d_embed = torch.nn.Linear(3, kwargs["token_dim"], bias=False)

        # To form initial representation
        self.init_repr_factory = FeatureFactory(
            feats=kwargs["feats_init_seq"],
            dim_feats_out=kwargs["token_dim"],
            use_ln_out=False,
            mode="seq",
            **kwargs,
        )

        # To get conditioning variables
        self.cond_factory = FeatureFactory(
            feats=kwargs["feats_cond_seq"],
            dim_feats_out=kwargs["dim_cond"],
            use_ln_out=False,
            mode="seq",
            **kwargs,
        )

        self.transition_c_1 = Transition(kwargs["dim_cond"], expansion_factor=2)
        self.transition_c_2 = Transition(kwargs["dim_cond"], expansion_factor=2)

        # To get pair representation
        if self.use_attn_pair_bias:
            self.pair_repr_builder = PairReprBuilder(
                feats_repr=kwargs["feats_pair_repr"],
                feats_cond=kwargs["feats_pair_cond"],
                dim_feats_out=kwargs["pair_repr_dim"],
                dim_cond_pair=kwargs["dim_cond"],
                **kwargs,
            )
        else:
            # If no pair bias no point in having a pair representation
            self.update_pair_repr = False

        # Trunk layers
        self.transformer_layers = torch.nn.ModuleList(
            [
                MultiheadAttnAndTransition(
                    dim_token=kwargs["token_dim"],
                    dim_pair=kwargs["pair_repr_dim"],
                    nheads=kwargs["nheads"],
                    dim_cond=kwargs["dim_cond"],
                    residual_mha=kwargs["residual_mha"],
                    residual_transition=kwargs["residual_transition"],
                    parallel_mha_transition=kwargs["parallel_mha_transition"],
                    use_attn_pair_bias=kwargs["use_attn_pair_bias"],
                    use_qkln=self.use_qkln,
                    apply_rotary=self.apply_rotary,
                )
                for _ in range(self.nlayers)
            ]
        )

        # To update pair representations if needed
        if self.update_pair_repr:
            self.pair_update_layers = torch.nn.ModuleList(
                [
                    (
                        PairReprUpdate(
                            token_dim=kwargs["token_dim"],
                            pair_dim=kwargs["pair_repr_dim"],
                            use_tri_mult=self.use_tri_mult,
                        )
                        if i % self.update_pair_repr_every_n == 0
                        else None
                    )
                    for i in range(self.nlayers - 1)
                ]
            )
            # For distogram pair prediction
            if self.num_buckets_predict_pair is not None:
                self.pair_head_prediction = torch.nn.Sequential(
                    torch.nn.LayerNorm(kwargs["pair_repr_dim"]),
                    torch.nn.Linear(
                        kwargs["pair_repr_dim"], self.num_buckets_predict_pair
                    ),
                )
        
        if self.ca_only:
            self.coors_3d_decoder = torch.nn.Sequential(
                torch.nn.LayerNorm(kwargs["token_dim"]),
                torch.nn.Linear(kwargs["token_dim"], 3, bias=False),
            )
        else:
            self.coors_3d_decoder = AtomAttentionDecoder(
                **kwargs.get("atom_decoder", {}),
                token_dim=kwargs["token_dim"],
                dim_cond=kwargs["dim_cond"],
                pair_repr_dim=kwargs["pair_repr_dim"],
            )
            
    def _set_registers(self, registers):
        self.num_registers = registers.shape[1]
        # Delete the old registers
        del self.registers
        self.register_buffer("registers", registers, persistent=False)

    def _extend_w_registers(self, seqs, pair, mask, cond_seq):
        """
        Extends the sequence representation, pair representation, mask and indices with registers.

        Args:
            - seqs: sequence representation, shape [b, n, dim_token]
            - pair: pair representation, shape [b, n, n, dim_pair]
            - mask: binary mask, shape [b, n]
            - cond_seq: tensor of shape [b, n, dim_cond]

        Returns:
            All elements above extended with registers / zeros.
        """
        if self.num_registers == 0:
            return seqs, pair, mask, cond_seq  # Do nothing

        b, n, _ = seqs.shape
        dim_pair = pair.shape[-1]
        r = self.num_registers
        dim_cond = cond_seq.shape[-1]

        # Concatenate registers to sequence
        reg_expanded = self.registers[None, :, :]  # [1, r, dim_token]
        reg_expanded = reg_expanded.expand(b, -1, -1)  # [b, r, dim_token]
        seqs = torch.cat([reg_expanded, seqs], dim=1)  # [b, r+n, dim_token]

        # Extend mask
        true_tensor = torch.ones(b, r, dtype=torch.bool, device=seqs.device)  # [b, r]
        mask = torch.cat([true_tensor, mask], dim=1)  # [b, r+n]

        # Extend pair representation with zeros; pair has shape [b, n, n, pair_dim] -> [b, r+n, r+n, pair_dim]
        # [b, n, n, pair_dim] -> [b, r+n, n, pair_dim]
        zero_pad_top = torch.zeros(
            b, r, n, dim_pair, device=seqs.device
        )  # [b, r, n, dim_pair]
        pair = torch.cat([zero_pad_top, pair], dim=1)  # [b, r+n, n, dim_pair]
        # [b, r+n, n, pair_dim] -> [b, r+n, r+n, pair_dim]
        zero_pad_left = torch.zeros(
            b, r + n, r, dim_pair, device=seqs.device
        )  # [b, r+n, r, dim_pair]
        pair = torch.cat([zero_pad_left, pair], dim=2)  # [b, r+n, r+n, dim+pair]

        # Extend cond
        zero_tensor = torch.zeros(
            b, r, dim_cond, device=seqs.device
        )  # [b, r, dim_cond]
        cond_seq = torch.cat([zero_tensor, cond_seq], dim=1)  # [b, r+n, dim_cond]

        return seqs, pair, mask, cond_seq

    def _undo_registers(self, seqs, pair, mask):
        """
        Undoes register padding.

        Args:
            - seqs: sequence representation, shape [b, r+n, dim_token]
            - pair: pair representation, shape [b, r+n, r+n, dim_pair]
            - mask: binary mask, shape [b, r+n]

        Returns:
            All three elements with the register padding removed.
        """
        if self.num_registers == 0:
            return seqs, pair, mask
        r = self.num_registers
        return seqs[:, r:, :], pair[:, r:, r:, :], mask[:, r:]

    def forward(self, batch_nn: Dict[str, torch.Tensor]):
        """
        Runs the network.

        Args:
            batch_nn: dictionary with keys
                - "x_t": tensor of shape [b, n, 3]
                - "t": tensor of shape [b]
                - "mask": binary tensor of shape [b, n]
                - "x_sc" (optional): tensor of shape [b, n, 3]
                - "cath_code" (optional): list of cath codes [b, ?]
                - And potentially others... All in the data batch.

        Returns:
            Predicted clean coordinates, shape [b, n, 3].
        """
        mask, coords_mask = batch_nn["mask"], batch_nn["coords_mask"]

        # Conditioning variables
        c = self.cond_factory(batch_nn)  # [b, n, dim_cond]
        c = self.transition_c_2(self.transition_c_1(c, mask), mask)  # [b, n, dim_cond]

        # Prepare input - coordinates and initial sequence representation from features
        coors_3d = batch_nn["x_t"] * coords_mask[..., None]  # [b, n, 3]
        
        # Pair representation
        pair_rep = None
        if self.use_attn_pair_bias:
            pair_rep = self.pair_repr_builder(batch_nn)  # [b, n, n, pair_dim]
        
        if not self.ca_only:
            coors_embed, q_skip, c_skip, p_skip, pair_mask = self.atom_encoder(coors_3d, pair_rep, c, coords_mask, mask)
        else:
            coors_embed = (
                self.linear_3d_embed(coors_3d) * coords_mask[..., None]
            )  # [b, n, token_dim]
        
        seq_f_repr = self.init_repr_factory(batch_nn)  # [b, n, token_dim]
        seqs = coors_embed + seq_f_repr  # [b, n, token_dim]
        seqs = seqs * mask[..., None]  # [b, n, token_dim]

        # Apply registers
        seqs, pair_rep, mask, c = self._extend_w_registers(seqs, pair_rep, mask, c)

        # Run trunk
        for i in range(self.nlayers):
            seqs = self.transformer_layers[i](
                seqs, pair_rep, c, mask
            )  # [b, n, token_dim]

            if self.update_pair_repr:
                if i < self.nlayers - 1:
                    if self.pair_update_layers[i] is not None:
                        pair_rep = self.pair_update_layers[i](
                            seqs, pair_rep, mask
                        )  # [b, n, n, pair_dim]

        # Undo registers
        seqs, pair_rep, mask = self._undo_registers(seqs, pair_rep, mask)

        # Get final coordinates
        if self.ca_only:
            final_coors = self.coors_3d_decoder(seqs) # [b, n, 3]
        else:
            final_coors = self.coors_3d_decoder(
                seqs,
                q_skip,
                c_skip,
                p_skip,
                coords_mask,
                pair_mask
            ) # [b, n, 3]
        nn_out = {}
        if self.update_pair_repr and self.num_buckets_predict_pair is not None:
            pair_pred = self.pair_head_prediction(pair_rep)
            final_coors = (
                final_coors + torch.mean(pair_pred) * 0.0
            )  # Does not affect loss but pytorch does not complain for unused params
            final_coors = final_coors * coords_mask[..., None]
            nn_out["pair_pred"] = pair_pred
        nn_out["coors_pred"] = final_coors
        return nn_out