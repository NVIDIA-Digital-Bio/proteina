# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.


import gzip
import math
import os
import random
from typing import Dict, List, Literal

import torch
from einops import rearrange
from loguru import logger
from torch.nn import functional as F
from torch_scatter import scatter_mean

from proteinfoundation.utils.ff_utils.pdb_utils import extract_cath_code_by_level
from proteinfoundation.utils.ff_utils.idx_emb_utils import get_index_embedding, get_time_embedding


# ################################
# # # Some auxiliary functions # #
# ################################


def bin_pairwise_distances(x, min_dist, max_dist, dim):
    """
    Takes coordinates and bins the pairwise distances.

    Args:
        x: Coordinates of shape [b, n, 3]
        min_dist: Right limit of first bin
        max_dist: Left limit of last bin
        dim: Dimension of the final one hot vectors

    Returns:
        Tensor of shape [b, n, n, dim] consisting of one-hot vectors
    """
    pair_dists_nm = torch.norm(x[:, :, None, :] - x[:, None, :, :], dim=-1)  # [b, n, n]
    bin_limits = torch.linspace(
        min_dist, max_dist, dim - 1, device=x.device
    )  # Open left and right
    return bin_and_one_hot(pair_dists_nm, bin_limits)  # [b, n, n, pair_dist_dim]


def bin_and_one_hot(tensor, bin_limits):
    """
    Converts a tensor of shape [*] to a tensor of shape [*, d] using the given bin limits.

    Args:
        tensor (Tensor): Input tensor of shape [*]
        bin_limits (Tensor): bin limits [l1, l2, ..., l_{d-1}]. d-1 limits define
            d-2 bins, and the first one is <l1, the last one is >l_{d-1}, giving a total of d bins.

    Returns:
        torch.Tensor: Output tensor of shape [*, d] where d = len(bin_limits) + 1
    """
    bin_indices = torch.bucketize(tensor, bin_limits)
    return torch.nn.functional.one_hot(bin_indices, len(bin_limits) + 1) * 1.0


def indices_force_start_w_one(pdb_idx, mask):
    """
    Takes a tensor with pdb indices for a batch and forces them all to start with the index 1.
    Masked elements are still assigned the index -1.

    Args:
        pdb_idx: tensor of increasing integers (except masked ones fixed to -1), shape [b, n]
        mask: binary tensor, shape [b, n]

    Returns:
        pdb_idx but now all rows start at 1, masked elements are still set to -1.
    """
    first_val = pdb_idx[:, 0][:, None]  # min val is the first one
    pdb_idx = pdb_idx - first_val + 1
    pdb_idx = torch.masked_fill(pdb_idx, ~mask, -1)  # set masked elements to -1
    return pdb_idx


################################
# # Classes for each feature # #
################################


class Feature(torch.nn.Module):
    """Base class for features."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def get_dim(self):
        return self.dim

    def forward(self, batch: Dict):
        pass  # Implemented by each class

    def assert_defaults_allowed(self, batch: Dict, ftype: str):
        """Raises error if default features should not be used to fill-up missing features in the current batch."""
        if "strict_feats" in batch:
            if batch["strict_feats"]:
                raise IOError(
                    f"{ftype} feature requested but no appropriate feature provided. "
                    "Make sure to include the relevant transform in the data config."
                )


class ZeroFeat(Feature):
    """Computes empty feature (zero) of shape [b, n, dim] or [b, n, n, dim],
    depending on sequence or pair features."""

    def __init__(self, dim_feats_out, mode: Literal["seq", "pair"]):
        super().__init__(dim=dim_feats_out)
        self.mode = mode

    def forward(self, batch):
        mask = batch["mask"]  # [b, n]
        b, n = mask.shape[0], mask.shape[1]
        if self.mode == "seq":
            return torch.zeros((b, n, self.dim), device=mask.device)
        elif self.mode == "pair":
            return torch.zeros((b, n, n, self.dim_feats_out), device=mask.device)
        else:
            raise IOError(f"Mode {self.mode} wrong for zero feature")


class FoldEmbeddingSeqFeat(Feature):
    """Computes fold class embedding and returns as sequence feature of shape [b, n, fold_emb_dim * 3]."""

    def __init__(
        self,
        fold_emb_dim,
        cath_code_dir,
        multilabel_mode="sample",
        fold_nhead=4,
        fold_nlayer=2,
        **kwargs,
    ):
        """
        multilabel_mode (["sample", "average", "transformer"]): Schemes to handle multiple fold labels
            "sample": randomly sample one label
            "average": average fold embeddings over all labels
            "transformer": pad labels together and feed into a transformer, take the average over the output
        """
        super().__init__(dim=fold_emb_dim * 3)
        self.create_mapping(cath_code_dir)
        self.embedding_C = torch.nn.Embedding(
            self.num_classes_C + 1, fold_emb_dim
        )  # The last class is left as null embedding
        self.embedding_A = torch.nn.Embedding(self.num_classes_A + 1, fold_emb_dim)
        self.embedding_T = torch.nn.Embedding(self.num_classes_T + 1, fold_emb_dim)
        self.register_buffer("_device_param", torch.tensor(0), persistent=False)
        assert multilabel_mode in ["sample", "average", "transformer"]
        self.multilabel_mode = multilabel_mode
        if multilabel_mode == "transformer":
            encoder_layer = torch.nn.TransformerEncoderLayer(
                fold_emb_dim * 3,
                nhead=fold_nhead,
                dim_feedforward=fold_emb_dim * 3,
                batch_first=True,
            )
            self.transformer = torch.nn.TransformerEncoder(encoder_layer, fold_nlayer)

    @property
    def device(self):
        return next(self.buffers()).device

    def create_mapping(self, cath_code_dir):
        """Create cath label vocabulary for C, A, T levels."""
        mapping_file = os.path.join(cath_code_dir, "cath_label_mapping.pt")
        if os.path.exists(mapping_file):
            class_mapping = torch.load(mapping_file)
        else:
            raise IOError(f"{mapping_file} does not exist...")

        self.class_mapping_C = class_mapping["C"]
        self.class_mapping_A = class_mapping["A"]
        self.class_mapping_T = class_mapping["T"]
        self.num_classes_C = len(self.class_mapping_C)
        self.num_classes_A = len(self.class_mapping_A)
        self.num_classes_T = len(self.class_mapping_T)

    def parse_label(self, cath_code_list):
        """Parse cath_code into corresponding indices at C, A, T levels

        Args:
            cath_code_list (List[List[str]]): List of cath codes for each protein. Each protein can have no, one or multiple labels.

        Return:
            results: for each label of each protein, return its C, A, T label indices
        """
        results = []
        for cath_codes in cath_code_list:
            result = []
            for cath_code in cath_codes:
                result.append(
                    [
                        self.class_mapping_C.get(
                            extract_cath_code_by_level(cath_code, "C"),
                            self.num_classes_C,
                        ),  # If unknown or masked, set as null
                        self.class_mapping_A.get(
                            extract_cath_code_by_level(cath_code, "A"),
                            self.num_classes_A,
                        ),
                        self.class_mapping_T.get(
                            extract_cath_code_by_level(cath_code, "T"),
                            self.num_classes_T,
                        ),
                    ]
                )
            if len(cath_codes) == 0:
                result = [
                    [self.num_classes_C, self.num_classes_A, self.num_classes_T]
                ]  # If no cath code is provided, return null
            results.append(result)
        return results  # [b, num_label, 3]

    def sample(self, cath_code_list):
        """Randomly sample one cath code"""
        results = []
        for cath_codes in cath_code_list:
            idx = random.randint(0, len(cath_codes) - 1)
            results.append(cath_codes[idx])
        return results

    def flatten(self, cath_code_list):
        """Flatten variable lengths of cath codes into a long cath code tensor"""
        results = []
        batch_id = []
        for i, cath_codes in enumerate(cath_code_list):
            results += cath_codes
            batch_id += [i] * len(cath_codes)
        results = torch.as_tensor(results, device=self.device)
        batch_id = torch.as_tensor(batch_id, device=self.device)
        return results, batch_id

    def pad(self, cath_code_list):
        """Pad variable lengths of cath codes into a batched cath code tensor"""
        results = []
        max_num_label = 0
        for cath_codes in cath_code_list:
            results.append(cath_codes)
            max_num_label = max(max_num_label, len(cath_codes))
        mask = []
        for i in range(len(results)):
            mask_i = [False] * len(results[i])
            if len(results[i]) < max_num_label:
                mask_i += [True] * (max_num_label - len(results[i]))
                results[i] += [
                    [self.num_classes_C, self.num_classes_A, self.num_classes_T]
                ] * (max_num_label - len(results[i]))
            mask.append(mask_i)
        results = torch.as_tensor(results, device=self.device)
        mask = torch.as_tensor(mask, device=self.device)
        return results, mask

    def forward(self, batch):
        mask = batch["mask"]  # [b, n]
        bs = mask.shape[0]
        n = mask.shape[1]
        if "cath_code" not in batch:
            cath_code = [
                ["x.x.x.x"]
            ] * bs  # If no cath code provided, return null embeddings
        else:
            cath_code = batch["cath_code"]

        cath_code_list = self.parse_label(cath_code)
        if self.multilabel_mode == "sample":
            cath_code_list = self.sample(
                cath_code_list
            )  # Random sample one label for each protein
            cath_code = torch.as_tensor(cath_code_list, device=self.device)  # [b, 3]
            fold_emb = torch.cat(
                [
                    self.embedding_C(cath_code[:, 0]),
                    self.embedding_A(cath_code[:, 1]),
                    self.embedding_T(cath_code[:, 2]),
                ],
                dim=-1,
            )  # [b, fold_emb_dim * 3]
        elif self.multilabel_mode == "average":
            cath_code, batch_id = self.flatten(cath_code_list)
            fold_emb = torch.cat(
                [
                    self.embedding_C(cath_code[:, 0]),
                    self.embedding_A(cath_code[:, 1]),
                    self.embedding_T(cath_code[:, 2]),
                ],
                dim=-1,
            )  # [num_code, fold_emb_dim * 3]
            fold_emb = scatter_mean(fold_emb, batch_id, dim=0, dim_size=bs)
        elif self.multilabel_mode == "transformer":
            cath_code, mask = self.pad(cath_code_list)
            fold_emb = torch.cat(
                [
                    self.embedding_C(cath_code[:, :, 0]),
                    self.embedding_A(cath_code[:, :, 1]),
                    self.embedding_T(cath_code[:, :, 2]),
                ],
                dim=-1,
            )  # [b, max_num_label, fold_emb_dim * 3]
            fold_emb = self.transformer(
                fold_emb, src_key_padding_mask=mask
            )  # [b, max_num_label, fold_emb_dim * 3]
            fold_emb = (fold_emb * (~mask[:, :, None]).float()).sum(dim=1) / (
                (~mask[:, :, None]).float().sum(dim=1) + 1e-10
            )  # [b, fold_emb_dim * 3]
        fold_emb = fold_emb[:, None, :]  # [b, 1, fold_emb_dim * 3]
        return fold_emb.expand(
            (fold_emb.shape[0], n, fold_emb.shape[2])
        )  # [b, n, fold_emb_dim * 3]


class TimeEmbeddingSeqFeat(Feature):
    """Computes time embedding and returns as sequence feature of shape [b, n, t_emb_dim]."""

    def __init__(self, t_emb_dim, **kwargs):
        super().__init__(dim=t_emb_dim)

    def forward(self, batch):
        t = batch["t"]  # [b]
        mask = batch["mask"]  # [b, n]
        n = mask.shape[1]
        t_emb = get_time_embedding(t, edim=self.dim)  # [b, t_emb_dim]
        t_emb = t_emb[:, None, :]  # [b, 1, t_emb_dim]
        return t_emb.expand((t_emb.shape[0], n, t_emb.shape[2]))  # [b, n, t_emb_dim]


class TimeEmbeddingPairFeat(Feature):
    """Computes time embedding and returns as pair feature of shape [b, n, n, t_emb_dim]."""

    def __init__(self, t_emb_dim, **kwargs):
        super().__init__(dim=t_emb_dim)

    def forward(self, batch):
        t = batch["t"]  # [b]
        mask = batch["mask"]  # [b, n]
        n = mask.shape[1]
        t_emb = get_time_embedding(t, edim=self.dim)  # [b, t_emb_dim]
        t_emb = t_emb[:, None, None, :]  # [b, 1, 1, t_emb_dim]
        return t_emb.expand((t_emb.shape[0], n, n, t_emb.shape[3]))  # [b, n, t_emb_dim]


class IdxEmbeddingSeqFeat(Feature):
    """Computes index embedding and returns sequence feature of shape [b, n, idx_emb]."""

    def __init__(self, idx_emb_dim, **kwargs):
        super().__init__(dim=idx_emb_dim)

    def forward(self, batch):
        # If it has the actual residue indices
        if "residue_pdb_idx" in batch:
            inds = batch["residue_pdb_idx"]  # [b, n]
            inds = indices_force_start_w_one(inds, batch["mask"])
        else:
            self.assert_defaults_allowed(batch, "Residue index sequence")
            mask = batch["mask"]  # [b, n]
            b, n = mask.shape[0], mask.shape[1]
            inds = torch.Tensor([[i + 1 for i in range(n)] for _ in range(b)]).long().to(
                mask.device
            )  # [b, n]
        
        return get_index_embedding(inds, edim=self.dim)  # [b, n, idx_embed_dim]


class ChainBreakPerResidueSeqFeat(Feature):
    """Computes a 1D sequence feature indicating if a residue is followed by a chain break, shape [b, n, 1]."""

    def __init__(self, **kwargs):
        super().__init__(dim=1)

    def forward(self, batch):
        # If it has the actual chain breaks
        if "chain_breaks_per_residue" in batch:
            chain_breaks = batch["chain_breaks_per_residue"] * 1.0  # [b, n]
        else:
            self.assert_defaults_allowed(batch, "Chain break sequence")
            mask = batch['mask']  # [b, n]
            b, n = mask.shape[0], mask.shape[1]
            chain_breaks = torch.zeros((b, n), device=mask.device) * 1.0  # [b, n]
        return chain_breaks[..., None]  # [b, n, 1]


class XscSeqFeat(Feature):
    """Computes feature from self conditining coordinates, seq feature of shape [b, n, 3]."""

    def __init__(self, **kwargs):
        super().__init__(dim=3)

    def forward(self, batch):
        if "x_sc" in batch:
            return batch["x_sc"]  # [b, n, 3]
        else:
            # If we do not provide self-conditioning as input to the nn
            mask = batch["mask"]  # [b, n]
            b, n = mask.shape[0], mask.shape[1]
            return torch.zeros(b, n, 3, device=mask.device)


class MotifX1SeqFeat(Feature):
    """Computes feature from motif coordinates if present, seq feature of shape [b, n, 3]."""

    def __init__(self, **kwargs):
        super().__init__(dim=3)

    def forward(self, batch):
        if "x_motif" in batch:
            return batch["x_motif"]  # [b, n, 3]
        else:
            # If no motif
            x = batch["x_t"]
            b, n = x.shape[0], x.shape[1]
            device = x.device
            return torch.zeros(b, n, 3, device=device)


class MotifMaskSeqFeat(Feature):
    """Computes feature from mask of the motif positions if present, seq feature of shape [b, n, 3]."""

    def __init__(self, **kwargs):
        super().__init__(dim=1)

    def forward(self, batch):
        if "motif_mask" in batch:
            return batch["motif_mask"].unsqueeze(-1)  # [b, n, 1]
        else:
            # If no motif
            x = batch["x_t"]
            b, n = x.shape[0], x.shape[1]
            device = x.device
            return torch.zeros(b, n, device=device).unsqueeze(-1)


class MotifStructureMaskFeat(Feature):
    """Computes feature of the pair wise motif mask of shape [b, n, n, seq_sep_dim]."""

    def __init__(self, **kwargs):
        super().__init__(dim=1)

    def forward(self, batch):
        if "fixed_structure_mask" in batch:
            # no need to force 1 since taking difference
            mask = batch["fixed_structure_mask"].unsqueeze(-1)  # [b, n]
        else:
            raise ValueError("No fixed_structure_mask")
        return mask


class MotifX1PairwiseDistancesPairFeat(Feature):
    """Computes pairwise distances for CA backbone atoms of motif atoms and returns feature of shape [b, n, n, dim_pair_dist]."""
    def __init__(
        self, x_motif_pair_dist_dim, x_motif_pair_dist_min, x_motif_pair_dist_max, **kwargs
    ):
        super().__init__(dim=x_motif_pair_dist_dim)
        self.min_dist = x_motif_pair_dist_min
        self.max_dist = x_motif_pair_dist_max
        
    def forward(self, batch):
        assert("x_motif" in batch)
        assert("fixed_structure_mask" in batch)
        return bin_pairwise_distances(
            x=batch["x_motif"],
            min_dist=self.min_dist,
            max_dist=self.max_dist,
            dim=self.dim,
        ) * batch["fixed_structure_mask"].unsqueeze(-1)# [b, n, n, pair_dist_dim]


class SequenceSeparationPairFeat(Feature):
    """Computes sequence separation and returns feature of shape [b, n, n, seq_sep_dim]."""

    def __init__(self, seq_sep_dim, **kwargs):
        super().__init__(dim=seq_sep_dim)

    def forward(self, batch):
        if "residue_pdb_idx" in batch:
            # no need to force 1 since taking difference
            inds = batch["residue_pdb_idx"]  # [b, n]
        else:
            self.assert_defaults_allowed(batch, "Relative sequence separation pair")
            mask = batch["mask"]  # [b, n]
            b, n = mask.shape[0], mask.shape[1]
            inds = torch.Tensor([[i + 1 for i in range(n)] for _ in range(b)]).to(
                mask.device
            )  # [b, n]

        seq_sep = inds[:, :, None] - inds[:, None, :]  # [b, n, n]

        # Dimension should be odd, bins limits [-(dim/2-1), ..., -1.5, -0.5, 0.5, 1.5, ..., dim/2-1]
        # gives dim-2 bins, and the first and last for values beyond the bin limits
        assert (
            self.dim % 2 == 1
        ), "Relative seq separation feature dimension must be odd and > 3"

        # Create bins limits [..., -3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.3, 3.5, ...]
        # Equivalent to binning relative sequence separation
        low = -(self.dim / 2.0 - 1)
        high = self.dim / 2.0 - 1
        bin_limits = torch.linspace(low, high, self.dim - 1, device=inds.device)

        return bin_and_one_hot(seq_sep, bin_limits)  # [b, n, n, seq_sep_dim]


# class SequenceSeparationPairFeat(Feature):
#     """Computes sequence separation and returns feature of shape [b, n, n, seq_sep_dim]."""

#     def __init__(self, seq_sep_dim, **kwargs):
#         super().__init__(dim=seq_sep_dim)
        
#         self.r_max = 32
#         self.s_max = 2
#         assert seq_sep_dim == 2 * (self.r_max + 1)

#     def forward(self, batch):
#         if "residue_pdb_idx" in batch:
#             # no need to force 1 since taking difference
#             inds = batch["residue_pdb_idx"]  # [b, n]
#         else:
#             self.assert_defaults_allowed(batch, "Relative sequence separation pair")
#             mask = batch["mask"]  # [b, n]
#             b, n = mask.shape[0], mask.shape[1]
#             inds = torch.Tensor([[i + 1 for i in range(n)] for _ in range(b)]).long().to(
#                 mask.device
#             )  # [b, n]

#         d_residue = torch.clip(
#             input=inds[:, :, None] - inds[:, None, :] + self.r_max,
#             min=0,
#             max=2 * self.r_max
#         )

#         a_rel_pos = F.one_hot(d_residue, 2 * (self.r_max + 1))
        
#         return a_rel_pos


class XtPairwiseDistancesPairFeat(Feature):
    """Computes pairwise distances and returns feature of shape [b, n, n, dim_pair_dist]."""

    def __init__(self, xt_pair_dist_dim, xt_pair_dist_min, xt_pair_dist_max, **kwargs):
        super().__init__(dim=xt_pair_dist_dim)
        self.min_dist = xt_pair_dist_min
        self.max_dist = xt_pair_dist_max
        self.ca_only = kwargs.get("ca_only", False)

    def forward(self, batch):
        if self.ca_only:
            x = batch["x_t"]
        else:
            x = rearrange(
                batch["x_t"],
                "b (n c) d -> b n c d",
                c=4
            )[..., 1, :]
        return bin_pairwise_distances(
            x=x,
            min_dist=self.min_dist,
            max_dist=self.max_dist,
            dim=self.dim,
        )  # [b, n, n, pair_dist_dim]


class XscPairwiseDistancesPairFeat(Feature):
    """Computes pairwise distances and returns feature of shape [b, n, n, dim_pair_dist]."""

    def __init__(
        self, x_sc_pair_dist_dim, x_sc_pair_dist_min, x_sc_pair_dist_max, **kwargs
    ):
        super().__init__(dim=x_sc_pair_dist_dim)
        self.min_dist = x_sc_pair_dist_min
        self.max_dist = x_sc_pair_dist_max

    def forward(self, batch):
        if "x_sc" in batch:
            return bin_pairwise_distances(
                x=batch["x_sc"],
                min_dist=self.min_dist,
                max_dist=self.max_dist,
                dim=self.dim,
            )  # [b, n, n, pair_dist_dim]
        else:
            # If we do not provide self-conditioning as input to the nn
            mask = batch["mask"]
            b, n = mask.shape[0], mask.shape[1]
            return torch.zeros(b, n, n, self.dim, device=mask.device)


####################################
# # Class that produces features # #
####################################


class FeatureFactory(torch.nn.Module):
    def __init__(
        self,
        feats: List[str],
        dim_feats_out: int,
        use_ln_out: bool,
        mode: Literal["seq", "pair"],
        **kwargs,
    ):
        """
        Sequence features include:
            - "res_seq_pdb_idx", requires transform ResidueSequencePositionPdbTransform
            - "time_emb"
            - "chain_break_per_res", requires transform ChainBreakPerResidueTransform
            - "fold_emb"
            - "x_sc"

        Pair features include:
            - "xt_pair_dists"
            - "x_sc_pair_dists"
            - "rel_seq_sep"
            - "time_emb"
        """
        super().__init__()
        self.mode = mode

        self.ret_zero = True if (feats is None or len(feats) == 0) else False
        if self.ret_zero:
            logger.info("No features requested")
            self.zero_creator = ZeroFeat(dim_feats_out=dim_feats_out, mode=mode)
            return

        self.feat_creators = torch.nn.ModuleList(
            [self.get_creator(f, **kwargs) for f in feats]
        )
        self.ln_out = (
            torch.nn.LayerNorm(dim_feats_out) if use_ln_out else torch.nn.Identity()
        )
        self.linear_out = torch.nn.Linear(
            sum([c.get_dim() for c in self.feat_creators]), dim_feats_out, bias=False
        )

    def get_creator(self, f, **kwargs):
        """Returns the right class for the requested feature f (a string)."""

        if self.mode == "seq":
            if f == "time_emb":
                return TimeEmbeddingSeqFeat(**kwargs)
            elif f == "res_seq_pdb_idx":
                return IdxEmbeddingSeqFeat(**kwargs)
            elif f == "chain_break_per_res":
                return ChainBreakPerResidueSeqFeat(**kwargs)
            elif f == "fold_emb":
                return FoldEmbeddingSeqFeat(**kwargs)
            elif f == "x_sc":
                return XscSeqFeat(**kwargs)
            elif f == "motif_x1":
                return MotifX1SeqFeat(**kwargs)
            elif f == "motif_sequence_mask":
                return MotifMaskSeqFeat(**kwargs)
            else:
                raise IOError(f"Sequence feature {f} not implemented.")

        elif self.mode == "pair":
            if f == "xt_pair_dists":
                return XtPairwiseDistancesPairFeat(**kwargs)
            elif f == "x_sc_pair_dists":
                return XscPairwiseDistancesPairFeat(**kwargs)
            elif f == "rel_seq_sep":
                return SequenceSeparationPairFeat(**kwargs)
            elif f == "time_emb":
                return TimeEmbeddingPairFeat(**kwargs)
            elif f == "motif_x1_pair_dists":
                return MotifX1PairwiseDistancesPairFeat(**kwargs)
            elif f == "motif_structure_mask":
                return MotifStructureMaskFeat(**kwargs)
            else:
                raise IOError(f"Pair feature {f} not implemented.")

        else:
            raise IOError(
                f"Wrong feature mode (creator): {self.mode}. Should be 'seq' or 'pair'."
            )

    def apply_padding_mask(self, feature_tensor, mask):
        """
        Applies mask to features.

        Args:
            feature_tensor: tensor with requested features, shape [b, n, d] of [b, n, n, d] depending on self.mode ('seq' or 'pair')
            mask: Binary mask, shape [b, n]

        Returns:
            Masked features, same shape as input tensor.
        """
        if self.mode == "seq":
            return feature_tensor * mask[..., None]  # [b, n, d]
        elif self.mode == "pair":
            mask_pair = mask[:, None, :] * mask[:, :, None]  # [b, n, n]
            return feature_tensor * mask_pair[..., None]  # [b, n, n, d]
        else:
            raise IOError(
                f"Wrong feature mode (pad mask): {self.mode}. Should be 'seq' or 'pair'."
            )

    def forward(self, batch):
        """Returns masked features, shape depends on mode, either 'seq' or 'pair'."""
        # If no features requested just return the zero tensor of appropriate dimensions
        if self.ret_zero:
            return self.zero_creator(batch)

        # Compute requested features
        feature_tensors = []
        for fcreator in self.feat_creators:
            feature_tensors.append(
                fcreator(batch)
            )  # [b, n, dim_f] or [b, n, n, dim_f] if seq or pair mode

        # Concatenate features and mask
        features = torch.cat(
            feature_tensors, dim=-1
        )  # [b, n, dim_f] or [b, n, n, dim_f]
        features = self.apply_padding_mask(
            features, batch["mask"]
        )  # [b, n, dim_f] or [b, n, n, dim_f]

        # Linear layer and mask
        features_proc = self.ln_out(
            self.linear_out(features)
        )  # [b, n, dim_f] or [b, n, n, dim_f]
        return self.apply_padding_mask(
            features_proc, batch["mask"]
        )  # [b, n, dim_f] or [b, n, n, dim_f]
