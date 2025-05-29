# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.


import torch
from einops import rearrange

nm_to_ang_scale = 10.0
ang_to_nm = lambda trans: trans / nm_to_ang_scale
nm_to_ang = lambda trans: trans * nm_to_ang_scale


def trans_nm_to_atom37(x_coors_nm, ca_only=True):
    """
    Converts CA positions (in nm) into atom37 representation (in Å).

    Args:
        ca_coors: CA coordinates in nm, shape [*, N, 3]

    Returns:
        Coordinates in atom37 representation (in Å)
    """
    return trans_ang_to_atom37(nm_to_ang(x_coors_nm), ca_only=ca_only)


def trans_ang_to_atom37(x_coors, ca_only=True):
    """
    Converts CA positions (in Å) into atom37 representation.

    Args:
        ca_coors: CA coordinates in Å, shape [*, N, 3]

    Returns:
        Coordinates in atom37 representation
    """
    if ca_only:
        original_shape = x_coors.shape  # [*, N, 3]
        atom37_shape = list(original_shape[:-1]) + [37, original_shape[-1]]  # [*, N, 37, 3]
        x_coors_atom37 = torch.zeros(
            atom37_shape, dtype=x_coors.dtype, device=x_coors.device
        )  # [*, N, 37, 3]
        x_coors_atom37[..., 1, :] = x_coors  # Sets correct positions for CA [*, N, 37, 3]
    else:
        x_coors = rearrange(x_coors, "b (n c) d -> b n c d", n=int(x_coors.shape[-2] / 4))  # [*, N, 4, 3]
        original_shape = x_coors.shape  # [*, N, 3]
        atom37_shape = list(original_shape[:-2]) + [37, original_shape[-1]]  # [*, N, 37, 3]
        x_coors_atom37 = torch.zeros(
            atom37_shape, dtype=x_coors.dtype, device=x_coors.device
        )  # [*, N, 37, 3]
        BB_INDEX = [0, 1, 2, 4]
        x_coors_atom37[..., BB_INDEX, :] = x_coors  # Sets correct positions for CA [*, N, 37, 3]
        # x_coors_atom37[..., 1, :] = x_coors[..., 1, :]  # Sets correct positions for CA [*, N, 37, 3]
    return x_coors_atom37
