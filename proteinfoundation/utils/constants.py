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
from typing import Dict
from openfold.np.residue_constants import atom_types


ATOM_NUMBERING: Dict[str, int] = {
    atom: i
    for i, atom in enumerate(
        [
            "N",
            "CA",
            "C",
            "O",
            "CB",
            "OG",
            "CG",
            "CD1",
            "CD2",
            "CE1",
            "CE2",
            "CZ",
            "OD1",
            "ND2",
            "CG1",
            "CG2",
            "CD",
            "CE",
            "NZ",
            "OD2",
            "OE1",
            "NE2",
            "OE2",
            "OH",
            "NE",
            "NH1",
            "NH2",
            "OG1",
            "SD",
            "ND1",
            "SG",
            "NE1",
            "CE3",
            "CZ2",
            "CZ3",
            "CH2",
            "OXT",
        ]
    )
}
"""Default ordering of atoms in (dimension 1 of) a protein structure tensor."""


# PDB and OpenFold have different atom ordering, these utils convert between the two
# PDB ordering: https://cdn.rcsb.org/wwpdb/docs/documentation/file-format/PDB_format_1992.pdf
# OpenFold ordering: https://github.com/aqlaboratory/openfold/blob/f6c875b3c8e3e873a932cbe3b31f94ae011f6fd4/openfold/np/residue_constants.py#L556
# more background: https://kdidi.netlify.app/blog/proteins/2024-02-03-protein-representations/
PDB_TO_OPENFOLD_INDEX_TENSOR = torch.tensor(
    [ATOM_NUMBERING[atom] for atom in atom_types]
)
OPENFOLD_TO_PDB_INDEX_TENSOR = torch.tensor(
    [atom_types.index(atom) for atom in ATOM_NUMBERING]
)
