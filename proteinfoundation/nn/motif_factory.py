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
from typing import List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import random

import biotite.structure.io as strucio
from proteinfoundation.utils.align_utils.align_utils import mean_w_mask
from proteinfoundation.utils.coors_utils import ang_to_nm

import itertools

def generate_combinations(min_cost, max_cost, ranges):
    result = []
    ranges = [[x] if isinstance(x, int) else range(x[0], x[1] + 1) for x in ranges]
    for combination in itertools.product(*ranges):
        total_cost = sum(combination)
        if min_cost <= total_cost <= max_cost:
            padded_combination = list(combination) + [0] * (len(ranges) - len(combination))
            result.append(padded_combination)
    return result


def generate_indices_and_mask_clean(contig: str, min_length: int, max_length: int) -> Tuple[int, List[int], np.ndarray]:
    """Index motif and scaffold positions by contig for sequence redesign.
    Args:
        contig (str): A string containing positions for scaffolds and motifs.

        Details:
        Scaffold parts: Contain a single integer.
        Motif parts: Start with a letter (chain ID) and contain either a single positions (e.g. A33) or a range of positions (e.g. A33-39).
        The numbers following chain IDs corresponds to the motif positions in native backbones, which are used to calculate motif reconstruction later on.
        e.g. "15/A45-65/20/A20-30"
        NOTE: The scaffold part should be DETERMINISTIC in this case as it contains information for the corresponding protein backbones.

    Raises:
        ValueError: Once a "-" is detected in scaffold parts, throws an error for the aforementioned reason.

    Returns:
        A Tuple containing:
            - overall_length (int): Total length of the sequence defined by the contig.
            - motif_indices (List[int]): List of indices where motifs are located.
            - motif_mask (np.ndarray): Boolean array where True indicates motif positions and False for scaffold positions.
    """
    ALPHABET = "ABCDEFGHJKLMNOPQRSTUVWXYZ"
    components = contig.split('/')
    success = False
    ranges = []
    current_position = 1  # Start positions at 1 for 1-based indexing
    motif_indices = []
    motif_mask = []
    output_string = ""
    motif_length = 0
    for part in components:
        if part[0] in ALPHABET:
            # Motif part
            if '-' in part:
                start, end = map(int, part[1:].split("-"))
            else: # Single motif
                start = end = int(part[1:])
            length = (end - start + 1)
            motif_length += length
        else:
            # Scaffold part
            if '-' in part:
                bounds = part.split('-')
                assert int(bounds[0]) <= int(bounds[-1])
                ranges.append((int(bounds[0]), int(bounds[-1])))
            else:
                length = int(part)
                ranges.append(length)
    combinations = generate_combinations(min_length - motif_length, max_length - motif_length, ranges)
    if len(combinations) == 0:
        raise ValueError("No Motif combinations to sample from please update the max and min lengths")
    combo = random.choice(combinations)
    combo_idx = 0
    current_position = 1  # Start positions at 1 for 1-based indexing
    motif_indices = []
    motif_mask = []
    output_string = ""
    for part in components:
        if part[0] in ALPHABET:
            # Motif part
            if '-' in part:
                start, end = map(int, part[1:].split("-"))
            else: # Single motif
                start = end = int(part[1:])
            length = (end - start + 1)
            motif_indices.extend(range(current_position, current_position + length))
            motif_mask.extend([True] * length)
            new_part = part[0] + str(current_position)
            if length > 1:
                new_part += "-" + str(current_position + length - 1)
            output_string +=  new_part + "/"
        else:
            # Scaffold part
            length = int(combo[combo_idx])
            combo_idx += 1
            motif_mask.extend([False] * length)
            output_string += str(length) + "/"
        current_position += length  # Update the current position after processing each part
    output_string = output_string[:-1]
    
    motif_mask = np.array(motif_mask, dtype=bool)
    overall_length = motif_mask.shape[0]
    return (overall_length, motif_indices, motif_mask, output_string)

def parse_motif(pdb_path, contig_str, nsamples = 1, make_tensor = False, motif_only = False, min_length = None, max_length = None):
    motif = motif_extract(contig_str, pdb_path, motif_only = motif_only)
    x_motif = torch.Tensor(np.array([x.coord for x in motif]))
    if nsamples == 1:
        length, motif_idx, mask, out_str = generate_indices_and_mask_clean(contig_str, min_length, max_length)
        mask = torch.Tensor(mask).bool()
        x_motif_full = torch.zeros((length, 3), dtype=x_motif.dtype)
        x_motif_full[mask] = x_motif
        return mask, x_motif_full, out_str
    else:
        lengths,motif_indices, masks, outstrs = [], [], [], []
        x_motif_all = []
        for nsample_idx in range(nsamples):
            print(nsample_idx)
            length, motif_idx, mask, out_str = generate_indices_and_mask_clean(contig_str, min_length, max_length)
            lengths.append(length)
            motif_indices.append(motif_idx)
            mask = torch.Tensor(mask).bool()
            masks.append(mask)
            outstrs.append(out_str)
            x_motif_full = torch.zeros((length, 3), dtype=x_motif.dtype)
            x_motif_full[mask] = x_motif
            x_motif_all.append(x_motif_full)
        if make_tensor:
            padded_masks = torch.nn.utils.rnn.pad_sequence(masks, batch_first=True, padding_value=False)
            padded_structures = torch.nn.utils.rnn.pad_sequence(x_motif_all, batch_first=True, padding_value=0)
            return padded_masks, padded_structures, outstrs
        else:
            return masks, x_motif_all, outstrs

def save_motif_csv(pdb_path, motif_task_name, contigs, outpath = None, segment_order = 'A'):
    pdb_name = pdb_path.split('/')[-1].split('.')[0]
    
    # Create a list of dictionaries to be converted into a DataFrame
    # Each dictionary represents a row in the CSV file
    data = [
        {
            'pdb_name': pdb_name, 
            'sample_num': index, 
            'contig': value,
            'redesign_positions': ' ',
            'segment_order': segment_order
        } 
        for index, value in enumerate(contigs)
    ]
    
    # Convert the list of dictionaries into a DataFrame
    df = pd.DataFrame(data)
    if outpath is None:
        outpath = f"./{motif_task_name}_motif_info.csv"
    
    # Save the DataFrame to a CSV file
    df.to_csv(outpath, index=False)
    


def generate_indices_and_mask(contig: str, min_length: int, max_length: int) -> Tuple[int, List[int], np.ndarray]:
    """Index motif and scaffold positions by contig for sequence redesign.
    Args:
        contig (str): A string containing positions for scaffolds and motifs.

        Details:
        Scaffold parts: Contain a single integer.
        Motif parts: Start with a letter (chain ID) and contain either a single positions (e.g. A33) or a range of positions (e.g. A33-39).
        The numbers following chain IDs corresponds to the motif positions in native backbones, which are used to calculate motif reconstruction later on.
        e.g. "15/A45-65/20/A20-30"
        NOTE: The scaffold part should be DETERMINISTIC in this case as it contains information for the corresponding protein backbones.

    Raises:
        ValueError: Once a "-" is detected in scaffold parts, throws an error for the aforementioned reason.

    Returns:
        A Tuple containing:
            - overall_length (int): Total length of the sequence defined by the contig.
            - motif_indices (List[int]): List of indices where motifs are located.
            - motif_mask (np.ndarray): Boolean array where True indicates motif positions and False for scaffold positions.
    """
    ALPHABET = "ABCDEFGHJKLMNOPQRSTUVWXYZ"
    components = contig.split('/')
    success = False
    while not success:
        current_position = 1  # Start positions at 1 for 1-based indexing
        motif_indices = []
        motif_mask = []
        output_string = ""
        for part in components:
            if part[0] in ALPHABET:
                # Motif part
                if '-' in part:
                    start, end = map(int, part[1:].split("-"))
                else: # Single motif
                    start = end = int(part[1:])
                length = (end - start + 1)
                motif_indices.extend(range(current_position, current_position + length))
                motif_mask.extend([True] * length)
                new_part = part[0] + str(current_position)
                if length > 1:
                    new_part += "-" + str(current_position + length - 1)
                output_string +=  new_part + "/"
            else:
                # Scaffold part
                if '-' in part:
                    bounds = part.split('-')
                    assert int(bounds[0]) <= int(bounds[-1])
                    length = random.randint(int(bounds[0]), int(bounds[-1]))
                else:
                    length = int(part)
                motif_mask.extend([False] * length)
                output_string += str(length) + "/"

            current_position += length  # Update the current position after processing each part
        if min_length is None and max_length is None:
            success = True
        elif (current_position-1) >= min_length and (current_position-1) <= max_length: #cur_pos starts at 1 so after adding 3 its 4
            success = True
    output_string = output_string[:-1]
    # Convert motif_mask to a numpy array for more efficient boolean operations
    motif_mask = np.array(motif_mask, dtype=bool)
    overall_length = motif_mask.shape[0]
    
    return (overall_length, motif_indices, motif_mask, output_string)

def motif_extract(
    position: str,
    structure_path,
    atom_part: Optional[str] = "CA",
    split_char: str = "/",
    motif_only = False,
):
    """Extracting motif positions from input protein structure.

    Args:
        position (str): Motif region of input protein. DEMO: "A1-7/A28-79" corresponds defines res1-7 and res28-79 in chain A to be motif.
        structure_path (Union[str, None]): Input protein structure, can either be a path or an AtomArray.
        atom_part (str, optional): _description_. Defaults to "all".
        split_char (str): Spliting character between discontinuous motifs. Defaults to "/".

    Returns:
        motif (biotite.structure.AtomArray): The motif positions extracted by user-specified way (all-atom / CA / backbone)
    """

    position = position.split(split_char)
    ALPHABET = "ABCDEFGHJKLMNOPQRSTUVWXYZ"
    if isinstance(structure_path, str):
        array = strucio.load_structure(structure_path, model=1)
    else:
        array = structure_path
    motif_array = []
    
    if motif_only:
        seen = set()
        for i in position:
            chain_id = i[0]
            if chain_id not in ALPHABET or chain_id in seen:
                continue
            seen.add(chain_id) # used for 1QJG multiple A chains
            if atom_part == "all-atom":
                motif_array.append(array[(array.chain_id==chain_id) & (array.hetero==False)])
            elif atom_part == "CA":
                motif_array.append(array[(array.chain_id==chain_id) & (array.hetero==False) & (array.atom_name=="CA")])
            elif atom_part == "backbone":
                motif_array.append(array[(array.chain_id==chain_id) & (array.hetero==False) & ((array.atom_name=="N") | (array.atom_name=="CA")| (array.atom_name=="C") | (array.atom_name=="O"))])
    else:
        for i in position:
            chain_id = i[0]
            if chain_id not in ALPHABET:
                continue
            i = i.replace(chain_id, "")
            if "-" not in i: # Single-residue motif
                start = end = int(i)
            else:
                start, end = i.split("-")
                start, end = int(start), int(end)

            if atom_part == "all-atom":
                motif_array.append(array[(array.chain_id==chain_id) & (array.res_id <= end) & (array.res_id >= start) & (array.hetero==False)])
            elif atom_part == "CA":
                motif_array.append(array[(array.chain_id==chain_id) & (array.res_id <= end) & (array.res_id >= start) & (array.hetero==False) & (array.atom_name=="CA")])
            elif atom_part == "backbone":
                motif_array.append(array[(array.chain_id==chain_id) & (array.res_id <= end) & (array.res_id >= start) & (array.hetero==False) & ((array.atom_name=="N") | (array.atom_name=="CA")| (array.atom_name=="C") | (array.atom_name=="O"))])

    motif = motif_array[0]
    for i in range(len(motif_array) - 1):
        motif += motif_array[i + 1]
    
    return motif


class SingleMotifFactory:
    def __init__(self, 
                 motif_prob: float =1.0, #0.8,
                 motif_min_pct_res: float = 0.05,
                 motif_max_pct_res: float = 0.5,
                 motif_min_n_seg: int = 1,
                 motif_max_n_seg: int = 4):
        """
        Initialize SingleMotifFactory with default parameters.

        Args:
            motif_prob (float, optional): Motif probability. Defaults to 0.8.
            motif_min_pct_res (float, optional): Minimum percentage of residues in motif. Defaults to 0.05.
            motif_max_pct_res (float, optional): Maximum percentage of residues in motif. Defaults to 0.5.
            motif_min_n_seg (int, optional): Minimum number of segments in motif. Defaults to 1.
            motif_max_n_seg (int, optional): Maximum number of segments in motif. Defaults to 4.
        """
        self.motif_prob = motif_prob
        self.motif_min_pct_res = motif_min_pct_res
        self.motif_max_pct_res = motif_max_pct_res
        self.motif_min_n_seg = motif_min_n_seg
        self.motif_max_n_seg = motif_max_n_seg
        
    def __call__(self, batch, zeroes=False):
        return self.create_batch_motif(batch, zeroes)
    
    def create_batch_motif(self, batch, zeroes = False):
        result = {}
        if "x_1" in batch:
            x_1 = batch["x_1"]  # [b, n, 3]
        else:
            x_1 = batch["coords"][:,:,1,:]  # [b, n, 3]
            x_1 = ang_to_nm(x_1)
        if "mask" in batch:
            mask = batch["mask"]
        else:
            mask = batch["mask_dict"]["coords"][..., 0, 0]  # [b, n] boolean
        batch_size, num_residues =mask.shape
        if zeroes or random.random() > self.motif_prob:
            motif_sequence_mask = torch.zeros((batch_size, num_residues), dtype = torch.bool)
            motif_structure_mask = torch.zeros((batch_size, num_residues, num_residues), dtype = torch.bool)
            result['fixed_sequence_mask'] = motif_sequence_mask.to(mask.device)
            result['fixed_structure_mask'] = motif_structure_mask.to(mask.device)
            result['x_motif'] = torch.zeros((batch_size, num_residues, 3)).to(mask.device)
            return result

        batch_num_residues = mask.sum(-1).cpu().numpy()
        
        motif_n_res_batch = (
            np.random.rand(len(batch_num_residues)) * (
                batch_num_residues * self.motif_max_pct_res - batch_num_residues * self.motif_min_pct_res
            ) + batch_num_residues * self.motif_min_pct_res
        ).astype(np.int64)

        motif_n_seg_batch = (
            np.random.rand(len(motif_n_res_batch)) * (
                np.minimum(motif_n_res_batch, np.full_like(motif_n_res_batch, self.motif_max_n_seg)) - self.motif_min_n_seg + 1
            ) + self.motif_min_n_seg
        ).astype(np.int64)
        
        indices_batch = []
        motif_seg_lens_batch = []
        for i in range(len(motif_n_res_batch)):
        
            indices = np.sort(np.random.choice(motif_n_res_batch[i] - 1, motif_n_seg_batch[i] - 1, replace=False) + 1)
            indices = np.concatenate([[0], indices, [motif_n_res_batch[i]]])
            indices_batch.append(indices)
            
            # Calculate motif segment lengths
            motif_seg_lens = indices[1:] - indices[:-1]
            motif_seg_lens_batch.append(motif_seg_lens)
        
        motif_sequence_masks = []
        # motif_structure_masks = []

        for i in range(len(motif_seg_lens_batch)):
            segs = [''.join(['1'] * l) for l in motif_seg_lens_batch[i]]
            segs.extend(['0'] * (batch_num_residues[i] - motif_n_res_batch[i]).astype(np.int64))
            random.shuffle(segs)
            motif_sequence_mask = torch.tensor([int(elt) for elt in ''.join(segs)], dtype=torch.bool)
            motif_sequence_masks.append(motif_sequence_mask)

        
        motif_sequence_masks = torch.nn.utils.rnn.pad_sequence(motif_sequence_masks, batch_first=True, padding_value=False)
        motif_structure_masks = motif_sequence_masks[:, :, None] * motif_sequence_masks[:, None, :]
        result['fixed_sequence_mask'] = motif_sequence_masks.to(mask.device)
        result['fixed_structure_mask'] = motif_structure_masks.to(mask.device)
        result['x_motif'] = x_1.clone()
        #! Center the conditional Motif
        result['x_motif'] = (result['x_motif'] - mean_w_mask(result['x_motif'], result['fixed_sequence_mask'], keepdim=True)) * result['fixed_sequence_mask'][..., None]
        #! Translate x_1 so that the motif is in the center
        batch["x_1"] = (x_1 - mean_w_mask(x_1, result['fixed_sequence_mask'], keepdim=True)) * mask[..., None]
        return result
        
        
    def create_motif(self, num_residues: int) -> dict:
        """
        Create a single motif and update the input features.

        Args:
            np_features (dict): Input features.

        Returns:
            dict: Updated features with motif information.
        """
        result = {}
        if random.random() > self.motif_prob:
            motif_sequence_mask = torch.zeros((num_residues))
            motif_structure_mask = torch.zeros((num_residues, num_residues))
            result['fixed_sequence_mask'] = motif_sequence_mask
            result['fixed_structure_mask'] = motif_structure_mask
            return result

        # Sample number of motif residues
        motif_n_res = np.random.randint(
            np.floor(num_residues * self.motif_min_pct_res),
            np.ceil(num_residues * self.motif_max_pct_res)
        )

        # Sample number of motif segments
        motif_n_seg = np.random.randint(
            self.motif_min_n_seg,
            min(self.motif_max_n_seg, motif_n_res) + 1
        )

        # Sample motif segments
        indices = sorted(np.random.choice(motif_n_res - 1, motif_n_seg - 1, replace=False) + 1)
        indices = [0] + indices + [motif_n_res]
        motif_seg_lens = [indices[i+1] - indices[i] for i in range(motif_n_seg)]

        # Generate motif mask
        segs = [''.join(['1'] * l) for l in motif_seg_lens]
        segs.extend(['0'] * (num_residues - motif_n_res))
        random.shuffle(segs)
        motif_sequence_mask = torch.tensor([int(elt) for elt in ''.join(segs)], dtype=torch.bool)
        motif_structure_mask = motif_sequence_mask[:, None] * motif_sequence_mask[None, :]
        motif_structure_mask = motif_structure_mask.bool()

        # Update
        result['fixed_sequence_mask'] = motif_sequence_mask
        result['fixed_structure_mask'] = motif_structure_mask

        return result

if __name__ == "__main__":
    # Test SingleMotifFactory
    factory = SingleMotifFactory()

    # Create a batch with a single sequence of 100 residues
    batch = {
        "mask": torch.ones(5, 100),  # (batch_size, num_residues)
    }
    batch['mask'][0, -10:] = 0
    batch['mask'][1, -5:] = 0
    batch['mask'][2, -7:] = 0
    batch['mask'][3, -8:] = 0
    
    # Create a motif and update the batch
    updated_batch = factory.create_motif(batch['mask'].shape[-1])
    # Check the shapes and types of the updated masks
    assert updated_batch["fixed_sequence_mask"].shape == (100,)
    assert updated_batch["fixed_structure_mask"].shape == (100, 100)
    assert updated_batch["fixed_sequence_mask"].dtype == torch.bool
    assert updated_batch["fixed_structure_mask"].dtype == torch.bool
    
    updated_batch = factory.create_batch_motif(batch)
    
    assert updated_batch["fixed_sequence_mask"].shape == (5, 100)
    assert updated_batch["fixed_structure_mask"].shape == (5, 100, 100)
    assert updated_batch["fixed_sequence_mask"].dtype == torch.bool
    assert updated_batch["fixed_structure_mask"].dtype == torch.bool

    