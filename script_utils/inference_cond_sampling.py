# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import os
import sys
from typing import List

root = os.path.abspath(".")
sys.path.append(root)  # Adds project's root directory

import argparse
from collections import defaultdict
import random

import hydra
import lightning as L
import torch
from dotenv import load_dotenv
from loguru import logger
from torch.utils.data import DataLoader, Dataset

from proteinfoundation.proteinflow.proteina import Proteina
from proteinfoundation.utils.ff_utils.pdb_utils import write_prot_to_pdb


# Another version of the generator dataset, which is a bit more flexible than the one in inference.py
class GenDataset(Dataset):
    """
    Dataset that indicates length of the proteins to generate,
    discretization step size, and number of samples per length,
    empirical (len, cath_code) joint distribution.
    """

    # Use this for proteins with lengths between 50 and 250
    bucket_min_len = 50
    bucket_max_len = 274
    bucket_step_size = 25
    bucket_size = (bucket_max_len - bucket_min_len) // bucket_step_size + 1

    # Use this for proteins with lengths between 250 and 800
    # bucket_min_len = 300
    # bucket_max_len = 824
    # bucket_step_size = 25
    # bucket_size = (bucket_max_len - bucket_min_len) // bucket_step_size + 1

    def __init__(
        self,
        nres=None,
        cath_codes=None,
        dt=0.005,
        nsamples=10,
        len_cath_codes=None,
        max_nsamples=30,
    ):
        """
        Dataset that indicates either length or cath codes to generate, discretization step size, and number of samples per length / cath code.
        Lengths are randomly sampled from empricial (len, cath_code) distribution.

        Args:
            nres (Optional[List[int]]): List of lengths to generate
            cath_codes (Optional[List[str]]): List of cath code to condition on
            nsamples (Optional[int]): Number of samples to generate for each cath code
            dt (Optional[float]): Discretization step size
            len_cath_codes (Optional[List[Tuple[int, List[str]]]]): List of (length, cath_codes) pairs for representing joint distribution
            max_nsamples (Optional[int]): Maximum number of samples for each batch, set this when used for single-gpu generation
            n_replicas (Optional[int]): Number of devices
        """
        super(GenDataset, self).__init__()
        if isinstance(nsamples, List):
            assert len(nsamples) == len(nres)
        elif isinstance(nsamples, int):
            if nres:
                nsamples = [nsamples] * len(nres)
            else:
                nsamples = [nsamples] * len(cath_codes)
        else:
            raise ValueError(f"Unknown type of nsamples {type(nsamples)}")
        self.dt = dt

        assert (nres is None) or (
            cath_codes is None
        ), "The dataset can only be either length-based or label-based."
        if nres is not None:
            # Length-based generation
            nres = [int(n) for n in nres]
            self.cath_codes_given_len_bucket, _ = self.bucketize(len_cath_codes)
            self.nres, self.cath_codes, self.nsamples = (
                self.generate_cath_code_given_len(nres, nsamples)
            )
        else:
            # Label-based generation
            _, self.len_bucket_given_cath_codes = self.bucketize(len_cath_codes)
            self.nres, self.cath_codes, self.nsamples = (
                self.generate_len_given_cath_code(cath_codes, nsamples)
            )

        # Make sure the nsamples for each data point is not greater than max_nsamples
        if max_nsamples:
            self.nres, self.cath_codes, self.nsamples = self.flatten(max_nsamples)

        assert all(
            [n <= max_nsamples for n in self.nsamples]
        ), f"The nsamples for each len shouldn't be greater than {max_nsamples}"

    def bucketize(self, len_cath_codes):
        """Build length buckets for cath_codes. Record the cath_code distribution given length bucket and the reverse"""
        if len_cath_codes is None:
            return None, None

        bucket = list(
            range(self.bucket_min_len, self.bucket_max_len, self.bucket_step_size)
        )
        cath_codes_given_len_bucket = [[] for _ in range(len(bucket))]
        _len_bucket_given_cath_codes = defaultdict(set)
        for _len, codes in len_cath_codes:
            if len(codes) == 0:
                continue
            bucket_idx = (_len - self.bucket_min_len) // self.bucket_step_size
            bucket_idx = min(bucket_idx, self.bucket_size - 1)  # Boundary cutoff
            bucket_idx = max(bucket_idx, 0)

            # Record all possible cath codes for each bucket
            cath_codes_given_len_bucket[bucket_idx].append(codes)

            # Record all possible len bucket for each cath code
            for code in codes:
                for level in ["C", "A", "T"]:
                    ns = {"C": 3, "A": 2, "T": 1}
                    level_code = code.rsplit(".", ns[level])[0] + ".x" * ns[level]
                    _len_bucket_given_cath_codes[level_code].add(bucket_idx)

        len_bucket_given_cath_codes = {}
        for k, v in _len_bucket_given_cath_codes.items():
            len_bucket_given_cath_codes[k] = tuple(v)
        return cath_codes_given_len_bucket, len_bucket_given_cath_codes

    def generate_cath_code_given_len(self, nres, nsamples):
        """Pre-generate corresponding cath codes for each length"""
        cath_codes = []
        for i in range(len(nres)):
            if self.cath_codes_given_len_bucket is None:
                cath_code = None
            else:
                if nres[i] <= self.bucket_max_len:
                    bucket_idx = (
                        nres[i] - self.bucket_min_len
                    ) // self.bucket_step_size
                else:
                    bucket_idx = -1
                cath_code = random.choices(
                    self.cath_codes_given_len_bucket[bucket_idx], k=nsamples[i]
                )
            cath_codes.append(cath_code)
        return nres, cath_codes, nsamples

    def generate_len_given_cath_code(self, cath_codes, nsamples):
        """Pre-generate corresponding lengths for each cath code, then gather proteins of the same length as one batch"""
        assert (
            self.len_bucket_given_cath_codes is not None
        ), "Need len_cath_code distribution for label-based dataset"
        tmp_nres = []
        tmp_cath_codes = []
        for i in range(len(cath_codes)):
            for _ in range(nsamples[i]):
                if cath_codes[i] not in self.len_bucket_given_cath_codes:
                    raise ValueError(
                        f"CATH code {cath_codes[i]} not in the empirical distribution"
                    )
                bucket_idx = random.choices(
                    self.len_bucket_given_cath_codes[cath_codes[i]], k=1
                )[0]
                _len = self.bucket_min_len + bucket_idx * self.bucket_step_size

                tmp_nres.append(_len)
                tmp_cath_codes.append([cath_codes[i]])

        # Gather the same lengths, as we need to generate proteins of the same length together
        len_bucket = defaultdict(list)
        out_nres, out_cath_codes, out_nsamples = [], [], []
        for n, code in zip(tmp_nres, tmp_cath_codes):
            len_bucket[n].append(code)

        for n, code in len_bucket.items():
            out_nres.append(n)
            out_cath_codes.append(code)
            out_nsamples.append(len(code))

        return out_nres, out_cath_codes, out_nsamples

    def flatten(self, max_nsamples):
        """Flatten the list to make sure each data point haveno more than max_nsamples"""
        nres, cath_codes, nsamples = [], [], []
        for i in range(len(self.nsamples)):
            for j in range(0, self.nsamples[i], max_nsamples):
                nres.append(self.nres[i])
                if self.cath_codes[i] is not None:
                    cath_codes.append(self.cath_codes[i][j : j + max_nsamples])
                else:
                    cath_codes.append(None)
                if j + max_nsamples <= self.nsamples[i]:
                    nsamples.append(max_nsamples)
                else:
                    nsamples.append(self.nsamples[i] - j)
        return nres, cath_codes, nsamples

    def __len__(self):
        return len(self.nres)

    def __getitem__(self, index):
        result = {
            "nres": self.nres[index],
            "dt": self.dt,
            "nsamples": self.nsamples[index],
        }
        if self.cath_codes[index] is not None:
            result["cath_code"] = self.cath_codes[index]
        return result
    

if __name__ == "__main__":
    load_dotenv()

    parser = argparse.ArgumentParser(description="Job info")
    parser.add_argument("--config_name", type=str, default="inference_cond_sampling_specific_codes", help="Name of the config yaml file.")
    parser.add_argument("--cath_codes", nargs="*", default=
        [
            "1.x.x.x", "2.x.x.x", "3.x.x.x",
        ], help="List of cath codes.")
    parser.add_argument("--nsamples", type=int, default=5, help="Number of samples for each cath code.")
    args = parser.parse_args()
    logger.info(" ".join(sys.argv))

    assert torch.cuda.is_available(), "CUDA not available"  # Needed for ESMfold and designability
    logger.add(sys.stdout, format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {file}:{line} | {message}")  # Send to stdout

    # Inference config
    config_path = "../configs/experiment_config"
    config_name = args.config_name
    with hydra.initialize(config_path, version_base=hydra.__version__):
        cfg = hydra.compose(config_name=config_name)
        logger.info(f"Inference config {cfg}")
        run_name = cfg.run_name_

    # Set root path for this inference run
    root_path = f"./inference/{config_name}"
    os.makedirs(root_path, exist_ok=True)

    # Load model from checkpoint
    ckpt_path = cfg.ckpt_path
    ckpt_file = os.path.join(ckpt_path, cfg.ckpt_name)
    logger.info(f"Using checkpoint {ckpt_file}")
    assert os.path.exists(ckpt_file), f"Not a valid checkpoint {ckpt_file}"

    model = Proteina.load_from_checkpoint(ckpt_file)

    # Set seed
    logger.info(f"Seeding everything to seed {cfg.seed}")
    L.seed_everything(cfg.seed)

    # Set inference variables and potentially load autoguidance
    nn_ag = None
    if cfg.get("autoguidance_ratio", 0.0) > 0 and cfg.get("guidance_weight", 1.0) != 1.0:
        assert cfg.autoguidance_ckpt_path is not None
        ckpt_ag_file = cfg.autoguidance_ckpt_path
        model_ag = Proteina.load_from_checkpoint(ckpt_ag_file)
        nn_ag = model_ag.nn
    
    model.configure_inference(cfg, nn_ag=nn_ag)

    cath_codes = args.cath_codes
    cath_codes_dict = None

    logger.info(f"Generating samples for classes {cath_codes}")
    len_cath_codes = torch.load(cfg.len_cath_code_path)
    dataset = GenDataset(cath_codes=cath_codes, nsamples=args.nsamples, dt=cfg.dt, len_cath_codes=len_cath_codes)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    # Leave batch_size=1, the number of samples in a "batch of size 1" is controlled by args.nsamples

    # Sample the model
    trainer = L.Trainer(accelerator="gpu", devices=1)
    predictions = trainer.predict(model, dataloader)

    # Create directory to store all samples
    samples_dir_fid = os.path.join(root_path, "samples_recls")
    os.makedirs(samples_dir_fid, exist_ok=True)

    # Store samples
    lens_sample = dataset.nres
    cath_codes = dataset.cath_codes
    list_of_pdbs = []
    for j, pred in enumerate(predictions):
        for i in range(pred.shape[0]):
            coors_atom37 = pred[i]  # pred - [b, n, 37, 3],  coors_atom37 - [n, 37, 3]
            idx = len(list_of_pdbs)
            pdb_path = os.path.join(samples_dir_fid, f"{idx}_{lens_sample[j]}_{cath_codes[j][i][0]}_recls.pdb")

            dir_name = os.path.basename(pdb_path)[:-10]
            os.makedirs(os.path.join(root_path, dir_name), exist_ok=True)
            write_prot_to_pdb(
                coors_atom37.numpy(),
                os.path.join(root_path, dir_name, dir_name+".pdb"),
                overwrite=True,
                no_indexing=True,
            )
            list_of_pdbs.append(dir_name)
