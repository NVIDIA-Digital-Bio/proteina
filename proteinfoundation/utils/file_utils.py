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
import pathlib
import tarfile


def flatten_directory(
    directory: pathlib.Path,
    replace_ending: bool = False,
    old_ending: str = ".ent",
    new_ending: str = ".pdb",
) -> None:
    """
    Flattens the directory structure by moving all files to the top level and optionally replacing file endings.

    Args:
        directory: The directory to flatten.
        replace_ending: A boolean flag indicating whether to replace the file ending. Default is False.
        old_ending: The old file ending to be replaced. Default is '.ent'.
        new_ending: The new file ending to replace the old one with. Default is '.pdb'.

    Returns:
        None
    """
    for file_path in directory.rglob("*"):
        if file_path.is_file():
            new_file_path = directory / (
                file_path.stem
                + (
                    file_path.suffix.replace(old_ending, new_ending)
                    if replace_ending
                    else file_path.suffix
                )
            )
            file_path.rename(new_file_path)

    for dir_path in directory.rglob("*"):
        if dir_path.is_dir() and not any(dir_path.iterdir()):
            dir_path.rmdir()


def extract_archive(
    tar_file: pathlib.Path, destination_dir: pathlib.Path, extracted_dir_name: str
) -> pathlib.Path:
    """
    Extracts the contents of the tar archive to a specified directory with a given name.

    Args:
        tar_file: The path to the tar archive file.
        destination_dir: The directory where the archive contents will be extracted.
        extracted_dir_name: The desired name for the directory containing the extracted contents.

    Returns:
        The path to the directory containing the extracted contents.
    """
    # Create the destination directory if it doesn't exist
    destination_dir.mkdir(parents=True, exist_ok=True)
    # Create the directory for the extracted contents
    renamed_extracted_dir = destination_dir / extracted_dir_name

    # Extract the contents of the archive to the extracted directory
    with tarfile.open(tar_file, "r:gz") as tar:
        top_dir = os.path.commonpath(tar.getnames())
        tar.extractall(path=destination_dir, filter="data")

    # Remove the tar archive file
    tar_file.unlink()

    extracted_dir = pathlib.Path(destination_dir) / top_dir
    extracted_dir.rename(renamed_extracted_dir)

    return renamed_extracted_dir
