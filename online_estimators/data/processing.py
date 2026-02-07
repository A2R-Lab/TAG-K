"""
Utilities for merging and post-processing experiment results stored as
NumPy ``.npz`` archives.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Sequence

import numpy as np


def merge_npz(
    files: Sequence[str | Path],
    noise_labels: Sequence[str],
    out: str | Path = "merged.npz",
) -> None:
    """Concatenate several ``.npz`` experiment result files along the
    trajectory axis and save a single merged archive.

    All files must share the same set of keys.  Keys whose first dimension
    matches ``len(traj_labels)`` are treated as *trajectory-indexed* and
    concatenated along ``axis=0``.  All remaining keys are copied verbatim
    from the first file.

    Parameters
    ----------
    files : sequence of str or Path
        Paths to the ``.npz`` archives.
    noise_labels : sequence of str
        Human-readable noise level for each file (e.g. ``["low", "high"]``).
        Must have the same length as *files*.
    out : str or Path
        Destination path for the merged archive.

    Raises
    ------
    ValueError
        If the files have different key sets or incompatible trailing shapes.
    """
    if len(files) != len(noise_labels):
        raise ValueError("len(files) must equal len(noise_labels)")
    if len(files) == 0:
        raise ValueError("Need at least one file")

    packs = [np.load(f, allow_pickle=False) for f in files]

    # Sanity: same key set
    keyset0 = set(packs[0].files)
    for i, p in enumerate(packs[1:], 1):
        if set(p.files) != keyset0:
            raise ValueError(
                f"Key set of file {i} differs from file 0: "
                f"extra={set(p.files) - keyset0}, missing={keyset0 - set(p.files)}"
            )

    keys: List[str] = list(packs[0].files)
    n_trajs = [len(p["traj_labels"]) for p in packs]

    def _is_traj_key(p: np.lib.npyio.NpzFile, k: str) -> bool:
        arr = p[k]
        return arr.ndim >= 1 and arr.shape[0] == len(p["traj_labels"])

    traj_keys = [k for k in keys if all(_is_traj_key(p, k) for p in packs)]

    # Check shape compatibility along non-axis-0 dims
    for k in traj_keys:
        ref_tail = packs[0][k].shape[1:]
        for i, p in enumerate(packs[1:], 1):
            if p[k].shape[1:] != ref_tail:
                raise ValueError(
                    f"Shape mismatch for key '{k}': file {i} has "
                    f"{p[k].shape} vs reference {packs[0][k].shape}"
                )

    meta_keys = [k for k in keys if k not in traj_keys]
    merged: dict[str, np.ndarray] = {}

    for k in traj_keys:
        merged[k] = np.concatenate([p[k] for p in packs], axis=0)
    for k in meta_keys:
        merged[k] = packs[0][k]

    max_label_len = max(len(n) for n in noise_labels)
    noise_per_file = [
        np.array([noise_labels[i]] * n_trajs[i], dtype=f"<U{max_label_len}")
        for i in range(len(packs))
    ]
    merged["noise_level_per_traj"] = np.concatenate(noise_per_file, axis=0)

    np.savez(out, **merged)  # type: ignore[arg-type]
