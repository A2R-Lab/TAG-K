# import numpy as np

# files = [
#     "paper_plots/exp1/high_noise/results.npz", 
#     "paper_plots/exp1/medium_noise/results.npz", 
#     "paper_plots/exp1/low_noise/results.npz"]

# # Inspect keys and shapes
# for f in files:
#     data = np.load(f)
#     print(f"\n{f}:")
#     for k in data.keys():
#         print(f"  {k}: shape={data[k].shape}, dtype={data[k].dtype}")

# import pdb; pdb.set_trace()

import numpy as np
from pathlib import Path

def merge_npz(files, noise_labels, out="merged.npz"):
    """
    files: list of 3 .npz paths
    noise_labels: list of strings, e.g. ["low", "medium", "high"]
    out: output .npz
    """
    assert len(files) == len(noise_labels) and len(files) > 0

    # Load all
    packs = [np.load(f, allow_pickle=False) for f in files]

    # Sanity: same key set across files
    keyset = [set(p.files) for p in packs]
    assert all(ks == keyset[0] for ks in keyset), "NPZ files have different key sets"

    # Use first file as reference
    ref = packs[0]
    keys = list(ref.files)

    # How many trajectories per file (use traj_labels length)
    n_trajs = [len(p["traj_labels"]) for p in packs]

    # Identify trajectory-indexed keys: first dim equals len(traj_labels)
    def is_traj_key(p, k):
        arr = p[k]
        return (hasattr(arr, "shape") and arr.ndim >= 1 and arr.shape[0] == len(p["traj_labels"]))

    traj_keys = [k for k in keys if all(is_traj_key(p, k) for p in packs)]
    # Optional: warn about keys that differ in shape (excluding axis 0)
    for k in traj_keys:
        ref_shape_rest = ref[k].shape[1:]
        for p in packs[1:]:
            if p[k].shape[1:] != ref_shape_rest:
                raise ValueError(f"Shape mismatch for key '{k}': {p[k].shape} vs {ref[k].shape}")

    # Meta keys = everything else
    meta_keys = [k for k in keys if k not in traj_keys]

    merged = {}

    # Concatenate all trajectory-indexed keys along axis 0 (works for float64/int/bool/str)
    for k in traj_keys:
        merged[k] = np.concatenate([p[k] for p in packs], axis=0)

    # Copy meta/bookkeeping keys from the first file (assuming identical across)
    for k in meta_keys:
        merged[k] = ref[k]

    # Add bookkeeping for noise per trajectory (so you can slice later)
    noise_per_file = [
        np.array([noise_labels[i]] * n_trajs[i], dtype=f"<U{max(len(n) for n in noise_labels)}")
        for i in range(len(packs))
    ]
    merged["noise_level_per_traj"] = np.concatenate(noise_per_file, axis=0)

    # Also store the original noise_level scalars for traceability (unchanged from each file)
    merged["noise_levels_sources"] = np.array([p["noise_level"][0] for p in packs])

    # Save
    np.savez(out, **merged)
    print(f"Saved merged to {out}")
    # Show a quick summary
    print("\nConcatenated keys (axis 0 extended):")
    for k in traj_keys:
        print(f"  {k}: {merged[k].shape}, {merged[k].dtype}")
    print("\nMeta keys (copied from first file):")
    for k in meta_keys:
        if k not in ("noise_level",):  # since it differs across files
            print(f"  {k}: {merged[k].shape if hasattr(merged[k],'shape') else type(merged[k])}")

# Example usage
files = [
    "paper_plots/exp1_v5/no_noise_wind5/results.npz",
    "paper_plots/exp1_v5/low_noise_wind5/results.npz",
    "paper_plots/exp1_v5/medium_noise_wind5/results.npz",
    "paper_plots/exp1_v5/high_noise_wind5/results.npz",
]
merge_npz(files, noise_labels=["none", "low", "medium", "high"], out="paper_plots/exp1_v5/merged_results.npz")