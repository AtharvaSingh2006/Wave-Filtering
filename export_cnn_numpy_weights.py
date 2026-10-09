"""Export the supplied CNN-SWS HDF5 weights to a NumPy-only NPZ file."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h5", type=Path, default=Path("tmp/upstream_cnn_sws/model/CNN_XKS.h5"))
    parser.add_argument("--output", type=Path, default=Path("models/CNN_XKS_numpy_weights.npz"))
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    arrays: dict[str, np.ndarray] = {}
    with h5py.File(args.h5, "r") as weights:
        layer_names = [name.decode("utf-8") for name in weights.attrs["layer_names"]]
        weighted_names = [name for name in layer_names if name.startswith(("conv1d_", "dense_"))]
        for index, name in enumerate(weighted_names):
            group = weights[name][name]
            arrays[f"layer_{index}_kernel"] = np.asarray(group["kernel:0"], dtype=np.float32)
            arrays[f"layer_{index}_bias"] = np.asarray(group["bias:0"], dtype=np.float32)
    if len(weighted_names) != 10:
        raise ValueError(f"Expected nine Conv1D layers and one dense layer; found {len(weighted_names)}")
    np.savez_compressed(args.output, **arrays)
    print(f"Exported {len(weighted_names)} layers to {args.output}")


if __name__ == "__main__":
    main()
