"""
data.py
-------
Download / load the Galaxy10 DECaLS dataset and produce a stratified
train / val / test split that is reused by every other script.

The dataset: 17,736 RGB galaxy cutouts (256x256x3) from DECaLS, labeled into
10 Galaxy Zoo morphology classes (0-9). Distributed as a single HDF5 file.

Usage:
    python data.py --data-dir ../data

This will:
  1. Try to fetch Galaxy10_DECals.h5 automatically via astroNN.
     If astroNN is not installed / the download fails, it will look for
     a manually-downloaded copy at <data-dir>/Galaxy10_DECals.h5
     (get it from https://data.galaxyzoo.org or the astroNN docs).
  2. Build a stratified 70/15/15 train/val/test split (stratified on class
     label so all 10 classes are represented in every split).
  3. Save the split as an index file (split_indices.npz) so baseline.py,
     cnn.py and analysis.py all train/evaluate on identical splits.
"""
import argparse
import os
import sys

import h5py
import numpy as np
from sklearn.model_selection import train_test_split

GALAXY10_CLASSES = {
    0: "Disturbed",
    1: "Merging",
    2: "Round Smooth",
    3: "In-between Round Smooth",
    4: "Cigar-shaped Smooth",
    5: "Barred Spiral",
    6: "Unbarred Tight Spiral",
    7: "Unbarred Loose Spiral",
    8: "Edge-on without Bulge",
    9: "Edge-on with Bulge",
}


def _https_opener():
    """
    Build a urllib opener that verifies HTTPS certs using certifi's CA
    bundle. Plain urllib.request.urlretrieve() relies on the OS trust
    store, which on a fresh macOS python.org install (or some corporate
    networks) is missing/incomplete and raises:
        [SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed:
        unable to get local issuer certificate
    Using certifi's bundle explicitly avoids depending on the OS store.
    """
    import ssl
    import urllib.request

    try:
        import certifi
        ctx = ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        print(
            "Note: 'certifi' is not installed (pip install certifi) -- "
            "falling back to the OS default SSL context. If you hit a "
            "CERTIFICATE_VERIFY_FAILED error, run:\n"
            "    pip install --upgrade certifi"
        )
        ctx = ssl.create_default_context()

    https_handler = urllib.request.HTTPSHandler(context=ctx)
    return urllib.request.build_opener(https_handler)


def download_with_astroNN(h5_path):
    """Try to fetch the file automatically using astroNN's downloader."""
    try:
        from astroNN.datasets.galaxy10 import Galaxy10DECals_URL  # noqa
    except Exception:
        pass
    try:
        from astroNN.shared.downloader_tools import TqdmUpTo
    except Exception:
        TqdmUpTo = None

    import urllib.request

    url = "https://www.astro.utoronto.ca/~hleung/shared/Galaxy10/Galaxy10_DECals.h5"
    print(f"Attempting download from {url} ...")

    opener = _https_opener()
    urllib.request.install_opener(opener)

    try:
        if TqdmUpTo is not None:
            with TqdmUpTo(unit="B", unit_scale=True, miniters=1, desc=url.split("/")[-1]) as t:
                urllib.request.urlretrieve(url, h5_path, reporthook=t.update_to)
        else:
            urllib.request.urlretrieve(url, h5_path)
        return True
    except Exception as e:
        print(f"Automatic download failed: {e}")
        print(
            "\nIf this was a CERTIFICATE_VERIFY_FAILED error, try:\n"
            "  1. pip install --upgrade certifi\n"
            "  2. On macOS with python.org's installer, run:\n"
            "       /Applications/Python\\ 3.x/Install\\ Certificates.command\n"
            "  3. Or just download the file manually from\n"
            "     https://data.galaxyzoo.org (or the astroNN docs) and place it at:\n"
            f"       {h5_path}"
        )
        return False


def load_raw(data_dir):
    """Return (images uint8 [N,256,256,3], labels int [N]) from the h5 file."""
    h5_path = os.path.join(data_dir, "Galaxy10_DECals.h5")
    if not os.path.exists(h5_path):
        os.makedirs(data_dir, exist_ok=True)
        ok = download_with_astroNN(h5_path)
        if not ok:
            raise FileNotFoundError(
                f"Could not find or download the dataset.\n"
                f"Please manually download 'Galaxy10_DECals.h5' "
                f"(see astroNN docs / data.galaxyzoo.org) and place it at:\n"
                f"  {h5_path}"
            )

    with h5py.File(h5_path, "r") as f:
        # astroNN's Galaxy10 DECaLS file stores images under 'images' and
        # labels under 'ans'. Fall back to scanning keys if names differ.
        keys = list(f.keys())
        img_key = "images" if "images" in keys else next(k for k in keys if "im" in k.lower())
        lab_key = "ans" if "ans" in keys else next(k for k in keys if "ans" in k.lower() or "label" in k.lower())
        images = f[img_key][:]
        labels = f[lab_key][:].astype(np.int64)

    return images, labels


def make_split(labels, seed=42):
    """Stratified 70/15/15 train/val/test split, returned as index arrays."""
    idx = np.arange(len(labels))
    train_idx, temp_idx = train_test_split(
        idx, test_size=0.30, stratify=labels, random_state=seed
    )
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.50, stratify=labels[temp_idx], random_state=seed
    )
    return train_idx, val_idx, test_idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="../data")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    images, labels = load_raw(args.data_dir)
    print(f"Loaded {len(images)} images, shape={images.shape[1:]}, dtype={images.dtype}")
    print("Class counts:")
    for c, name in GALAXY10_CLASSES.items():
        print(f"  {c} ({name}): {(labels == c).sum()}")

    train_idx, val_idx, test_idx = make_split(labels, seed=args.seed)
    out_path = os.path.join(args.data_dir, "split_indices.npz")
    np.savez(out_path, train_idx=train_idx, val_idx=val_idx, test_idx=test_idx)
    print(f"\nSaved split ({len(train_idx)} train / {len(val_idx)} val / {len(test_idx)} test) -> {out_path}")


if __name__ == "__main__":
    main()
