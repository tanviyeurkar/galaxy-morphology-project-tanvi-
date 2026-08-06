"""
features.py
------------
Hand-crafted, per-galaxy features computed directly from pixels:

  mean_brightness   - mean flux inside the detected galaxy mask
  size_px            - effective radius (pixels) of the galaxy mask, i.e.
                        "how big does this galaxy look in the cutout"
  ellipticity         - 1 - (minor/major axis ratio) of the light distribution
  axis_ratio (q)      - minor/major axis ratio (b/a); low q ~ edge-on / elongated
  concentration       - C = 5*log10(R80/R20), a standard morphology proxy
                        (higher C = light more centrally concentrated, e.g.
                        ellipticals; lower C = more extended, e.g. spirals)

These five numbers are (a) the feature vector for the baseline classifier,
and (b) the "apparent size / brightness / inclination" axes used later in
analysis.py to test whether the CNN's mistakes are systematic.

No deep learning here on purpose -- this is meant to be a dumb, fast,
interpretable baseline the CNN has to beat.
"""
import numpy as np
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops


def _largest_component_mask(binary):
    """Keep only the largest connected component touching/near the center."""
    lbl = label(binary)
    if lbl.max() == 0:
        return binary
    h, w = binary.shape
    cy, cx = h // 2, w // 2
    # prefer the component that covers the image center; fall back to largest
    center_label = lbl[cy, cx]
    if center_label != 0:
        return lbl == center_label
    props = regionprops(lbl)
    biggest = max(props, key=lambda p: p.area)
    return lbl == biggest.label


def _radial_profile_fractions(gray, mask, cy, cx):
    """Return R20, R80: radii (px) enclosing 20% / 80% of the flux in mask."""
    ys, xs = np.nonzero(mask)
    if len(ys) < 5:
        return 1.0, 2.0
    r = np.sqrt((ys - cy) ** 2 + (xs - cx) ** 2)
    flux = gray[ys, xs]
    flux = np.clip(flux, 0, None)
    order = np.argsort(r)
    r_sorted = r[order]
    flux_sorted = flux[order]
    cum = np.cumsum(flux_sorted)
    total = cum[-1] if cum[-1] > 0 else 1.0
    cum_frac = cum / total
    r20 = np.interp(0.20, cum_frac, r_sorted)
    r80 = np.interp(0.80, cum_frac, r_sorted)
    r20 = max(r20, 1e-3)
    return r20, r80


def extract_features_single(img):
    """
    img: HxWx3 uint8 or float array.
    Returns dict of 5 hand-crafted features.
    """
    if img.dtype != np.float64 and img.dtype != np.float32:
        gray = rgb2gray(img.astype(np.float32) / 255.0)
    else:
        gray = rgb2gray(img)

    try:
        thresh = threshold_otsu(gray)
    except ValueError:
        thresh = gray.mean()
    binary = gray > thresh
    mask = _largest_component_mask(binary)

    if mask.sum() < 10:
        # degenerate case (very faint / failed segmentation): fall back to
        # a generous central mask so features are still defined
        mask = gray > np.percentile(gray, 70)
        if mask.sum() < 10:
            mask = np.ones_like(gray, dtype=bool)

    ys, xs = np.nonzero(mask)
    npix = mask.sum()
    size_px = np.sqrt(npix / np.pi)  # effective radius of equal-area circle
    mean_brightness = float(gray[mask].mean())

    cy, cx = ys.mean(), xs.mean()
    cov = np.cov(np.vstack([ys - cy, xs - cx]))
    eigvals = np.linalg.eigvalsh(cov)
    eigvals = np.clip(eigvals, 1e-6, None)
    lam_minor, lam_major = np.sort(eigvals)
    axis_ratio = float(np.sqrt(lam_minor / lam_major))  # b/a in [0,1]
    ellipticity = float(1.0 - axis_ratio)

    r20, r80 = _radial_profile_fractions(gray, mask, cy, cx)
    concentration = float(5.0 * np.log10(max(r80 / r20, 1.01)))

    return {
        "mean_brightness": mean_brightness,
        "size_px": float(size_px),
        "ellipticity": ellipticity,
        "axis_ratio": axis_ratio,
        "concentration": concentration,
    }


def extract_features_batch(images, verbose=True):
    """images: NxHxWx3 uint8 array -> pandas-ready dict of arrays."""
    from tqdm import tqdm

    keys = ["mean_brightness", "size_px", "ellipticity", "axis_ratio", "concentration"]
    out = {k: np.zeros(len(images), dtype=np.float64) for k in keys}
    it = tqdm(range(len(images)), disable=not verbose, desc="Extracting hand-crafted features")
    for i in it:
        feats = extract_features_single(images[i])
        for k in keys:
            out[k][i] = feats[k]
    return out
