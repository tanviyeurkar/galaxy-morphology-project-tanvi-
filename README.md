# Project 1 — Galaxy morphology: where does the classifier fail?

**Question.** Can a small CNN classify galaxy shapes? Are its mistakes random or
systematic — does it fail on faint, small, or edge-on galaxies?

**Dataset.** Galaxy10 DECaLS — 17,736 galaxy images (256×256, 3 bands), 10 Galaxy
Zoo morphology classes, distributed as one `.h5` file.

## Layout

```
galaxy10_project/
├── code/
│   ├── data.py        # download Galaxy10_DECals.h5, build stratified 70/15/15 split
│   ├── features.py     # 5 hand-crafted features per galaxy (size, brightness,
│   │                    # ellipticity/axis-ratio, concentration)
│   ├── baseline.py      # logistic regression + random forest on hand-crafted
│   │                     # features only — the number the CNN must beat
│   ├── cnn.py            # small 4-block conv net (PyTorch) on raw pixels (128x128)
│   ├── analysis.py       # confusion matrix + binned-accuracy + "is it secretly
│   │                      # a size classifier?" tests
│   └── run_all.py        # runs the four steps above in order
├── data/       # h5 file, feature cache, split indices, prediction dumps go here
├── models/     # trained CNN weights
├── figures/    # confusion matrix + binned accuracy plots + report.txt
└── requirements.txt
```

## How to run

This needs real compute + internet access to fetch the ~2.5 GB dataset and install
PyTorch, so it's meant to run on your own machine (a laptop CPU is fine, GPU is
faster), not in this sandbox.

```bash
cd galaxy10_project
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cd code
python run_all.py --epochs 25
```

Or step by step:

```bash
python data.py            # downloads Galaxy10_DECals.h5, writes split_indices.npz
python baseline.py        # hand-crafted-feature baseline (logreg + random forest)
python cnn.py --epochs 25 # trains the CNN, dumps test predictions
python analysis.py        # confusion matrix + failure-mode report
```

If the automatic download in `data.py` fails (some networks block the
astro.utoronto.ca mirror), manually download `Galaxy10_DECals.h5` from
[data.galaxyzoo.org](https://data.galaxyzoo.org) or the
[astroNN docs](https://astronn.readthedocs.io/en/latest/galaxy10.html) and drop
it in `data/Galaxy10_DECals.h5`.

## What each piece does

**Baseline (must beat this).** `features.py` segments each galaxy from its
background (Otsu threshold), then computes: mean brightness inside the mask,
an effective radius (`size_px`, i.e. apparent size in the cutout), an axis
ratio from the second moments of the light distribution (ellipticity /
inclination proxy — low axis ratio ≈ edge-on or elongated), and a
concentration index `C = 5·log10(R80/R20)`. A logistic regression and a
random forest are trained on just these 5 numbers.

**CNN.** A small 4-block conv net (32→64→128→256 channels, batchnorm, global
average pool, dropout head) trained on 128×128 RGB crops with random
flips/rotations (galaxy orientation on sky is arbitrary, so this is a free
augmentation). Early stopping on validation accuracy.

**Failure-mode analysis.** The deliverable isn't a single accuracy number —
`analysis.py` produces:
1. A row-normalized confusion matrix heatmap.
2. CNN vs. baseline accuracy broken into tercile bins (small/medium/large,
   faint/medium/bright, edge-on/mid/face-on).
3. A specific "is it secretly a size classifier?" test:
   - correlation between each class's mean apparent size and the CNN's
     per-class recall,
   - whether a trivial model using only (size, brightness, axis ratio,
     concentration) can recover the CNN's *own predicted* labels well above
     chance,
   - whether the CNN's wrong guesses tend to land on the class whose typical
     apparent size is closest to the true galaxy's own apparent size.

All of it is written to `figures/report.txt` plus PNGs, so the finding is a
readable claim ("the CNN's errors correlate with apparent size at r=X and
mostly fail on Y") backed by numbers, not just a top-line accuracy.
# galaxy-morphology-project-tanvi-
