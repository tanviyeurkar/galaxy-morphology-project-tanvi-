"""
run_all.py
-----------
Runs the full pipeline end to end:
  1. data.py      -- download + split
  2. baseline.py   -- hand-crafted-feature baseline
  3. cnn.py         -- small CNN
  4. analysis.py    -- confusion / failure-mode analysis

Usage:
    python run_all.py --data-dir ../data --fig-dir ../figures --epochs 25
"""
import argparse
import subprocess
import sys


def run(cmd):
    print(f"\n$ {' '.join(cmd)}\n")
    subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="../data")
    ap.add_argument("--fig-dir", default="../figures")
    ap.add_argument("--epochs", type=int, default=25)
    args = ap.parse_args()

    py = sys.executable
    run([py, "data.py", "--data-dir", args.data_dir])
    run([py, "baseline.py", "--data-dir", args.data_dir])
    run([py, "cnn.py", "--data-dir", args.data_dir, "--epochs", str(args.epochs)])
    run([py, "analysis.py", "--data-dir", args.data_dir, "--fig-dir", args.fig_dir])


if __name__ == "__main__":
    main()
