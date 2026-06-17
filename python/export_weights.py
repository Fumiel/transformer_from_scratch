from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=False)
    parser.add_argument("--out", type=Path, default=Path("weights/tiny_weights.json"))
    args = parser.parse_args()

    print("TODO: export PyTorch checkpoint weights to JSON or binary format")
    print(f"checkpoint={args.checkpoint}, out={args.out}")


if __name__ == "__main__":
    main()
