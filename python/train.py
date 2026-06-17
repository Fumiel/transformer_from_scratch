from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/tiny_corpus.txt"))
    parser.add_argument("--steps", type=int, default=1000)
    args = parser.parse_args()

    print("TODO: implement training loop")
    print(f"data={args.data}, steps={args.steps}")


if __name__ == "__main__":
    main()
