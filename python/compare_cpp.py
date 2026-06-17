from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--python-logits", type=Path, required=False)
    parser.add_argument("--cpp-logits", type=Path, required=False)
    args = parser.parse_args()

    print("TODO: compare Python logits and C++ logits")
    print(f"python_logits={args.python_logits}, cpp_logits={args.cpp_logits}")


if __name__ == "__main__":
    main()
