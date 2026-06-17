from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default="hello")
    parser.add_argument("--max-new-tokens", type=int, default=100)
    args = parser.parse_args()

    print("TODO: implement text generation")
    print(f"prompt={args.prompt!r}, max_new_tokens={args.max_new_tokens}")


if __name__ == "__main__":
    main()
