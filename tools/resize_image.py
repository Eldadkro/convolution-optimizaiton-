#!/usr/bin/env python3

import argparse
from pathlib import Path

from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Resize an image and save it to an output file.")
    parser.add_argument("--input", dest="input_option", type=Path, help="Path to the source image.")
    parser.add_argument("args", nargs="+", help="Either input width height output or width height output with --input.")
    parsed = parser.parse_args()

    if parsed.input_option is None:
        if len(parsed.args) != 4:
            parser.error("expected: input width height output")
        input_path, width, height, output_path = parsed.args
    else:
        if len(parsed.args) != 3:
            parser.error("expected with --input: width height output")
        input_path = parsed.input_option
        width, height, output_path = parsed.args

    try:
        parsed.width = int(width)
        parsed.height = int(height)
    except ValueError:
        parser.error("width and height must be integers")

    parsed.input = Path(input_path)
    parsed.output = Path(output_path)
    return parsed


def main() -> int:
    args = parse_args()

    if args.width <= 0 or args.height <= 0:
        raise ValueError("width and height must be positive integers")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    with Image.open(args.input) as image:
        resized = image.resize((args.width, args.height), Image.Resampling.LANCZOS)
        resized.save(args.output)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
