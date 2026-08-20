"""Build a contact sheet from an evaluator's deterministic selection manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def build_contact_sheet(selection_path: Path, output_path: Path, tile_size: int) -> None:
    with selection_path.open("r", encoding="utf-8") as handle:
        selection = json.load(handle)
    cases = selection.get("cases", [])
    if not cases:
        raise ValueError("selection manifest contains no cases")
    model_names = sorted(cases[0]["predictions"])
    columns = 1 + len(model_names)
    label_height = 34
    canvas = Image.new("RGB", (columns * tile_size, len(cases) * (tile_size + label_height)), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    for row, case in enumerate(cases):
        entries = [("GT", case["gt_path"])] + [
            (model_name, case["predictions"][model_name]["path"])
            for model_name in model_names
        ]
        for column, (label, image_path) in enumerate(entries):
            with Image.open(image_path) as opened:
                image = opened.convert("RGB")
                image.thumbnail((tile_size, tile_size), Image.Resampling.LANCZOS)
                tile = Image.new("RGB", (tile_size, tile_size), "#202020")
                tile.paste(image, ((tile_size - image.width) // 2, (tile_size - image.height) // 2))
            x = column * tile_size
            y = row * (tile_size + label_height)
            canvas.paste(tile, (x, y))
            text = f"{case['id']} | {label}"
            draw.text((x + 4, y + tile_size + 4), text, fill="black", font=font)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--tile-size", type=int, default=256)
    args = parser.parse_args()
    if args.tile_size < 32:
        parser.error("--tile-size must be at least 32")
    try:
        build_contact_sheet(args.selection, args.out, args.tile_size)
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(2, f"contact-sheet error: {exc}\n")
    print(args.out.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
