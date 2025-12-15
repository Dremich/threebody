"""Assemble per-orbit GIFs into a single mosaic GIF (image processing only).

Policy:
- No solver imports.
- Deterministic ordering and output.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys

# Allow running directly from a src-layout repo without installation.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def _parse_grid(s: str) -> tuple[int, int]:
    parts = s.lower().split("x")
    if len(parts) != 2:
        raise ValueError("grid must be like '4x5'")
    return int(parts[0]), int(parts[1])


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Make a mosaic GIF from individual orbit GIFs")
    p.add_argument("--input-dir", type=str, default="renders", help="Directory containing *.gif")
    p.add_argument("--out", type=str, default="mosaic.gif", help="Output mosaic GIF")
    p.add_argument("--grid", type=str, default=None, help="RowsxCols (default: auto)")
    p.add_argument("--fps", type=float, default=30.0, help="Output FPS (default: 30)")
    p.add_argument("--tile-size", type=int, default=256, help="Tile size in pixels (default: 256)")
    return p.parse_args()


def _read_gif_frames(path: Path):
    from PIL import Image

    im = Image.open(path)
    frames = []
    try:
        i = 0
        while True:
            im.seek(i)
            frames.append(im.convert("RGBA"))
            i += 1
    except EOFError:
        pass
    return frames


def _resample(frames, n: int):
    if len(frames) == n:
        return frames
    out = []
    L = len(frames)
    for k in range(n):
        idx = int(round((k * (L - 1)) / max(1, n - 1)))
        out.append(frames[idx])
    return out


def main() -> int:
    args = _parse_args()

    try:
        from PIL import Image
    except Exception as e:
        raise SystemExit("Pillow is required for mosaic export. Install with: pip install pillow") from e

    in_dir = Path(args.input_dir)
    gifs = sorted(p for p in in_dir.glob("*.gif") if p.is_file() and p.name != Path(args.out).name)
    if not gifs:
        raise SystemExit(f"No .gif files found in {in_dir}")

    clips = [(_read_gif_frames(p), p.stem) for p in gifs]
    max_len = max(len(frames) for frames, _ in clips)

    tile = int(args.tile_size)

    if args.grid is None:
        cols = int(math.ceil(math.sqrt(len(clips))))
        rows = int(math.ceil(len(clips) / cols))
    else:
        rows, cols = _parse_grid(args.grid)

    W = cols * tile
    H = rows * tile

    # Normalize all clips to common frame count and resolution.
    norm = []
    for frames, _name in clips:
        frames = _resample(frames, max_len)
        frames = [f.resize((tile, tile), resample=Image.BICUBIC) for f in frames]
        norm.append(frames)

    mosaic_frames = []
    for k in range(max_len):
        canvas = Image.new("RGBA", (W, H), (0, 0, 0, 255))
        for idx, frames in enumerate(norm):
            r = idx // cols
            c = idx % cols
            if r >= rows:
                break
            canvas.paste(frames[k], (c * tile, r * tile))
        mosaic_frames.append(canvas.convert("P"))

    out = Path(args.out)
    duration_ms = int(round(1000.0 / float(args.fps)))
    mosaic_frames[0].save(
        out,
        save_all=True,
        append_images=mosaic_frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=False,
    )
    print(f"Wrote: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
