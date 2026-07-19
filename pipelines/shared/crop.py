"""Bird crop helpers: padded square crop + fixed 256×256 resize."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image


@dataclass(frozen=True)
class CropResult:
    path: Path
    box_w: float
    box_h: float
    pad_px: float
    size: int = 256


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(v, hi))


def square_padded_box(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    img_w: int,
    img_h: int,
    pad_ratio: float = 0.1,
) -> tuple[int, int, int, int]:
    """Expand bbox by ``pad_ratio`` then grow to a square clipped to the image."""
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    side = max(bw, bh) * (1.0 + 2.0 * pad_ratio)
    half = side / 2.0
    sx1 = cx - half
    sy1 = cy - half
    sx2 = cx + half
    sy2 = cy + half
    # Shift into frame without shrinking below available space when possible
    if sx1 < 0:
        sx2 -= sx1
        sx1 = 0
    if sy1 < 0:
        sy2 -= sy1
        sy1 = 0
    if sx2 > img_w:
        sx1 -= sx2 - img_w
        sx2 = img_w
    if sy2 > img_h:
        sy1 -= sy2 - img_h
        sy2 = img_h
    sx1 = _clamp(sx1, 0, img_w - 1)
    sy1 = _clamp(sy1, 0, img_h - 1)
    sx2 = _clamp(sx2, sx1 + 1, img_w)
    sy2 = _clamp(sy2, sy1 + 1, img_h)
    # Final square inside remaining rect
    rw, rh = sx2 - sx1, sy2 - sy1
    side = min(rw, rh)
    cx = (sx1 + sx2) / 2.0
    cy = (sy1 + sy2) / 2.0
    half = side / 2.0
    return (
        int(round(cx - half)),
        int(round(cy - half)),
        int(round(cx + half)),
        int(round(cy + half)),
    )


def crop_bird_to_square(
    src: Path,
    dest: Path,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    *,
    size: int = 256,
    pad_ratio: float = 0.1,
    jpeg_quality: int = 90,
) -> CropResult:
    """Crop the bird box, pad to square, resize to ``size``×``size``, save JPEG."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src) as im:
        im = im.convert("RGB")
        w, h = im.size
        box = square_padded_box(x1, y1, x2, y2, w, h, pad_ratio=pad_ratio)
        crop = im.crop(box)
        # Letterbox to square without distorting (edge case if clip made non-square)
        cw, ch = crop.size
        side = max(cw, ch)
        canvas = Image.new("RGB", (side, side), (0, 0, 0))
        canvas.paste(crop, ((side - cw) // 2, (side - ch) // 2))
        out = canvas.resize((size, size), Image.Resampling.LANCZOS)
        out.save(dest, format="JPEG", quality=jpeg_quality, optimize=True)
    return CropResult(
        path=dest,
        box_w=x2 - x1,
        box_h=y2 - y1,
        pad_px=pad_ratio * max(x2 - x1, y2 - y1),
        size=size,
    )
