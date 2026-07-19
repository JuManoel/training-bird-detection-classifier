"""Crop helper tests — fixed 256×256 output."""

from pathlib import Path

from PIL import Image

from pipelines.shared.crop import crop_bird_to_square, square_padded_box


def test_square_padded_box_inside_frame():
    box = square_padded_box(10, 20, 50, 80, img_w=200, img_h=200, pad_ratio=0.1)
    x1, y1, x2, y2 = box
    assert x1 >= 0 and y1 >= 0 and x2 <= 200 and y2 <= 200
    assert abs((x2 - x1) - (y2 - y1)) <= 1


def test_crop_always_256(tmp_path: Path):
    src = tmp_path / "bird.jpg"
    Image.new("RGB", (640, 480), color=(10, 20, 30)).save(src)
    dest = tmp_path / "crop.jpg"
    result = crop_bird_to_square(src, dest, 100, 80, 300, 280, size=256)
    assert result.path.exists()
    with Image.open(result.path) as im:
        assert im.size == (256, 256)
