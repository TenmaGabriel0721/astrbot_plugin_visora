from __future__ import annotations

import random
from io import BytesIO

import pytest
from astrbot_plugin_visora.delivery_images import prepare_qq_image
from PIL import Image


def png_bytes(image, *, compress_level=6):
    buffer = BytesIO()
    image.save(buffer, format="PNG", compress_level=compress_level)
    return buffer.getvalue()


def test_small_transparent_image_is_preserved_exactly():
    original = png_bytes(Image.new("RGBA", (96, 96), (240, 180, 200, 128)))

    prepared, resized = prepare_qq_image(original)

    assert prepared == original
    assert resized is None


def test_oversized_png_uses_lossless_compression_without_changing_dimensions():
    original = png_bytes(
        Image.new("RGBA", (1024, 1024), (240, 180, 200, 128)), compress_level=0
    )
    assert len(original) > 64 * 1024

    prepared, resized = prepare_qq_image(original, max_bytes=64 * 1024)

    assert len(prepared) <= 64 * 1024
    assert resized is None
    with Image.open(BytesIO(prepared)) as image:
        assert image.format == "PNG"
        assert image.size == (1024, 1024)
        assert image.getpixel((512, 512)) == (240, 180, 200, 128)


def test_complex_image_uses_jpeg_before_reducing_dimensions():
    pixels = random.Random(42).randbytes(384 * 384 * 3)
    original = png_bytes(Image.frombytes("RGB", (384, 384), pixels))

    prepared, resized = prepare_qq_image(original, max_bytes=160 * 1024)

    assert len(prepared) <= 160 * 1024
    assert resized is None
    with Image.open(BytesIO(prepared)) as image:
        assert image.format == "JPEG"
        assert image.size == (384, 384)


def test_unavoidable_resizing_preserves_aspect_ratio_and_reports_real_dimensions():
    pixels = random.Random(42).randbytes(512 * 768 * 3)
    original = png_bytes(Image.frombytes("RGB", (512, 768), pixels))

    prepared, resized = prepare_qq_image(original, max_bytes=32 * 1024)

    assert len(prepared) <= 32 * 1024
    assert resized is not None
    with Image.open(BytesIO(prepared)) as image:
        assert image.size == resized
        assert image.width < 512
        assert abs(image.width / image.height - 512 / 768) < 0.01


def test_transparency_is_flattened_to_white_when_jpeg_is_required():
    pixels = random.Random(42).randbytes(256 * 256 * 3)
    image = Image.frombytes("RGB", (256, 256), pixels).convert("RGBA")
    image.putalpha(0)
    original = png_bytes(image)

    prepared, resized = prepare_qq_image(original, max_bytes=4 * 1024)

    assert len(prepared) <= 4 * 1024
    assert resized is None
    with Image.open(BytesIO(prepared)) as result:
        assert result.format == "JPEG"
        assert result.getpixel((128, 128)) == (255, 255, 255)


def test_invalid_oversized_image_is_rejected_before_upload():
    with pytest.raises(OSError):
        prepare_qq_image(b"invalid image" * 1024, max_bytes=1024)
