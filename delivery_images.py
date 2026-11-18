"""Prepare oversized generated images for QQ Official media uploads."""

from __future__ import annotations

import logging
from io import BytesIO

from PIL import Image, ImageOps

logger = logging.getLogger(__name__)

# Leave headroom for base64 expansion within the QQ upload request size limit.
QQ_IMAGE_MAX_BYTES = 4 * 1024 * 1024


def prepare_qq_image(
    image_bytes: bytes, *, max_bytes: int = QQ_IMAGE_MAX_BYTES
) -> tuple[bytes, tuple[int, int] | None]:
    """Fit an upload copy into the QQ byte budget, preserving dimensions first.

    Args:
        image_bytes: Original generated image bytes.
        max_bytes: Maximum decoded image size; base64 requests need extra headroom.

    Returns:
        Upload bytes and the new dimensions only when resizing was necessary.

    Raises:
        ValueError: The byte budget is invalid or the image cannot fit it.
        OSError: The oversized input cannot be decoded as an image.
    """
    if max_bytes < 1024:
        raise ValueError("The QQ image byte budget must be at least 1024 bytes")
    if len(image_bytes) <= max_bytes:
        return image_bytes, None

    with Image.open(BytesIO(image_bytes)) as source:
        image_format = source.format
        source.load()
        image = ImageOps.exif_transpose(source)
    original_size = image.size

    if image_format == "PNG":
        output = BytesIO()
        image.save(output, format="PNG", optimize=True)
        lossless = output.getvalue()
        if len(lossless) <= max_bytes:
            logger.info(
                "Prepared lossless QQ image: bytes=%d -> %d, dimensions=%s",
                len(image_bytes),
                len(lossless),
                original_size,
            )
            return lossless, None

    if image.mode in {"RGBA", "LA"} or "transparency" in image.info:
        rgba = image.convert("RGBA")
        image = Image.new("RGB", rgba.size, "white")
        image.paste(rgba, mask=rgba.getchannel("A"))
    else:
        image = image.convert("RGB")

    while True:
        for quality in (95, 90, 85):
            output = BytesIO()
            image.save(output, format="JPEG", quality=quality, optimize=True)
            encoded = output.getvalue()
            if len(encoded) <= max_bytes:
                logger.info(
                    "Prepared QQ image: bytes=%d -> %d, dimensions=%s -> %s, quality=%d",
                    len(image_bytes),
                    len(encoded),
                    original_size,
                    image.size,
                    quality,
                )
                return encoded, image.size if image.size != original_size else None
        width, height = image.size
        if max(width, height) <= 64:
            raise ValueError("The image cannot fit the QQ upload byte budget")
        image = image.resize(
            (max(1, int(width * 0.8)), max(1, int(height * 0.8))),
            Image.Resampling.LANCZOS,
        )
