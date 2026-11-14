from __future__ import annotations

from io import BytesIO
from unittest.mock import patch

from astrbot_plugin_visora.image_generator import ImageGenerator
from PIL import Image


def _png_bytes(width: int, height: int, *, compress_level: int = 6) -> bytes:
    image = Image.new("RGB", (width, height), color=(12, 34, 56))
    output = BytesIO()
    image.save(output, format="PNG", compress_level=compress_level)
    return output.getvalue()


def _generator() -> ImageGenerator:
    return ImageGenerator(
        api_keys=["test-key"],
        base_url="https://images.example/v1",
        model="image-model",
        api_request_formats=["v1/images/generations"],
        max_retry_attempts=1,
    )


def test_4k_request_warns_when_result_is_only_2048_square():
    image_data = _png_bytes(2048, 2048)

    with patch("astrbot_plugin_visora.image_generator.logger") as mock_logger:
        _generator()._log_result_image_metadata(
            [image_data],
            requested_size="4K",
            model="image-model",
            task_id="task-low-resolution",
        )

    mock_logger.warning.assert_called()
    warning_text = " ".join(
        str(call.args[0]) for call in mock_logger.warning.call_args_list
    )
    assert "2048x2048" in warning_text


def test_4k_request_accepts_2880_square_and_logs_metadata():
    image_data = _png_bytes(2880, 2880)

    with patch("astrbot_plugin_visora.image_generator.logger") as mock_logger:
        _generator()._log_result_image_metadata(
            [image_data],
            requested_size="4K",
            model="image-model",
            task_id="task-valid-resolution",
        )

    mock_logger.warning.assert_not_called()
    mock_logger.info.assert_called()
    info_text = " ".join(str(call.args[0]) for call in mock_logger.info.call_args_list)
    assert "4K" in info_text
    assert "2880x2880" in info_text
    assert str(len(image_data)) in info_text
