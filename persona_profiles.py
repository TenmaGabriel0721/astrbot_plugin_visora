"""Bot-bound persona profiles and persistent identity reference images."""

from __future__ import annotations

import asyncio
import copy
import json
import re
import shutil
import uuid
from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image

DEFAULT_PERSONA_IDENTITY = ""
PROFILE_STYLES = {"photo", "anime", "follow"}
PROFILE_RESOLUTIONS = {"inherit", "1K", "2K", "4K"}
REFERENCE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}
MAX_IDENTITY_REFERENCES = 4


class PersonaProfileStore:
    """Keep identity files separate for each configured Bot profile."""

    def __init__(self, plugin: Any, data_dir: Path, legacy_dir: Path) -> None:
        """Create a store without touching existing identity files.

        Args:
            plugin: Plugin exposing configuration access and persistence.
            data_dir: Persistent plugin data directory.
            legacy_dir: Previous shared reference image directory.
        """
        self.plugin = plugin
        self.root = Path(data_dir) / "persona_profiles"
        self.legacy_dir = Path(legacy_dir)
        self.migration_file = self.root / ".legacy-import.json"
        self._locks: dict[str, asyncio.Lock] = {}
        self._migration_lock = asyncio.Lock()

    @staticmethod
    def normalize_profiles(raw: Any) -> list[dict[str, str]]:
        """Validate profile identifiers, bindings, styles, and text.

        Args:
            raw: Configuration value containing profile entries.

        Returns:
            Detached normalized entries with stable identifiers.

        Raises:
            ValueError: Profiles contain duplicate bindings or invalid fields.
        """
        if not isinstance(raw, list):
            raise ValueError("自拍资料必须是列表")
        if len(raw) > 32:
            raise ValueError("自拍资料最多保留 32 套")
        result = []
        identifiers: set[str] = set()
        bindings: set[str] = set()
        for index, entry in enumerate(raw):
            if not isinstance(entry, dict):
                raise ValueError("自拍资料必须是对象")
            profile_id = str(
                entry.get("id")
                or uuid.uuid5(
                    uuid.NAMESPACE_URL,
                    f"visora-persona:{index}:{entry.get('bot_id', '')}:{entry.get('name', '')}",
                ).hex
            ).strip()
            if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", profile_id):
                raise ValueError("自拍资料 ID 不合法")
            if profile_id in identifiers:
                raise ValueError("自拍资料 ID 不能重复")
            identifiers.add(profile_id)
            bot_id = str(entry.get("bot_id") or "").strip()
            if len(bot_id) > 128:
                raise ValueError("Bot ID 过长")
            if bot_id and bot_id in bindings:
                raise ValueError("每个 Bot 只能绑定一套自拍资料")
            if bot_id:
                bindings.add(bot_id)
            name = str(entry.get("name") or "自拍资料").strip()
            identity = str(entry.get("identity_prompt") or "").strip()
            style = str(entry.get("style") or "photo").strip()
            resolution = str(entry.get("resolution") or "inherit").strip()
            if not name or len(name) > 80 or len(identity) > 4096:
                raise ValueError("自拍资料名称或身份描述长度不合法")
            if style not in PROFILE_STYLES or resolution not in PROFILE_RESOLUTIONS:
                raise ValueError("自拍画风或分辨率不合法")
            result.append(
                {
                    "__template_key": "profile",
                    "id": profile_id,
                    "name": name,
                    "bot_id": bot_id,
                    "identity_prompt": identity,
                    "style": style,
                    "resolution": resolution,
                }
            )
        return result

    def profiles(self) -> list[dict[str, str]]:
        """Return validated snapshots of the current profile configuration.

        Returns:
            Detached profile dictionaries.
        """
        return self.normalize_profiles(self.plugin._config_get("persona_profiles", []))

    def resolve(self, event: Any) -> dict[str, str] | None:
        """Select identity by the receiving Bot rather than an LLM argument.

        Args:
            event: Message event carrying the platform instance ID.

        Returns:
            The profile bound to this Bot, or None when it is not configured.
        """
        getter = getattr(event, "get_platform_id", None)
        bot_id = str(getter() or "") if callable(getter) else ""
        if not bot_id:
            bot_id = str(getattr(event, "unified_msg_origin", "") or "").split(":", 1)[
                0
            ]
        for profile in self.profiles():
            if profile["bot_id"] and profile["bot_id"] == bot_id:
                return copy.deepcopy(profile)
        return None

    def get(self, profile_id: str) -> dict[str, str]:
        """Look up a configured profile by its internal identifier.

        Args:
            profile_id: Stable profile ID.

        Returns:
            A detached profile.

        Raises:
            ValueError: The profile does not exist.
        """
        for profile in self.profiles():
            if profile["id"] == profile_id:
                return profile
        raise ValueError("自拍资料不存在，请刷新页面")

    def reference_dir(self, profile_id: str) -> Path:
        """Resolve a reference directory within persistent plugin storage.

        Args:
            profile_id: Stable profile ID.

        Returns:
            The profile's reference image directory.

        Raises:
            ValueError: The ID or resolved path escapes the store.
        """
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", profile_id):
            raise ValueError("自拍资料 ID 不合法")
        directory = self.root / profile_id
        if directory.is_symlink() or not directory.resolve().is_relative_to(
            self.root.resolve()
        ):
            raise ValueError("自拍资料路径不合法")
        return directory

    def reference_files(self, profile_id: str) -> list[Path]:
        """List regular identity files in a stable order.

        Args:
            profile_id: Stable profile ID.

        Returns:
            Sorted supported files, excluding symlinks.
        """
        directory = self.reference_dir(profile_id)
        if not directory.exists():
            return []
        return sorted(
            (
                path
                for path in directory.iterdir()
                if path.is_file()
                and not path.is_symlink()
                and path.suffix.lower() in REFERENCE_SUFFIXES
            ),
            key=lambda path: path.name.lower(),
        )

    async def migrate_legacy(self) -> None:
        """Import the former shared identity into one explicit legacy profile.

        Existing files and newer persistent profiles are retained. An unbound
        profile is created when ownership has not yet been configured.

        Raises:
            OSError: Reference files or configuration cannot be persisted.
        """
        async with self._migration_lock:
            if self.migration_file.exists():
                return
            files = sorted(
                path
                for path in self.legacy_dir.glob("*")
                if path.is_file()
                and not path.is_symlink()
                and path.suffix.lower() in REFERENCE_SUFFIXES
            )
            if not files:
                return
            profiles = self.profiles()
            legacy = next(
                (entry for entry in profiles if entry["id"] == "legacy"), None
            )
            if legacy is None:
                legacy = {
                    "__template_key": "profile",
                    "id": "legacy",
                    "name": "原有自拍资料",
                    "bot_id": "",
                    "style": "photo",
                    "identity_prompt": DEFAULT_PERSONA_IDENTITY,
                    "resolution": "inherit",
                }
                profiles.append(legacy)

            def copy_files() -> None:
                destination = self.reference_dir(legacy["id"])
                destination.mkdir(mode=0o700, parents=True, exist_ok=True)
                for source in files:
                    target = destination / source.name
                    if not target.exists():
                        shutil.copy2(source, target)
                        target.chmod(0o600)

            await asyncio.to_thread(copy_files)
            previous = copy.deepcopy(self.plugin._config_get("persona_profiles", []))
            self.plugin._config_set("persona_profiles", profiles)
            try:
                await asyncio.to_thread(self.plugin.config.save_config)
            except Exception:
                self.plugin._config_set("persona_profiles", previous)
                raise
            await asyncio.to_thread(
                self.migration_file.write_text,
                json.dumps({"version": 1, "profile_id": "legacy"}, ensure_ascii=False),
                encoding="utf-8",
            )

    async def load_references(
        self, profile: dict[str, str], *, limit: int, max_bytes: int
    ) -> list[tuple[bytes, str]]:
        """Read bounded identity images under the selected profile's lock.

        Args:
            profile: Previously resolved identity snapshot.
            limit: Maximum identity references needed by the current task.
            max_bytes: Maximum bytes per reference.

        Returns:
            Validated image bytes and their actual MIME types.
        """
        profile_id = profile["id"]
        async with self._locks.setdefault(profile_id, asyncio.Lock()):

            def read() -> list[tuple[bytes, str]]:
                result = []
                for path in self.reference_files(profile_id):
                    if len(result) >= min(limit, MAX_IDENTITY_REFERENCES):
                        break
                    try:
                        if path.stat().st_size > max_bytes:
                            continue
                        data = path.read_bytes()
                        if not data or len(data) > max_bytes:
                            continue
                        with Image.open(BytesIO(data)) as image:
                            image.verify()
                            mime = Image.MIME.get(image.format)
                        if mime in {"image/png", "image/jpeg", "image/webp"}:
                            result.append((data, mime))
                    except (OSError, ValueError, Image.DecompressionBombError):
                        continue
                return result

            return await asyncio.to_thread(read)

    async def add_references(
        self, profile_id: str, images: list[tuple[bytes, str]], *, max_bytes: int
    ) -> list[str]:
        """Validate and append a bounded batch of identity images.

        Args:
            profile_id: Configured profile receiving the images.
            images: Raw uploaded images and declared MIME types.
            max_bytes: Maximum bytes per image.

        Returns:
            Names of the files added.

        Raises:
            ValueError: Images are invalid or exceed the profile's limits.
            OSError: Storage cannot be written.
        """
        self.get(profile_id)
        if not images or len(images) > MAX_IDENTITY_REFERENCES:
            raise ValueError("每套自拍资料最多上传 4 张参考图")
        async with self._locks.setdefault(profile_id, asyncio.Lock()):

            def save() -> list[str]:
                if (
                    len(self.reference_files(profile_id)) + len(images)
                    > MAX_IDENTITY_REFERENCES
                ):
                    raise ValueError(
                        "每套自拍资料最多保留 4 张参考图，请先删除不需要的图片"
                    )
                validated = []
                for data, _ in images:
                    if not data or len(data) > max_bytes:
                        raise ValueError("参考图为空或超过图片大小限制")
                    try:
                        with Image.open(BytesIO(data)) as image:
                            image.verify()
                            image_format = image.format
                    except (OSError, ValueError, Image.DecompressionBombError) as exc:
                        raise ValueError("参考图不是有效的 PNG/JPEG/WebP 图片") from exc
                    extension = {"PNG": ".png", "JPEG": ".jpg", "WEBP": ".webp"}.get(
                        image_format
                    )
                    if not extension:
                        raise ValueError("参考图仅支持 PNG/JPEG/WebP")
                    validated.append((data, extension))
                directory = self.reference_dir(profile_id)
                directory.mkdir(mode=0o700, parents=True, exist_ok=True)
                added: list[Path] = []
                try:
                    for data, extension in validated:
                        path = directory / f"ref_{uuid.uuid4().hex[:12]}{extension}"
                        with path.open("xb") as output:
                            added.append(path)
                            path.chmod(0o600)
                            output.write(data)
                except BaseException:
                    for path in added:
                        path.unlink(missing_ok=True)
                    raise
                return [path.name for path in added]

            return await asyncio.to_thread(save)

    async def delete_references(self, profile_id: str, names: list[str]) -> int:
        """Delete only explicitly selected files belonging to one profile.

        Args:
            profile_id: Configured profile owning the files.
            names: Existing basenames selected by an administrator.

        Returns:
            Number of deleted files.

        Raises:
            ValueError: A requested file does not belong to the profile.
        """
        self.get(profile_id)
        async with self._locks.setdefault(profile_id, asyncio.Lock()):

            def delete() -> int:
                existing = {
                    path.name: path for path in self.reference_files(profile_id)
                }
                if not names or any(name not in existing for name in names):
                    raise ValueError("参考图不存在或不属于当前自拍资料")
                for name in set(names):
                    existing[name].unlink()
                return len(set(names))

            return await asyncio.to_thread(delete)
