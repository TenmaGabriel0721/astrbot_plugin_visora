"""Unified reference-image selection for commands and LLM tools."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import astrbot.api.message_components as Comp
from astrbot.api import logger
from astrbot.core.utils.quoted_message.extractor import extract_quoted_message_images


@dataclass(frozen=True, slots=True)
class ReferenceSelection:
    sources: tuple[str, ...] = ()
    source_type: str = "none"
    requested: bool = False
    reference_count: int = 0
    avatar_user_ids: tuple[str, ...] = ()
    source_groups: tuple[tuple[str, ...], ...] = ()

    @property
    def candidate_groups(self) -> tuple[tuple[str, ...], ...]:
        """Candidate sources grouped by logical image, with legacy fallback."""
        if self.source_groups:
            return self.source_groups
        return tuple((source,) for source in self.sources)


def _dedupe_strings(values: Iterable[Any], limit: int | None = None) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        if not isinstance(value, str):
            continue
        cleaned = value.strip()
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        result.append(cleaned)
        if limit is not None and len(result) >= limit:
            break
    return result


def get_image_component_sources(component: Any) -> list[str]:
    """Return every usable field instead of locking onto one stale path."""
    return _dedupe_strings(
        getattr(component, attr, None) for attr in ("url", "path", "file")
    )


def get_raw_onebot_image_sources(event: Any) -> tuple[list[str], int]:
    groups = get_raw_onebot_image_source_groups(event)
    return _dedupe_strings(source for group in groups for source in group), len(groups)


def get_raw_onebot_image_source_groups(event: Any) -> list[tuple[str, ...]]:
    """Return one candidate-source group for each raw OneBot image segment."""
    raw_message = getattr(getattr(event, "message_obj", None), "raw_message", None)
    segments = getattr(raw_message, "message", None)
    if segments is None and isinstance(raw_message, dict):
        segments = raw_message.get("message")
    if not isinstance(segments, list):
        return []

    groups: list[tuple[str, ...]] = []
    for segment in segments:
        if not isinstance(segment, dict) or segment.get("type") != "image":
            continue
        data = segment.get("data")
        if not isinstance(data, dict):
            groups.append(())
            continue
        groups.append(
            tuple(_dedupe_strings(data.get(key) for key in ("url", "path", "file")))
        )
    return groups


def _get_self_id(event: Any) -> str:
    try:
        value = event.get_self_id()
        if value is not None:
            return str(value).strip()
    except Exception:
        pass
    value = getattr(getattr(event, "message_obj", None), "self_id", "")
    return str(value or "").strip()


def sanitize_avatar_user_ids(
    values: Iterable[Any],
    *,
    self_id: str = "",
    limit: int = 4,
) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        cleaned = str(value or "").strip()
        if (
            not cleaned
            or cleaned == "all"
            or not cleaned.isdigit()
            or cleaned == self_id
            or cleaned in seen
        ):
            continue
        seen.add(cleaned)
        result.append(cleaned)
        if len(result) >= limit:
            break
    return result


def selection_from_sources(
    sources: Iterable[Any],
    *,
    max_images: int,
    source_type: str = "explicit",
) -> ReferenceSelection:
    normalized = _dedupe_strings(sources, max_images * 3)
    selected = normalized[:max_images]
    return ReferenceSelection(
        sources=tuple(selected),
        source_groups=tuple((source,) for source in selected),
        source_type=source_type if selected else "none",
        requested=bool(selected),
        reference_count=len(selected),
    )


async def collect_reference_selection(
    event: Any,
    *,
    explicit_user_ids: Iterable[Any] | None = None,
    max_images: int = 4,
    max_avatars: int = 4,
    include_sender_avatar: bool = False,
    include_avatars: bool = True,
) -> ReferenceSelection:
    """Select one source class using message > quote > avatar priority."""
    max_images = min(max(1, int(max_images or 4)), 10)
    max_avatars = min(max(1, int(max_avatars or 4)), 10)
    components = list(getattr(getattr(event, "message_obj", None), "message", []) or [])

    current_components = [item for item in components if isinstance(item, Comp.Image)]
    component_groups = [
        tuple(get_image_component_sources(component))
        for component in current_components
    ]
    raw_groups = get_raw_onebot_image_source_groups(event)
    raw_image_count = len(raw_groups)
    if current_components or raw_image_count:
        logical_count = min(max(len(component_groups), raw_image_count), max_images)
        current_groups: list[tuple[str, ...]] = []
        for index in range(logical_count):
            candidates = [
                *(component_groups[index] if index < len(component_groups) else ()),
                *(raw_groups[index] if index < len(raw_groups) else ()),
            ]
            current_groups.append(tuple(_dedupe_strings(candidates)))
        normalized_current = _dedupe_strings(
            source for group in current_groups for source in group
        )
        return ReferenceSelection(
            sources=tuple(normalized_current),
            source_groups=tuple(current_groups),
            source_type="message",
            requested=True,
            reference_count=logical_count,
        )

    reply_components = [item for item in components if isinstance(item, Comp.Reply)]
    if reply_components:
        chain_groups: list[tuple[str, ...]] = []
        for reply in reply_components:
            for component in getattr(reply, "chain", None) or []:
                if isinstance(component, Comp.Image):
                    chain_groups.append(tuple(get_image_component_sources(component)))

        quoted_sources: list[str] = []
        try:
            quoted_sources = list(await extract_quoted_message_images(event) or [])
        except Exception as exc:
            logger.debug(
                f"[Visora] Failed to parse quoted images; falling back to Reply.chain: {exc}"
            )

        if quoted_sources:
            normalized_quote = _dedupe_strings(quoted_sources)
            if chain_groups:
                # The core preprocessor turns Reply.chain images into local JPEGs,
                # while get_msg commonly resolves the same images to remote URLs.
                # Pair the first remote-only refs with chain images by message order,
                # then keep any remaining refs as additional logical images.
                chain_source_set = {
                    source for group in chain_groups for source in group
                }
                remote_only = [
                    source
                    for source in normalized_quote
                    if source not in chain_source_set
                ]
                merged_groups = [list(group) for group in chain_groups]
                pair_count = min(len(merged_groups), len(remote_only))
                for index in range(pair_count):
                    merged_groups[index].append(remote_only[index])
                final_groups = [
                    tuple(_dedupe_strings(group)) for group in merged_groups
                ]
                final_groups.extend((source,) for source in remote_only[pair_count:])
                final_groups = final_groups[:max_images]
                flattened = _dedupe_strings(
                    source for group in final_groups for source in group
                )
                return ReferenceSelection(
                    sources=tuple(flattened),
                    source_groups=tuple(final_groups),
                    source_type="quoted",
                    requested=True,
                    reference_count=len(final_groups),
                )

            normalized_quote = normalized_quote[:max_images]
            return ReferenceSelection(
                sources=tuple(normalized_quote),
                source_groups=tuple((source,) for source in normalized_quote),
                source_type="quoted",
                requested=True,
                reference_count=len(normalized_quote),
            )

        if chain_groups:
            chain_groups = chain_groups[:max_images]
            normalized_chain = _dedupe_strings(
                source for group in chain_groups for source in group
            )
            return ReferenceSelection(
                sources=tuple(normalized_chain),
                source_groups=tuple(chain_groups),
                source_type="quoted",
                requested=True,
                reference_count=len(chain_groups),
            )

    if not include_avatars:
        return ReferenceSelection()

    self_id = _get_self_id(event)
    if explicit_user_ids is None:
        normalized_explicit_ids: list[Any] = []
    elif isinstance(explicit_user_ids, (str, int)):
        normalized_explicit_ids = [explicit_user_ids]
    else:
        normalized_explicit_ids = list(explicit_user_ids)

    mention_ids = [item.qq for item in components if isinstance(item, Comp.At)]
    avatar_ids = sanitize_avatar_user_ids(
        [*normalized_explicit_ids, *mention_ids],
        self_id=self_id,
        limit=min(max_images, max_avatars),
    )

    if not avatar_ids and include_sender_avatar:
        try:
            sender_id = event.get_sender_id()
        except Exception:
            sender_id = getattr(
                getattr(getattr(event, "message_obj", None), "sender", None),
                "user_id",
                "",
            )
        avatar_ids = sanitize_avatar_user_ids([sender_id], self_id=self_id, limit=1)

    if avatar_ids:
        return ReferenceSelection(
            sources=tuple(
                f"https://q4.qlogo.cn/headimg_dl?dst_uin={user_id}&spec=640"
                for user_id in avatar_ids
            ),
            source_groups=tuple(
                (f"https://q4.qlogo.cn/headimg_dl?dst_uin={user_id}&spec=640",)
                for user_id in avatar_ids
            ),
            source_type="avatar",
            requested=True,
            reference_count=len(avatar_ids),
            avatar_user_ids=tuple(avatar_ids),
        )

    return ReferenceSelection()
