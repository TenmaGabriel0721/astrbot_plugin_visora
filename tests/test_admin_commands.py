from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import astrbot.api.message_components as Comp
import pytest
from astrbot_plugin_visora.main import VisoraPlugin


def make_event(content="充值 target 5", *, admin=True, mentions=()):
    return SimpleNamespace(
        message_str=content,
        message_obj=SimpleNamespace(
            message=[Comp.Plain(text=content), *[Comp.At(qq=i) for i in mentions]]
        ),
        get_sender_id=lambda: "official-owner",
        get_self_id=lambda: "official-bot",
        is_admin=Mock(return_value=admin),
        plain_result=lambda text: text,
    )


def make_plugin():
    plugin = VisoraPlugin.__new__(VisoraPlugin)
    plugin.context = SimpleNamespace(
        get_config=Mock(return_value={"admins_id": ["onebot-owner"]})
    )
    plugin._config_get = Mock(return_value=True)
    plugin.points_name = "积分"
    plugin.data_manager = SimpleNamespace(add_points=AsyncMock(return_value=105))
    return plugin


@pytest.mark.parametrize("admin", [True, False])
def test_admin_role_uses_active_event_instead_of_default_qq_admin_list(admin):
    plugin = make_plugin()
    event = make_event(admin=admin)
    assert plugin.is_global_admin(event) is admin
    plugin.context.get_config.assert_not_called()


@pytest.mark.asyncio
async def test_non_admin_recharge_never_changes_the_balance():
    plugin = make_plugin()
    results = [r async for r in plugin.recharge_command(make_event(admin=False))]
    assert results == ["⚠️ 无权执行此操作"]
    plugin.data_manager.add_points.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("content", "mentions", "target"),
    [
        ("~充值 target 5", (), "target"),
        ("充值 5", ("onebot-user",), "onebot-user"),
        ("充值 5 <@OPENID_USER>", ("official-bot",), "OPENID_USER"),
        ("充值 <@!OPENID_USER> 5", ("qq_official",), "OPENID_USER"),
        ("充值 <@official-bot> OPENID_USER 5", ("official-bot",), "OPENID_USER"),
    ],
)
async def test_admin_recharge_parses_onebot_and_official_mentions(
    content, mentions, target
):
    plugin = make_plugin()
    results = [
        r async for r in plugin.recharge_command(make_event(content, mentions=mentions))
    ]
    assert "充值成功" in results[0]
    plugin.data_manager.add_points.assert_awaited_once_with(target, 5)


@pytest.mark.asyncio
async def test_currency_name_requires_current_event_admin_permission():
    plugin = make_plugin()
    plugin._config_set = Mock()
    results = [
        r
        async for r in plugin.currency_name_command(
            make_event("货币 金币", admin=False)
        )
    ]
    assert results == ["⚠️ 无权执行此操作"]
    plugin._config_set.assert_not_called()
