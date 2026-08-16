"""Regression tests for channel-aware ``/model`` status output."""

import pytest

from gateway.config import ChannelOverride, GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import MessageEvent, MessageType
from gateway.run import GatewayRunner
from gateway.session import SessionSource


@pytest.mark.asyncio
async def test_model_status_reports_channel_override(monkeypatch):
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={
            Platform.BLUEBUBBLES: PlatformConfig(
                enabled=True,
                channel_overrides={
                    "group-1": ChannelOverride(
                        model="z-ai/glm-5.2",
                        provider="nous",
                    ),
                },
            ),
        },
    )
    runner._session_model_overrides = {}
    runner._adapter_for_source = lambda _source: None
    runner._normalize_source_for_session_key = lambda source: source
    runner._session_key_for_source = lambda _source: "group-session"

    monkeypatch.setattr(
        "gateway.run._load_gateway_config",
        lambda: {
            "model": {
                "default": "grok-4.20",
                "provider": "xai-oauth",
                "base_url": "https://api.x.ai/v1",
            },
        },
    )
    captured = {}

    def list_providers(**kwargs):
        captured.update(kwargs)
        return []

    monkeypatch.setattr(
        "hermes_cli.model_switch.list_authenticated_providers",
        list_providers,
    )
    event = MessageEvent(
        text="/model",
        message_type=MessageType.TEXT,
        source=SessionSource(
            platform=Platform.BLUEBUBBLES,
            chat_id="group-1",
            chat_type="group",
        ),
    )

    result = await runner._handle_model_command(event)

    assert result is not None
    assert "z-ai/glm-5.2" in result
    assert "Nous" in result
    assert captured["current_model"] == "z-ai/glm-5.2"
    assert captured["current_provider"] == "nous"
    assert captured["current_base_url"] == ""
