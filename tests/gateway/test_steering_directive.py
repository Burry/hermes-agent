"""Steering directives must ride the system prompt, never the user's message."""
import json
from types import SimpleNamespace

import pytest

from gateway.config import Platform
from gateway.run import GatewayRunner
from tools.steer_session_tool import steering_marker_path


@pytest.fixture
def steering_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    import hermes_constants
    monkeypatch.setattr(hermes_constants, "get_hermes_home", lambda: tmp_path)
    return tmp_path


def _write_marker(platform, chat_id, instruction):
    path = steering_marker_path(platform, chat_id)
    path.write_text(json.dumps({"instruction": instruction}), encoding="utf-8")


def _source(chat_id="any;-;+15551112222"):
    return SimpleNamespace(platform=Platform.BLUEBUBBLES, chat_id=chat_id)


def _runner():
    return GatewayRunner.__new__(GatewayRunner)


def test_no_marker_yields_no_prompt(steering_home):
    assert _runner()._steering_directive_prompt(_source()) == ""


def test_directive_is_rendered_as_operator_instruction(steering_home):
    _write_marker("bluebubbles", "any;-;+15551112222", "Reply only in haiku.")

    prompt = _runner()._steering_directive_prompt(_source())

    assert "Reply only in haiku." in prompt
    assert "OPERATOR DIRECTIVE" in prompt
    # It must instruct the model not to narrate the directive back to the user,
    # which is what produced the visible "extra reply" behaviour.
    assert "do not mention" in prompt.lower()


def test_directive_is_scoped_to_its_own_chat(steering_home):
    _write_marker("bluebubbles", "any;-;+15551112222", "Be terse.")

    other = _source(chat_id="any;-;+15559998888")

    assert _runner()._steering_directive_prompt(other) == ""


def test_blank_instruction_clears_the_directive(steering_home):
    _write_marker("bluebubbles", "any;-;+15551112222", "   ")

    assert _runner()._steering_directive_prompt(_source()) == ""


def test_missing_chat_id_is_not_an_error(steering_home):
    assert _runner()._steering_directive_prompt(
        SimpleNamespace(platform=Platform.BLUEBUBBLES, chat_id="")
    ) == ""


def test_webhook_no_longer_mutates_inbound_text():
    """The directive must not be smuggled into the user's message text."""
    import inspect
    from gateway.platforms import bluebubbles

    src = inspect.getsource(bluebubbles.BlueBubblesAdapter._handle_webhook)

    assert "read_steering_marker" not in src
    assert "OWNER DIRECTIVE" not in src
