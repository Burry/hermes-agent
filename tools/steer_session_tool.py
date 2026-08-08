"""Steer Session Tool -- inject an owner directive into another session's next replies.

Lets a "default"-profile session leave a standing instruction for a specific
BlueBubbles conversation (looked up by session_id, e.g. a public-profile
friend DM) that gets prepended to that conversation's inbound messages as a
clearly-delimited owner directive, until cleared or replaced. There is
deliberately no live cross-session/cross-process hook here: the marker is a
small JSON file, written by this tool and read by
gateway/platforms/bluebubbles.py's webhook handler on each inbound message
for that chat -- filesystem-mediated, so it works regardless of whether the
target profile happens to be running in this process or a separate one.

Only the default profile's own physical BlueBubbles connection exists (other
profiles are reached via post-receipt routing, not their own adapter), so
markers are always stored under the default profile's home
(HERMES_HOME/steering/), keyed by a hash of (platform, chat_id) -- never by
session_id, since a session can end and a fresh one begin for the same
ongoing conversation.
"""

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def _steering_dir() -> Path:
    from hermes_constants import get_hermes_home

    path = get_hermes_home() / "steering"
    path.mkdir(parents=True, exist_ok=True)
    return path


def steering_marker_path(platform: str, chat_id: str) -> Path:
    """Return the marker path for a given platform + chat_id.

    Shared between the write side (this tool) and the read side
    (gateway/platforms/bluebubbles.py's webhook handler) -- both must derive
    the same path from the same inputs.
    """
    key = f"{platform}:{chat_id}".encode("utf-8")
    digest = hashlib.sha256(key).hexdigest()[:24]
    return _steering_dir() / f"{platform}__{digest}.json"


def read_steering_marker(platform: str, chat_id: str) -> Optional[dict]:
    """Return the active steering marker for (platform, chat_id), or None."""
    path = steering_marker_path(platform, chat_id)
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        logger.warning("steering marker at %s is unreadable, ignoring", path, exc_info=True)
        return None
    instruction = data.get("instruction")
    if not isinstance(instruction, str) or not instruction.strip():
        return None
    return data


def _resolve_session(session_id: str, profile: Optional[str]):
    """Resolve session_id -> (chat_id, platform, profile_name), or raise ValueError."""
    from tools.session_search_tool import _resolve_profile_db, _locate_session_db

    db = None
    resolved_profile = profile
    try:
        if profile:
            db = _resolve_profile_db(profile)
        if db is not None:
            meta = db.get_session(session_id)
            if not meta:
                db.close()
                db, resolved_profile = _locate_session_db(session_id)
        else:
            db, resolved_profile = _locate_session_db(session_id)
            if db is None:
                from hermes_state import SessionDB
                db = SessionDB()
                resolved_profile = "default"

        meta = db.get_session(session_id) if db is not None else None
        if not meta:
            raise ValueError(f"session_id not found in any profile: {session_id}")
        chat_id = meta.get("chat_id")
        platform = meta.get("source")
        if not chat_id or not platform:
            raise ValueError(f"session {session_id} has no chat_id/platform on record")
        # A session's own profile_name column is authoritative over which
        # profile's DB it was physically found in -- session storage isn't
        # guaranteed to be per-profile-directory (e.g. a multiplexed gateway's
        # messaging-platform sessions can all live in the default profile's
        # DB, tagged by profile_name, rather than split across directories).
        return chat_id, platform, meta.get("profile_name") or resolved_profile or "default"
    finally:
        if db is not None:
            db.close()


def _check_steer_session():
    from hermes_cli.profiles import get_active_profile_name

    return get_active_profile_name() == "default"


STEER_SESSION_SCHEMA = {
    "name": "steer_session",
    "description": (
        "Leave (or clear) a standing directive for another BlueBubbles conversation, most "
        "commonly a friend's session on the public profile. The directive is installed in "
        "that conversation's system prompt as an operator instruction and applies on every "
        "future turn, until you clear it or set a new one -- it is NOT a one-time nudge. "
        "Best suited to persistent tone, persona, and topic guidance (e.g. 'reply in "
        "Spanish', 'keep answers to two sentences', 'never discuss my work schedule'). "
        "Use session_search first to find the session_id you want to steer. Only available "
        "from the default profile."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "session_id": {
                "type": "string",
                "description": "The target session's id, from session_search (e.g. against profile='public').",
            },
            "instruction": {
                "type": "string",
                "description": (
                    "The directive to install for that conversation going forward, written "
                    "as an instruction to the assistant (e.g. 'Be dry and sarcastic; skip "
                    "pleasantries'). Pass an empty string to clear any standing directive "
                    "for that conversation."
                ),
            },
            "profile": {
                "type": "string",
                "description": "Profile the session_id belongs to (e.g. 'public'). Omit to search every profile.",
            },
        },
        "required": ["session_id", "instruction"],
    },
}


def steer_session_tool(args, **kw):
    session_id = (args.get("session_id") or "").strip()
    instruction = args.get("instruction")
    profile = (args.get("profile") or "").strip() or None

    if not session_id:
        return json.dumps({"error": "session_id is required"})
    if instruction is None:
        return json.dumps({"error": "instruction is required (pass '' to clear)"})

    try:
        chat_id, platform, resolved_profile = _resolve_session(session_id, profile)
    except ValueError as e:
        return json.dumps({"error": str(e)})

    path = steering_marker_path(platform, chat_id)
    instruction = instruction.strip()

    if not instruction:
        if path.is_file():
            path.unlink()
            return json.dumps({
                "success": True,
                "cleared": True,
                "session_id": session_id,
                "profile": resolved_profile,
            })
        return json.dumps({
            "success": True,
            "cleared": True,
            "note": "no standing directive was set",
            "session_id": session_id,
            "profile": resolved_profile,
        })

    from gateway.session_context import get_session_env

    marker = {
        "instruction": instruction,
        "set_at": time.time(),
        "set_by_session": get_session_env("HERMES_SESSION_ID", "") or None,
        "target_session_id": session_id,
        "target_profile": resolved_profile,
        "platform": platform,
    }
    path.write_text(json.dumps(marker, ensure_ascii=False, indent=2), encoding="utf-8")

    return json.dumps({
        "success": True,
        "session_id": session_id,
        "profile": resolved_profile,
        "platform": platform,
        "note": "Directive will be prepended to this conversation's inbound messages until cleared or replaced.",
    })


# --- Registry ---
from tools.registry import registry

registry.register(
    name="steer_session",
    toolset="messaging",
    schema=STEER_SESSION_SCHEMA,
    handler=steer_session_tool,
    check_fn=_check_steer_session,
    emoji="🕹️",
)
