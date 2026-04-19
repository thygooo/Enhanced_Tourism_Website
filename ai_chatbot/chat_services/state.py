import time


def _to_int(value, default=0):
    try:
        if value in ("", None):
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def load_chat_state(request, *, session_key, ttl_seconds):
    session = getattr(request, "session", None)
    if session is None:
        return {}

    state = session.get(session_key, {})
    if not isinstance(state, dict):
        session.pop(session_key, None)
        return {}

    last_updated_epoch = _to_int(state.get("last_updated_epoch"), default=0)
    if last_updated_epoch <= 0:
        session.pop(session_key, None)
        return {}

    if int(time.time()) - last_updated_epoch > _to_int(ttl_seconds, default=0):
        session.pop(session_key, None)
        return {}

    return state


def save_chat_state(request, state, *, session_key):
    session = getattr(request, "session", None)
    if session is None:
        return

    payload = state if isinstance(state, dict) else {}
    payload["last_updated_epoch"] = int(time.time())
    session[session_key] = payload
    session.modified = True


def clear_chat_state(request, *, session_key):
    session = getattr(request, "session", None)
    if session is None:
        return
    if session_key in session:
        session.pop(session_key, None)
        session.modified = True

