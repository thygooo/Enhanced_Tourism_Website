import json
import os
import uuid
from datetime import datetime
from pathlib import Path

from django.conf import settings
from django.core.files.storage import default_storage


DATA_FILE_NAME = "mainpage_content.json"
LOGO_DIR = "mainpage_assets/logos"
HERO_DIR = "mainpage_assets/heroes"


def _data_file_path() -> Path:
    media_root = Path(getattr(settings, "MEDIA_ROOT", "") or "")
    if not media_root:
        media_root = Path(settings.BASE_DIR) / "media"
    media_root.mkdir(parents=True, exist_ok=True)
    return media_root / DATA_FILE_NAME


def _default_state():
    return {
        "logos": [],
        "heroes": [],
    }


def load_state():
    path = _data_file_path()
    if not path.exists():
        return _default_state()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return _default_state()
        data.setdefault("logos", [])
        data.setdefault("heroes", [])
        return data
    except Exception:
        return _default_state()


def save_state(state):
    path = _data_file_path()
    path.write_text(json.dumps(state, indent=2), encoding="utf-8")


def _normalize_upload_name(original_name):
    base, ext = os.path.splitext(original_name or "")
    ext = ext.lower() if ext else ".jpg"
    return f"{uuid.uuid4().hex}{ext}"


def _to_media_url(stored_path):
    if not stored_path:
        return ""
    media_url = str(getattr(settings, "MEDIA_URL", "/media/") or "/media/")
    if not media_url.endswith("/"):
        media_url += "/"
    return f"{media_url}{stored_path}"


def _mark_only_active(rows, entry_id):
    for row in rows:
        row["is_active"] = str(row.get("id")) == str(entry_id)


def upload_logo(image_file, title="", set_active=False):
    state = load_state()
    file_name = _normalize_upload_name(getattr(image_file, "name", "logo.jpg"))
    stored_path = default_storage.save(f"{LOGO_DIR}/{file_name}", image_file)
    row = {
        "id": uuid.uuid4().hex,
        "title": str(title or "Website Logo").strip() or "Website Logo",
        "path": stored_path,
        "is_active": bool(set_active),
        "created_at": datetime.utcnow().isoformat(),
    }
    state["logos"].append(row)
    if row["is_active"] or len(state["logos"]) == 1:
        _mark_only_active(state["logos"], row["id"])
    save_state(state)
    return row


def upload_hero(image_file, title="", display_order=1, set_active=False):
    state = load_state()
    file_name = _normalize_upload_name(getattr(image_file, "name", "hero.jpg"))
    stored_path = default_storage.save(f"{HERO_DIR}/{file_name}", image_file)
    try:
        order_num = int(display_order)
    except (TypeError, ValueError):
        order_num = 1
    row = {
        "id": uuid.uuid4().hex,
        "title": str(title or "Hero Image").strip() or "Hero Image",
        "display_order": max(order_num, 1),
        "path": stored_path,
        "is_active": bool(set_active),
        "created_at": datetime.utcnow().isoformat(),
    }
    state["heroes"].append(row)
    if row["is_active"] or len(state["heroes"]) == 1:
        _mark_only_active(state["heroes"], row["id"])
    save_state(state)
    return row


def set_active_logo(entry_id):
    state = load_state()
    _mark_only_active(state["logos"], entry_id)
    save_state(state)


def set_active_hero(entry_id):
    state = load_state()
    _mark_only_active(state["heroes"], entry_id)
    save_state(state)


def _remove_file_if_exists(path_value):
    if not path_value:
        return
    try:
        if default_storage.exists(path_value):
            default_storage.delete(path_value)
    except Exception:
        pass


def delete_logo(entry_id):
    state = load_state()
    logos = state.get("logos", [])
    keep = []
    removed_active = False
    for row in logos:
        if str(row.get("id")) == str(entry_id):
            removed_active = bool(row.get("is_active"))
            _remove_file_if_exists(row.get("path"))
            continue
        keep.append(row)
    state["logos"] = keep
    if removed_active and keep:
        keep[0]["is_active"] = True
        for row in keep[1:]:
            row["is_active"] = False
    save_state(state)


def delete_hero(entry_id):
    state = load_state()
    heroes = state.get("heroes", [])
    keep = []
    removed_active = False
    for row in heroes:
        if str(row.get("id")) == str(entry_id):
            removed_active = bool(row.get("is_active"))
            _remove_file_if_exists(row.get("path"))
            continue
        keep.append(row)
    state["heroes"] = keep
    if removed_active and keep:
        keep[0]["is_active"] = True
        for row in keep[1:]:
            row["is_active"] = False
    save_state(state)


def get_admin_context():
    state = load_state()
    logos = state.get("logos", [])
    heroes = sorted(state.get("heroes", []), key=lambda r: (int(r.get("display_order", 9999)), r.get("created_at", "")))
    for row in logos:
        row["url"] = _to_media_url(row.get("path"))
    for row in heroes:
        row["url"] = _to_media_url(row.get("path"))
    return {
        "logos": logos,
        "heroes": heroes,
    }


def get_public_assets():
    state = load_state()
    logos = state.get("logos", [])
    heroes = state.get("heroes", [])

    active_logo = next((r for r in logos if r.get("is_active")), None)
    if active_logo is None and logos:
        active_logo = logos[0]

    # Use active hero first, then ordered heroes list.
    active_hero = next((r for r in heroes if r.get("is_active")), None)
    ordered_heroes = sorted(heroes, key=lambda r: (int(r.get("display_order", 9999)), r.get("created_at", "")))

    hero_urls = []
    if active_hero:
        hero_urls.append(_to_media_url(active_hero.get("path")))
    for row in ordered_heroes:
        url = _to_media_url(row.get("path"))
        if url and url not in hero_urls:
            hero_urls.append(url)

    return {
        "active_logo_url": _to_media_url(active_logo.get("path")) if active_logo else "",
        "hero_urls": [u for u in hero_urls if u],
    }

