from __future__ import annotations

import os
import pickle
import logging
from datetime import date, datetime
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Iterable, List, Optional

from django.conf import settings
from django.utils import timezone
from django.db.models import F, Q

try:
    import pandas as pd
except ModuleNotFoundError:
    pd = None

from admin_app.models import Accomodation, Room
from tour_app.models import Tour_Schedule

_DECISION_TREE_MODEL_CACHE = None
_DECISION_TREE_MODEL_PATH_CACHE = None
_DECISION_TREE_MODEL_SOURCE_CACHE = "unknown"
_DECISION_TREE_EXPECTED_PARAMS = {
    "criterion": "entropy",
    "max_depth": 8,
    "min_samples_leaf": 5,
    "min_samples_split": 30,
    "max_features": None,
}
logger = logging.getLogger(__name__)

_TOUR_PREFERENCE_ALIASES = {
    "sea": ("sea", "beach", "coastal", "ocean", "island", "shore", "seaside"),
    "nature": ("nature", "natural", "falls", "waterfall", "forest", "mountain", "river", "scenic"),
    "culture": ("culture", "cultural", "heritage", "food", "culinary", "tradition", "museum"),
    "city": ("city", "urban", "proper", "downtown", "plaza"),
    "highlights": ("highlight", "highlights", "day tour", "must-see"),
    "adventure": ("adventure", "hike", "trek", "trail", "outdoor", "camp"),
    "family": ("family", "kids", "child", "children", "friendly"),
    "river": ("river", "riverside"),
}


@dataclass
class RecommendationResult:
    title: str
    subtitle: str
    score: float
    meta: dict


def _to_int(value, default=0):
    try:
        if value in ("", None):
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _to_decimal(value, default=Decimal("0")):
    try:
        if value in ("", None):
            return default
        return Decimal(str(value))
    except Exception:
        return default


def _normalize(value: float, min_value: float, max_value: float) -> float:
    if max_value <= min_value:
        return 0.0
    return max(0.0, min(1.0, (value - min_value) / (max_value - min_value)))


def _cnn_score(features: List[float]) -> float:
    """
    Lightweight 1D convolution-style heuristic scorer with a fixed kernel.
    This is intentionally simple and dependency-free, and is not a trained CNN model.
    """
    if not features:
        return 0.0
    if len(features) < 3:
        return sum(features) / len(features)

    kernel = [0.25, 0.5, 0.25]
    conv = []
    for i in range(len(features) - 2):
        window = features[i:i + 3]
        conv.append(sum(w * k for w, k in zip(window, kernel)))
    return sum(conv) / len(conv)


def _decision_tree_penalty(conditions: Iterable[bool]) -> float:
    """
    Simple Decision Tree proxy: penalize failing hard constraints.
    """
    if any(conditions):
        return -10.0
    return 0.0


def _extract_tour_preference_tags_from_text(text: str) -> set[str]:
    lowered = str(text or "").strip().lower()
    if not lowered:
        return set()
    detected = set()
    for canonical, markers in _TOUR_PREFERENCE_ALIASES.items():
        if any(marker in lowered for marker in markers):
            detected.add(canonical)
    return detected


def _collect_requested_tour_preferences(params: dict) -> set[str]:
    requested = set()
    for key in ("preference", "tour_type", "preferred_type", "interest"):
        value = str(params.get(key) or "").strip().lower()
        if value:
            requested.update(_extract_tour_preference_tags_from_text(value))

    tags_value = params.get("preference_tags")
    if isinstance(tags_value, list):
        for item in tags_value:
            tag = str(item or "").strip().lower()
            if tag:
                requested.update(_extract_tour_preference_tags_from_text(tag))

    # Keep only canonical tags + known explicit values derived above.
    return {tag for tag in requested if tag}


def _tokenize_preference_phrase(text: str) -> set[str]:
    raw = str(text or "").strip().lower()
    if not raw:
        return set()
    stopwords = {
        "i", "we", "prefer", "want", "like", "tour", "tours", "package", "packages",
        "day", "trip", "please", "show", "me", "a", "an", "the", "in", "near",
        "and", "or", "how", "about", "what", "right", "now", "there", "any", "are",
    }
    tokens = set()
    for token in raw.replace("/", " ").replace("-", " ").split():
        cleaned = "".join(ch for ch in token if ch.isalnum())
        if len(cleaned) >= 3 and cleaned not in stopwords:
            tokens.add(cleaned)
    return tokens


def get_unavailable_tour_matches(params: dict, limit: int = 3) -> list[str]:
    """
    Return matching published tour titles that currently have no upcoming schedules.
    This supports UX messaging when a preference appears valid but no future run exists.
    """
    now = timezone.now()
    requested_preferences = _collect_requested_tour_preferences(params if isinstance(params, dict) else {})
    requested_preference_text = str(
        (params or {}).get("preference_text")
        or (params or {}).get("preference")
        or (params or {}).get("tour_type")
        or (params or {}).get("preferred_type")
        or (params or {}).get("interest")
        or ""
    ).strip().lower()
    requested_preference_tokens = _tokenize_preference_phrase(requested_preference_text)
    if not requested_preferences and not requested_preference_tokens:
        return []

    schedules = (
        Tour_Schedule.objects.select_related("tour")
        .filter(tour__publication_status="published")
        .exclude(status="cancelled")
    )
    tour_availability: dict[str, bool] = {}
    tour_objects: dict[str, object] = {}
    for schedule in schedules:
        tour_key = str(schedule.tour_id)
        tour_objects[tour_key] = schedule.tour
        has_future = bool(schedule.end_time and schedule.end_time >= now)
        tour_availability[tour_key] = bool(tour_availability.get(tour_key)) or has_future

    scored = []
    for tour_key, tour_obj in tour_objects.items():
        if tour_availability.get(tour_key):
            continue
        name = str(getattr(tour_obj, "tour_name", "") or "").lower()
        desc = str(getattr(tour_obj, "description", "") or "").lower()
        detected_tour_tags = _extract_tour_preference_tags_from_text(f"{name} {desc}")
        tour_tokens = _tokenize_preference_phrase(f"{name} {desc}")

        tag_match_ratio = 0.0
        token_match_ratio = 0.0
        if requested_preferences:
            overlap = requested_preferences.intersection(detected_tour_tags)
            tag_match_ratio = len(overlap) / max(1, len(requested_preferences))
        if requested_preference_tokens:
            token_overlap = requested_preference_tokens.intersection(tour_tokens)
            token_match_ratio = len(token_overlap) / max(1, len(requested_preference_tokens))

        match_score = max(tag_match_ratio, token_match_ratio)
        if match_score > 0:
            scored.append((match_score, str(getattr(tour_obj, "tour_name", "") or "").strip()))

    scored.sort(key=lambda row: (-float(row[0]), row[1].lower()))
    names = []
    for _score, title in scored:
        if title and title not in names:
            names.append(title)
        if len(names) >= max(1, int(limit or 3)):
            break
    return names


def _to_bool_env(value, default=False):
    raw = str(value or "").strip().lower()
    if not raw:
        return bool(default)
    if raw in ("1", "true", "yes", "on", "y"):
        return True
    if raw in ("0", "false", "no", "off", "n"):
        return False
    return bool(default)


def _owner_exclusion_keywords() -> list[str]:
    raw = str(
        os.getenv(
            "CHATBOT_OWNER_EXCLUDE_KEYWORDS",
            getattr(settings, "CHATBOT_OWNER_EXCLUDE_KEYWORDS", "smoke"),
        )
        or ""
    ).strip()
    if not raw:
        return []
    tokens = [str(v).strip().lower() for v in raw.split(",")]
    return [token for token in tokens if token]


def _allow_demo_artifact_fallback():
    override = os.getenv("CHATBOT_ALLOW_DEMO_ARTIFACT_FALLBACK")
    if override not in (None, ""):
        return _to_bool_env(override, default=False)
    settings_override = getattr(settings, "CHATBOT_ALLOW_DEMO_ARTIFACT_FALLBACK", None)
    if settings_override not in (None, ""):
        return _to_bool_env(settings_override, default=False)
    return False


def _resolve_decision_tree_model_path() -> tuple[Path, str]:
    configured = str(
        os.getenv(
            "CHATBOT_DECISION_TREE_MODEL_PATH",
            getattr(settings, "CHATBOT_DECISION_TREE_MODEL_PATH", ""),
        )
        or ""
    ).strip()
    if configured:
        configured_path = Path(configured)
        if not configured_path.is_absolute():
            configured_path = Path(getattr(settings, "BASE_DIR", Path.cwd())) / configured_path
        return configured_path, "configured_env"

    artifacts_root = Path(__file__).resolve().parent.parent / "artifacts"
    if _allow_demo_artifact_fallback():
        demo_path = artifacts_root / "decision_tree_demo" / "decision_tree_demo.pkl"
        if demo_path.exists():
            return demo_path, "demo_fallback"

    final_path = artifacts_root / "decision_tree_final" / "decision_tree_final.pkl"
    if final_path.exists():
        return final_path, "final_default"

    return final_path, "final_required_missing"


def _default_decision_tree_model_path() -> Path:
    path, _source = _resolve_decision_tree_model_path()
    return path


def _load_decision_tree_model(model_path: Optional[Path] = None):
    global _DECISION_TREE_MODEL_CACHE, _DECISION_TREE_MODEL_PATH_CACHE, _DECISION_TREE_MODEL_SOURCE_CACHE

    if model_path is None:
        resolved_path, resolved_source = _resolve_decision_tree_model_path()
    else:
        resolved_path = Path(model_path)
        resolved_source = "manual_override"
    exists = resolved_path.exists()
    demo_allowed = _allow_demo_artifact_fallback()
    fallback_used = resolved_source.startswith("demo") or resolved_source.startswith("surrogate")
    logger.info(
        "DecisionTree load attempt | path=%s | source=%s | file_exists=%s | demo_fallback_allowed=%s | fallback_used=%s",
        str(resolved_path),
        resolved_source,
        exists,
        demo_allowed,
        fallback_used,
    )
    if resolved_source.startswith("demo") or resolved_source.startswith("final_required_missing"):
        logger.warning(
            "DecisionTree fallback/non-final source active | source=%s | path=%s",
            resolved_source,
            str(resolved_path),
        )
    if not resolved_path.exists():
        _DECISION_TREE_MODEL_SOURCE_CACHE = resolved_source
        return None, f"model_not_found:{resolved_path}"

    resolved_str = str(resolved_path)
    if _DECISION_TREE_MODEL_CACHE is not None and _DECISION_TREE_MODEL_PATH_CACHE == resolved_str:
        _DECISION_TREE_MODEL_SOURCE_CACHE = resolved_source
        return _DECISION_TREE_MODEL_CACHE, None

    try:
        with resolved_path.open("rb") as f:
            model = pickle.load(f)
        _DECISION_TREE_MODEL_CACHE = model
        _DECISION_TREE_MODEL_PATH_CACHE = resolved_str
        _DECISION_TREE_MODEL_SOURCE_CACHE = resolved_source
        params = _extract_decision_tree_params(model)
        logger.info(
            "DecisionTree loaded | source=%s | params=%s",
            resolved_source,
            params,
        )
        return model, None
    except Exception as exc:
        _DECISION_TREE_MODEL_SOURCE_CACHE = resolved_source
        return None, f"model_load_error:{exc}"


def _extract_decision_tree_params(model) -> dict:
    classifier = model
    if hasattr(model, "named_steps"):
        classifier = model.named_steps.get("model", model)
    params = {}
    for key in ("criterion", "max_depth", "min_samples_leaf", "min_samples_split", "max_features"):
        params[key] = getattr(classifier, key, None)
    return params


def get_decision_tree_runtime_status(*, force_reload=False) -> dict:
    resolved_path, resolved_source = _resolve_decision_tree_model_path()
    if force_reload:
        model, load_error = _load_decision_tree_model(model_path=resolved_path)
    else:
        model, load_error = _load_decision_tree_model()
    file_exists = resolved_path.exists()
    mtime = ""
    if file_exists:
        try:
            mtime = datetime.fromtimestamp(resolved_path.stat().st_mtime).isoformat()
        except Exception:
            mtime = ""
    params = _extract_decision_tree_params(model) if model is not None else {}
    fallback_used = bool(
        resolved_source.startswith("demo")
        or resolved_source.startswith("final_required_missing")
        or str(load_error or "").startswith("model_not_found")
        or str(load_error or "").startswith("model_load_error")
    )
    expected_match = True
    if params:
        expected_match = all(params.get(k) == v for k, v in _DECISION_TREE_EXPECTED_PARAMS.items())

    status = {
        "path": str(resolved_path),
        "source": resolved_source,
        "file_exists": bool(file_exists),
        "fallback_used": fallback_used,
        "demo_fallback_allowed": bool(_allow_demo_artifact_fallback()),
        "loaded_model_params": params,
        "expected_pruned_params": dict(_DECISION_TREE_EXPECTED_PARAMS),
        "expected_params_match": bool(expected_match) if params else False,
        "file_last_modified": mtime,
        "load_error": str(load_error or ""),
    }
    if fallback_used:
        logger.warning("DecisionTree runtime status indicates fallback usage | status=%s", status)
    return status


def _parse_date(value):
    if isinstance(value, date):
        return value
    raw = str(value or "").strip()
    if not raw:
        return None
    try:
        return datetime.strptime(raw, "%Y-%m-%d").date()
    except ValueError:
        return None


def _resolve_nights_requested(params: dict) -> int:
    nights = _to_int(params.get("nights"), default=0)
    if nights > 0:
        return nights
    check_in = _parse_date(params.get("check_in"))
    check_out = _parse_date(params.get("check_out"))
    if check_in and check_out:
        delta = (check_out - check_in).days
        if delta > 0:
            return delta
    return 1


def _surrogate_decision_tree_score(
    room: Room,
    params: dict,
    *,
    shown_rank: int,
    cnn_confidence: float,
) -> float:
    requested_budget = _to_decimal(params.get("budget"), default=Decimal("0"))
    requested_guests = _to_int(params.get("guests"), default=1)
    requested_location = str(params.get("location") or "").strip().lower()
    requested_type = str(
        params.get("company_type")
        or params.get("predicted_accommodation_type")
        or ""
    ).strip().lower()
    room_price = _to_decimal(getattr(room, "price_per_night", 0), default=Decimal("0"))
    room_capacity = _to_int(getattr(room, "person_limit", 0), default=0)
    room_location = str(getattr(room.accommodation, "location", "") or "").strip().lower()
    company_type = str(getattr(room.accommodation, "company_type", "") or "").strip().lower()

    score = 0.10
    if requested_budget > 0:
        if room_price <= requested_budget:
            score += 0.32
        else:
            score -= 0.25
    else:
        score += 0.08

    if requested_location:
        if requested_location in room_location:
            score += 0.24
        else:
            score -= 0.16
    else:
        score += 0.05

    if requested_type and requested_type != "either":
        if requested_type in company_type:
            score += 0.16
        else:
            score -= 0.12
    else:
        score += 0.05

    if requested_guests > 0 and room_capacity > 0:
        if requested_guests <= room_capacity:
            score += 0.18
        else:
            score -= 0.30

    if shown_rank <= 3:
        score += 0.06
    elif shown_rank <= 6:
        score += 0.03

    score += min(0.08, max(0.0, float(cnn_confidence)) * 0.08)
    return max(0.0, min(1.0, round(float(score), 6)))


def _decision_tree_relevance_score(room: Room, params: dict, *, shown_rank: int, cnn_confidence: float):
    model, _error = _load_decision_tree_model()
    if model is None:
        return _surrogate_decision_tree_score(
            room,
            params,
            shown_rank=shown_rank,
            cnn_confidence=cnn_confidence,
        ), f"surrogate:{_DECISION_TREE_MODEL_SOURCE_CACHE}"

    requested_type = str(
        params.get("company_type")
        or params.get("predicted_accommodation_type")
        or "either"
    ).strip().lower()

    feature_row = {
        "requested_guests": _to_int(params.get("guests"), default=1),
        "requested_budget": float(_to_decimal(params.get("budget"), default=Decimal("0"))),
        "requested_location": str(params.get("location") or "").strip().lower(),
        "requested_accommodation_type": requested_type,
        "room_price_per_night": float(room.price_per_night or 0),
        "room_capacity": _to_int(room.person_limit, default=0),
        "room_available": _to_int(room.current_availability, default=0),
        "accom_location": str(getattr(room.accommodation, "location", "") or "").strip().lower(),
        "company_type": str(getattr(room.accommodation, "company_type", "") or "").strip().lower(),
        "nights_requested": _resolve_nights_requested(params),
        "cnn_confidence": float(max(0.0, min(1.0, cnn_confidence))),
        "shown_rank": int(max(1, shown_rank)),
    }

    try:
        model_input = [feature_row]
        if pd is not None and hasattr(model, "named_steps"):
            # sklearn Pipeline + ColumnTransformer expects named columns.
            model_input = pd.DataFrame([feature_row])

        if hasattr(model, "predict_proba"):
            classes = [str(c).strip().lower() for c in getattr(model, "classes_", [])]
            if not classes and hasattr(model, "named_steps"):
                clf = model.named_steps.get("model")
                classes = [str(c).strip().lower() for c in getattr(clf, "classes_", [])]
            probabilities = model.predict_proba(model_input)[0]
            if "relevant" in classes:
                idx = classes.index("relevant")
                return float(probabilities[idx]), f"model:{_DECISION_TREE_MODEL_SOURCE_CACHE}"
            if probabilities is not None and len(probabilities):
                return float(max(probabilities)), f"model:{_DECISION_TREE_MODEL_SOURCE_CACHE}"

        predicted = model.predict(model_input)[0]
        predicted_label = str(predicted).strip().lower()
        if predicted_label in ("relevant", "1", "true", "yes"):
            return 1.0, f"model:{_DECISION_TREE_MODEL_SOURCE_CACHE}"
        return 0.0, f"model:{_DECISION_TREE_MODEL_SOURCE_CACHE}"
    except Exception:
        return _surrogate_decision_tree_score(
            room,
            params,
            shown_rank=shown_rank,
            cnn_confidence=cnn_confidence,
        ), f"surrogate:{_DECISION_TREE_MODEL_SOURCE_CACHE}"


def predict_accommodation_relevance_from_features(feature_row: dict) -> dict:
    """
    Runtime helper for direct Decision Tree inference from a feature payload.
    Used by internal validation utilities and backend integration checks.
    """
    normalized = {
        "requested_guests": _to_int(feature_row.get("requested_guests"), default=1),
        "requested_budget": float(_to_decimal(feature_row.get("requested_budget"), default=Decimal("0"))),
        "requested_location": str(feature_row.get("requested_location") or "").strip().lower(),
        "requested_accommodation_type": str(feature_row.get("requested_accommodation_type") or "either").strip().lower(),
        "room_price_per_night": float(_to_decimal(feature_row.get("room_price_per_night"), default=Decimal("0"))),
        "room_capacity": _to_int(feature_row.get("room_capacity"), default=1),
        "room_available": _to_int(feature_row.get("room_available"), default=0),
        "accom_location": str(feature_row.get("accom_location") or "").strip().lower(),
        "company_type": str(feature_row.get("company_type") or "").strip().lower(),
        "nights_requested": max(1, _to_int(feature_row.get("nights_requested"), default=1)),
        "cnn_confidence": float(max(0.0, min(1.0, float(feature_row.get("cnn_confidence") or 0.0)))),
        "shown_rank": max(1, _to_int(feature_row.get("shown_rank"), default=1)),
    }

    model, error = _load_decision_tree_model()
    if model is None:
        return {
            "score": 0.0,
            "predicted_label": "unknown",
            "source": f"unavailable:{_DECISION_TREE_MODEL_SOURCE_CACHE}",
            "error": error or "model_unavailable",
            "normalized_features": normalized,
        }

    try:
        model_input = [normalized]
        if pd is not None and hasattr(model, "named_steps"):
            model_input = pd.DataFrame([normalized])

        if hasattr(model, "predict_proba"):
            classes = [str(c).strip().lower() for c in getattr(model, "classes_", [])]
            if not classes and hasattr(model, "named_steps"):
                clf = model.named_steps.get("model")
                classes = [str(c).strip().lower() for c in getattr(clf, "classes_", [])]
            proba = model.predict_proba(model_input)[0]
            if "relevant" in classes:
                idx = classes.index("relevant")
                score = float(proba[idx])
            else:
                score = float(max(proba)) if len(proba) else 0.0
            label = "relevant" if score >= 0.5 else "not_relevant"
            return {
                "score": score,
                "predicted_label": label,
                "source": f"model:{_DECISION_TREE_MODEL_SOURCE_CACHE}",
                "error": "",
                "normalized_features": normalized,
            }

        pred = str(model.predict(model_input)[0]).strip().lower()
        score = 1.0 if pred in ("relevant", "1", "true", "yes") else 0.0
        label = "relevant" if score >= 0.5 else "not_relevant"
        return {
            "score": score,
            "predicted_label": label,
            "source": f"model:{_DECISION_TREE_MODEL_SOURCE_CACHE}",
            "error": "",
            "normalized_features": normalized,
        }
    except Exception as exc:
        return {
            "score": 0.0,
            "predicted_label": "unknown",
            "source": f"error:{_DECISION_TREE_MODEL_SOURCE_CACHE}",
            "error": str(exc),
            "normalized_features": normalized,
        }


def _normalize_amenity_tokens(value) -> List[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        raw = " ".join(str(item) for item in value)
    else:
        raw = str(value)
    return [token.strip().lower() for token in raw.replace(";", ",").split(",") if token.strip()]


def _normalize_amenity_alias(token: str) -> str:
    alias_map = {
        "ac": "aircon",
        "air conditioning": "aircon",
        "air-conditioned": "aircon",
        "airconditioned": "aircon",
    }
    lowered = str(token or "").strip().lower()
    return alias_map.get(lowered, lowered)


def _to_bool(value, default=False):
    if isinstance(value, bool):
        return value
    if value in (None, ""):
        return default
    lowered = str(value).strip().lower()
    if lowered in ("1", "true", "yes", "y", "on"):
        return True
    if lowered in ("0", "false", "no", "n", "off"):
        return False
    return default


def build_accommodation_recommendation_trace(room: Room, params: dict) -> dict:
    """
    Build explainable recommendation reasons from existing accommodation logic.
    """
    guests = _to_int(params.get("guests"), default=1)
    budget = _to_decimal(params.get("budget"), default=Decimal("0"))
    location = str(params.get("location") or "").strip()
    location_anchor = str(params.get("location_anchor") or "").strip()
    company_type = str(params.get("company_type") or "").strip().lower()

    accom = room.accommodation
    price = Decimal(str(room.price_per_night or 0))
    reasons: List[str] = []
    score = 0.0

    if budget > 0:
        if price <= budget:
            reasons.append(f"Within your budget (PHP {price:.2f} <= PHP {budget:.2f})")
            score += 0.35
        else:
            reasons.append(f"Above your budget (PHP {price:.2f} > PHP {budget:.2f})")
            score -= 0.45
    else:
        reasons.append("No budget limit provided")
        score += 0.10

    location_match = True
    if location_anchor:
        reasons.append(f"Map anchor considered: {location_anchor} (city-proper coverage)")
        score += 0.05
    if location:
        if location.lower() in str(accom.location or "").lower():
            reasons.append(f"Location match: {accom.location}")
            score += 0.30
        else:
            location_match = False
            reasons.append(f"Outside preferred location: {accom.location}")
            score -= 0.25
    else:
        reasons.append(f"Located in: {accom.location}")
        score += 0.05

    type_match = True
    if company_type:
        if company_type in str(accom.company_type or "").lower():
            reasons.append(f"{company_type.title()} type match")
            score += 0.20
        else:
            type_match = False
            reasons.append(f"Different type: {accom.company_type}")
            score -= 0.20
    else:
        reasons.append(f"Type: {accom.company_type}")
        score += 0.05

    guest_fit = True
    if room.person_limit and guests > 0:
        if guests <= room.person_limit:
            reasons.append(f"Fits your guest count ({guests}/{room.person_limit} pax)")
            score += 0.15
        else:
            reasons.append(f"Capacity limit ({room.person_limit} pax)")
            guest_fit = False
            score -= 0.40

    requested_amenities = _normalize_amenity_tokens(
        params.get("amenities") or params.get("amenity")
    )
    amenity_match_ratio = 1.0
    if requested_amenities:
        normalized_requested = sorted(
            { _normalize_amenity_alias(token) for token in requested_amenities if str(token).strip() }
        )
        searchable_text = " ".join(
            [
                str(room.room_name or ""),
                str(accom.company_name or ""),
                str(accom.location or ""),
                str(accom.company_type or ""),
            ]
        ).lower()
        matched = [token for token in normalized_requested if token in searchable_text]
        amenity_match_ratio = (len(matched) / len(normalized_requested)) if normalized_requested else 0.0
        if amenity_match_ratio >= 1.0:
            reasons.append(f"Amenity match: {', '.join(matched)}")
            score += 0.20
        elif amenity_match_ratio > 0:
            reasons.append(
                f"Partial amenity match ({len(matched)}/{len(normalized_requested)}): {', '.join(matched)}"
            )
            score += 0.08
        else:
            reasons.append("No amenity keyword match found in available details")
            score -= 0.15

    preference_tags = (
        params.get("preference_tags")
        if isinstance(params.get("preference_tags"), list)
        else []
    )
    normalized_preference_tags = sorted(
        {
            str(tag or "").strip().lower()
            for tag in preference_tags
            if str(tag or "").strip()
        }
    )
    preference_match_ratio = 1.0
    if normalized_preference_tags:
        searchable_text = " ".join(
            [
                str(room.room_name or ""),
                str(accom.company_name or ""),
                str(accom.location or ""),
                str(accom.company_type or ""),
                str(getattr(accom, "description", "") or ""),
                str(getattr(accom, "accommodation_amenities", "") or ""),
            ]
        ).lower()
        preference_aliases = {
            "quiet": ("quiet", "peaceful", "calm", "serene", "relax"),
            "nature": ("nature", "green", "garden", "fresh air", "view", "scenic", "river", "mountain"),
            "family": ("family", "kids", "children", "group", "suite", "spacious"),
            "clean": ("clean", "hygienic", "sanitary", "well-maintained", "tidy"),
            "accessible": ("terminal", "transport", "commute", "downtown", "highway"),
        }
        matched_preferences = []
        for tag in normalized_preference_tags:
            markers = preference_aliases.get(tag, (tag,))
            if any(marker in searchable_text for marker in markers):
                matched_preferences.append(tag)
        preference_match_ratio = (
            len(matched_preferences) / len(normalized_preference_tags)
            if normalized_preference_tags
            else 0.0
        )
        if preference_match_ratio >= 1.0:
            reasons.append(f"Preference match: {', '.join(matched_preferences)}")
            score += 0.18
        elif preference_match_ratio > 0:
            reasons.append(
                f"Partial preference match ({len(matched_preferences)}/{len(normalized_preference_tags)}): "
                f"{', '.join(matched_preferences)}"
            )
            score += 0.08
        else:
            reasons.append("No strong text match for your descriptive preferences yet")
            score -= 0.10

    if bool(params.get("prefer_low_price")) and budget <= 0:
        if price <= Decimal("1500"):
            reasons.append("Budget-friendly option")
            score += 0.12
        elif price <= Decimal("2500"):
            reasons.append("Mid-range price option")
            score += 0.05
        else:
            reasons.append("Premium-priced option")
            score -= 0.08

    normalized_score = round(max(0.0, min(score, 1.0)), 4)
    if normalized_score >= 0.80:
        match_strength = "High"
    elif normalized_score >= 0.55:
        match_strength = "Medium"
    else:
        match_strength = "Low"

    return {
        "match_score": normalized_score,
        "match_strength": match_strength,
        "reasons": reasons,
        "location_match": location_match,
        "type_match": type_match,
        "guest_fit": guest_fit,
        "amenity_match_ratio": round(float(amenity_match_ratio), 3),
        "preference_match_ratio": round(float(preference_match_ratio), 3),
    }


def recommend_tours(params: dict, limit: int = 3) -> List[RecommendationResult]:
    now = timezone.now()
    guests = _to_int(params.get("guests"), default=1)
    budget = _to_decimal(params.get("budget"), default=Decimal("0"))
    duration = _to_int(params.get("duration_days"), default=0)
    requested_preferences = _collect_requested_tour_preferences(params)
    requested_preference_text = str(
        params.get("preference_text")
        or params.get("preference")
        or params.get("tour_type")
        or params.get("preferred_type")
        or params.get("interest")
        or ""
    ).strip().lower()
    requested_preference_tokens = _tokenize_preference_phrase(requested_preference_text)

    schedules = (
        Tour_Schedule.objects.select_related("tour")
        .filter(end_time__gte=now)
        .exclude(status="cancelled")
        .annotate(slots_left=F("slots_available") - F("slots_booked"))
    )

    prices = [float(s.price) for s in schedules] or [0.0]
    min_price, max_price = min(prices), max(prices)

    results = []
    for schedule in schedules:
        slots_left = max(schedule.slots_left, 0)
        if slots_left < guests:
            continue

        price = float(schedule.price)
        price_fit = 1.0 if budget <= 0 else float(min(budget / Decimal(price), 1))
        duration_fit = 1.0 if duration and schedule.duration_days == duration else 0.5 if duration and abs(schedule.duration_days - duration) == 1 else 0.0
        if not duration:
            duration_fit = 0.5

        name = (schedule.tour.tour_name or "").lower()
        desc = (schedule.tour.description or "").lower()
        tour_tokens = _tokenize_preference_phrase(f"{name} {desc}")
        detected_tour_tags = _extract_tour_preference_tags_from_text(f"{name} {desc}")
        tag_match_ratio = 0.0
        token_match_ratio = 0.0
        if requested_preferences:
            overlap = requested_preferences.intersection(detected_tour_tags)
            tag_match_ratio = len(overlap) / max(1, len(requested_preferences))
        if requested_preference_tokens:
            token_overlap = requested_preference_tokens.intersection(tour_tokens)
            token_match_ratio = len(token_overlap) / max(1, len(requested_preference_tokens))
        if requested_preferences or requested_preference_tokens:
            preference_fit = max(tag_match_ratio, token_match_ratio)
        else:
            preference_fit = 0.3
        availability_fit = min(slots_left, 10) / 10.0
        price_norm = 1.0 - _normalize(price, min_price, max_price)

        features = [price_fit, duration_fit, preference_fit, availability_fit, price_norm]
        cnn_score = _cnn_score(features)

        penalty = _decision_tree_penalty([
            budget > 0 and price > float(budget),
            duration > 0 and schedule.duration_days not in (duration, duration - 1, duration + 1),
        ])

        score = cnn_score + penalty
        results.append(
            RecommendationResult(
                title=schedule.tour.tour_name,
                subtitle=f"{schedule.sched_id} | PHP {schedule.price} per guest | {schedule.duration_days} day(s)",
                score=score,
                meta={
                    "sched_id": schedule.sched_id,
                    "detected_tour_tags": sorted(detected_tour_tags),
                    "requested_tour_preferences": sorted(requested_preferences),
                    "requested_preference_tokens": sorted(requested_preference_tokens),
                    "matched_preference_tokens": sorted(
                        requested_preference_tokens.intersection(tour_tokens)
                    ),
                    "tag_match_ratio": round(float(tag_match_ratio), 4),
                    "token_match_ratio": round(float(token_match_ratio), 4),
                },
            )
        )

    results.sort(key=lambda item: item.score, reverse=True)
    return results[:limit]


def recommend_accommodations(params: dict, limit: int = 3) -> List[RecommendationResult]:
    results, _diagnostics = recommend_accommodations_with_diagnostics(params, limit=limit)
    return results


def _build_accommodation_room_queryset(
    params: dict,
    *,
    apply_location: bool,
    apply_company_type: bool,
    apply_budget: bool,
):
    guests = _to_int(params.get("guests"), default=1)
    budget = _to_decimal(params.get("budget"), default=Decimal("0"))
    location = str(params.get("location") or "").strip().lower()
    company_type = str(params.get("company_type") or "").strip().lower()

    room_qs = (
        Room.objects.select_related("accommodation")
        .filter(status="AVAILABLE")
        .filter(current_availability__gte=1)
        .filter(accommodation__approval_status="accepted")
        .filter(accommodation__is_active=True)
        .filter(accommodation__owner__isnull=False)
        .filter(accommodation__owner__is_active=True)
        .filter(accommodation__owner__groups__name__iexact="accommodation_owner")
        .exclude(accommodation__owner__groups__name__iexact="accommodation_owner_pending")
        .exclude(accommodation__owner__groups__name__iexact="accommodation_owner_declined")
    )

    for keyword in _owner_exclusion_keywords():
        room_qs = room_qs.exclude(accommodation__owner__email__icontains=keyword)
        room_qs = room_qs.exclude(accommodation__owner__username__icontains=keyword)
        room_qs = room_qs.exclude(accommodation__owner__first_name__icontains=keyword)
        room_qs = room_qs.exclude(accommodation__owner__last_name__icontains=keyword)
        room_qs = room_qs.exclude(accommodation__company_name__icontains=keyword)

    if company_type:
        if apply_company_type:
            if company_type == "either":
                room_qs = room_qs.filter(
                    Q(accommodation__company_type__icontains="hotel") |
                    Q(accommodation__company_type__icontains="inn")
                )
            else:
                room_qs = room_qs.filter(accommodation__company_type__icontains=company_type)
    else:
        room_qs = room_qs.filter(
            Q(accommodation__company_type__icontains="hotel") |
            Q(accommodation__company_type__icontains="inn")
        )

    if apply_location and location:
        room_qs = room_qs.filter(accommodation__location__icontains=location)

    # If the user explicitly provides a budget, apply a strict DB-level filter so
    # over-budget rooms are not returned in the recommendation list.
    if apply_budget and budget > 0:
        room_qs = room_qs.filter(price_per_night__lte=budget)

    return room_qs.distinct(), guests


def _build_accommodation_results(room_qs, *, guests: int, params: dict) -> List[RecommendationResult]:
    results = []
    predicted_type = str(params.get("predicted_accommodation_type") or "").strip().lower()
    cnn_confidence = float(max(0.0, min(1.0, float(params.get("predicted_accommodation_confidence") or 0.0))))

    for shown_rank, room in enumerate(room_qs, start=1):
        accom = room.accommodation
        if room.person_limit and guests > room.person_limit:
            continue

        trace = build_accommodation_recommendation_trace(room, params)
        base_score = float(trace.get("match_score") or 0.0)
        room_type = str(getattr(accom, "company_type", "") or "").strip().lower()
        cnn_type_match = bool(predicted_type and predicted_type in room_type)
        cnn_alignment = 0.0
        if predicted_type and cnn_confidence > 0:
            weight = min(0.20, 0.30 * cnn_confidence)
            cnn_alignment = weight if cnn_type_match else (-0.5 * weight)

        dt_score, dt_source = _decision_tree_relevance_score(
            room,
            params,
            shown_rank=shown_rank,
            cnn_confidence=cnn_confidence,
        )
        if dt_score is None:
            score = max(0.0, min(1.0, base_score + cnn_alignment))
            scoring_mode = "hybrid_fallback_heuristic"
        else:
            score = max(0.0, min(1.0, (0.55 * float(dt_score)) + (0.35 * base_score) + (0.10 * cnn_alignment)))
            scoring_mode = (
                "hybrid_textcnn_decisiontree"
                if str(dt_source).startswith("model")
                else "hybrid_textcnn_surrogate_tree"
            )

        trace["decision_tree_score"] = None if dt_score is None else round(float(dt_score), 4)
        trace["decision_tree_source"] = dt_source
        trace["cnn_alignment"] = round(float(cnn_alignment), 4)
        trace["cnn_predicted_type"] = predicted_type or ""
        trace["cnn_confidence"] = round(float(cnn_confidence), 4)
        trace["cnn_type_match"] = bool(cnn_type_match)
        trace["scoring_mode"] = scoring_mode
        results.append(
            RecommendationResult(
                title=f"{accom.company_name} - {room.room_name}",
                subtitle=f"{accom.location} | PHP {room.price_per_night} per night | {room.person_limit} pax",
                score=score,
                meta={
                    "room_id": room.room_id,
                    "accom_id": accom.accom_id,
                    "trace": trace,
                    "decision_tree_score": None if dt_score is None else round(float(dt_score), 6),
                    "decision_tree_source": dt_source,
                    "cnn_alignment": round(float(cnn_alignment), 6),
                    "scoring_mode": scoring_mode,
                },
            )
        )

    results.sort(key=lambda item: item.score, reverse=True)
    return results


def recommend_accommodations_with_diagnostics(params: dict, limit: int = 3):
    diagnostics = {
        "fallback_applied": "none",
        "no_match_reasons": [],
        "suggested_budget_min": None,
    }

    broaden_location = _to_bool(params.get("broaden_location"), default=False)
    broaden_type = _to_bool(params.get("broaden_company_type"), default=False)

    passes = [("strict", True, True)]
    if broaden_location:
        passes.append(("relaxed_location", False, True))
    if broaden_type:
        passes.append(("relaxed_type", True, False))
    if broaden_location and broaden_type:
        passes.append(("relaxed_location_and_type", False, False))

    seen_passes = set()
    final_results: List[RecommendationResult] = []
    min_score_by_pass = {
        "strict": 0.10,
        "relaxed_location": 0.15,
        "relaxed_type": 0.15,
        "relaxed_location_and_type": 0.20,
    }

    for pass_name, apply_location, apply_company_type in passes:
        room_qs, guests = _build_accommodation_room_queryset(
            params,
            apply_location=apply_location,
            apply_company_type=apply_company_type,
            apply_budget=True,
        )
        results = _build_accommodation_results(room_qs, guests=guests, params=params)
        if results:
            top_score = float(results[0].score) if results else 0.0
            min_required = float(min_score_by_pass.get(pass_name, 0.0))
            if top_score >= min_required:
                diagnostics["fallback_applied"] = pass_name
                final_results = results[:limit]
                break
        seen_passes.add((apply_location, apply_company_type))

    if final_results:
        return final_results, diagnostics

    budget = _to_decimal(params.get("budget"), default=Decimal("0"))
    location = str(params.get("location") or "").strip()
    company_type = str(params.get("company_type") or "").strip().lower()

    # Analyze constraints without budget to provide actionable guidance.
    analysis_qs, guests = _build_accommodation_room_queryset(
        params,
        apply_location=True,
        apply_company_type=True,
        apply_budget=False,
    )
    analysis_candidates = list(analysis_qs)
    analysis_candidates = [
        room for room in analysis_candidates
        if not room.person_limit or guests <= room.person_limit
    ]

    if budget > 0 and analysis_candidates:
        min_price = min(Decimal(str(room.price_per_night or 0)) for room in analysis_candidates)
        if min_price > budget:
            diagnostics["no_match_reasons"].append("budget_too_low")
            diagnostics["suggested_budget_min"] = float(min_price)

    if location:
        location_relaxed_qs, location_guests = _build_accommodation_room_queryset(
            params,
            apply_location=False,
            apply_company_type=True,
            apply_budget=True,
        )
        location_relaxed_candidates = [
            room for room in location_relaxed_qs
            if not room.person_limit or location_guests <= room.person_limit
        ]
        if location_relaxed_candidates:
            diagnostics["no_match_reasons"].append("location_too_narrow")

    if company_type:
        type_relaxed_qs, type_guests = _build_accommodation_room_queryset(
            params,
            apply_location=True,
            apply_company_type=False,
            apply_budget=True,
        )
        type_relaxed_candidates = [
            room for room in type_relaxed_qs
            if not room.person_limit or type_guests <= room.person_limit
        ]
        if type_relaxed_candidates:
            diagnostics["no_match_reasons"].append("type_too_narrow")

    if not diagnostics["no_match_reasons"]:
        diagnostics["no_match_reasons"].append("no_available_match")

    return [], diagnostics


def calculate_accommodation_billing(room: Room, check_in, check_out) -> Decimal:
    nights = max((check_out - check_in).days, 1)
    return Decimal(room.price_per_night) * Decimal(nights)
