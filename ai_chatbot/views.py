import json
import importlib
import hashlib
import logging
import os
import re
import time
import uuid
from collections import Counter
from datetime import datetime, timedelta
from decimal import Decimal
from difflib import get_close_matches
from pathlib import Path
from urllib.parse import urlencode

from django.conf import settings
from django.contrib.auth.models import Group
from django.db import transaction
from django.db.models import DecimalField, ExpressionWrapper, F, Q, Sum
from django.http import JsonResponse
from django.urls import reverse
from django.utils import timezone
from django.views.decorators.csrf import csrf_exempt

try:
    from openai import OpenAI
except ModuleNotFoundError:
    OpenAI = None

try:
    from google import genai  # type: ignore
except Exception:
    try:
        genai = importlib.import_module("google.genai")
    except Exception:
        genai = None

try:
    import numpy as np
    import pandas as pd
    import tensorflow as tf
except ModuleNotFoundError:
    np = None
    pd = None
    tf = None

from tour_app.models import Admission_Rates, Tour_Schedule
from admin_app.models import Accomodation, Employee, Room, TourismInformation, TourAssignment
from accom_app.models import AuthoritativeRoomDetails
from guest_app.models import AccommodationBooking, Billing, Guest, TourBooking, Pending
from guest_app.booking_integrity import create_accommodation_booking_with_integrity
from .recommenders import (
    recommend_tours,
    recommend_accommodations_with_diagnostics,
    calculate_accommodation_billing,
    get_decision_tree_runtime_status,
    get_unavailable_tour_matches,
)
from .llm_translation import (
    translate_to_english,
    translate_to_user_language,
    translation_runtime_health,
)
from .chat_services.state import (
    load_chat_state as _load_chat_state_service,
    save_chat_state as _save_chat_state_service,
    clear_chat_state as _clear_chat_state_service,
)
from .chat_services.response_templates import (
    get_accommodation_slot_question,
    format_acknowledged_details,
    build_personalization_offer_text as _template_personalization_offer_text,
    PERSONALIZATION_PROMPT_DEFAULT,
)
from .models import (
    ChatbotLog,
    RecommendationEvent,
    RecommendationResult,
    SystemMetricLog,
    UsabilitySurveyResponse,
)

logger = logging.getLogger(__name__)

_TEXT_CNN_MODEL_CACHE = None
_TEXT_CNN_MODEL_PATH_CACHE = None
_ACCOM_LOCATION_CACHE = None
_MAP_REFERENCE_PLACE_CACHE = None
_CHAT_STATE_SESSION_KEY = "ai_chatbot_state"
_CHAT_PREFERENCE_SESSION_KEY = "ai_chatbot_saved_preferences"
_CHAT_STATE_TTL_SECONDS = 30 * 60
_PENDING_BOOKING_TTL_SECONDS = 10 * 60
_VALID_DATA_SOURCES = {"unlabeled", "demo_seeded", "pilot_test", "real_world"}
_ALLOWED_INTENTS = {
    "get_recommendation",
    "get_tourism_information",
    "calculate_billing",
    "get_accommodation_recommendation",
    "calculate_accommodation_billing",
    "book_accommodation",
}
_INTENT_LABEL_ALIASES = {
    "gettourrecommendation": "get_recommendation",
    "tour_recommendation": "get_recommendation",
    "tour recommendation": "get_recommendation",
    "gettourisminformation": "get_tourism_information",
    "tourism_information": "get_tourism_information",
    "tourism information": "get_tourism_information",
    "tourist_information": "get_tourism_information",
    "tourist information": "get_tourism_information",
    "tourist_spot_info": "get_tourism_information",
    "tourist spot info": "get_tourism_information",
    "attraction_information": "get_tourism_information",
    "attraction information": "get_tourism_information",
    "calculatetourbilling": "calculate_billing",
    "tour_billing": "calculate_billing",
    "tour billing": "calculate_billing",
    "gethotelrecommendation": "get_accommodation_recommendation",
    "hotel_recommendation": "get_accommodation_recommendation",
    "hotel recommendation": "get_accommodation_recommendation",
    "calculatehotelbilling": "calculate_accommodation_billing",
    "hotel_billing": "calculate_accommodation_billing",
    "hotel billing": "calculate_accommodation_billing",
    "bookhotel": "book_accommodation",
    "book_hotel": "book_accommodation",
    "reserve_accommodation": "book_accommodation",
}
_MAP_ANCHOR_ALIASES = {
    "Bayawan City Public Terminal": (
        "terminal",
        "public terminal",
        "city terminal",
        "bus terminal",
        "bayawan terminal",
        "bayawan city public terminal",
        "near terminal",
        "near bus terminal",
    ),
    "Bayawan City Plaza": (
        "plaza",
        "city plaza",
        "bayawan plaza",
        "bayawan city plaza",
        "plaza area",
        "near plaza",
    ),
    "Bayawan City Public Market": (
        "market",
        "public market",
        "city market",
        "bayawan market",
        "bayawan city public market",
        "wet market",
        "near market",
    ),
    "Hayahay Square": (
        "hayahay",
        "hayahay square",
        "near hayahay",
        "hayahay area",
    ),
    "Eskina Restaurant": (
        "eskina",
        "eskina restaurant",
        "near eskina",
        "eskina area",
    ),
    "Catholic Church": (
        "church",
        "catholic church",
        "parish church",
        "near church",
        "near catholic church",
    ),
    "Puregold Grocery Section": (
        "puregold",
        "puregold bayawan",
        "near puregold",
        "grocery near puregold",
    ),
}
_TERMINAL_SPECIFIC_MARKERS = (
    "bus terminal",
    "public terminal",
    "city terminal",
    "trike terminal",
    "tricycle terminal",
    "tricyle terminal",
    "pedicab terminal",
    "motorcab terminal",
)
_SUS_CODES = [f"SUS_Q{i}" for i in range(1, 11)]
_TAM_CODES = [f"PU_Q{i}" for i in range(1, 5)] + [f"PEU_Q{i}" for i in range(1, 5)]
_DIFFICULTY_CODES = ["DIFF_DISCOVER", "DIFF_MATCH", "DIFF_PLAN", "DIFF_BOOKPAY"]
_FULL_SURVEY_CODES = set(_SUS_CODES + _TAM_CODES)
_GUEST_FUNNEL_EVENT_MAP = {
    "chatbot_opened": {"item_ref": "chat:funnel_chatbot_opened", "event_type": "view"},
    "quick_start_clicked": {"item_ref": "chat:funnel_quick_start_clicked", "event_type": "click"},
    "recommendation_shown": {"item_ref": "chat:funnel_recommendation_shown", "event_type": "view"},
    "recommendation_card_clicked": {"item_ref": "chat:funnel_recommendation_card_clicked", "event_type": "click"},
    "book_button_clicked": {"item_ref": "chat:funnel_book_button_clicked", "event_type": "click"},
    "booking_flow_started": {"item_ref": "chat:funnel_booking_flow_started", "event_type": "save"},
    "billing_link_shown": {"item_ref": "chat:funnel_billing_link_shown", "event_type": "view"},
    "billing_link_clicked": {"item_ref": "chat:funnel_billing_link_clicked", "event_type": "click"},
    "booking_completed": {"item_ref": "chat:funnel_booking_completed", "event_type": "book"},
}


def _to_bool_env(value, default=False):
    raw = str(value or "").strip().lower()
    if not raw:
        return bool(default)
    if raw in ("1", "true", "yes", "on", "y"):
        return True
    if raw in ("0", "false", "no", "off", "n"):
        return False
    return bool(default)


def _to_bool(value, default=False):
    return _to_bool_env(value, default=default)


def _known_accommodation_locations(force_reload=False):
    global _ACCOM_LOCATION_CACHE
    if _ACCOM_LOCATION_CACHE is not None and not force_reload:
        return _ACCOM_LOCATION_CACHE
    try:
        rows = (
            Accomodation.objects.filter(is_active=True, approval_status="accepted")
            .exclude(location__isnull=True)
            .exclude(location__exact="")
            .values_list("location", flat=True)
            .distinct()
        )
        normalized = []
        seen = set()
        for row in rows:
            value = " ".join(str(row or "").strip().lower().split())
            if not value or value in seen:
                continue
            seen.add(value)
            normalized.append(value)
        _ACCOM_LOCATION_CACHE = normalized[:300]
    except Exception:
        _ACCOM_LOCATION_CACHE = []
    return _ACCOM_LOCATION_CACHE


def _normalize_chat_text(value):
    text = " ".join(str(value or "").strip().lower().split())
    if not text:
        return ""
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return " ".join(text.split())


def _load_map_reference_place_entries(force_reload=False):
    global _MAP_REFERENCE_PLACE_CACHE
    if _MAP_REFERENCE_PLACE_CACHE is not None and not force_reload:
        return _MAP_REFERENCE_PLACE_CACHE

    entries = []
    seen = set()
    try:
        template_path = Path(__file__).resolve().parent.parent / "admin_app" / "templates" / "map.html"
        raw = template_path.read_text(encoding="utf-8", errors="ignore")
        for name in re.findall(r'name:\s*"([^"]+)"', raw):
            canonical = " ".join(str(name or "").split()).strip()
            normalized = _normalize_chat_text(canonical)
            if not canonical or not normalized or normalized in seen:
                continue
            seen.add(normalized)
            entries.append(
                {
                    "name": canonical,
                    "normalized": normalized,
                }
            )
    except Exception:
        entries = []

    _MAP_REFERENCE_PLACE_CACHE = entries[:500]
    return _MAP_REFERENCE_PLACE_CACHE


def _match_map_reference_place(raw_location):
    matches = _match_map_reference_places(raw_location, limit=1)
    return matches[0] if matches else {}


def _match_map_reference_places(raw_location, *, limit=5):
    candidate = _normalize_chat_text(raw_location)
    if not candidate:
        return []

    ranked = []
    seen = set()

    def _push(entry, rank):
        if not isinstance(entry, dict):
            return
        name = " ".join(str(entry.get("name") or "").split()).strip()
        normalized = _normalize_chat_text(entry.get("normalized") or name)
        if not name or not normalized:
            return
        key = normalized
        if key in seen:
            return
        seen.add(key)
        ranked.append(
            {
                "name": name,
                "normalized": normalized,
                "_rank": int(rank),
            }
        )

    # Resolve known user wordings first (e.g., "trike terminal").
    for anchor_name, aliases in _MAP_ANCHOR_ALIASES.items():
        for alias in aliases:
            alias_norm = _normalize_chat_text(alias)
            if not alias_norm:
                continue
            if anchor_name == "Bayawan City Public Terminal":
                if any(
                    marker in candidate
                    for marker in (
                        "trike terminal",
                        "tricycle terminal",
                        "tricyle terminal",
                        "pedicab terminal",
                        "motorcab terminal",
                    )
                ):
                    # Keep trike/pedicab terminals from collapsing to the bus/public anchor.
                    continue
            if candidate == alias_norm:
                _push(
                    {
                        "name": anchor_name,
                        "normalized": _normalize_chat_text(anchor_name),
                    },
                    rank=0,
                )
            elif candidate in alias_norm or alias_norm in candidate:
                _push(
                    {
                        "name": anchor_name,
                        "normalized": _normalize_chat_text(anchor_name),
                    },
                    rank=1,
                )

    entries = _load_map_reference_place_entries()
    if not entries:
        trimmed = sorted(ranked, key=lambda row: (row.get("_rank", 99), row.get("name", "")))
        return [{k: v for k, v in row.items() if not str(k).startswith("_")} for row in trimmed[: max(1, int(limit or 1))]]

    for entry in entries:
        if candidate == entry["normalized"]:
            _push(entry, rank=0)

    for entry in entries:
        normalized_name = str(entry.get("normalized") or "")
        if candidate in normalized_name or normalized_name in candidate:
            _push(entry, rank=1)

    pool = [str(entry.get("normalized") or "") for entry in entries]
    close = get_close_matches(candidate, pool, n=max(1, int(limit or 1)), cutoff=0.8)
    for best in close:
        for entry in entries:
            if entry.get("normalized") == best:
                _push(entry, rank=2)
                break

    trimmed = sorted(ranked, key=lambda row: (row.get("_rank", 99), row.get("name", "")))
    return [{k: v for k, v in row.items() if not str(k).startswith("_")} for row in trimmed[: max(1, int(limit or 1))]]


def _map_place_to_location_hint(place_name):
    normalized = _normalize_chat_text(place_name)
    if not normalized:
        return ""

    hint_rules = [
        ("villareal", "villareal"),
        ("villarreal", "villareal"),
        ("tinago", "tinago"),
        ("boyco", "boyco"),
        ("ubos", "ubos"),
        ("suba", "suba"),
        ("poblacion", "poblacion"),
        ("public terminal", "tinago"),
        ("terminal", "tinago"),
        ("public market", "boyco"),
        ("market", "boyco"),
        ("plaza", "poblacion"),
        ("catholic church", "ubos"),
        ("church", "ubos"),
        ("hayahay", "suba"),
        ("eskina", "suba"),
        ("city hall", "bayawan city"),
        ("bayawan", "bayawan city"),
    ]
    for needle, hint in hint_rules:
        if needle in normalized:
            return hint
    return "bayawan city"


def _is_generic_terminal_reference(value):
    normalized = _normalize_chat_text(value)
    if "terminal" not in normalized:
        return False
    return not any(marker in normalized for marker in _TERMINAL_SPECIFIC_MARKERS)


def _allow_demo_artifact_fallback():
    override = os.getenv("CHATBOT_ALLOW_DEMO_ARTIFACT_FALLBACK")
    if override not in (None, ""):
        return _to_bool_env(override, default=False)
    return bool(getattr(settings, "DEBUG", False))


def _resolve_model_artifact_path(*, env_var, final_relative_path, demo_relative_path=None):
    configured = str(os.getenv(env_var, "")).strip()
    if configured:
        return Path(configured), "configured_env"

    artifacts_root = Path(__file__).resolve().parent.parent / "artifacts"
    final_path = artifacts_root / final_relative_path
    if final_path.exists():
        return final_path, "final_default"

    if demo_relative_path and _allow_demo_artifact_fallback():
        demo_path = artifacts_root / demo_relative_path
        if demo_path.exists():
            return demo_path, "demo_fallback"

    return final_path, "final_required_missing"


def _resolve_accommodation_text_cnn_model_path():
    return _resolve_model_artifact_path(
        env_var="CHATBOT_ACCOM_CNN_MODEL_PATH",
        final_relative_path="text_cnn_accommodation/text_cnn_accommodation.keras",
        demo_relative_path="text_cnn_demo/text_cnn_demo.keras",
    )


def _default_text_cnn_model_path():
    path, _source = _resolve_accommodation_text_cnn_model_path()
    return path


def _resolve_intent_text_cnn_model_path():
    return _resolve_model_artifact_path(
        env_var="CHATBOT_INTENT_CNN_MODEL_PATH",
        final_relative_path="text_cnn_intent/text_cnn_intent.h5",
        demo_relative_path="text_cnn_demo/text_cnn_demo.keras",
    )


def _default_intent_text_cnn_model_path():
    path, _source = _resolve_intent_text_cnn_model_path()
    return path


def _resolve_text_cnn_model_source(model_path):
    try:
        candidate = Path(model_path).resolve()
    except Exception:
        return "manual_override"

    accom_path, accom_source = _resolve_accommodation_text_cnn_model_path()
    intent_path, intent_source = _resolve_intent_text_cnn_model_path()
    try:
        if candidate == accom_path.resolve():
            return f"accommodation:{accom_source}"
    except Exception:
        pass
    try:
        if candidate == intent_path.resolve():
            return f"intent:{intent_source}"
    except Exception:
        pass
    return "manual_override"


def _default_label_map_path_for_model(model_path):
    path = Path(model_path)
    return path.parent / "label_map.json"


def _default_vocab_path_candidates_for_model(model_path):
    path = Path(model_path)
    return [
        path.parent / "text_cnn_intent_vocab.json",
        path.parent / f"{path.stem}_vocab.json",
        path.parent / "text_cnn_vocab.json",
    ]


def _load_saved_vectorizer_vocab(model_path):
    for vocab_path in _default_vocab_path_candidates_for_model(model_path):
        if not vocab_path.exists():
            continue
        try:
            payload = json.loads(vocab_path.read_text(encoding="utf-8"))
            if isinstance(payload, list) and payload:
                return [str(token) for token in payload], str(vocab_path)
        except Exception:
            continue
    return None, ""


def _default_text_cnn_repair_dataset_path():
    configured = str(os.getenv("CHATBOT_TEXT_CNN_REPAIR_DATASET", "")).strip()
    if configured:
        return Path(configured)
    root = Path(__file__).resolve().parent.parent / "thesis_data_templates"
    preferred = root / "text_cnn_messages_final_expanded_v3_clean.csv"
    if preferred.exists():
        return preferred
    return root / "text_cnn_messages_final_merged.csv"


def _load_repair_corpus(dataset_path):
    if pd is None:
        return [], "pandas_not_installed"
    path = Path(dataset_path)
    if not path.exists():
        return [], f"repair_dataset_not_found:{path}"
    try:
        # Pandas keeps this robust with UTF-8/BOM CSV variants.
        df = pd.read_csv(path, encoding="utf-8-sig")
    except Exception as exc:
        return [], f"repair_dataset_error:{exc}"
    if "message_text" not in df.columns:
        return [], "repair_dataset_missing_message_text"
    corpus = (
        df["message_text"]
        .fillna("")
        .astype(str)
        .str.strip()
        .tolist()
    )
    corpus = [row for row in corpus if row]
    if not corpus:
        return [], "repair_dataset_empty"
    return corpus, ""


def _repair_text_vectorization_table(loaded_model, model_path=None):
    if tf is None:
        return None, "tensorflow_not_installed"

    try:
        old_vectorizer = None
        for layer in loaded_model.layers:
            if isinstance(layer, tf.keras.layers.TextVectorization):
                old_vectorizer = layer
                break
        if old_vectorizer is None:
            return None, "repair_vectorizer_missing"

        embedding_layer = loaded_model.get_layer("embedding")
        conv_layer = loaded_model.get_layer("conv1d")
        dense_layer = loaded_model.get_layer("class_probs")

        emb_weights = embedding_layer.get_weights()
        conv_weights = conv_layer.get_weights()
        dense_weights = dense_layer.get_weights()
        if not emb_weights or not conv_weights or not dense_weights:
            return None, "repair_weights_missing"

        vocab_size, embedding_dim = emb_weights[0].shape
        kernel_size, conv_in_dim, conv_filters = conv_weights[0].shape
        if conv_in_dim != embedding_dim:
            return None, "repair_shape_mismatch"

        dense_in, class_count = dense_weights[0].shape
        if dense_in != conv_filters:
            return None, "repair_dense_shape_mismatch"

        vec_cfg = old_vectorizer.get_config()
        sequence_length = int(vec_cfg.get("output_sequence_length") or 24)
        max_tokens = int(vec_cfg.get("max_tokens") or vocab_size)
        standardize = vec_cfg.get("standardize") or "lower_and_strip_punctuation"

        text_input = tf.keras.Input(shape=(1,), dtype=tf.string, name="text")
        vectorizer = tf.keras.layers.TextVectorization(
            max_tokens=max_tokens,
            output_mode="int",
            output_sequence_length=sequence_length,
            standardize=standardize,
            name=old_vectorizer.name,
        )
        x = vectorizer(text_input)
        x = tf.keras.layers.Embedding(
            input_dim=vocab_size,
            output_dim=embedding_dim,
            name="embedding",
        )(x)
        x = tf.keras.layers.Conv1D(
            filters=conv_filters,
            kernel_size=kernel_size,
            activation="relu",
            name="conv1d",
        )(x)
        x = tf.keras.layers.GlobalMaxPooling1D(name="global_max_pooling1d")(x)
        out = tf.keras.layers.Dense(
            class_count,
            activation="softmax",
            name="class_probs",
        )(x)
        repaired_model = tf.keras.Model(inputs=text_input, outputs=out)
        repaired_model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

        saved_vocab, _saved_vocab_path = _load_saved_vectorizer_vocab(model_path) if model_path else (None, "")
        if saved_vocab:
            vectorizer.set_vocabulary(saved_vocab)
        else:
            corpus, corpus_err = _load_repair_corpus(_default_text_cnn_repair_dataset_path())
            if not corpus:
                return None, corpus_err
            vectorizer.adapt(tf.data.Dataset.from_tensor_slices(corpus).batch(32))
        repaired_model.get_layer("embedding").set_weights(emb_weights)
        repaired_model.get_layer("conv1d").set_weights(conv_weights)
        repaired_model.get_layer("class_probs").set_weights(dense_weights)
        # Smoke test: fail fast if lookup table is still not initialized.
        repaired_model.predict(np.array(["healthcheck"], dtype=object), verbose=0)
        return repaired_model, ""
    except Exception as exc:
        return None, f"repair_failed:{exc}"


def _load_text_cnn_model(model_path=None):
    global _TEXT_CNN_MODEL_CACHE, _TEXT_CNN_MODEL_PATH_CACHE

    if tf is None:
        return None, "tensorflow_not_installed"

    resolved_path = Path(model_path or _default_text_cnn_model_path())
    artifact_source = _resolve_text_cnn_model_source(resolved_path)
    if not resolved_path.exists():
        logger.warning(
            "Text-CNN artifact missing | path=%s | source=%s",
            str(resolved_path),
            artifact_source,
        )
        return None, f"model_not_found:{resolved_path}"

    if _TEXT_CNN_MODEL_CACHE is not None and _TEXT_CNN_MODEL_PATH_CACHE == str(resolved_path):
        logger.info(
            "Text-CNN model cache hit | path=%s | source=%s",
            str(resolved_path),
            artifact_source,
        )
        return _TEXT_CNN_MODEL_CACHE, None

    try:
        loaded = tf.keras.models.load_model(resolved_path, compile=False)
        try:
            loaded.predict(np.array(["healthcheck"], dtype=object), verbose=0)
        except Exception as predict_exc:
            lowered = str(predict_exc).lower()
            if "table not initialized" in lowered:
                repaired, repair_err = _repair_text_vectorization_table(
                    loaded,
                    model_path=resolved_path,
                )
                if repaired is None:
                    return None, f"model_predict_error:{predict_exc}|{repair_err}"
                loaded = repaired
            else:
                return None, f"model_predict_error:{predict_exc}"
        _TEXT_CNN_MODEL_CACHE = loaded
        _TEXT_CNN_MODEL_PATH_CACHE = str(resolved_path)
        logger.info(
            "Text-CNN model loaded | path=%s | source=%s",
            str(resolved_path),
            artifact_source,
        )
        return _TEXT_CNN_MODEL_CACHE, None
    except Exception as exc:
        logger.exception(
            "Text-CNN load failed | path=%s | source=%s | error=%s",
            str(resolved_path),
            artifact_source,
            str(exc),
        )
        return None, f"model_load_error:{exc}"


def _load_text_cnn_labels(label_map_path=None):
    resolved_label_map = Path(label_map_path or _default_label_map_path_for_model(_default_text_cnn_model_path()))
    label_map_path = resolved_label_map
    if not label_map_path.exists():
        return None, f"label_map_not_found:{label_map_path}"
    try:
        payload = json.loads(label_map_path.read_text(encoding="utf-8"))
        classes = payload.get("classes") or []
        if not isinstance(classes, list) or not classes:
            return None, "invalid_label_map"
        return [str(c) for c in classes], None
    except Exception as exc:
        return None, f"label_map_error:{exc}"


def _predict_text_cnn_labels(*, text, model_path, label_map_path):
    message = str(text or "").strip()
    if not message:
        return None, "empty_text"

    model, model_err = _load_text_cnn_model(model_path=model_path)
    if model is None:
        return None, model_err

    classes, label_err = _load_text_cnn_labels(label_map_path=label_map_path)
    if classes is None:
        return None, label_err

    try:
        probs = model.predict(np.array([message], dtype=object), verbose=0)[0]
        pred_idx = int(np.argmax(probs))
        top_idx = np.argsort(probs)[::-1][:3]
        return {
            "predicted_class": classes[pred_idx],
            "confidence": float(probs[pred_idx]),
            "top_3": [
                {"label": classes[int(i)], "confidence": float(probs[int(i)])}
                for i in top_idx
            ],
            "label_space": [str(c) for c in classes],
        }, None
    except Exception as exc:
        return None, f"predict_error:{exc}"


def _predict_accommodation_class_from_text(text):
    model_path, artifact_source = _resolve_accommodation_text_cnn_model_path()
    label_map_path = _default_label_map_path_for_model(model_path)
    payload, err = _predict_text_cnn_labels(
        text=text,
        model_path=model_path,
        label_map_path=label_map_path,
    )
    if payload is None:
        return None, err
    payload.pop("label_space", None)
    payload["artifact_source"] = artifact_source
    return payload, None


def _format_cnn_prediction_for_chat(cnn_prediction):
    if not cnn_prediction:
        return ""

    def _to_domain_label(raw_label):
        label = str(raw_label or "").strip().lower()
        remap = {
            "hostel": "hotel",
            "transient_house": "hotel",
            "transient house": "hotel",
        }
        return remap.get(label, label or "unknown")

    predicted = _to_domain_label(cnn_prediction.get("predicted_class", "unknown"))
    confidence = float(cnn_prediction.get("confidence", 0.0))
    top_3 = cnn_prediction.get("top_3") or []

    lines = [
        "No DB match yet for the current filters.",
        f"CNN predicted type: {predicted}",
        f"Confidence: {confidence:.3f}",
    ]

    if top_3:
        lines.append("Top 3 classes:")
        for item in top_3[:3]:
            label = _to_domain_label(item.get("label", "unknown"))
            score = float(item.get("confidence", 0.0))
            lines.append(f"- {label}: {score:.3f}")

    lines.append(
        "Note: Current accommodation DB results depend on available hotel/inn room records and filters."
    )
    return "\n".join(lines)


def _normalize_data_source(value):
    source = str(value or "").strip().lower()
    return source if source in _VALID_DATA_SOURCES else "unlabeled"


def _resolve_data_source(request=None, payload=None):
    if isinstance(payload, dict):
        payload_source = payload.get("data_source")
        if payload_source not in (None, ""):
            return _normalize_data_source(payload_source)

    if request is not None:
        try:
            header_source = request.headers.get("X-Data-Source", "")
        except Exception:
            header_source = ""
        if header_source:
            return _normalize_data_source(header_source)

    default_source = os.getenv("CHATBOT_DATA_SOURCE", "unlabeled")
    return _normalize_data_source(default_source)


def _safe_log_system_metric(
    *,
    endpoint,
    response_time_ms,
    success_flag,
    status_code=None,
    error_message="",
    request=None,
):
    try:
        SystemMetricLog.objects.create(
            module="chat",
            endpoint=endpoint,
            response_time_ms=max(int(response_time_ms), 0),
            success_flag=bool(success_flag),
            status_code=status_code,
            error_message=(error_message or "")[:1000],
            data_source=_resolve_data_source(request=request),
        )
    except Exception:
        # Logging must never break the chatbot response path.
        pass


def _safe_log_recommendation_event(request, intent):
    try:
        user = getattr(request, "user", None)
        if not user or not getattr(user, "is_authenticated", False):
            return

        item_ref = "chat:tour_recommendation_request"
        if intent in ("get_accommodation_recommendation", "gethotelrecommendation"):
            item_ref = "chat:accommodation_recommendation_request"

        session_key = ""
        if hasattr(request, "session"):
            session_key = request.session.session_key or ""

        RecommendationEvent.objects.create(
            user=user,
            event_type="view",
            item_ref=item_ref,
            session_id=session_key,
            data_source=_resolve_data_source(request=request),
        )
    except Exception:
        # Event logging is optional and should not affect chatbot behavior.
        pass


def _compose_click_item_ref(payload):
    if not isinstance(payload, dict):
        return "chat:accommodation_recommendation_click"

    room_id = _to_int(payload.get("room_id"), default=0)
    accom_id = _to_int(payload.get("accom_id"), default=0)
    rank = _to_int(payload.get("rank"), default=0)
    mode = str(payload.get("scoring_mode") or "").strip().lower()[:40]

    parts = []
    if room_id > 0:
        parts.append(f"room:{room_id}")
    if accom_id > 0:
        parts.append(f"accom:{accom_id}")
    if rank > 0:
        parts.append(f"rank:{rank}")
    if mode:
        parts.append(f"mode:{mode}")

    return "|".join(parts) if parts else "chat:accommodation_recommendation_click"


def _safe_log_recommendation_click(request, payload):
    try:
        user = getattr(request, "user", None)
        if not user or not getattr(user, "is_authenticated", False):
            return False, ""

        session_key = ""
        if hasattr(request, "session"):
            session_key = request.session.session_key or ""

        rating_score = _to_int(payload.get("rating_score"), default=0) if isinstance(payload, dict) else 0
        dwell_time = _to_int(payload.get("dwell_time_sec"), default=0) if isinstance(payload, dict) else 0
        item_ref = _compose_click_item_ref(payload)

        RecommendationEvent.objects.create(
            user=user,
            event_type="click",
            item_ref=item_ref[:100],
            rating_score=rating_score if 1 <= rating_score <= 5 else None,
            dwell_time_sec=dwell_time if dwell_time >= 0 else None,
            session_id=session_key,
            data_source=_resolve_data_source(request=request, payload=payload),
        )
        return True, item_ref
    except Exception:
        return False, ""


def _safe_log_chat_step_event(
    request,
    *,
    event_type="view",
    item_ref="",
    rating_score=None,
    dwell_time_sec=None,
):
    try:
        user = getattr(request, "user", None)
        if not user or not getattr(user, "is_authenticated", False):
            return
        normalized_type = str(event_type or "").strip().lower()
        if normalized_type not in {"view", "click", "save", "rate", "book"}:
            normalized_type = "view"
        session_key = ""
        if hasattr(request, "session"):
            session_key = request.session.session_key or ""
        RecommendationEvent.objects.create(
            user=user,
            event_type=normalized_type,
            item_ref=str(item_ref or "chat:step").strip()[:100],
            rating_score=rating_score if isinstance(rating_score, int) and 1 <= rating_score <= 5 else None,
            dwell_time_sec=dwell_time_sec if isinstance(dwell_time_sec, int) and dwell_time_sec >= 0 else None,
            session_id=session_key,
            data_source=_resolve_data_source(request=request),
        )
    except Exception:
        pass


def _safe_log_chat_runtime_event(request, *, event_key, detail=""):
    normalized_key = str(event_key or "").strip().lower().replace(" ", "_")[:80]
    if not normalized_key:
        return
    _safe_log_chat_step_event(
        request,
        event_type="view",
        item_ref=f"chat:{normalized_key}",
    )
    try:
        _safe_log_system_metric(
            endpoint=f"{getattr(request, 'path', '/api/chat/')}#event:{normalized_key}",
            response_time_ms=0,
            success_flag=True,
            status_code=200,
            error_message=str(detail or "")[:200],
            request=request,
        )
    except Exception:
        pass


def _safe_log_step_events_from_response(request, *, intent, response_payload):
    if not isinstance(response_payload, dict):
        return
    actor = _resolve_chat_actor(request)
    is_guest = actor.get("role") == "guest"
    if response_payload.get("recommendation_trace"):
        if intent in ("get_accommodation_recommendation", "gethotelrecommendation"):
            _safe_log_chat_step_event(
                request,
                event_type="view",
                item_ref="chat:accommodation_recommendation_rendered",
            )
            if is_guest:
                funnel = _GUEST_FUNNEL_EVENT_MAP.get("recommendation_shown") or {}
                _safe_log_chat_step_event(
                    request,
                    event_type=str(funnel.get("event_type") or "view"),
                    item_ref=str(funnel.get("item_ref") or "chat:funnel_recommendation_shown"),
                )
        elif intent in ("get_recommendation", "gettourrecommendation"):
            _safe_log_chat_step_event(
                request,
                event_type="view",
                item_ref="chat:tour_recommendation_rendered",
            )

    if response_payload.get("booking_id") and response_payload.get("show_feedback_prompt"):
        _safe_log_chat_step_event(
            request,
            event_type="book",
            item_ref="chat:accommodation_booking_confirmed",
        )
    elif response_payload.get("booking_id"):
        _safe_log_chat_step_event(
            request,
            event_type="save",
            item_ref="chat:accommodation_booking_draft_or_pending",
        )
    if is_guest and response_payload.get("room_id") and intent in (
        "book_accommodation",
        "bookhotel",
        "book_hotel",
        "reserve_accommodation",
    ):
        click_ev = _GUEST_FUNNEL_EVENT_MAP.get("book_button_clicked") or {}
        flow_ev = _GUEST_FUNNEL_EVENT_MAP.get("booking_flow_started") or {}
        _safe_log_chat_step_event(
            request,
            event_type=str(click_ev.get("event_type") or "click"),
            item_ref=str(click_ev.get("item_ref") or "chat:funnel_book_button_clicked"),
        )
        _safe_log_chat_step_event(
            request,
            event_type=str(flow_ev.get("event_type") or "save"),
            item_ref=str(flow_ev.get("item_ref") or "chat:funnel_booking_flow_started"),
        )

    if response_payload.get("billing_link"):
        _safe_log_chat_step_event(
            request,
            event_type="view",
            item_ref="chat:lgu_payment_handoff_ready",
        )
        if is_guest:
            billing_ev = _GUEST_FUNNEL_EVENT_MAP.get("billing_link_shown") or {}
            _safe_log_chat_step_event(
                request,
                event_type=str(billing_ev.get("event_type") or "view"),
                item_ref=str(billing_ev.get("item_ref") or "chat:funnel_billing_link_shown"),
            )
    if is_guest and response_payload.get("booking_id"):
        completed_ev = _GUEST_FUNNEL_EVENT_MAP.get("booking_completed") or {}
        _safe_log_chat_step_event(
            request,
            event_type=str(completed_ev.get("event_type") or "book"),
            item_ref=str(completed_ev.get("item_ref") or "chat:funnel_booking_completed"),
        )


def _safe_log_recommendation_result(request, intent, reply, params, cnn_prediction=None):
    return _safe_log_recommendation_result_with_metadata(
        request,
        intent,
        reply,
        params,
        cnn_prediction=cnn_prediction,
    )


def _safe_log_recommendation_result_with_metadata(
    request,
    intent,
    reply,
    params,
    cnn_prediction=None,
    *,
    message_text="",
    recommended_items=None,
    booking_linkage=None,
):
    try:
        user = getattr(request, "user", None)
        if not user or not getattr(user, "is_authenticated", False):
            return

        if intent not in (
            "get_recommendation",
            "gettourrecommendation",
            "get_tourism_information",
            "get_accommodation_recommendation",
            "gethotelrecommendation",
            "book_accommodation",
            "bookhotel",
            "book_hotel",
            "reserve_accommodation",
        ):
            return

        context_payload = {
            "intent": intent,
            "params": params if isinstance(params, dict) else {},
        }
        if message_text:
            context_payload["message_text"] = str(message_text)[:2000]
        if hasattr(request, "session"):
            context_payload["session_id"] = request.session.session_key or ""
        if cnn_prediction:
            context_payload["cnn_prediction"] = cnn_prediction
        if booking_linkage and isinstance(booking_linkage, dict):
            context_payload["booking_outcome"] = booking_linkage
        context_payload["data_source"] = _resolve_data_source(request=request)

        items_payload = []
        if isinstance(recommended_items, list):
            for item in recommended_items[:10]:
                if isinstance(item, dict):
                    items_payload.append(item)

        # Fallback to reply text if structured recommendation items are unavailable.
        if not items_payload:
            items_payload = [{"reply_text": str(reply or "")}]

        mode_counts = Counter()
        dt_source_counts = Counter()
        dt_scores = []
        for item in items_payload:
            if not isinstance(item, dict):
                continue
            mode = str(item.get("scoring_mode") or "").strip().lower()
            meta = item.get("meta") if isinstance(item.get("meta"), dict) else {}
            trace = meta.get("trace") if isinstance(meta.get("trace"), dict) else {}
            if not mode:
                mode = str(trace.get("scoring_mode") or "").strip().lower()
            if mode:
                mode_counts[mode] += 1

            dt_score = item.get("decision_tree_score")
            if not isinstance(dt_score, (int, float)):
                dt_score = trace.get("decision_tree_score")
            if isinstance(dt_score, (int, float)):
                dt_scores.append(float(dt_score))
            dt_source = item.get("decision_tree_source")
            if not isinstance(dt_source, str) or not dt_source.strip():
                dt_source = trace.get("decision_tree_source")
            dt_source = str(dt_source or "").strip().lower()
            if dt_source:
                dt_source_counts[dt_source] += 1

        if mode_counts:
            context_payload["hybrid_mode_counts"] = dict(mode_counts)
            context_payload["top_mode"] = mode_counts.most_common(1)[0][0]
        if dt_source_counts:
            context_payload["decision_tree_source_counts"] = dict(dt_source_counts)
            context_payload["decision_tree_top_source"] = dt_source_counts.most_common(1)[0][0]
        if dt_scores:
            context_payload["decision_tree_score_avg"] = round(sum(dt_scores) / len(dt_scores), 6)
            context_payload["decision_tree_score_count"] = len(dt_scores)
        if isinstance(cnn_prediction, dict):
            context_payload["cnn_artifact_source"] = str(cnn_prediction.get("artifact_source") or "")[:80]
            context_payload["cnn_predicted_class"] = str(cnn_prediction.get("predicted_class") or "")[:80]
            context_payload["cnn_confidence"] = float(cnn_prediction.get("confidence", 0.0) or 0.0)

        clicked_item_ref = ""
        if booking_linkage and isinstance(booking_linkage, dict):
            room_id = booking_linkage.get("room_id")
            booking_id = booking_linkage.get("booking_id")
            if room_id:
                clicked_item_ref = f"room:{room_id}"
            if room_id and booking_id:
                clicked_item_ref = f"room:{room_id}|booking:{booking_id}"

        RecommendationResult.objects.create(
            user=user,
            algorithm_version="v1-chat",
            context_json=context_payload,
            recommended_items_json=items_payload,
            top_k=max(len(items_payload), 1),
            clicked_item_ref=clicked_item_ref,
            data_source=_resolve_data_source(request=request),
        )
    except Exception:
        # Result logging should not affect chatbot behavior.
        pass


def _safe_log_chatbot_interaction(
    request,
    *,
    user_message,
    resolved_intent="",
    resolved_params=None,
    bot_response="",
    intent_classifier=None,
    response_nlg_source="",
    fallback_used=False,
    provenance=None,
):
    try:
        user = getattr(request, "user", None)
        if not user or not getattr(user, "is_authenticated", False):
            user = None

        params_payload = resolved_params if isinstance(resolved_params, dict) else {}
        params_payload = {str(k)[:60]: v for k, v in params_payload.items()}

        intent_classifier = intent_classifier if isinstance(intent_classifier, dict) else {}
        provenance_payload = provenance if isinstance(provenance, dict) else {}
        # Keep provenance compact and avoid accidentally logging large/sensitive payloads.
        compact_provenance = {
            "intent_source": str(intent_classifier.get("source") or "")[:80],
            "intent_confidence": float(intent_classifier.get("confidence", 0.0) or 0.0),
            "intent_error": str(intent_classifier.get("error") or "")[:160],
            "intent_artifact_source": str(intent_classifier.get("artifact_source") or "")[:80],
            "extra": provenance_payload,
        }

        ChatbotLog.objects.create(
            user=user,
            user_message=str(user_message or "")[:4000],
            resolved_intent=str(resolved_intent or "")[:80],
            resolved_params_json=params_payload,
            bot_response=str(bot_response or "")[:8000],
            intent_classifier_source=str(intent_classifier.get("source") or "")[:80],
            response_nlg_source=str(response_nlg_source or "")[:80],
            fallback_used=bool(fallback_used),
            provenance_json=compact_provenance,
            data_source=_resolve_data_source(request=request),
        )
    except Exception:
        # Canonical chat logging must never break chatbot responses.
        pass


def _chat_json_response(request, start_time, payload, status=200, error_message=""):
    response_payload = payload if isinstance(payload, dict) else payload
    try:
        if isinstance(response_payload, dict):
            if response_payload.get("needs_clarification"):
                existing_qr = response_payload.get("quick_replies")
                if not (isinstance(existing_qr, list) and existing_qr):
                    missing_slot = str(response_payload.get("missing_slot") or "").strip().lower()
                    clarification_qr = _slot_quick_replies(missing_slot)
                    if not clarification_qr:
                        clarification_qr = [{"label": "Help", "value": "help"}]
                    response_payload["quick_replies"] = clarification_qr
            fallback_text = str(response_payload.get("fulfillmentText") or "").strip()
            if not fallback_text:
                if response_payload.get("billing_link"):
                    response_payload["fulfillmentText"] = (
                        "I found a page/action for your request. Use the button below to continue."
                    )
                elif isinstance(response_payload.get("recommendation_trace"), list) and response_payload.get("recommendation_trace"):
                    response_payload["fulfillmentText"] = (
                        "Here are the recommendations I found based on your request."
                    )
                elif isinstance(response_payload.get("quick_replies"), list) and response_payload.get("quick_replies"):
                    response_payload["fulfillmentText"] = (
                        "Please choose one of the quick options below, or type your request."
                    )
                else:
                    response_payload["fulfillmentText"] = (
                        "I can help with your request. Please try rephrasing it in one sentence."
                    )

        context = getattr(request, "_chatbot_log_context", None)
        if isinstance(context, dict) and isinstance(response_payload, dict):
            bot_text = str(response_payload.get("fulfillmentText") or "").strip()
            context_nlg_source = str(
                response_payload.get("response_nlg_source")
                or context.get("response_nlg_source")
                or ""
            ).strip()
            if (
                bot_text
                and not context_nlg_source
                and status < 400
            ):
                resolved_intent = str(context.get("resolved_intent") or "").strip()
                user_message = str(context.get("user_message") or "").strip()
                finalized_text, finalized_source = generate_final_ai_response(
                    request=request,
                    intent=resolved_intent,
                    user_message=user_message,
                    backend_reply=bot_text,
                )
                if str(finalized_text or "").strip():
                    response_payload["fulfillmentText"] = str(finalized_text).strip()
                    bot_text = str(finalized_text).strip()
                context_nlg_source = str(finalized_source or "").strip()
                if context_nlg_source:
                    response_payload["response_nlg_source"] = context_nlg_source
                    context["response_nlg_source"] = context_nlg_source
            provenance = context.get("provenance") if isinstance(context.get("provenance"), dict) else {}
            detected_language = str(provenance.get("detected_language") or "").strip().lower()
            already_back_translated = "gemini_back_translate" in context_nlg_source
            if bot_text and detected_language not in ("", "en", "english") and not already_back_translated:
                translated = translate_to_user_language(bot_text, detected_language)
                translated_text = str(translated or "").strip()
                if translated_text:
                    response_payload["fulfillmentText"] = translated_text
                    if context_nlg_source:
                        context_nlg_source = f"{context_nlg_source}|gemini_back_translate"
                    else:
                        context_nlg_source = "gemini_back_translate"
                    response_payload["response_nlg_source"] = context_nlg_source
                    context["response_nlg_source"] = context_nlg_source
                    provenance["response_translated_to_user_language"] = True
                    context["provenance"] = provenance
    except Exception:
        # Translation/normalization should not block response delivery.
        response_payload = payload

    response = JsonResponse(response_payload, status=status)
    elapsed_ms = int((time.perf_counter() - start_time) * 1000)
    _safe_log_system_metric(
        endpoint=request.path,
        response_time_ms=elapsed_ms,
        success_flag=(200 <= status < 400),
        status_code=status,
        error_message=error_message,
        request=request,
    )
    try:
        context = getattr(request, "_chatbot_log_context", None)
        if isinstance(context, dict):
            payload_for_logs = response_payload if isinstance(response_payload, dict) else {}
            if payload_for_logs.get("needs_clarification"):
                _safe_log_chat_runtime_event(
                    request,
                    event_key="clarification_triggered",
                    detail=str(payload_for_logs.get("missing_slot") or ""),
                )
            if isinstance(payload_for_logs.get("no_match_reasons"), list) and payload_for_logs.get("no_match_reasons"):
                _safe_log_chat_runtime_event(
                    request,
                    event_key="no_match_database_result",
                    detail=",".join([str(v) for v in payload_for_logs.get("no_match_reasons", [])][:4]),
                )
            nlg_source_payload = str(payload_for_logs.get("response_nlg_source") or context.get("response_nlg_source") or "")
            if "guardrail_fallback" in nlg_source_payload:
                _safe_log_chat_runtime_event(
                    request,
                    event_key="output_guardrail_triggered",
                    detail=nlg_source_payload[:120],
                )
            provenance_payload = context.get("provenance") if isinstance(context.get("provenance"), dict) else {}
            runtime_models = provenance_payload.get("runtime_models") if isinstance(provenance_payload.get("runtime_models"), dict) else {}
            if "gemini" in nlg_source_payload:
                runtime_models["llm_provider"] = "gemini"
                runtime_models["gemini_model"] = str(os.getenv("GEMINI_MODEL", "gemini-1.5-flash") or "").strip() or "gemini-1.5-flash"
            elif "openai" in nlg_source_payload:
                runtime_models["llm_provider"] = "openai"
                runtime_models["openai_model"] = str(os.getenv("OPENAI_MODEL", "gpt-4o-mini") or "").strip() or "gpt-4o-mini"
            if nlg_source_payload:
                runtime_models["nlg_source"] = nlg_source_payload[:120]

            intent_classifier = context.get("intent_classifier") if isinstance(context.get("intent_classifier"), dict) else {}
            artifact_source = str(intent_classifier.get("artifact_source") or "").strip()
            if artifact_source:
                runtime_models["intent_cnn_artifact_source"] = artifact_source[:80]
            try:
                dt_status = get_decision_tree_runtime_status(force_reload=False)
                runtime_models["decision_tree_artifact_source"] = str(dt_status.get("source") or "")[:80]
                runtime_models["decision_tree_fallback_used"] = bool(dt_status.get("fallback_used"))
            except Exception:
                pass

            if runtime_models:
                provenance_payload["runtime_models"] = runtime_models
                context["provenance"] = provenance_payload
            bot_text = str(payload_for_logs.get("fulfillmentText") or "")
            _safe_log_chatbot_interaction(
                request,
                user_message=context.get("user_message", ""),
                resolved_intent=context.get("resolved_intent", ""),
                resolved_params=context.get("resolved_params", {}),
                bot_response=bot_text,
                intent_classifier=context.get("intent_classifier", {}),
                response_nlg_source=(
                    payload_for_logs.get("response_nlg_source")
                    or context.get("response_nlg_source", "")
                ),
                fallback_used=bool(
                    context.get("fallback_used")
                    or payload_for_logs.get("recommendation_fallback")
                    or (
                        context.get("response_nlg_source")
                        and context.get("response_nlg_source") != "openai_nlg"
                    )
                ),
                provenance=context.get("provenance", {}),
            )
    except Exception:
        pass
    return response


def _normalize_survey_response_items(raw_items):
    normalized = []
    if not isinstance(raw_items, list):
        return normalized
    for item in raw_items:
        if not isinstance(item, dict):
            continue
        code = str(item.get("statement_code") or "").strip().upper()[:30]
        score = _to_int(item.get("likert_score"), default=0)
        comment = str(item.get("comment") or "").strip()[:1000]
        if not code or score < 1 or score > 5:
            continue
        normalized.append(
            {
                "statement_code": code,
                "likert_score": score,
                "comment": comment,
            }
        )
    return normalized


def _load_chat_state(request):
    return _load_chat_state_service(
        request,
        session_key=_CHAT_STATE_SESSION_KEY,
        ttl_seconds=_CHAT_STATE_TTL_SECONDS,
    )


def _save_chat_state(request, state):
    _save_chat_state_service(
        request,
        state,
        session_key=_CHAT_STATE_SESSION_KEY,
    )


def _clear_chat_state(request):
    _clear_chat_state_service(
        request,
        session_key=_CHAT_STATE_SESSION_KEY,
    )


def _load_saved_chat_preferences(request):
    raw = request.session.get(_CHAT_PREFERENCE_SESSION_KEY, {})
    return raw if isinstance(raw, dict) else {}


def _save_saved_chat_preferences(request, prefs):
    payload = prefs if isinstance(prefs, dict) else {}
    request.session[_CHAT_PREFERENCE_SESSION_KEY] = payload
    request.session.modified = True


def _clear_saved_chat_preferences(request):
    if _CHAT_PREFERENCE_SESSION_KEY in request.session:
        del request.session[_CHAT_PREFERENCE_SESSION_KEY]
        request.session.modified = True


def _is_reset_command(message):
    text = str(message or "").strip().lower()
    if not text:
        return False
    reset_phrases = {
        "reset",
        "start over",
        "startover",
        "clear",
        "clear chat",
        "clear state",
    }
    return text in reset_phrases or any(phrase in text for phrase in (" reset", "start over", "startover"))


def _normalize_iso_date(value):
    raw = str(value or "").strip()
    if not raw:
        return ""
    raw = re.sub(r"\s+", " ", raw).strip()
    raw = re.sub(r",\s*", ", ", raw)

    match = re.fullmatch(r"(\d{4})-(\d{1,2})-(\d{1,2})", raw)
    if match:
        year = int(match.group(1))
        month = int(match.group(2))
        day = int(match.group(3))
        try:
            return datetime(year, month, day).date().isoformat()
        except Exception:
            return ""

    for fmt in ("%B %d, %Y", "%b %d, %Y", "%B %d %Y", "%b %d %Y"):
        try:
            return datetime.strptime(raw, fmt).date().isoformat()
        except ValueError:
            continue
    return ""


def _has_stay_details(params):
    if not isinstance(params, dict):
        return False
    has_dates = bool(str(params.get("check_in") or "").strip()) and bool(
        str(params.get("check_out") or "").strip()
    )
    nights = _to_int(params.get("nights"), default=0)
    return has_dates or nights > 0


def _next_accommodation_clarifying_question(params):
    if not isinstance(params, dict):
        params = {}

    missing_fields = []

    company_type = str(params.get("company_type") or "").strip().lower()
    if company_type not in ("hotel", "inn", "either"):
        missing_fields.append(("company_type", "Accommodation type (hotel / inn / either)"))

    location = str(params.get("location") or "").strip()
    location_anchor = str(params.get("location_anchor") or "").strip()
    preference_tags = params.get("preference_tags") if isinstance(params.get("preference_tags"), list) else []
    if (not location) and (not location_anchor) and (not preference_tags):
        missing_fields.append(("location", "Preferred area/location in Bayawan"))

    guests = _to_int(params.get("guests"), default=0)
    if guests <= 0:
        missing_fields.append(("guests", "Number of guests"))

    if not _has_stay_details(params):
        missing_fields.append(
            ("stay_details", "Stay details: check-in/check-out dates (YYYY-MM-DD) or number of nights")
        )

    budget = _to_int(params.get("budget"), default=0)
    if budget <= 0:
        missing_fields.append(("budget", "Budget per night in PHP"))

    if len(missing_fields) == 1:
        field = missing_fields[0][0]
        return field, _build_dynamic_accommodation_slot_question(field, params)

    if len(missing_fields) > 1:
        acknowledged_parts = []
        if company_type in ("hotel", "inn", "either"):
            if company_type == "either":
                acknowledged_parts.append("accommodation type: hotel or inn")
            else:
                acknowledged_parts.append(f"accommodation type: {company_type}")
        if location:
            acknowledged_parts.append(f"location: {location}")
        if budget > 0:
            acknowledged_parts.append(f"budget: PHP {budget}")
        if guests > 0:
            acknowledged_parts.append(f"guests: {guests}")
        if _has_stay_details(params):
            acknowledged_parts.append("stay details received")

        next_field = missing_fields[0][0]
        next_question = _build_dynamic_accommodation_slot_question(next_field, params)
        if acknowledged_parts:
            return (next_field, format_acknowledged_details(acknowledged_parts, next_question))
        return next_field, next_question

    return None, ""


def _stable_choice(options, seed_text):
    if not isinstance(options, list) or not options:
        return ""
    seed = str(seed_text or "")
    if not seed:
        return str(options[0])
    digest = hashlib.md5(seed.encode("utf-8")).hexdigest()
    idx = int(digest[:8], 16) % len(options)
    return str(options[idx])


def _build_dynamic_accommodation_slot_question(field, params):
    slot = str(field or "").strip().lower()
    params = params if isinstance(params, dict) else {}
    company_type = str(params.get("company_type") or "").strip().lower()
    location = str(params.get("location") or "").strip()
    guests = _to_int(params.get("guests"), default=0)
    budget = _to_int(params.get("budget"), default=0)
    prefix_bits = []
    if company_type in ("hotel", "inn"):
        prefix_bits.append(company_type)
    if location:
        prefix_bits.append(location)
    if guests > 0:
        prefix_bits.append(f"{guests} guest(s)")
    if budget > 0:
        prefix_bits.append(f"PHP {budget}")
    prefix = ""
    if prefix_bits:
        prefix = f"So far: {', '.join(prefix_bits)}. "

    if slot == "company_type":
        variants = [
            "Do you prefer a hotel, an inn, or either?",
            "For your stay, should I prioritize hotel options, inn options, or both?",
            "Which accommodation type do you want first: hotel, inn, or either?",
        ]
        return prefix + _stable_choice(variants, f"{slot}|{company_type}|{location}")
    if slot == "location":
        variants = [
            "Which area in Bayawan should I prioritize?",
            "What area in Bayawan would you like me to focus on?",
            "Please share your preferred area in Bayawan so I can narrow results.",
        ]
        return prefix + _stable_choice(variants, f"{slot}|{company_type}|{guests}")
    if slot == "budget":
        variants = [
            "What is your budget per night in PHP?",
            "Please share your preferred nightly budget in PHP.",
            "How much is your target budget per night (PHP)?",
        ]
        return prefix + _stable_choice(variants, f"{slot}|{location}|{guests}")
    if slot == "guests":
        variants = [
            "How many guests will stay in the room?",
            "Please indicate the total number of guests.",
            "How many people should the room accommodate?",
        ]
        return prefix + _stable_choice(variants, f"{slot}|{company_type}|{budget}")
    if slot == "stay_details":
        variants = [
            "Please provide your check-in/check-out dates (YYYY-MM-DD), or the number of nights.",
            "What are your stay dates (check-in/check-out), or how many nights do you plan to stay?",
            "To continue, share your check-in/check-out dates (YYYY-MM-DD) or tell me how many nights.",
        ]
        return prefix + _stable_choice(variants, f"{slot}|{location}|{budget}")
    return prefix + get_accommodation_slot_question(slot)


def _looks_like_slot_update(params):
    if not isinstance(params, dict):
        return False
    slot_keys = {
        "budget",
        "guests",
        "adults",
        "children",
        "location",
        "company_type",
        "check_in",
        "check_out",
        "nights",
        "amenities",
        "amenity",
        "preference_tags",
        "prefer_low_price",
        "clear_budget",
        "clear_location_anchor",
        "broaden_location",
        "broaden_company_type",
    }
    return any(key in params for key in slot_keys)


def _looks_like_tour_request(message):
    text = str(message or "").strip().lower()
    if not text:
        return False
    tour_keywords = ("tour", "itinerary", "schedule", "destination", "package")
    if any(keyword in text for keyword in tour_keywords):
        return True
    timeframe_hint = _extract_tour_timeframe_hint(text)
    if timeframe_hint and any(
        marker in text
        for marker in ("event", "events", "activity", "activities", "what can i do", "things to do", "happening")
    ):
        return True
    return False


def _extract_tour_timeframe_hint(message):
    text = str(message or "").strip().lower()
    if not text:
        return ""
    if "this weekend" in text or re.search(r"\bweekend\b", text):
        return "this weekend"
    if re.search(r"\btours?\s+today\b", text) or re.search(r"\bevents?\s+today\b", text) or re.search(r"\btoday\b", text):
        return "today"
    if "this week" in text or re.search(r"\bevents?\s+this week\b", text):
        return "this week"
    if "upcoming" in text:
        return "upcoming schedules"
    return ""


def _message_mentions_guest_count(message):
    text = str(message or "").strip().lower()
    if not text:
        return False
    return bool(re.search(r"\b\d+\s*(guest|guests|people|person|pax|bisita|katao)\b", text))


def _is_personalization_decline_message(message):
    text = str(message or "").strip().lower()
    if not text:
        return False
    decline_phrases = {
        "no",
        "nope",
        "nah",
        "different budget",
        "no thanks",
        "not now",
    }
    if text in decline_phrases:
        return True
    if re.search(r"^(no|nope|nah)\b", text):
        return True
    return ("different budget" in text) or ("give list" in text) or ("show list" in text)


def _is_personalization_accept_message(message):
    text = str(message or "").strip().lower()
    if not text:
        return False
    accept_phrases = {
        "yes",
        "y",
        "yes please",
        "sure",
        "ok",
        "okay",
        "same",
        "similar",
        "go ahead",
    }
    return (text in accept_phrases) or bool(re.search(r"^(yes|y)\b", text))


def _is_booking_confirmation_decline(message):
    text = str(message or "").strip().lower()
    if not text:
        return False
    return text in {"no", "nope", "cancel", "stop", "do not book", "don't book"}


def _is_booking_confirmation_accept(message):
    text = str(message or "").strip().lower()
    if not text:
        return False
    return text in {"yes", "y", "yes please", "confirm", "book it", "proceed"}


def _infer_user_accommodation_baseline(user):
    if not user or not getattr(user, "is_authenticated", False):
        return {}

    today = timezone.now().date()
    past_bookings = list(
        AccommodationBooking.objects.select_related("accommodation", "room")
        .filter(guest=user, check_out__lt=today)
    )
    if not past_bookings:
        return {}

    nightly_rates = []
    type_counter = Counter()
    location_counter = Counter()

    for booking in past_bookings:
        room = getattr(booking, "room", None)
        if room is not None and getattr(room, "price_per_night", None) not in (None, ""):
            nightly_rates.append(Decimal(str(room.price_per_night)))
        else:
            nights = max((booking.check_out - booking.check_in).days, 1)
            if booking.total_amount not in (None, "") and nights > 0:
                nightly_rates.append(Decimal(str(booking.total_amount)) / Decimal(nights))

        accom = getattr(booking, "accommodation", None)
        if accom is None:
            continue

        raw_type = str(getattr(accom, "company_type", "") or "").strip().lower()
        normalized_type = ""
        if "hotel" in raw_type and "inn" in raw_type:
            normalized_type = "either"
        elif "hotel" in raw_type:
            normalized_type = "hotel"
        elif "inn" in raw_type:
            normalized_type = "inn"
        if normalized_type:
            type_counter[normalized_type] += 1

        location_value = str(getattr(accom, "location", "") or "").strip().lower()
        if location_value:
            location_counter[location_value] += 1

    if not nightly_rates:
        return {}

    sorted_rates = sorted(nightly_rates)
    mid = len(sorted_rates) // 2
    typical_rate = sorted_rates[mid]

    common_type = type_counter.most_common(1)[0][0] if type_counter else ""
    common_location = location_counter.most_common(1)[0][0] if location_counter else ""

    return {
        "typical_budget": int(typical_rate),
        "budget_min": int(sorted_rates[0]),
        "budget_max": int(sorted_rates[-1]),
        "common_company_type": common_type,
        "common_location": common_location,
        "sample_size": len(past_bookings),
    }


def _build_personalization_defaults(params, baseline):
    if not isinstance(params, dict) or not isinstance(baseline, dict):
        return {}

    defaults = {}

    budget = _to_int(params.get("budget"), default=0)
    if budget <= 0 and _to_int(baseline.get("typical_budget"), default=0) > 0:
        defaults["budget"] = _to_int(baseline.get("typical_budget"), default=0)

    company_type = str(params.get("company_type") or "").strip().lower()
    baseline_type = str(baseline.get("common_company_type") or "").strip().lower()
    if company_type not in ("hotel", "inn", "either") and baseline_type in ("hotel", "inn", "either"):
        defaults["company_type"] = baseline_type

    location = str(params.get("location") or "").strip().lower()
    baseline_location = str(baseline.get("common_location") or "").strip().lower()
    if not location and baseline_location:
        defaults["location"] = baseline_location

    return defaults


def _build_personalization_offer_text(defaults, baseline):
    parts = []
    budget_value = _to_int(defaults.get("budget"), default=0)
    if budget_value > 0:
        parts.append(f"around PHP {budget_value}")

    company_type = str(defaults.get("company_type") or "").strip().lower()
    if company_type in ("hotel", "inn", "either"):
        if company_type == "either":
            parts.append("hotel or inn")
        else:
            parts.append(f"{company_type} stays")

    location = str(defaults.get("location") or "").strip()
    if location:
        parts.append(f"in {location.title()}")

    if not parts:
        return ""

    basis_text = "I can speed up your booking search with suggested defaults"
    defaults_text = ", ".join(parts)
    return _template_personalization_offer_text(basis_text, defaults_text)


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


def _resolve_guests(params):
    guests = _to_int(params.get("guests"), default=0)
    if guests > 0:
        return guests

    adults = _to_int(params.get("adults"), default=0)
    children = _to_int(params.get("children"), default=0)
    total = adults + children
    return total if total > 0 else 0


def _compose_accommodation_context_summary(params):
    if not isinstance(params, dict):
        return ""
    parts = []
    company_type = str(params.get("company_type") or "").strip().lower()
    if company_type in ("hotel", "inn"):
        parts.append(company_type)
    elif company_type == "either":
        parts.append("hotel/inn")

    location = str(params.get("location") or "").strip()
    location_anchor = str(params.get("location_anchor") or "").strip()
    if location:
        parts.append(f"near {location}")
    if location_anchor:
        parts.append(f"anchored to map place: {location_anchor}")

    guests = _to_int(params.get("guests"), default=0)
    if guests > 0:
        parts.append(f"for {guests} guest(s)")

    budget = _to_int(params.get("budget"), default=0)
    if budget > 0:
        parts.append(f"under PHP {budget}")
    elif bool(params.get("prefer_low_price")):
        parts.append("favoring budget-friendly options")

    preference_tags = params.get("preference_tags") if isinstance(params.get("preference_tags"), list) else []
    if preference_tags:
        parts.append(f"with preferences: {', '.join(str(tag) for tag in preference_tags[:3])}")

    check_in = str(params.get("check_in") or "").strip()
    check_out = str(params.get("check_out") or "").strip()
    nights = _to_int(params.get("nights"), default=0)
    if check_in and check_out:
        parts.append(f"from {check_in} to {check_out}")
    elif nights > 0:
        parts.append(f"for {nights} night(s)")

    return ", ".join(parts)


def _format_known_accommodation_details(*, company_type="", location="", budget=Decimal("0"), guests=0):
    parts = []
    normalized_type = str(company_type or "").strip().lower()
    normalized_location = str(location or "").strip()
    normalized_budget = _to_decimal(budget, default=Decimal("0"))
    normalized_guests = _to_int(guests, default=0)

    if normalized_type in ("hotel", "inn", "either"):
        parts.append(f"type: {normalized_type}")
    if normalized_location:
        parts.append(f"location: {normalized_location}")
    if normalized_budget > 0:
        parts.append(f"budget: PHP {normalized_budget:.2f}")
    if normalized_guests > 0:
        parts.append(f"guests: {normalized_guests}")
    return "; ".join(parts)


def _inject_recommendation_context(response_payload, params=None, default_summary=""):
    if not isinstance(response_payload, dict):
        return
    summary = _compose_accommodation_context_summary(params if isinstance(params, dict) else {})
    if summary:
        response_payload["recommendation_context_summary"] = summary
        return
    fallback = str(default_summary or "").strip()
    if fallback:
        response_payload["recommendation_context_summary"] = fallback


def _resolve_accommodation_result_limit(params):
    requested = _to_int((params or {}).get("result_limit"), default=3)
    if requested <= 0:
        requested = 3
    return max(1, min(requested, 10))


def _get_recommendations(params):
    results = recommend_tours(params, limit=3)
    timeframe_hint = str((params or {}).get("tour_timeframe_hint") or "").strip()
    if not results:
        guests = _resolve_guests(params)
        budget = _to_int(params.get("budget"), default=0)
        pref_type = str(params.get("tour_type") or params.get("preferred_type") or "").strip()
        known_bits = []
        if guests > 0:
            known_bits.append(f"guests: {guests}")
        if budget > 0:
            known_bits.append(f"budget: PHP {budget}")
        if pref_type:
            known_bits.append(f"tour type: {pref_type}")
        known_text = (
            f"Thank you. I have recorded the following details: {'; '.join(known_bits)}.\n"
            if known_bits else ""
        )
        no_match_text = (
            "I could not find a strong tour match at this time. "
            "You may try increasing your budget, changing the tour type, or sharing a preferred destination."
        )
        if timeframe_hint:
            no_match_text = (
                f"For {timeframe_hint}, I could not find a strong currently available tour match at this time. "
                "You may try a broader timeframe, increasing your budget, changing the tour type, or sharing a preferred destination."
            )
        return (f"{known_text}{no_match_text}"), []

    lines = []
    if timeframe_hint:
        lines.append(
            f"You asked about {timeframe_hint}. Here are currently available tours that may match your timeframe."
        )
    lines.append("Top recommendations for you (hybrid score-based ranking):")
    lines.append("Here are currently available tours you may like:")
    items_payload = []
    any_strong_preference_match = False
    has_requested_preference = False
    for idx, item in enumerate(results, 1):
        lines.append(f"{idx}. {item.title} | {item.subtitle}")
        meta = item.meta if isinstance(item.meta, dict) else {}
        requested_tags = {
            str(tag).strip().lower()
            for tag in (meta.get("requested_tour_preferences") or [])
            if str(tag).strip()
        }
        detected_tags = {
            str(tag).strip().lower()
            for tag in (meta.get("detected_tour_tags") or [])
            if str(tag).strip()
        }
        matched_tags = sorted(requested_tags.intersection(detected_tags))
        matched_tokens = [
            str(token).strip().lower()
            for token in (meta.get("matched_preference_tokens") or [])
            if str(token).strip()
        ]
        requested_tokens = [
            str(token).strip().lower()
            for token in (meta.get("requested_preference_tokens") or [])
            if str(token).strip()
        ]
        tag_match_ratio = float(meta.get("tag_match_ratio") or 0.0)
        token_match_ratio = float(meta.get("token_match_ratio") or 0.0)
        if requested_tags or requested_tokens:
            has_requested_preference = True
        if tag_match_ratio > 0 or token_match_ratio > 0:
            any_strong_preference_match = True
        if matched_tags:
            lines.append(f"   Match: {', '.join(matched_tags)}")
        elif matched_tokens:
            lines.append(f"   Match: {', '.join(matched_tokens)}")
        elif requested_tags or requested_tokens:
            lines.append("   Match: closest overall fit (no exact preference keyword match)")
        items_payload.append(
            {
                "rank": idx,
                "title": item.title,
                "subtitle": item.subtitle,
                "score": round(float(item.score), 6),
                "meta": meta,
            }
        )
    if has_requested_preference and not any_strong_preference_match:
        unavailable_titles = get_unavailable_tour_matches(params, limit=2)
        if unavailable_titles:
            lines.append(
                "Note: Matching tour(s) found but currently without upcoming schedules: "
                + ", ".join(unavailable_titles)
                + "."
            )
        else:
            lines.append(
                "Note: A matching tour title may exist in the system, but only tours with upcoming schedules are shown in this list."
            )
    return "\n".join(lines), items_payload


def _get_accommodation_recommendations(params):
    limit = _resolve_accommodation_result_limit(params)
    results, diagnostics = recommend_accommodations_with_diagnostics(params, limit=limit)
    context_summary = _compose_accommodation_context_summary(params)
    requested_amenities = []
    if isinstance(params, dict):
        amenity_raw = params.get("amenities") or params.get("amenity")
        if isinstance(amenity_raw, list):
            requested_amenities = [str(v).strip().lower() for v in amenity_raw if str(v).strip()]
        elif str(amenity_raw or "").strip():
            requested_amenities = [str(v).strip().lower() for v in re.split(r"[;,]", str(amenity_raw)) if str(v).strip()]
    if not results:
        no_match_reasons = diagnostics.get("no_match_reasons") or []
        suggested_budget_min = diagnostics.get("suggested_budget_min")
        location = str(params.get("location") or "").strip()
        budget = _to_decimal(params.get("budget"), default=Decimal("0"))
        guests = _to_int(params.get("guests"), default=0)
        company_type = str(params.get("company_type") or "").strip().lower()
        location_anchor = str(params.get("location_anchor") or "").strip()
        location_scope_note = str(params.get("location_scope_note") or "").strip()

        known_text = _format_known_accommodation_details(
            company_type=company_type,
            location=location,
            budget=budget,
            guests=guests,
        )

        lines = []
        if context_summary:
            lines.append(f"I checked available rooms {context_summary}.")
        if known_text:
            lines.append(f"Recorded details: {known_text}.")
        lines.append("I could not find a strong hotel/inn match at this time.")
        if location_anchor:
            lines.append(
                f"I used '{location_anchor}' as your map anchor (city-proper coverage) and checked accommodation records."
            )
        if location_scope_note:
            lines.append(location_scope_note)
        if requested_amenities:
            lines.append(
                "I also applied your amenity request "
                f"({', '.join(sorted(set(requested_amenities))[:5])}) where searchable details were available."
            )
            lines.append(
                "If exact amenity metadata is incomplete for some listings, results may be based on partial keyword matches."
            )
        quick_replies = []
        if "budget_too_low" in no_match_reasons and suggested_budget_min:
            min_budget_value = Decimal(str(suggested_budget_min))
            quick_replies.append(f"budget {int(min_budget_value)}")
            quick_replies.append("broaden location")
            quick_replies.append("include both hotel and inn")
            if location and budget > 0:
                lines.append(
                    f"No rooms in {location.title()} are available under PHP {budget:.2f}. "
                    f"Cheapest matching option starts at PHP {min_budget_value:.2f}. "
                    f"Would you like to use PHP {min_budget_value:.0f} as your budget?"
                )
            else:
                lines.append(
                    f"Cheapest matching option starts at PHP {min_budget_value:.2f}. "
                    f"Would you like to use PHP {min_budget_value:.0f} as your budget?"
                )
        elif "location_too_narrow" in no_match_reasons:
            lines.append(
                "I found options outside your current location filter. "
                "If you want, send: broaden location"
            )
            quick_replies.append("broaden location")
            quick_replies.append("include both hotel and inn")
        elif "type_too_narrow" in no_match_reasons:
            lines.append(
                "I found options under a different accommodation type. "
                "If you want, send: include both hotel and inn"
            )
            quick_replies.append("include both hotel and inn")
            quick_replies.append("broaden location")

        if not quick_replies:
            quick_replies.append("show default hotel suggestions")
            quick_replies.append("broaden location")

        return "\n\n".join(lines), [], {
            "no_match_reasons": no_match_reasons,
            "suggested_budget_min": suggested_budget_min,
            "fallback_applied": diagnostics.get("fallback_applied", "none"),
            "quick_replies": quick_replies,
        }

    lines = ["Top hotel/inn recommendations for you (CNN + Decision Tree):"]
    location_anchor = str(params.get("location_anchor") or "").strip()
    location_scope_note = str(params.get("location_scope_note") or "").strip()
    if location_anchor:
        lines.insert(
            0,
            f"Using map anchor: {location_anchor} (city-proper map coverage).",
        )
    if location_scope_note:
        lines.insert(0, location_scope_note)
    fallback_applied = str(diagnostics.get("fallback_applied") or "none").strip().lower()
    if fallback_applied in ("relaxed_type", "relaxed_location", "relaxed_location_and_type"):
        fallback_map = {
            "relaxed_type": "I broadened accommodation type to include both hotels and inns so you can still see viable options.",
            "relaxed_location": "I broadened the location filter so you can still see viable options near your target area.",
            "relaxed_location_and_type": "I broadened both type and location filters to return viable options from current records.",
        }
        lines.insert(0, fallback_map.get(fallback_applied, "I broadened filters to return viable options."))
    if context_summary:
        lines.insert(0, f"I found {len(results)} option(s) based on your request {context_summary}.")
    items_payload = []
    amenity_confident_match = False
    for idx, item in enumerate(results, 1):
        item_meta = item.meta if isinstance(item.meta, dict) else {}
        room_id = item_meta.get("room_id")
        room_id_label = f" | Room ID: {room_id}" if room_id not in (None, "") else ""
        trace = item_meta.get("trace") if isinstance(item_meta.get("trace"), dict) else {}
        reasons = trace.get("reasons") if isinstance(trace.get("reasons"), list) else []
        lowered_reasons = [str(r or "").strip().lower() for r in reasons]
        if any(
            ("amenity" in r and "partial" not in r and "no amenity" not in r)
            or "wifi" in r
            or "aircon" in r
            for r in lowered_reasons
        ):
            amenity_confident_match = True
        match_score = trace.get("match_score")
        match_strength = str(trace.get("match_strength") or "").strip()
        lines.append(f"{idx}. {item.title}{room_id_label} | {item.subtitle}")
        if match_strength:
            lines.append(f"   Match: {match_strength}")
        if reasons:
            lines.append(f"   Key reason: {str(reasons[0])}")
        items_payload.append(
            {
                "rank": idx,
                "title": item.title,
                "subtitle": item.subtitle,
                "score": round(float(item.score), 6),
                "ranking_score": round(float(item.score), 6),
                "room_id": item_meta.get("room_id"),
                "accom_id": item_meta.get("accom_id"),
                "match_score": match_score,
                "match_strength": match_strength,
                "decision_tree_score": trace.get("decision_tree_score"),
                "cnn_alignment": trace.get("cnn_alignment"),
                "scoring_mode": trace.get("scoring_mode"),
                "reasons": [str(r) for r in reasons],
                "meta": item_meta,
            }
        )
    if requested_amenities and not amenity_confident_match:
        lines.append(
            "I considered your amenity request, but some listings have limited amenity details, "
            "so results may be based more strongly on budget, location, and guest fit."
        )
    lines.append("Reply with: compare top 3 or why option <number>.")
    return "\n".join(lines), items_payload, {
        "fallback_applied": diagnostics.get("fallback_applied", "none"),
    }


def _safe_get_accommodation_recommendations(params):
    try:
        return _get_accommodation_recommendations(params)
    except Exception:
        fallback_text = (
            "I encountered a temporary issue while loading hotel/inn recommendations.\n"
            "Please try again, or refine your request with location, budget, and number of guests."
        )
        return fallback_text, [], {
            "fallback_applied": "runtime_error",
            "quick_replies": ["show default hotel suggestions"],
        }


def _get_default_accommodation_suggestions(limit=3):
    room_qs = (
        Room.objects.select_related("accommodation")
        .filter(status="AVAILABLE")
        .filter(current_availability__gte=1)
        .filter(accommodation__approval_status="accepted")
        .filter(
            Q(accommodation__company_type__icontains="hotel") |
            Q(accommodation__company_type__icontains="inn")
        )
    )

    rooms = list(room_qs.order_by("?")[:limit])

    # Fallback to any available accommodation rooms if no hotel/inn tag matches in DB.
    if not rooms:
        rooms = list(
            Room.objects.select_related("accommodation")
            .filter(status="AVAILABLE")
            .filter(current_availability__gte=1)
            .filter(accommodation__approval_status="accepted")
            .order_by("?")[:limit]
        )

    if not rooms:
        return (
            "I don't have available hotel/inn room records to suggest yet. "
            "Please try again later or ask for a specific hotel/room."
        )

    lines = ["Accommodation recommendations (suggested stays):"]
    suggestion_items = []
    for idx, room in enumerate(rooms, 1):
        accom = room.accommodation
        subtitle = (
            f"{accom.location} | PHP {room.price_per_night} per night "
            f"| up to {room.person_limit} guests"
        )
        lines.append(f"{idx}. {accom.company_name} - {room.room_name} | {subtitle}")
        suggestion_items.append(
            {
                "rank": idx,
                "title": f"{accom.company_name} - {room.room_name}",
                "subtitle": subtitle,
                "room_id": room.room_id,
                "accom_id": room.accommodation_id,
                "match_strength": "Suggested",
                "price_per_night": str(room.price_per_night),
                "person_limit": room.person_limit,
                "location": accom.location,
                "reasons": [
                    f"Capacity: up to {room.person_limit} guests",
                    f"Current availability: {room.current_availability}",
                    "Balanced option for common short-stay needs",
                ],
            }
        )
    lines.append(
        "To book, send: book room <room_id> for <guests> guests from <YYYY-MM-DD> to <YYYY-MM-DD>."
    )
    lines.append(
        "Example: book room 12 for 2 guests from 2026-03-10 to 2026-03-12."
    )
    return "\n".join(lines), suggestion_items


def _calculate_billing(params):
    guests = _resolve_guests(params)
    sched_id = str(params.get("sched_id") or params.get("schedule_id") or "").strip()
    tour_name = str(params.get("tour_name") or "").strip()

    schedule = None
    if sched_id:
        schedule = Tour_Schedule.objects.select_related("tour").filter(sched_id=sched_id).first()

    if schedule is None and tour_name:
        now = timezone.now()
        schedule = (
            Tour_Schedule.objects.select_related("tour")
            .filter(tour__tour_name__icontains=tour_name, end_time__gte=now)
            .exclude(status="cancelled")
            .order_by("start_time")
            .first()
        )

    if schedule is None:
        return (
            "I couldn't find that schedule. Please provide a valid sched_id "
            "or exact tour name."
        )

    base = Decimal(schedule.price) * guests
    admission_per_guest = (
        Admission_Rates.objects.filter(tour_id=schedule.tour).aggregate(total=Sum("price"))["total"]
        or Decimal("0")
    )
    admission_total = Decimal(admission_per_guest) * guests
    grand_total = base + admission_total

    return (
        f"Billing Summary for {schedule.tour.tour_name} ({schedule.sched_id}):\n"
        f"Guests: {guests}\n"
        f"Base fare: PHP {schedule.price} x {guests} = PHP {base:.2f}\n"
        f"Admission fees: PHP {admission_per_guest:.2f} x {guests} = PHP {admission_total:.2f}\n"
        f"Total amount due: PHP {grand_total:.2f}"
    )


def _calculate_accommodation_billing(params):
    raw_room_id = params.get("room_id")
    room_id = _to_int(raw_room_id, default=0)
    if room_id <= 0:
        room_id = None
    check_in = _normalize_iso_date(params.get("check_in"))
    check_out = _normalize_iso_date(params.get("check_out"))
    nights = _to_int(params.get("nights"), default=0)

    room = None
    if room_id:
        room = (
            Room.objects.select_related("accommodation")
            .filter(room_id=room_id, status="AVAILABLE", accommodation__approval_status="accepted")
            .first()
        )
    if room is None and params.get("accom_name"):
        room = (
            Room.objects.select_related("accommodation")
            .filter(
                status="AVAILABLE",
                accommodation__approval_status="accepted",
                accommodation__company_name__icontains=params.get("accom_name"),
            )
            .first()
        )

    if room is None:
        return "I couldn't find that room. Please provide a valid room ID or accommodation name."

    guests = _resolve_guests(params)
    if guests <= 0:
        return (
            f"Before I calculate billing for {room.accommodation.company_name} - {room.room_name}, "
            "please provide the number of guests (example: 2 guests)."
        )

    if check_in and check_out:
        try:
            from datetime import datetime
            check_in_dt = datetime.strptime(check_in, "%Y-%m-%d").date()
            check_out_dt = datetime.strptime(check_out, "%Y-%m-%d").date()
            total = calculate_accommodation_billing(room, check_in_dt, check_out_dt)
            nights = max((check_out_dt - check_in_dt).days, 1)
        except Exception:
            return "Please provide dates in YYYY-MM-DD format for check-in and check-out."
    else:
        nights = max(nights, 1)
        total = Decimal(room.price_per_night) * Decimal(nights)

    lgu_payment_note = (
        "Payment is securely processed through the LGU Tourism Office system."
        if str(getattr(settings, "TOURISM_OFFICE_BILLING_URL", "") or os.getenv("TOURISM_OFFICE_BILLING_URL", "")).strip()
        else "Payment processing is handled by the LGU Tourism Office system."
    )

    return (
        f"Billing Summary for {room.accommodation.company_name} - {room.room_name}:\n"
        f"Guests: {guests}\n"
        f"Nights: {nights}\n"
        f"Rate: PHP {room.price_per_night} per night\n"
        f"Total amount due: PHP {total:.2f}\n"
        f"Payment Note: {lgu_payment_note}\n"
        "Next Step: Complete booking first to proceed to payment."
    )


def _build_find_another_accommodation_prompt(params):
    if not isinstance(params, dict):
        return "find another hotel"

    guests = _resolve_guests(params)
    budget = _to_int(params.get("budget"), default=0)
    location = str(params.get("location") or "").strip()
    company_type = str(params.get("company_type") or "").strip().lower()

    type_text = "hotel"
    if company_type in ("hotel", "inn"):
        type_text = company_type
    elif company_type == "either":
        type_text = "hotel/inn"

    parts = [f"find another {type_text}"]
    if location:
        parts.append(f"in {location}")
    if guests > 0:
        parts.append(f"for {guests} guests")
    if budget > 0:
        parts.append(f"under {budget}")

    prompt = " ".join(parts).strip()
    return prompt if prompt else "find another hotel"


def _build_book_from_billing_prompt(room, params):
    room_id = getattr(room, "room_id", "")
    guests = _resolve_guests(params)
    check_in = _normalize_iso_date(params.get("check_in"))
    check_out = _normalize_iso_date(params.get("check_out"))

    if check_in and check_out:
        if guests > 0:
            return f"book room {room_id} for {guests} guests from {check_in} to {check_out}"
        return f"book room {room_id} from {check_in} to {check_out}"
    if guests > 0:
        return f"book room {room_id} for {guests} guests"
    return f"book room {room_id}"


def _build_booking_receipt_text(*, booking_id, hotel_name, room_name, room_id, check_in, check_out, nights, guests, rate, total):
    return (
        "IBAYAW TOUR - ACCOMMODATION BOOKING RECEIPT\n"
        "-------------------------------------------\n"
        f"Booking ID: #{booking_id}\n"
        f"Hotel/Inn: {hotel_name}\n"
        f"Room: {room_name} (Room {room_id})\n"
        f"Check-in: {check_in}\n"
        f"Check-out: {check_out}\n"
        f"Nights: {nights}\n"
        f"Guests: {guests}\n"
        f"Rate per night: PHP {rate:.2f}\n"
        f"Estimated Total: PHP {total:.2f}\n"
        "Booking Status: Pending\n"
        "Payment Status: Unpaid\n"
        "\n"
        "Payment and verification are processed through the LGU Tourism Office system."
    )


def _sanitize_quick_replies(items, *, limit=4):
    if not isinstance(items, list):
        return []
    normalized = []
    for item in items:
        if isinstance(item, dict):
            value = str(item.get("value") or "").strip()
            if not value:
                continue
            label = str(item.get("label") or value).strip() or value
            normalized.append({"label": label[:80], "value": value[:300]})
            continue

        value = str(item or "").strip()
        if value:
            normalized.append(value[:300])
    return normalized[:limit]


def _find_accommodation_room(params):
    room_id = params.get("room_id")
    guests = _resolve_guests(params)
    budget = _to_decimal(params.get("budget"), default=None)
    location = str(params.get("location") or "").strip()
    accom_name = str(params.get("accom_name") or params.get("hotel_name") or "").strip()

    qs = Room.objects.select_related("accommodation").filter(
        status="AVAILABLE",
        accommodation__approval_status="accepted",
    )

    if room_id not in ("", None):
        try:
            return qs.filter(room_id=int(room_id)).first()
        except (TypeError, ValueError):
            pass

    if guests > 0:
        qs = qs.filter(person_limit__gte=guests)
    if budget is not None:
        qs = qs.filter(price_per_night__lte=budget)
    if location:
        qs = qs.filter(accommodation__location__icontains=location)
    if accom_name:
        qs = qs.filter(accommodation__company_name__icontains=accom_name)

    return qs.order_by("price_per_night", "room_id").first()


def _build_accommodation_billing_link(request, room, check_in, check_out, num_guests, booking_id=None):
    external_billing_url = str(
        getattr(settings, "TOURISM_OFFICE_BILLING_URL", "") or os.getenv("TOURISM_OFFICE_BILLING_URL", "")
    ).strip()

    if external_billing_url:
        query = {
            "room_id": getattr(room, "room_id", ""),
            "hotel": getattr(getattr(room, "accommodation", None), "company_name", ""),
            "check_in": check_in,
            "check_out": check_out,
            "num_guests": num_guests,
            "from_chatbot": 1,
        }
        if booking_id:
            query["booking_id"] = booking_id
        separator = "&" if "?" in external_billing_url else "?"
        return f"{external_billing_url}{separator}{urlencode(query)}"

    base_path = reverse("accommodation_page")
    query = {
        "room_id": getattr(room, "room_id", ""),
        "check_in": check_in,
        "check_out": check_out,
        "num_guests": num_guests,
        "from_chatbot": 1,
        "focus": "booking",
    }
    if booking_id:
        query["booking_id"] = booking_id
    relative_url = f"{base_path}?{urlencode(query)}"
    relative_url = f"{relative_url}#bookingForm"
    return request.build_absolute_uri(relative_url) if hasattr(request, "build_absolute_uri") else relative_url


def _book_accommodation_from_chat(request, params, *, commit=True):
    room_id_raw = params.get("room_id")
    accom_name_raw = str(params.get("accom_name") or "").strip()

    if room_id_raw not in ("", None):
        try:
            int(str(room_id_raw).strip())
        except (TypeError, ValueError):
            return {
                "reply": (
                    f"I received room ID '{room_id_raw}', but it should be numeric. "
                    "Please send a number like: room 12."
                ),
                "room_id": None,
                "accom_id": None,
            }
    elif not accom_name_raw:
        return {
            "reply": (
                "I can proceed with booking once you provide a room reference.\n"
                "Send either a room ID or a clearer room/hotel detail.\n"
                "Example: book room 12 for 2 guests from 2026-03-10 to 2026-03-12."
            ),
            "room_id": None,
            "accom_id": None,
        }

    room = _find_accommodation_room(params)
    if room is None:
        return {
            "reply": (
                "I couldn’t map that booking request to a valid room yet.\n"
                "Please provide a room ID or clearer hotel details.\n"
                "Example: book room 12 for 2 guests from 2026-03-10 to 2026-03-12."
            ),
            "room_id": None,
            "accom_id": None,
        }

    check_in = str(params.get("check_in") or "").strip()
    check_out = str(params.get("check_out") or "").strip()
    num_guests = _resolve_guests(params)

    if num_guests <= 0:
        known_bits = [f"room: {room.accommodation.company_name} - {room.room_name} (Room {room.room_id})"]
        if check_in:
            known_bits.append(f"check-in: {check_in}")
        if check_out:
            known_bits.append(f"check-out: {check_out}")
        return {
            "reply": (
                "I can prepare your booking request, but I still need the number of guests.\n\n"
                "Details received:\n"
                + "\n".join(f"- {item}" for item in known_bits)
                + "\n\nNext step:\n"
                "Please send the guest count (example: 2 guests)."
            ),
            "room_id": room.room_id,
            "accom_id": getattr(room.accommodation, "accom_id", None),
            "requires_confirmation": False,
            "missing_slot": "guests",
            "prepared_params": {
                "room_id": room.room_id,
                "check_in": check_in,
                "check_out": check_out,
            },
        }

    if not check_in or not check_out:
        known_bits = [
            f"room: {room.accommodation.company_name} - {room.room_name} (Room {room.room_id})",
            f"guests: {num_guests}",
        ]
        if check_in and not check_out:
            ask_line = (
                f"I already have your check-in date ({check_in}). "
                "Please provide your check-out date in YYYY-MM-DD format."
            )
        elif check_out and not check_in:
            ask_line = (
                f"I already have your check-out date ({check_out}). "
                "Please provide your check-in date in YYYY-MM-DD format."
            )
        else:
            ask_line = (
                "Please provide both check-in and check-out dates in YYYY-MM-DD format "
                "(example: 2026-03-10 to 2026-03-12)."
            )
        prepared_params = {
            "room_id": room.room_id,
            "guests": num_guests,
        }
        if check_in:
            prepared_params["check_in"] = check_in
        if check_out:
            prepared_params["check_out"] = check_out
        return {
            "reply": (
                "I can prepare your booking request. I only need your stay date details.\n\n"
                "Details received:\n"
                + "\n".join(f"- {item}" for item in known_bits)
                + "\n\nNext step:\n"
                f"{ask_line}"
            ),
            "room_id": room.room_id,
            "accom_id": getattr(room.accommodation, "accom_id", None),
            "requires_confirmation": False,
            "missing_slot": "stay_details",
            "prepared_params": prepared_params,
        }

    try:
        check_in_dt = datetime.strptime(check_in, "%Y-%m-%d").date()
        check_out_dt = datetime.strptime(check_out, "%Y-%m-%d").date()
    except Exception:
        return {
            "reply": (
                "I couldn’t read the dates you sent.\n"
                "Please use YYYY-MM-DD format for both check-in and check-out "
                "(example: 2026-03-10 to 2026-03-12)."
            ),
            "room_id": room.room_id,
            "accom_id": getattr(room.accommodation, "accom_id", None),
            "requires_confirmation": False,
            "missing_slot": "stay_details",
            "prepared_params": {
                "room_id": room.room_id,
                "guests": num_guests,
            },
        }

    if check_out_dt <= check_in_dt:
        return {
            "reply": (
                f"I received check-in {check_in_dt.isoformat()} and check-out {check_out_dt.isoformat()}.\n"
                "Check-out must be later than check-in. Please send updated dates."
            ),
            "room_id": room.room_id,
            "accom_id": getattr(room.accommodation, "accom_id", None),
            "requires_confirmation": False,
            "missing_slot": "stay_details",
            "prepared_params": {
                "room_id": room.room_id,
                "guests": num_guests,
            },
        }

    if room.person_limit and num_guests > room.person_limit:
        return {
            "reply": (
                f"I found a capacity mismatch: you requested {num_guests} guest(s), "
                f"but this room allows up to {room.person_limit}.\n"
                "Please reduce guests or choose another room."
            ),
            "room_id": room.room_id,
            "accom_id": getattr(room.accommodation, "accom_id", None),
        }

    total = calculate_accommodation_billing(room, check_in_dt, check_out_dt)
    nights = max((check_out_dt - check_in_dt).days, 1)

    booking = None
    booking_error = None
    if commit:
        user = getattr(request, "user", None)
        if user and getattr(user, "is_authenticated", False):
            try:
                booking, booking_error = create_accommodation_booking_with_integrity(
                    guest=user,
                    room=room,
                    check_in=check_in_dt,
                    check_out=check_out_dt,
                    num_guests=num_guests,
                    total_amount=total,
                    status="pending",
                    companions=[],
                )
            except Exception:
                booking = None
                booking_error = "booking_error"

    if booking_error == "room_unavailable":
        return {
            "reply": (
                "This room is no longer available for booking right now "
                "(status/approval/availability changed).\n"
                "Please choose another room."
            ),
            "room_id": getattr(room, "room_id", None),
            "accom_id": getattr(getattr(room, "accommodation", None), "accom_id", None),
        }
    if booking_error == "date_overlap":
        return {
            "reply": (
                "That room already has a booking overlap for the dates you selected.\n"
                "Please choose different dates or another room."
            ),
            "room_id": getattr(room, "room_id", None),
            "accom_id": getattr(getattr(room, "accommodation", None), "accom_id", None),
        }

    billing_link = _build_accommodation_billing_link(
        request=request,
        room=room,
        check_in=check_in_dt.isoformat(),
        check_out=check_out_dt.isoformat(),
        num_guests=num_guests,
        booking_id=getattr(booking, "booking_id", None),
    )

    if booking and commit:
        header = "Booking Receipt / Summary (Accommodation)"
        booking_line = f"Booking ID: #{booking.booking_id}"
        status_line = f"Booking Status: {booking.status.title()}"
    elif commit:
        header = "Booking Summary (Draft - not yet saved)"
        booking_line = "Booking ID: Not created (please log in if needed)"
        status_line = "Booking Status: Draft"
    else:
        header = "Booking Draft (Not yet saved)"
        booking_line = "Booking ID: Pending confirmation"
        status_line = "Booking Status: Awaiting confirmation"

    external_billing_configured = bool(
        str(getattr(settings, "TOURISM_OFFICE_BILLING_URL", "") or os.getenv("TOURISM_OFFICE_BILLING_URL", "")).strip()
    )

    reply = (
        f"{header}\n"
        f"{booking_line}\n"
        f"Hotel: {room.accommodation.company_name}\n"
        f"Room: {room.room_name} (Room {room.room_id})\n"
        f"Check-in: {check_in_dt.isoformat()}\n"
        f"Check-out: {check_out_dt.isoformat()}\n"
        f"Nights: {nights}\n"
        f"Guests: {num_guests}\n"
        f"Rate per night: PHP {room.price_per_night:.2f}\n"
        f"Estimated Total: PHP {total:.2f}\n"
        f"{status_line}\n"
        "Payment Status: Unpaid\n" + (
            (
                f"Estimated total: PHP {total:.2f} for {nights} nights.\n"
                "Please confirm this booking to continue to billing."
            )
            if not commit
            else (
                "Your booking has been successfully created.\n"
                +
                (
                    "Billing / Payment Link: use the button/link provided in the chat."
                    if external_billing_configured
                    else "Billing Link: use the button/link provided in the chat to continue in-system billing details."
                )
            )
        )
    )

    receipt_text = ""
    receipt_filename = ""
    if booking and commit:
        receipt_text = _build_booking_receipt_text(
            booking_id=booking.booking_id,
            hotel_name=room.accommodation.company_name,
            room_name=room.room_name,
            room_id=room.room_id,
            check_in=check_in_dt.isoformat(),
            check_out=check_out_dt.isoformat(),
            nights=nights,
            guests=num_guests,
            rate=Decimal(str(room.price_per_night)),
            total=Decimal(str(total)),
        )
        receipt_filename = f"ibayaw_booking_receipt_{booking.booking_id}.png"

    return {
        "reply": reply,
        "billing_link": billing_link if commit else "",
        "billing_link_label": (
            "Proceed to LGU Payment"
            if booking and commit and external_billing_configured
            else "Open Billing Details"
        ),
        "booking_id": getattr(booking, "booking_id", None),
        "booking_status": (
            getattr(booking, "status", "draft" if booking is None else "")
            if commit
            else "awaiting_confirmation"
        ),
        "room_id": getattr(room, "room_id", None),
        "accom_id": getattr(getattr(room, "accommodation", None), "accom_id", None),
        "requires_confirmation": (not commit),
        "prepared_params": {
            "room_id": getattr(room, "room_id", None),
            "check_in": check_in_dt.isoformat(),
            "check_out": check_out_dt.isoformat(),
            "guests": num_guests,
        },
        "receipt_text": receipt_text,
        "receipt_filename": receipt_filename,
        "quick_replies": (
            (
                ["view my accommodation bookings"]
                if booking and commit
                else ["Yes", "No"]
            )
            if (booking and commit) or (not commit)
            else []
        ),
    }


def _safe_log_chat_booking_linkage(request, message_text, params, booking_result):
    try:
        user = getattr(request, "user", None)
        if not user or not getattr(user, "is_authenticated", False):
            return

        if not isinstance(booking_result, dict):
            return

        room_id = booking_result.get("room_id")
        accom_id = booking_result.get("accom_id")
        booking_id = booking_result.get("booking_id")
        booking_status = booking_result.get("booking_status") or (
            "booked" if booking_id else "draft_or_not_created"
        )
        billing_link = booking_result.get("billing_link") or ""

        session_key = ""
        if hasattr(request, "session"):
            session_key = request.session.session_key or ""

        item_ref_parts = []
        if room_id:
            item_ref_parts.append(f"room:{room_id}")
        if accom_id:
            item_ref_parts.append(f"accom:{accom_id}")
        if booking_id:
            item_ref_parts.append(f"booking:{booking_id}")
        item_ref = "|".join(item_ref_parts) or "chat:accommodation_booking_attempt"

        RecommendationEvent.objects.create(
            user=user,
            event_type="book",
            item_ref=item_ref,
            session_id=session_key,
            data_source=_resolve_data_source(request=request),
        )

        _safe_log_recommendation_result_with_metadata(
            request,
            "book_accommodation",
            booking_result.get("reply", ""),
            params if isinstance(params, dict) else {},
            message_text=message_text,
            recommended_items=[
                {
                    "type": "booking_outcome",
                    "room_id": room_id,
                    "accom_id": accom_id,
                    "booking_id": booking_id,
                    "booking_status": booking_status,
                    "billing_link": billing_link,
                }
            ],
            booking_linkage={
                "room_id": room_id,
                "accom_id": accom_id,
                "booking_id": booking_id,
                "booking_status": booking_status,
                "billing_link": billing_link,
            },
        )
    except Exception:
        pass


def _extract_params_with_confidence(message):
    text = (message or "").strip().lower()
    params = {}
    confidence = 1.0
    needs_clarification = False
    clarification_question = ""
    clarification_field = ""
    clarification_options = []
    numeric_only = re.fullmatch(
        r"\s*([0-9][0-9,]*(?:\.[0-9]+)?k?)\s*",
        text,
        flags=re.IGNORECASE,
    )
    known_location_map = {
        "terminal": "terminal area",
        "terminal area": "terminal area",
        "public terminal": "tinago",
        "bus terminal": "tinago",
        "trike terminal": "tinago",
        "tricycle terminal": "tinago",
        "pedicab terminal": "tinago",
        "motorcab terminal": "tinago",
        "tinago": "tinago",
        "boyco": "boyco",
        "ubos": "ubos",
        "poblacion": "poblacion",
        "bayawan": "bayawan",
        "bayawan city": "bayawan city",
        "villareal": "villareal",
        "villarreal": "villareal",
        "suba": "suba",
    }
    map_location_anchor = ""

    def _resolve_location_value(raw_location):
        candidate = " ".join(str(raw_location or "").split()).strip().lower()
        candidate = re.sub(r"^the\s+", "", candidate)
        if not candidate:
            return ""
        if candidate in known_location_map:
            return known_location_map.get(candidate, candidate)
        alias_place_match = _match_map_reference_place(candidate)
        if alias_place_match:
            return _map_place_to_location_hint(alias_place_match.get("name"))
        if "trike terminal" in candidate or "tricycle terminal" in candidate or "tricyle terminal" in candidate:
            return "tinago"
        if "pedicab terminal" in candidate or "motorcab terminal" in candidate:
            return "tinago"
        if "bus terminal" in candidate or "public terminal" in candidate or "city terminal" in candidate:
            return "tinago"
        if "terminal" in candidate:
            return "terminal area"
        if "public market" in candidate or candidate == "market":
            return "boyco"
        if "plaza" in candidate:
            return "poblacion"
        db_locations = _known_accommodation_locations()
        close = get_close_matches(candidate, list(known_location_map.keys()), n=1, cutoff=0.82)
        if close:
            return known_location_map.get(close[0], close[0])
        close_db = get_close_matches(candidate, db_locations, n=1, cutoff=0.78)
        if close_db:
            return close_db[0]
        return candidate

    def _set_clarification(field, question, penalty=0.3):
        nonlocal confidence, needs_clarification, clarification_question, clarification_field
        needs_clarification = True
        confidence = max(0.0, confidence - float(penalty))
        if not clarification_question:
            clarification_question = question
            clarification_field = field

    def _set_location_ambiguity_clarification(raw_value, place_matches):
        nonlocal clarification_options
        candidates = []
        for row in place_matches or []:
            name = " ".join(str((row or {}).get("name") or "").split()).strip()
            if name and name not in candidates:
                candidates.append(name)
        if not candidates:
            return
        pick_list = ", ".join(candidates[:4])
        _set_clarification(
            "location",
            (
                f"I found multiple places matching '{raw_value}'. "
                f"Which one do you mean: {pick_list}?"
            ),
            penalty=0.2,
        )
        clarification_options = [
            {"label": name[:50], "value": name[:80]}
            for name in candidates[:4]
        ]

    def _set_terminal_clarification(raw_value):
        nonlocal clarification_options
        _set_clarification(
            "location",
            (
                f"I found multiple terminal matches for '{raw_value}'. "
                "Do you mean Bayawan City Public Terminal (bus/public) or the Trike Terminal?"
            ),
            penalty=0.2,
        )
        clarification_options = [
            {"label": "Bus/Public Terminal", "value": "near Bayawan City Public Terminal"},
            {"label": "Trike Terminal", "value": "near trike terminal"},
        ]

    def _parse_compact_number(raw_value):
        value = str(raw_value or "").strip().lower().replace(",", "")
        if not value:
            return None
        multiplier = 1
        if value.endswith("k"):
            multiplier = 1000
            value = value[:-1].strip()
        try:
            parsed = Decimal(value)
        except Exception:
            return None
        if parsed < 0:
            return None
        return int(parsed * multiplier)

    # Extract schedule ID like Sched00001.
    sched_match = re.search(r"(sched\d+)", text, flags=re.IGNORECASE)
    if sched_match:
        params["sched_id"] = sched_match.group(1)

    # Extract guest count from "<n> guest(s)/people/person/pax/bisita/katao".
    guest_match = re.search(r"(\d+)\s*(guest|guests|people|person|pax|bisita|katao|ka\s*bisita)", text)
    if guest_match:
        params["guests"] = int(guest_match.group(1))

    # Extract requested recommendation list size (e.g., "give 10 inns available", "top 5 hotels").
    list_size_match = re.search(
        r"\b(?:give|show|list|recommend|top)\s+(\d{1,2})\b.*\b(?:hotel|hotels|inn|inns|accommodation|accommodations)\b",
        text,
        flags=re.IGNORECASE,
    )
    if list_size_match:
        params["result_limit"] = max(1, min(int(list_size_match.group(1)), 10))

    # Extract budget from forms like:
    # - budget 1500
    # - budget 1,500
    # - budget 1.5k
    # - under/below/less than 2000
    budget_match = re.search(
        r"(?:budget|under|below|less than)\s*[:\-]?\s*([0-9][0-9,]*(?:\.[0-9]+)?k?)",
        text,
        flags=re.IGNORECASE,
    )
    budget_tail_match = re.search(
        r"([0-9][0-9,]*(?:\.[0-9]+)?k?)\s*(?:php|peso|pesos)?\s*(?:ang\s+)?budget\b",
        text,
        flags=re.IGNORECASE,
    )
    if budget_match:
        budget_value = _parse_compact_number(budget_match.group(1))
        if budget_value is not None:
            if "total" in text and not re.search(r"(per\s*night|nightly|/night)", text):
                _set_clarification(
                    "budget",
                    "Is that amount your budget per night in PHP?",
                    penalty=0.35,
                )
            else:
                params["budget"] = budget_value
    elif budget_tail_match:
        budget_value = _parse_compact_number(budget_tail_match.group(1))
        if budget_value is not None:
            params["budget"] = budget_value
    # Numeric-only fallback is handled contextually in the main chat flow
    # to avoid misreading guest-count replies as budget.

    # Budget-clearing commands ("remove budget", "without budget", etc.)
    if re.search(
        r"\b(remove|clear|reset|ignore|without|no)\s+(the\s+)?budget\b",
        text,
        flags=re.IGNORECASE,
    ) or re.search(
        r"\bwithout\s+minding\s+the\s+budget\b",
        text,
        flags=re.IGNORECASE,
    ):
        params["clear_budget"] = True
        params["budget"] = 0

    # Extract duration from "<n> day(s)".
    duration_match = re.search(r"(\d+)\s*day", text)
    if duration_match:
        params["duration_days"] = int(duration_match.group(1))

    # Extract nights from "<n> night(s)".
    nights_match = re.search(r"(\d+)\s*night", text)
    if nights_match:
        params["nights"] = int(nights_match.group(1))

    # Extract location from phrases like "in bayawan", "near terminal", "around poblacion".
    # Stop before common trailing constraint phrases so we don't swallow guests/budget text.
    loc_match = re.search(
        r"\b(in|near|around|sa)\s+([a-z\s]+?)(?=\s+(?:for|under|below|budget|with|from)\b|$)",
        text,
    )
    if loc_match:
        loc_prefix = str(loc_match.group(1) or "").strip()
        raw_location = " ".join(loc_match.group(2).split()).strip()
        if raw_location:
            if loc_prefix in ("near", "around") and _is_generic_terminal_reference(raw_location):
                _set_terminal_clarification(raw_location)
                raw_location = ""
        if raw_location:
            normalized_location = _resolve_location_value(raw_location)
            if normalized_location in known_location_map.values() or raw_location.lower() == normalized_location.lower():
                params["location"] = normalized_location
                # Prevent stale anchor carry-over when location changed without a map anchor.
                params["clear_location_anchor"] = True
                if loc_prefix in ("near", "around"):
                    place_matches = _match_map_reference_places(raw_location, limit=4)
                    if not place_matches:
                        place_matches = _match_map_reference_places(normalized_location, limit=4)
                    if (not place_matches) and normalized_location == "terminal area":
                        place_matches = [{"name": "Bayawan City Public Terminal"}]
                    if len(place_matches) > 1:
                        _set_location_ambiguity_clarification(raw_location, place_matches)
                    elif place_matches:
                        place_match = place_matches[0]
                        map_location_anchor = str(place_match.get("name") or "").strip()
                        params["location_anchor"] = map_location_anchor
                        params["location_anchor_source"] = "map_reference"
                        params["location"] = _map_place_to_location_hint(map_location_anchor)
                        params["broaden_location"] = True
                        params["location_scope_note"] = (
                            "Here are accommodations in/near "
                            f"{params.get('location')} based on available records (matched by city-proper map anchor)."
                        )
                        params.pop("clear_location_anchor", None)
            elif loc_prefix in ("near", "around"):
                place_matches = _match_map_reference_places(raw_location, limit=4)
                if len(place_matches) > 1:
                    _set_location_ambiguity_clarification(raw_location, place_matches)
                elif place_matches:
                    place_match = place_matches[0]
                    map_location_anchor = str(place_match.get("name") or "").strip()
                    params["location_anchor"] = map_location_anchor
                    params["location_anchor_source"] = "map_reference"
                    params["location"] = _map_place_to_location_hint(map_location_anchor)
                    params["broaden_location"] = True
                    params["location_scope_note"] = (
                        "Here are accommodations in/near "
                        f"{params.get('location')} based on available records (matched by city-proper map anchor)."
                    )
                else:
                    _set_clarification(
                        "location",
                        "I couldn't map that place yet. Please specify a barangay/area in Bayawan, or use a known city-proper landmark.",
                        penalty=0.35,
                    )
            else:
                params["location"] = raw_location
            if params.get("location") and not params.get("location_scope_note"):
                params["location_scope_note"] = (
                    f"Here are accommodations in/near {params.get('location')} based on available records."
                )

    # Capture common location mentions even without "in/near/around".
    if "location" not in params:
        for known_location in known_location_map:
            if known_location in text:
                params["location"] = known_location_map.get(known_location, known_location)
                params["clear_location_anchor"] = True
                break

    # Allow map-place anchoring even when user did not use explicit "in/near/around".
    if (not params.get("location_anchor")) and (
        any(token in text for token in ("near", "nearest", "close to", "walking distance"))
        or any(token in text for token in ("terminal", "trike", "pedicab", "market", "plaza", "mall", "church", "hayahay", "eskina", "puregold"))
    ):
        if _is_generic_terminal_reference(text):
            _set_terminal_clarification(text)
        place_matches = _match_map_reference_places(text, limit=4)
        if len(place_matches) > 1 and not needs_clarification:
            _set_location_ambiguity_clarification(text, place_matches)
        elif place_matches and not needs_clarification:
            place_match = place_matches[0]
            map_location_anchor = str(place_match.get("name") or "").strip()
            params["location_anchor"] = map_location_anchor
            params["location_anchor_source"] = "map_reference"
            params["location"] = _map_place_to_location_hint(map_location_anchor)
            params.setdefault("broaden_location", True)
            params.setdefault(
                "location_scope_note",
                "Matched by city-proper map anchor and accommodation records.",
            )

    # Extract room reference:
    # - valid numeric room id: "room 12"
    # - invalid/tampered token: "room xyz" (captured for safe validation messaging)
    room_token_match = re.search(r"\broom\s+([a-z0-9\-]+)\b", text, flags=re.IGNORECASE)
    if room_token_match:
        room_token = str(room_token_match.group(1) or "").strip()
        if room_token:
            params["room_id"] = room_token

    # Extract check-in/check-out dates:
    # - YYYY-M-D / YYYY-MM-DD
    # - Month DD, YYYY (e.g., March 27, 2026)
    # - Mon DD, YYYY (e.g., Mar 27, 2026)
    month_regex = (
        r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
        r"jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
    )
    raw_date_matches = re.findall(
        rf"(\d{{4}}-\d{{1,2}}-\d{{1,2}}|{month_regex}\s+\d{{1,2}}(?:,\s*\d{{4}}|\s+\d{{4}}))",
        text,
        flags=re.IGNORECASE,
    )
    normalized_dates = [_normalize_iso_date(v) for v in raw_date_matches]
    date_matches = [v for v in normalized_dates if v]
    if len(date_matches) >= 2:
        params["check_in"] = date_matches[0]
        params["check_out"] = date_matches[1]
    elif len(raw_date_matches) == 1:
        _set_clarification(
            "date_range",
            "Please provide both check-in and check-out dates (YYYY-MM-DD or Month DD, YYYY).",
            penalty=0.35,
        )

    # Detect month-day ranges without year and ask for clarification instead of guessing.
    month_day_range_no_year = re.search(
        r"\b(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|"
        r"aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
        r"\s+\d{1,2}\s*(?:-|to)\s*\d{1,2}\b",
        text,
        flags=re.IGNORECASE,
    )
    if month_day_range_no_year and not re.search(r"\b\d{4}\b", text):
        _set_clarification(
            "date_range",
            "Please include the year for your check-in/check-out dates (e.g., 2026-03-27 or March 27, 2026).",
            penalty=0.4,
        )

    # Relative date resolution for conversational prompts:
    # today, tomorrow, this weekend, next week, and "for N nights starting tomorrow".
    if not params.get("check_in") and not params.get("check_out"):
        rel_check_in, rel_check_out, _ = _resolve_relative_stay_window(text)
        if rel_check_in and rel_check_out:
            params["check_in"] = rel_check_in.isoformat()
            params["check_out"] = rel_check_out.isoformat()

    # Extract accommodation/hotel name from "at <name>" or "hotel <name>".
    at_match = re.search(r"\bat\s+([a-z0-9\s\-&]+)", text)
    if at_match and "check-in" not in at_match.group(1):
        at_candidate = " ".join(str(at_match.group(1) or "").split()).strip()
        if at_candidate and not re.search(
            r"\b(near|around|in|for|budget|guest|guests|from|to)\b",
            at_candidate,
            flags=re.IGNORECASE,
        ):
            params.setdefault("accom_name", at_candidate)

    hotel_name_match = re.search(r"\b(?:hotel|inn)\s+([a-z0-9\s\-&]+)", text)
    if hotel_name_match:
        name_candidate = " ".join(str(hotel_name_match.group(1) or "").split()).strip()
        if name_candidate and not re.search(
            r"\b(near|around|in|for|budget|guest|guests|from|to)\b",
            name_candidate,
            flags=re.IGNORECASE,
        ):
            params.setdefault("accom_name", name_candidate)

    # Respect explicit accommodation type words in the user's prompt.
    if "inn" in text and "hotel" not in text:
        params.setdefault("company_type", "inn")
    elif "hotel" in text and "inn" not in text:
        params.setdefault("company_type", "hotel")
    elif "hotel" in text and "inn" in text:
        params.setdefault("company_type", "either")

    # Tour-type/preference extraction for follow-up prompts such as:
    # "i prefer nature tours", "culture tour", "falls package".
    tour_preference_aliases = {
        "sea": ("sea", "beach", "coastal", "ocean", "island", "shore", "seaside"),
        "nature": ("nature", "natural", "falls", "waterfall", "eco", "scenic"),
        "culture": ("culture", "cultural", "heritage", "food", "culinary"),
        "city": ("city", "urban", "downtown", "proper", "plaza"),
        "highlights": ("highlight", "highlights", "must-see"),
        "river": ("river", "riverside"),
        "adventure": ("adventure", "hike", "trek", "outdoor"),
        "family": ("family", "kid", "kids", "child-friendly"),
    }
    matched_tour_type = ""
    for canonical_tour_type, markers in tour_preference_aliases.items():
        if any(re.search(rf"\b{re.escape(marker)}\b", text) for marker in markers):
            matched_tour_type = canonical_tour_type
            break
    prefer_phrase_match = re.search(
        r"\b(?:i\s+prefer|prefer|i\s+want|want|how\s+about|what\s+about)\s+([a-z0-9\s\-]+)",
        text,
        flags=re.IGNORECASE,
    )
    if prefer_phrase_match:
        preference_phrase = str(prefer_phrase_match.group(1) or "").strip().lower()
        # Trim generic suffix words while keeping meaningful terms like "city highlights".
        preference_phrase = re.sub(
            r"\b(tour|tours|tour package|tour packages|package|packages|trip|trips)\b",
            "",
            preference_phrase,
            flags=re.IGNORECASE,
        ).strip()
        if preference_phrase:
            params["preference_text"] = preference_phrase
    if matched_tour_type:
        params["tour_type"] = matched_tour_type
        params["preference"] = matched_tour_type
    else:
        # Backward-compatible fallback for short legacy keywords.
        for keyword in ["river", "mountain", "sea", "sunset", "forest"]:
            if keyword in text:
                params["preference"] = keyword
                break
    timeframe_hint = _extract_tour_timeframe_hint(text)
    if timeframe_hint:
        params["tour_timeframe_hint"] = timeframe_hint

    # Tourism information query hint extraction.
    tourism_query_matchers = [
        r"(?:tell me about|information about|info about|details about)\s+([a-z0-9\s\-'&]+)",
        r"(?:operating hours for|opening hours for|contact for)\s+([a-z0-9\s\-'&]+)",
        r"(?:where is)\s+([a-z0-9\s\-'&]+)",
    ]
    for matcher in tourism_query_matchers:
        m = re.search(matcher, text, flags=re.IGNORECASE)
        if not m:
            continue
        candidate = " ".join(str(m.group(1) or "").split()).strip(" .,!?:;")
        if candidate:
            params["tourism_query"] = candidate
            break

    # Amenity keyword extraction for accommodation refinement prompts like "with pool".
    known_amenities = (
        "pool",
        "wifi",
        "parking",
        "breakfast",
        "aircon",
        "ac",
        "kitchen",
        "balcony",
        "gym",
    )
    matched_amenities = [token for token in known_amenities if re.search(rf"\b{re.escape(token)}\b", text)]
    if matched_amenities:
        # Keep as list; recommender normalizes both list and string inputs.
        params["amenities"] = matched_amenities

    preference_profile = _extract_preference_profile(text)
    preference_tags = preference_profile.get("preference_tags") or []
    prefer_low_price = bool(preference_profile.get("prefer_low_price"))

    if (
        not preference_tags
        and not prefer_low_price
        and any(token in text for token in ("hotel", "inn", "accommodation", "stay", "place"))
    ):
        gemini_profile = _extract_preference_profile_with_gemini(text)
        gemini_tags = gemini_profile.get("preference_tags") if isinstance(gemini_profile.get("preference_tags"), list) else []
        merged_tags = []
        for tag in (preference_tags + gemini_tags):
            tag_val = str(tag or "").strip().lower()
            if tag_val and tag_val not in merged_tags:
                merged_tags.append(tag_val)
        preference_tags = merged_tags
        prefer_low_price = prefer_low_price or bool(gemini_profile.get("prefer_low_price"))

    if preference_tags:
        params["preference_tags"] = preference_tags
    if prefer_low_price:
        params["prefer_low_price"] = True

    # Explicit scope broadening controls (strict by default).
    if re.search(r"\b(broaden|expand|widen)\s+(the\s+)?location\b", text):
        params["broaden_location"] = True
    if re.search(r"\b(include|allow|show)\s+(both\s+)?(hotel\s+and\s+inn|inn\s+and\s+hotel)\b", text):
        params["broaden_company_type"] = True
    if re.search(r"\b(broaden|expand)\s+(filters|search|scope)\b", text):
        params["broaden_location"] = True
        params["broaden_company_type"] = True

    return {
        "params": params,
        "confidence": round(confidence, 3),
        "needs_clarification": needs_clarification,
        "clarification_question": clarification_question,
        "clarification_field": clarification_field,
        "clarification_options": clarification_options,
    }


def _extract_params_from_message(message):
    parsed = _extract_params_with_confidence(message)
    return parsed.get("params", {})


def _extract_preference_profile(text):
    raw = str(text or "").strip().lower()
    if not raw:
        return {"preference_tags": [], "prefer_low_price": False}

    preference_map = {
        "quiet": [
            "quiet", "peaceful", "calm", "relaxing", "serene", "less noise", "not noisy",
            "chill", "tahimik", "walang ingay", "hindi maingay", "mingaw",
        ],
        "nature": [
            "nature", "green", "garden", "fresh air", "good environment", "cool place",
            "cool environment", "scenic", "view", "mountain", "river", "presko", "luntian",
        ],
        "family": [
            "family", "family-friendly", "kids", "children", "group", "spacious",
            "pang pamilya", "for family",
        ],
        "clean": [
            "clean", "sanitary", "hygienic", "well-maintained", "tidy",
            "malinis", "limpyo",
        ],
        "accessible": [
            "near terminal", "accessible", "easy transport", "commute", "near transport",
            "near downtown", "city proper", "walking distance", "near highway",
            "malapit", "duol",
        ],
        "romantic": [
            "romantic", "honeymoon", "couple", "date place", "for couples",
        ],
    }
    low_price_markers = (
        "cheap", "affordable", "budget-friendly", "budget friendly", "low price", "economical",
        "value for money", "sulit", "barato", "murag barato",
    )

    matched_tags = []
    for tag, markers in preference_map.items():
        if any(marker in raw for marker in markers):
            matched_tags.append(tag)

    return {
        "preference_tags": matched_tags,
        "prefer_low_price": any(marker in raw for marker in low_price_markers),
    }


def _extract_json_payload(raw_text):
    text = str(raw_text or "").strip()
    if not text:
        return {}
    try:
        return json.loads(text)
    except Exception:
        pass
    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        return {}
    try:
        return json.loads(match.group(0))
    except Exception:
        return {}


def _gemini_preference_parsing_enabled():
    raw = str(os.getenv("CHATBOT_GEMINI_PREFERENCE_PARSING_ENABLED", "1") or "").strip().lower()
    return raw in ("1", "true", "yes", "on")


def _extract_preference_profile_with_gemini(text):
    raw = str(text or "").strip()
    if not raw:
        return {"preference_tags": [], "prefer_low_price": False}
    if not _gemini_preference_parsing_enabled():
        return {"preference_tags": [], "prefer_low_price": False}
    if genai is None:
        return {"preference_tags": [], "prefer_low_price": False}
    api_key = str(os.getenv("GEMINI_API_KEY", "") or "").strip()
    if not api_key:
        return {"preference_tags": [], "prefer_low_price": False}

    model = str(os.getenv("GEMINI_MODEL", "gemini-1.5-flash") or "").strip() or "gemini-1.5-flash"
    prompt = (
        "Extract accommodation preference intent from user text.\n"
        "Return JSON only with keys:\n"
        "- preference_tags: array from this closed list only: "
        "[quiet, nature, family, clean, accessible, romantic]\n"
        "- prefer_low_price: boolean\n"
        "Rules:\n"
        "- Do not include other keys.\n"
        "- If uncertain, return empty array and false.\n"
        f"User text:\n{raw}"
    )
    try:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=model,
            contents=prompt,
        )
        payload = _extract_json_payload(getattr(response, "text", ""))
        tags = payload.get("preference_tags") if isinstance(payload.get("preference_tags"), list) else []
        normalized_tags = []
        allowed = {"quiet", "nature", "family", "clean", "accessible", "romantic"}
        for tag in tags:
            tag_str = str(tag or "").strip().lower()
            if tag_str in allowed and tag_str not in normalized_tags:
                normalized_tags.append(tag_str)
        return {
            "preference_tags": normalized_tags,
            "prefer_low_price": bool(payload.get("prefer_low_price")),
        }
    except Exception:
        return {"preference_tags": [], "prefer_low_price": False}


def _intent_from_message(message):
    text = (message or "").lower()
    recommendation_keywords = [
        "recommend", "suggest", "show", "find", "looking for", "search",
        "best place", "where should i go", "saan magandang", "saan maganda", "gumala",
        "affordable", "cheap", "budget-friendly", "relaxation", "family trip", "solo traveler",
        "upcoming tours", "upcoming tour", "tours today", "events this week", "this weekend",
    ]
    billing_keywords = [
        "bill",
        "billing",
        "total",
        "price",
        "cost",
        "how much",
        "amount due",
        "magkano",
        "bayad",
    ]
    booking_keywords = [
        "book",
        "booking",
        "reserve",
        "reservation",
        "mag-book",
        "book online",
        "availability",
        "available pa",
        "check availability",
        "pay later",
    ]
    accommodation_keywords = [
        "hotel", "inn", "room", "accommodation", "place to stay", "stay", "matutuluyan", "tulugan",
        "persons", "pax",
    ]
    tourism_info_keywords = [
        "tourism information",
        "tourism info",
        "tourist spot",
        "tourist spots",
        "attraction",
        "attractions",
        "landmark",
        "landmarks",
        "operating hours",
        "opening hours",
        "pasyalan",
        "puntahan",
        "gala",
        "galaan",
        "beach",
        "nature spot",
        "city proper",
    ]
    if ("available" in text or "availability" in text) and ("room" in text or "hotel" in text or "inn" in text):
        return "get_accommodation_recommendation"
    if any(keyword in text for keyword in recommendation_keywords) and any(
        keyword in text for keyword in accommodation_keywords
    ):
        return "get_accommodation_recommendation"
    if any(keyword in text for keyword in booking_keywords) and any(keyword in text for keyword in accommodation_keywords):
        return "book_accommodation"
    if any(keyword in text for keyword in billing_keywords):
        if any(keyword in text for keyword in accommodation_keywords):
            return "calculate_accommodation_billing"
        return "calculate_billing"
    if any(keyword in text for keyword in accommodation_keywords):
        return "get_accommodation_recommendation"
    if any(keyword in text for keyword in tourism_info_keywords):
        return "get_tourism_information"
    return "get_recommendation"


def _extract_tourism_search_tokens(message):
    raw = str(message or "").strip().lower()
    if not raw:
        return []
    cleaned = re.sub(r"[^a-z0-9\s\-]", " ", raw)
    stop_words = {
        "the", "and", "for", "with", "about", "please", "show", "tell", "me", "what",
        "where", "when", "how", "is", "are", "in", "on", "at", "of", "to", "a", "an",
        "tourism", "information", "info", "tourist", "spot", "spots", "attraction",
        "attractions", "landmark", "landmarks", "operating", "opening", "hours", "contact",
    }
    tokens = []
    for token in cleaned.split():
        token = token.strip("-")
        if len(token) < 3:
            continue
        if token in stop_words:
            continue
        tokens.append(token)
    deduped = []
    seen = set()
    for token in tokens:
        if token in seen:
            continue
        seen.add(token)
        deduped.append(token)
    return deduped[:8]


def _get_tourism_information(params, user_message):
    qs = TourismInformation.objects.published()
    if not qs.exists():
        return (
            "I don't have published tourism information yet. "
            "Please check back later once the Tourism Office publishes records."
        )

    location = str(params.get("location") or "").strip()
    if location:
        qs = qs.filter(location__icontains=location)

    tourism_query = str(params.get("tourism_query") or "").strip()
    if tourism_query:
        direct_qs = qs.filter(
            Q(spot_name__icontains=tourism_query)
            | Q(description__icontains=tourism_query)
            | Q(location__icontains=tourism_query)
        )
        if direct_qs.exists():
            qs = direct_qs

    if not tourism_query:
        tokens = _extract_tourism_search_tokens(user_message)
        if tokens:
            token_query = Q()
            for token in tokens:
                token_query |= (
                    Q(spot_name__icontains=token)
                    | Q(description__icontains=token)
                    | Q(location__icontains=token)
                )
            token_qs = qs.filter(token_query).distinct()
            if token_qs.exists():
                qs = token_qs

    rows = list(qs.order_by("spot_name")[:3])
    if not rows:
        return (
            "I couldn't find a published tourism information match for that query. "
            "Please try another spot name or location."
        )

    lines = ["Here are the tourism information records I found:"]
    for idx, row in enumerate(rows, 1):
        lines.append(f"{idx}. {row.spot_name}")
        lines.append(f"   Location: {row.location or 'Not specified'}")
        if row.operating_hours:
            lines.append(f"   Operating hours: {row.operating_hours}")
        if row.contact_information:
            lines.append(f"   Contact: {row.contact_information}")
        if row.description:
            lines.append(f"   Description: {row.description}")
    return "\n".join(lines)


def _normalize_intent_label(raw_label):
    normalized = str(raw_label or "").strip().lower()
    if not normalized:
        return ""
    normalized = _INTENT_LABEL_ALIASES.get(normalized, normalized)
    return normalized if normalized in _ALLOWED_INTENTS else ""


def _intent_confidence_threshold():
    raw = str(os.getenv("CHATBOT_INTENT_CNN_CONFIDENCE_THRESHOLD", "")).strip()
    if not raw:
        return 0.60
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return 0.60
    return max(0.0, min(1.0, value))


def _classify_intent_with_text_cnn(message):
    model_path, artifact_source = _resolve_intent_text_cnn_model_path()
    label_map_path = _default_label_map_path_for_model(model_path)
    prediction, err = _predict_text_cnn_labels(
        text=message,
        model_path=model_path,
        label_map_path=label_map_path,
    )
    if prediction is None:
        return {
            "intent": "",
            "source": "text_cnn_unavailable",
            "confidence": 0.0,
            "top_3": [],
            "error": err or "unknown_error",
            "artifact_source": artifact_source,
        }

    normalized_intent = _normalize_intent_label(prediction.get("predicted_class"))
    normalized_top3 = []
    raw_top3 = prediction.get("top_3") if isinstance(prediction.get("top_3"), list) else []
    for item in raw_top3:
        if not isinstance(item, dict):
            continue
        raw_label = item.get("label")
        mapped = _normalize_intent_label(raw_label)
        normalized_top3.append(
            {
                "raw_label": str(raw_label or ""),
                "intent": mapped,
                "confidence": float(item.get("confidence", 0.0) or 0.0),
            }
        )

    if normalized_intent:
        confidence = float(prediction.get("confidence", 0.0) or 0.0)
        threshold = _intent_confidence_threshold()
        if confidence < threshold:
            return {
                "intent": "",
                "source": "text_cnn_low_confidence",
                "confidence": confidence,
                "top_3": normalized_top3,
                "error": f"low_confidence:{confidence:.4f}<threshold:{threshold:.4f}",
                "artifact_source": artifact_source,
            }
        return {
            "intent": normalized_intent,
            "source": "text_cnn_intent",
            "confidence": confidence,
            "top_3": normalized_top3,
            "error": "",
            "artifact_source": artifact_source,
        }

    # Controlled compatibility fallback:
    # Existing deployments may still have accommodation-type labels (hotel/inn/etc.).
    return {
        "intent": "",
        "source": "text_cnn_incompatible_label_space",
        "confidence": 0.0,
        "top_3": normalized_top3,
        "error": "incompatible_label_space",
        "artifact_source": artifact_source,
    }


def _classify_intent_and_extract_params(message):
    extracted = _extract_params_with_confidence(message)
    params = extracted.get("params", {}) if isinstance(extracted.get("params"), dict) else {}
    cnn_result = _classify_intent_with_text_cnn(message)

    if cnn_result.get("intent"):
        return {
            "intent": cnn_result.get("intent"),
            "params": params,
            "source": cnn_result.get("source", "text_cnn_intent"),
            "confidence": float(cnn_result.get("confidence", 0.0) or 0.0),
            "needs_clarification": bool(extracted.get("needs_clarification")),
            "clarification_question": extracted.get("clarification_question", ""),
            "clarification_field": extracted.get("clarification_field", ""),
            "clarification_options": extracted.get("clarification_options", []),
            "intent_classifier": cnn_result,
        }

    heuristic_intent = _intent_from_message(message)
    return {
        "intent": heuristic_intent,
        "params": params,
        "source": "heuristic_intent_fallback",
        "confidence": float(extracted.get("confidence", 1.0) or 1.0),
        "needs_clarification": bool(extracted.get("needs_clarification")),
        "clarification_question": extracted.get("clarification_question", ""),
        "clarification_field": extracted.get("clarification_field", ""),
        "clarification_options": extracted.get("clarification_options", []),
        "intent_classifier": cnn_result,
    }


def _is_my_accommodation_booking_status_command(message):
    text = (message or "").strip().lower()
    my_accommodation_booking_phrases = [
        "show my bookings",
        "view my bookings",
        "check my bookings",
        "my bookings",
        "show bookings",
        "booking status",
        "my booking status",
        "show my hotel bookings",
        "show my inn bookings",
        "show my accommodation bookings",
        "show accommodation bookings",
        "show my room bookings",
        "view my hotel bookings",
        "view my accommodation bookings",
        "view accommodation bookings",
        "check my hotel bookings",
        "check accommodation bookings",
        "check my booking status for hotel",
        "my hotel bookings",
        "my accommodation bookings",
        "accommodation bookings",
        "hotel booking status",
        "inn booking status",
        "accommodation booking status",
        "reservation already confirmed",
        "is my reservation already confirmed",
        "is my booking confirmed",
        "check if my reservation is confirmed",
    ]
    return any(phrase in text for phrase in my_accommodation_booking_phrases)


def _is_guest_booking_requirements_command(message):
    text = str(message or "").strip().lower()
    if not text:
        return False
    return any(
        phrase in text
        for phrase in (
            "what details do i need to provide for booking",
            "what details do i need for booking",
            "what details are needed for booking",
            "what do i need to provide for booking",
            "requirements for booking",
            "booking requirements",
            "ano kailangan para mag book",
            "unsa kinahanglan para mag book",
        )
    )


def _is_guest_password_help_command(message):
    text = str(message or "").strip().lower()
    if not text:
        return False
    return any(
        phrase in text
        for phrase in (
            "forgot my password",
            "forgot password",
            "reset my password",
            "reset password",
            "can't log in",
            "cannot log in",
            "cant log in",
            "di maka login",
            "hindi makalogin",
            "nakalimutan ko password",
            "nalimot nako password",
        )
    )


def _is_guest_booking_cancel_support_command(message):
    text = str(message or "").strip().lower()
    if not text:
        return False
    return any(
        phrase in text
        for phrase in (
            "cancel my booking",
            "cancel booking",
            "cancel reservation",
            "cancel my reservation",
            "i want to cancel my booking",
            "i want to cancel my reservation",
            "pwede i-cancel",
            "icancel booking",
            "kansela booking",
        )
    )


def _is_guest_booking_change_date_command(message):
    text = str(message or "").strip().lower()
    if not text:
        return False
    return any(
        phrase in text
        for phrase in (
            "change my check-in date",
            "change check-in date",
            "change check in date",
            "change booking date",
            "reschedule my booking",
            "reschedule booking",
            "move my check-in",
            "move check-in",
            "baguhin check-in",
            "usab check-in",
        )
    )


def _is_guest_payment_methods_command(message):
    text = str(message or "").strip().lower()
    if not text:
        return False
    return any(
        phrase in text
        for phrase in (
            "payment method",
            "payment methods",
            "how can i pay",
            "how do i pay",
            "ways to pay",
            "mode of payment",
            "modes of payment",
            "paano magbayad",
            "unsaon pagbayad",
        )
    )


def _is_guest_down_payment_command(message):
    text = str(message or "").strip().lower()
    if not text:
        return False
    return any(
        phrase in text
        for phrase in (
            "down payment",
            "deposit required",
            "is down payment required",
            "need a deposit",
            "kailangan ba ng down payment",
            "need ba og downpayment",
        )
    )


def _is_guest_room_availability_command(message):
    text = str(message or "").strip().lower()
    if not text:
        return False
    if "available" in text and ("room" in text or "rooms" in text):
        return True
    return any(
        phrase in text
        for phrase in (
            "available pa tomorrow",
            "may available room",
            "available room ba",
            "are there available rooms today",
            "available rooms today",
        )
    )


def _resolve_relative_stay_window(message):
    text = str(message or "").strip().lower()
    today = timezone.localdate()
    nights_hint_match = re.search(r"\b(\d+)\s*night", text)
    nights_hint = _to_int(nights_hint_match.group(1), default=0) if nights_hint_match else 0

    starting_tomorrow_match = re.search(r"\bfor\s+(\d+)\s*nights?\s+starting\s+tomorrow\b", text)
    if starting_tomorrow_match:
        nights = max(_to_int(starting_tomorrow_match.group(1), default=1), 1)
        check_in = today + timedelta(days=1)
        return check_in, check_in + timedelta(days=nights), (
            f"starting tomorrow for {nights} night(s) "
            f"({check_in.isoformat()} to {(check_in + timedelta(days=nights)).isoformat()})"
        )

    if "today" in text:
        return today, today + timedelta(days=1), f"today ({today.isoformat()})"
    if "tomorrow" in text:
        check_in = today + timedelta(days=1)
        nights = max(nights_hint, 1)
        return check_in, check_in + timedelta(days=nights), (
            f"tomorrow ({check_in.isoformat()} to {(check_in + timedelta(days=nights)).isoformat()})"
        )
    if "weekend" in text:
        days_until_saturday = (5 - today.weekday()) % 7
        saturday = today + timedelta(days=days_until_saturday)
        monday = saturday + timedelta(days=2)
        return saturday, monday, f"this weekend ({saturday.isoformat()} to {monday.isoformat()})"
    if "next week" in text:
        days_until_next_monday = (7 - today.weekday()) % 7
        if days_until_next_monday == 0:
            days_until_next_monday = 7
        next_monday = today + timedelta(days=days_until_next_monday)
        nights = max(nights_hint, 2)
        return next_monday, next_monday + timedelta(days=nights), (
            f"next week ({next_monday.isoformat()} to {(next_monday + timedelta(days=nights)).isoformat()})"
        )
    return None, None, ""


def _build_guest_room_availability_summary(message, params):
    check_in, check_out, period_label = _resolve_relative_stay_window(message)
    guests = _to_int((params or {}).get("guests"), default=0)
    budget = _to_decimal((params or {}).get("budget"), default=Decimal("0"))
    location = str((params or {}).get("location") or "").strip()

    qs = (
        Room.objects.select_related("accommodation")
        .filter(status="AVAILABLE", current_availability__gte=1, accommodation__approval_status="accepted")
        .filter(
            Q(accommodation__company_type__icontains="hotel")
            | Q(accommodation__company_type__icontains="inn")
        )
    )
    if guests > 0:
        qs = qs.filter(person_limit__gte=guests)
    if budget > 0:
        qs = qs.filter(price_per_night__lte=budget)
    if location:
        qs = qs.filter(accommodation__location__icontains=location)
    if check_in and check_out:
        qs = qs.exclude(
            guest_bookings__status__in=["pending", "confirmed"],
            guest_bookings__check_in__lt=check_out,
            guest_bookings__check_out__gt=check_in,
        )

    rows = list(qs.order_by("price_per_night", "room_id")[:3])
    total = qs.count()
    filters = []
    if guests > 0:
        filters.append(f"for {guests} guest(s)")
    if budget > 0:
        filters.append(f"under PHP {int(budget)}")
    if location:
        filters.append(f"in/near {location}")
    filter_text = f" ({', '.join(filters)})" if filters else ""
    period_text = period_label or "the requested period"

    if total <= 0:
        return (
            f"I checked available hotel/inn rooms for {period_text}{filter_text}, but none matched.\n"
            "Try adjusting budget, guest count, or location so I can suggest alternatives."
        )

    lines = [f"I found {total} available room option(s) for {period_text}{filter_text}."]
    lines.append("Top options right now:")
    for idx, room in enumerate(rows, 1):
        lines.append(
            (
                f"{idx}. {room.accommodation.company_name} - {room.room_name} "
                f"(Room {room.room_id}) | PHP {room.price_per_night}/night | "
                f"up to {room.person_limit} guests"
            )
        )
    lines.append("Reply with the room number (e.g., 1) or say: book room <id> from <check-in> to <check-out>.")
    return "\n".join(lines)


def _is_accommodation_bookings_page_command(message):
    text = str(message or "").strip().lower()
    if not text:
        return False
    strong_phrases = (
        "accommodation bookings",
        "hotel bookings",
        "inn bookings",
        "room bookings",
        "booking status for hotel",
        "pending accommodation bookings",
    )
    if any(phrase in text for phrase in strong_phrases):
        return True
    return bool(
        re.search(r"\b(accommodation|hotel|inn|room)\b.*\bbooking", text)
        or re.search(r"\bbooking.*\b(accommodation|hotel|inn|room)\b", text)
    )


def _is_accommodation_owner_user(user, request=None):
    if not user or not getattr(user, "is_authenticated", False):
        return False
    try:
        if user.groups.filter(name__iexact="accommodation_owner").exists():
            return True
    except Exception:
        pass

    role_value = str(getattr(user, "role", "") or "").strip().lower()
    if role_value in {"accommodation_owner", "accommodation owner", "owner"}:
        return True

    if request is not None:
        try:
            session_user_type = str(request.session.get("user_type") or "").strip().lower()
            if session_user_type in {"accomodation", "accommodation", "establishment"}:
                return True
        except Exception:
            pass

    try:
        if hasattr(user, "owned_accommodations") and user.owned_accommodations.exists():
            return True
    except Exception:
        pass

    return False


def _resolve_chat_actor(request):
    user = getattr(request, "user", None)
    if user and getattr(user, "is_authenticated", False):
        # Prioritize elevated/staff roles before falling back to guest mode.
        try:
            if bool(getattr(user, "is_superuser", False)) or bool(getattr(user, "is_staff", False)):
                return {
                    "is_allowed": True,
                    "role": "admin",
                    "user": user,
                    "employee": None,
                    "display_name": str(getattr(user, "first_name", "") or getattr(user, "username", "") or "Admin").strip(),
                }
        except Exception:
            pass

        try:
            if user.groups.filter(name__iexact="admin").exists() or user.groups.filter(name__iexact="administrator").exists():
                return {
                    "is_allowed": True,
                    "role": "admin",
                    "user": user,
                    "employee": None,
                    "display_name": str(getattr(user, "first_name", "") or getattr(user, "username", "") or "Admin").strip(),
                }
            if user.groups.filter(name__iexact="employee").exists():
                return {
                    "is_allowed": True,
                    "role": "employee",
                    "user": user,
                    "employee": None,
                    "display_name": str(getattr(user, "first_name", "") or getattr(user, "username", "") or "Staff").strip(),
                }
        except Exception:
            pass

        try:
            session_user_type = str(request.session.get("user_type") or "").strip().lower()
            if session_user_type == "employee":
                return {
                    "is_allowed": True,
                    "role": "admin" if bool(request.session.get("is_admin")) else "employee",
                    "user": user,
                    "employee": None,
                    "display_name": str(getattr(user, "first_name", "") or getattr(user, "username", "") or "Staff").strip(),
                }
        except Exception:
            pass

        if _is_accommodation_owner_user(user, request=request):
            return {
                "is_allowed": True,
                "role": "owner",
                "user": user,
                "employee": None,
                "display_name": str(getattr(user, "first_name", "") or getattr(user, "username", "") or "Owner").strip(),
            }
        return {
            "is_allowed": True,
            "role": "guest",
            "user": user,
            "employee": None,
            "display_name": str(getattr(user, "first_name", "") or getattr(user, "username", "") or "Guest").strip(),
        }

    try:
        session_user_type = str(request.session.get("user_type") or "").strip().lower()
        employee_id = request.session.get("employee_id")
    except Exception:
        session_user_type = ""
        employee_id = None

    if session_user_type == "employee" and employee_id:
        employee = Employee.objects.filter(emp_id=employee_id).first()
        if employee and str(getattr(employee, "status", "") or "").strip().lower() == "accepted":
            is_admin = bool(request.session.get("is_admin")) or str(getattr(employee, "role", "")).strip().lower() == "admin"
            return {
                "is_allowed": True,
                "role": "admin" if is_admin else "employee",
                "user": None,
                "employee": employee,
                "display_name": str(getattr(employee, "first_name", "") or "Staff").strip(),
            }

    return {
        "is_allowed": False,
        "role": "anonymous",
        "user": None,
        "employee": None,
        "display_name": "User",
    }


def _is_help_or_greeting_command(message):
    text = str(message or "").strip().lower()
    if not text:
        return False
    quick = {
        "help",
        "menu",
        "commands",
        "start",
        "hello",
        "hi",
        "hey",
    }
    if text in quick:
        return True
    return any(
        phrase in text
        for phrase in (
            "what can you do",
            "how can you help",
            "assist me",
            "show commands",
            "show menu",
        )
    )


def _contains_any_phrase(message, phrases):
    raw_text = str(message or "").strip().lower()
    if not raw_text:
        return False
    text = re.sub(r"[^a-z0-9\s]", " ", raw_text)
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return False

    # Keep direct substring matching first for backwards compatibility.
    for phrase in phrases:
        phrase_text = str(phrase or "").strip().lower()
        if not phrase_text:
            continue
        if phrase_text in raw_text:
            return True

    text_tokens = set(text.split())
    stopwords = {
        "a", "an", "the", "to", "for", "in", "on", "at", "of", "is", "are", "am",
        "my", "your", "our", "ako", "akong", "nako", "ko", "ang", "sa", "si", "ng",
        "yung", "ito", "yan", "kini", "kani", "ug", "and", "or",
    }
    for phrase in phrases:
        phrase_text = re.sub(r"[^a-z0-9\s]", " ", str(phrase or "").strip().lower())
        phrase_text = re.sub(r"\s+", " ", phrase_text).strip()
        if not phrase_text:
            continue
        if phrase_text in text:
            return True
        phrase_tokens = [tok for tok in phrase_text.split() if tok not in stopwords]
        # Only use unordered-token fallback when phrase still has useful signal.
        if len(phrase_tokens) >= 3 and all(token in text_tokens for token in phrase_tokens):
            return True
    return False


def _is_remember_preferences_command(message):
    return _contains_any_phrase(
        message,
        (
            "remember my preferences",
            "save my preferences",
            "remember this preference",
            "remember these preferences",
            "save this preference",
            "save these preferences",
        ),
    )


def _is_forget_preferences_command(message):
    return _contains_any_phrase(
        message,
        (
            "forget my preferences",
            "clear my preferences",
            "remove my preferences",
            "reset my preferences",
            "forget preferences",
        ),
    )


def _extract_memory_preference_payload(params):
    if not isinstance(params, dict):
        return {}
    payload = {}
    for key in (
        "company_type",
        "location",
        "budget",
        "guests",
        "preference_tags",
        "prefer_low_price",
        "amenities",
    ):
        value = params.get(key)
        if value in (None, "", [], {}):
            continue
        payload[key] = value
    return payload


def _apply_saved_preferences_to_params(params, saved_prefs):
    if not isinstance(params, dict):
        params = {}
    if not isinstance(saved_prefs, dict):
        return params
    merged = dict(params)
    for key, value in saved_prefs.items():
        if key not in (
            "company_type",
            "location",
            "budget",
            "guests",
            "preference_tags",
            "prefer_low_price",
            "amenities",
        ):
            continue
        if merged.get(key) in (None, "", [], {}):
            merged[key] = value
    return merged


def _build_role_operational_snapshot(actor):
    role = str(actor.get("role") or "").strip().lower()
    user = actor.get("user")
    employee = actor.get("employee")

    try:
        if role == "owner" and user is not None:
            owner_accom_qs = Accomodation.objects.filter(owner=user, is_active=True)
            accepted_accom_qs = owner_accom_qs.filter(approval_status="accepted")
            room_qs = Room.objects.filter(accommodation__in=accepted_accom_qs)
            booking_qs = AccommodationBooking.objects.filter(accommodation__in=accepted_accom_qs)
            return (
                "Current snapshot: "
                f"{owner_accom_qs.count()} accommodation(s), "
                f"{room_qs.count()} room(s), "
                f"{booking_qs.filter(status='pending').count()} pending booking(s)."
            )

        if role == "admin":
            pending_accom = Accomodation.objects.filter(is_active=True, approval_status="pending").count()
            pending_owner_group, _ = Group.objects.get_or_create(name="accommodation_owner_pending")
            pending_owner = pending_owner_group.user_set.count()
            today = timezone.localdate()
            today_bookings = AccommodationBooking.objects.filter(booking_date__date=today).count()
            return (
                "Current snapshot: "
                f"{pending_accom} pending accommodation approval(s), "
                f"{pending_owner} pending owner account(s), "
                f"{today_bookings} booking event(s) today."
            )

        if role == "employee":
            active_sched = Tour_Schedule.objects.filter(status="active").count()
            upcoming_sched = Tour_Schedule.objects.filter(start_time__date__gte=timezone.localdate()).count()
            name = str(getattr(employee, "first_name", "") or "").strip() if employee else ""
            prefix = f"{name}, " if name else ""
            return (
                f"Current snapshot: {prefix}"
                f"{active_sched} active schedule(s), "
                f"{upcoming_sched} upcoming schedule(s)."
            )
    except Exception:
        return ""

    return ""


def _build_role_help_payload(actor):
    role = str(actor.get("role") or "").strip().lower()
    snapshot = _build_role_operational_snapshot(actor)
    if role == "owner":
        snapshot_block = f"{snapshot}\n" if snapshot else ""
        return {
            "fulfillmentText": (
                "Owner assistant mode is active. I can help you check your business side.\n"
                f"{snapshot_block}"
                "- Show my accommodations\n"
                "- Show my rooms\n"
                "- Show my bookings\n"
                "- Open reports and analytics\n"
                "- Open Owner Hub\n"
                "- Show tourism information about <place>\n\n"
                "For registration/edits, use Owner Hub."
            ),
            "quick_replies": [
                "Show my accommodations",
                "Show my rooms",
                "Show my bookings",
                "Open reports and analytics",
                "Open Owner Hub",
            ],
        }
    if role == "admin":
        snapshot_block = f"{snapshot}\n" if snapshot else ""
        return {
            "fulfillmentText": (
                "Admin assistant mode is active.\n"
                f"{snapshot_block}"
                "I can answer tourism information, show moderation summaries, and help with quick navigation.\n"
                "Use the Admin Dashboard for approvals, encoding, and reports."
            ),
            "quick_replies": [
                "Open dashboard",
                "Open map",
                "Open discounts",
                "Open traveler surveys",
                "Show pending accommodations",
                "Show pending owner accounts",
                "Open accommodation bookings",
            ],
        }
    if role == "employee":
        snapshot_block = f"{snapshot}\n" if snapshot else ""
        return {
            "fulfillmentText": (
                "Employee assistant mode is active.\n"
                f"{snapshot_block}"
                "I can answer tourism information and guide role-based navigation.\n"
                "Use the Employee Dashboard for operational tasks."
            ),
            "quick_replies": [
                "Open dashboard",
                "Open tour list",
                "Open assigned tours",
                "Open tour calendar",
                "Open map",
                "Open profile",
            ],
        }
    return {
        "fulfillmentText": (
            "Guest assistant mode is active. I can help you find and book hotels/inns in Bayawan.\n"
            "Try: recommend a hotel in Bayawan for 2 guests under 2000."
        ),
        "quick_replies": [
            "Recommend a hotel in Bayawan for 2 guests under 2000",
            "Open map",
            "Show tourism information in Bayawan",
            "Show my bookings",
            "Remember my preferences",
            "Forget my preferences",
        ],
    }


def _build_out_of_scope_payload(actor):
    role = str(actor.get("role") or "").strip().lower()
    if role == "owner":
        return {
            "fulfillmentText": (
                "That request is outside the scope of this owner assistant.\n"
                "I can help with owner operations such as rooms, bookings, performance summaries, and owner navigation."
            ),
            "quick_replies": ["Help", "Show my rooms", "Show my bookings", "Open Owner Hub"],
        }
    if role == "admin":
        return {
            "fulfillmentText": (
                "That request is outside the scope of this admin assistant.\n"
                "I can help with moderation, approvals, accommodation bookings, survey results, and admin navigation."
            ),
            "quick_replies": ["Help", "Show pending accommodations", "Show pending owner accounts", "Open dashboard"],
        }
    if role == "employee":
        return {
            "fulfillmentText": (
                "That request is outside the scope of this employee assistant.\n"
                "I can help with assigned tours, tour calendar, accommodations, profile, and employee navigation."
            ),
            "quick_replies": ["Help", "Open assigned tours", "Open tour calendar", "Open dashboard"],
        }
    return {
        "fulfillmentText": (
            "That request is outside this system's scope.\n"
            "I can assist with Bayawan tourism information, hotel/inn recommendations, booking, billing, and booking status."
        ),
        "quick_replies": [
            "Help",
            "Recommend a hotel in Bayawan for 2 guests under 2000",
            "Show tourism information in Bayawan",
            "View my accommodation bookings",
        ],
    }


def _is_open_dashboard_command(message):
    text = str(message or "").strip().lower()
    if not text:
        return False
    return any(
        phrase in text
        for phrase in ("open dashboard", "go to dashboard", "dashboard")
    )


def _is_out_of_scope_message(message):
    text = str(message or "").strip().lower()
    if not text:
        return False
    if re.fullmatch(r"\d{1,2}", text):
        return False

    domain_keywords = (
        "bayawan",
        "tour",
        "itinerary",
        "schedule",
        "tourism",
        "hotel",
        "inn",
        "accommodation",
        "room",
        "booking",
        "bookings",
        "billing",
        "payment",
        "dashboard",
        "owner",
        "employee",
        "admin",
        "reports",
        "analytics",
        "survey",
        "map",
        "spot",
        "attraction",
    )
    if any(keyword in text for keyword in domain_keywords):
        return False

    out_scope_markers = (
        "poem",
        "joke",
        "lyrics",
        "translate",
        "who is",
        "what is",
        "where is",
        "when did",
        "why is",
        "history of",
        "solve",
        "math",
        "code this",
    )
    if any(marker in text for marker in out_scope_markers):
        return True

    return False


def _is_owner_room_overview_command(message):
    text = str(message or "").strip().lower()
    if not text:
        return False
    phrases = (
        "show my rooms",
        "view my rooms",
        "list my rooms",
        "check my rooms",
        "my rooms",
        "show rooms",
        "room list",
        "room status",
    )
    if any(phrase in text for phrase in phrases):
        return True
    return bool(
        re.search(r"\b(show|view|list|check)\b.*\b(my\s+)?rooms?\b", text)
    )


def _is_owner_accommodation_overview_command(message):
    text = str(message or "").strip().lower()
    if not text:
        return False
    phrases = (
        "show my accommodations",
        "view my accommodations",
        "list my accommodations",
        "my accommodations",
        "my accommodation",
        "show my inns",
        "show my hotels",
        "show my businesses",
    )
    if any(phrase in text for phrase in phrases):
        return True
    return bool(
        re.search(r"\b(show|view|list|check)\b.*\b(my\s+)?(accommodation|accommodations|hotel|inn|business)\b", text)
    )


def _is_owner_hub_command(message):
    return _contains_any_phrase(
        message,
        (
            "open owner hub",
            "go to owner hub",
            "owner hub",
            "owner dashboard",
            "open accommodation owner hub",
            "adto owner hub",
            "ablihi owner hub",
            "owner panel",
        ),
    )


def _is_owner_register_accommodation_command(message):
    return _contains_any_phrase(
        message,
        (
            "register accommodation",
            "add accommodation",
            "create accommodation",
            "open accommodation registration",
            "register hotel",
            "register inn",
            "add hotel",
            "add inn",
            "parehistro ug accommodation",
            "parehistro ug hotel",
            "parehistro ug inn",
            "mag register ng accommodation",
        ),
    )


def _is_owner_performance_summary_command(message):
    return _contains_any_phrase(
        message,
        (
            "owner performance",
            "my performance",
            "show my summary",
            "show my business summary",
            "show occupancy",
            "occupancy summary",
            "room occupancy",
            "show revenue",
            "revenue summary",
            "income summary",
            "booking performance",
            "performance snapshot",
            "pakita occupancy",
            "pakita revenue",
            "pakita summary",
        ),
    )


def _is_owner_reports_analytics_command(message):
    return _contains_any_phrase(
        message,
        (
            "open reports and analytics",
            "open reports",
            "owner reports",
            "hotel reports",
            "reports analytics",
            "show reports and analytics",
            "show reports",
            "open analytics",
            "owner analytics",
            "pakita reports",
            "pakita analytics",
        ),
    )


def _detect_owner_support_topic(message):
    text = str(message or "").strip().lower()
    if not text:
        return ""
    compact_text = re.sub(r"[^a-z0-9\s]", " ", text)
    compact_text = re.sub(r"\s+", " ", compact_text).strip()

    if _contains_any_phrase(text, ("forgot my password", "forgot password", "reset password", "can't log in", "cannot log in", "cant log in", "nakalimutan ko password", "di makalogin", "dili ko ka log in")):
        return "owner_password_help"
    if _contains_any_phrase(text, ("listing not showing", "listing is missing", "my listing is missing", "not showing on the website", "hindi lumalabas ang listing", "listing missing", "wala nagpakita akong listing", "di makita listing")):
        return "owner_listing_visibility"
    if _contains_any_phrase(text, ("cannot update room", "can't update room", "cant update room", "cannot edit room", "i cannot update my room details", "i-edit ang room details", "edit room details after posting", "di ko ma update room", "dili ma edit room")):
        return "owner_room_update_issue"

    if _contains_any_phrase(text, ("register my hotel", "register my inn", "register accommodation", "add my inn", "add my hotel", "i-register ang hotel", "i register ang hotel", "parehistro sa akong hotel", "irehistro ko ang inn")):
        return "owner_register_listing"
    if _contains_any_phrase(text, ("requirements", "need to submit", "what requirements", "list my accommodation", "unsa requirements", "ano requirements")):
        return "owner_listing_requirements"
    if _contains_any_phrase(text, ("update my accommodation information", "edit my hotel profile", "edit my accommodation", "update accommodation information", "update listing", "edit listing profile", "i update ang listing profile", "usba akong accommodation info")):
        return "owner_listing_update"

    if _contains_any_phrase(text, ("add a new room", "add room", "register room", "mag add room", "dugang kwarto")):
        return "owner_add_room"
    if _contains_any_phrase(text, ("update room price", "update my room rates", "update room rates", "update price", "room price", "room rates", "usba presyo sa kwarto", "iupdate ko ang room rate")):
        return "owner_update_room_price"
    if _contains_any_phrase(text, ("change the room capacity", "room capacity", "change capacity", "capacity of room", "update room capacity", "usba capacity sa kwarto", "ilan capacity ng room")):
        return "owner_update_room_capacity"
    if _contains_any_phrase(text, ("mark a room as unavailable", "room unavailable", "mark unavailable", "close a room for maintenance", "maintenance", "i-mark unavailable ang room", "isarado ang room for maintenance")):
        return "owner_mark_room_unavailable"
    if _contains_any_phrase(text, ("edit the amenities of a room", "edit amenities", "room amenities", "amenities sa kwarto", "amenities ng room")):
        return "owner_edit_room_amenities"

    if any(
        phrase in text
        for phrase in (
            "view bookings",
            "show bookings",
            "check bookings",
            "bookings for my accommodation",
            "makikita ang bookings",
            "bookings ng accommodation ko",
        )
    ):
        return "owner_view_bookings"
    if _contains_any_phrase(text, ("who reserved", "who booked", "guest reservation details", "kinsa nagbook", "sino nag book")):
        return "owner_booking_guest_details"
    if _contains_any_phrase(text, ("confirm a guest reservation", "confirm reservation", "approve reservation", "i confirm ang reservation", "aprubahan reservation")):
        return "owner_confirm_reservation"
    if _contains_any_phrase(text, ("pending reservations", "pending bookings", "pending ang bookings", "pending reservations ko")):
        return "owner_pending_reservations"
    if _contains_any_phrase(text, ("booking is cancelled", "cancelled booking", "canceled booking", "kanselado booking", "cancelled na reservation")):
        return "owner_cancelled_reservations"

    if _contains_any_phrase(text, ("update room availability", "availability update", "reopen a room", "re-open a room", "reopen room", "iupdate availability sa room", "ablihi balik ang kwarto")):
        return "owner_update_availability"
    if _contains_any_phrase(text, ("still showing as available", "why is my room still showing as available", "still available", "nganong available gihapon", "bakit available pa rin")):
        return "owner_room_still_available_issue"
    if _contains_any_phrase(text, ("block dates", "block date", "close dates", "unavailable dates", "i block ang dates", "isarado ang petsa")):
        return "owner_block_dates"

    if _contains_any_phrase(text, ("payment status of reservations", "check if a guest already paid", "payment status", "already paid", "bayad na ba ang guest", "status sa bayad")):
        return "owner_payment_status"
    if _contains_any_phrase(text, ("billing details for a booking", "view billing details", "billing details", "detalye sa billing", "billing details ng booking")):
        return "owner_billing_details"
    if _contains_any_phrase(text, ("booking transactions", "transactions", "booking transaction", "transaksyon sa booking", "transaction history")):
        return "owner_transactions"
    if any(
        phrase in text
        for phrase in (
            "guests book directly from my listing",
            "book directly from my listing",
            "users book from my accommodation page",
            "how do guests book my rooms",
            "reserve directly from my listing",
            "booking work for my accommodation",
            "can guests book directly",
            "pwede ba magbook diretso sa listing ko",
            "makabook ba diretso ang guests sa listing nako",
        )
    ):
        return "owner_direct_booking_flow"

    if compact_text in {"add room", "update price", "view bookings", "listing not showing", "room unavailable how"}:
        mapping = {
            "add room": "owner_add_room",
            "update price": "owner_update_room_price",
            "view bookings": "owner_view_bookings",
            "listing not showing": "owner_listing_visibility",
            "room unavailable how": "owner_mark_room_unavailable",
        }
        return mapping.get(compact_text, "")

    return ""


def _build_owner_manage_rooms_link_payload(request, *, text, label="Open Manage Rooms"):
    user = getattr(request, "user", None)
    accepted = (
        Accomodation.objects.filter(owner=user, approval_status="accepted", is_active=True)
        .order_by("company_name", "accom_id")
        .first()
    )
    if accepted is None:
        return _build_link_payload(
            request,
            text=f"{text}\nI could not find an accepted accommodation yet. Open Owner Hub first.",
            route_name="admin_app:owner_hub",
            label="Open Owner Hub",
        )
    manage_link = reverse("admin_app:owner_manage_rooms", kwargs={"accom_id": accepted.accom_id})
    if hasattr(request, "build_absolute_uri"):
        manage_link = request.build_absolute_uri(manage_link)
    return {
        "fulfillmentText": text,
        "billing_link": manage_link,
        "billing_link_label": label,
        "open_in_new_tab": True,
    }


def _build_owner_booking_payment_summary(user, *, max_rows=5):
    owner_bookings = (
        AccommodationBooking.objects.select_related("accommodation", "room")
        .filter(accommodation__owner=user, accommodation__is_active=True)
        .order_by("-booking_date", "-booking_id")
    )
    total = owner_bookings.count()
    if total <= 0:
        return (
            "No booking records found for your accommodations yet.\n"
            "Once guests create bookings, booking and payment status will appear here."
        )

    status_counts = Counter(
        str(status or "").strip().lower()
        for status in owner_bookings.values_list("status", flat=True)
    )
    payment_counts = Counter(
        str(status or "").strip().lower()
        for status in owner_bookings.values_list("payment_status", flat=True)
    )
    lines = [
        (
            f"Owner booking summary: Total {total}, "
            f"Pending {status_counts.get('pending', 0)}, "
            f"Confirmed {status_counts.get('confirmed', 0)}, "
            f"Declined {status_counts.get('declined', 0)}, "
            f"Cancelled {status_counts.get('cancelled', 0)}."
        ),
        (
            f"Payment summary: Unpaid {payment_counts.get('unpaid', 0)}, "
            f"Partial {payment_counts.get('partial', 0)}, "
            f"Paid {payment_counts.get('paid', 0)}."
        ),
        "Recent bookings:",
    ]
    shown = 0
    for booking in owner_bookings[: max(max_rows, 1)]:
        if shown >= max_rows:
            break
        accom_name = str(getattr(booking.accommodation, "company_name", "") or "Accommodation").strip()
        room = getattr(booking, "room", None)
        room_label = f"Room {getattr(room, 'room_id', '')}" if room is not None else "No room"
        lines.append(
            (
                f"- Booking ID {booking.booking_id} | {accom_name} | {room_label} | "
                f"{booking.check_in} to {booking.check_out} | "
                f"Status: {str(booking.status).title()} | Payment: {str(booking.payment_status).title()} | "
                f"Paid PHP {Decimal(booking.amount_paid):.2f} / PHP {Decimal(booking.total_amount):.2f}"
            )
        )
        shown += 1
    return "\n".join(lines)


def _build_owner_billing_details_summary(user, *, max_rows=5):
    billing_qs = (
        Billing.objects.select_related("booking", "booking__accommodation", "booking__room")
        .filter(booking__accommodation__owner=user, booking__accommodation__is_active=True)
        .order_by("-billing_date", "-billing_id")
    )
    total = billing_qs.count()
    if total <= 0:
        return "No billing records found yet for your accommodations."
    lines = [f"Billing records found: {total}. Recent billing entries:"]
    shown = 0
    for billing in billing_qs[: max(max_rows, 1)]:
        if shown >= max_rows:
            break
        booking = getattr(billing, "booking", None)
        accom = getattr(booking, "accommodation", None) if booking is not None else None
        accom_name = str(getattr(accom, "company_name", "") or "Accommodation").strip()
        room = getattr(booking, "room", None) if booking is not None else None
        room_id = getattr(room, "room_id", "")
        lines.append(
            (
                f"- Billing ID {billing.billing_id} | Booking ID {getattr(booking, 'booking_id', '')} | "
                f"{accom_name} | Room {room_id} | "
                f"Status: {str(billing.payment_status).title()} | "
                f"Method: {str(billing.payment_method or 'Not Set').replace('_', ' ').title()} | "
                f"Paid PHP {Decimal(billing.amount_paid):.2f} / PHP {Decimal(billing.total_amount):.2f}"
            )
        )
        shown += 1
    return "\n".join(lines)


def _build_owner_listing_visibility_diagnostic(user):
    accom_qs = Accomodation.objects.filter(owner=user, is_active=True)
    total = accom_qs.count()
    if total <= 0:
        return (
            "I could not find any active accommodation listing under your account.\n"
            "Please register your accommodation first from Owner Hub."
        )
    accepted = accom_qs.filter(approval_status="accepted")
    pending = accom_qs.filter(approval_status="pending")
    declined = accom_qs.filter(approval_status="declined")

    lines = [
        (
            f"Listing visibility check: Total {total}, "
            f"Accepted {accepted.count()}, Pending {pending.count()}, Declined {declined.count()}."
        )
    ]
    if pending.exists():
        lines.append("- Some listings are still pending admin approval; they may not appear publicly yet.")
    if declined.exists():
        lines.append("- Some listings were declined. Please open Owner Hub and update required details before re-submission.")
    if accepted.exists():
        accepted_rooms = Room.objects.filter(accommodation__in=accepted).count()
        lines.append(f"- Accepted listings currently have {accepted_rooms} room record(s).")
        if accepted_rooms <= 0:
            lines.append("- Add rooms to your accepted listing so it can appear in stay recommendations.")
    lines.append("Use Owner Hub and Manage Rooms to refresh listing details.")
    return "\n".join(lines)


def _build_owner_accommodations_summary(user, *, max_rows=8):
    owner_accommodations = list(
        Accomodation.objects.filter(owner=user, is_active=True).order_by("-submitted_at", "company_name")
    )
    if not owner_accommodations:
        return (
            "You don't have any accommodation record yet. "
            "Open Owner Hub and click Register New Accommodation."
        )

    status_counts = Counter(
        str(status or "").strip().lower()
        for status in Accomodation.objects.filter(owner=user, is_active=True).values_list("approval_status", flat=True)
    )
    lines = [
        (
            f"You currently have {len(owner_accommodations)} accommodation(s): "
            f"Accepted {status_counts.get('accepted', 0)}, "
            f"Pending {status_counts.get('pending', 0)}, "
            f"Declined {status_counts.get('declined', 0)}."
        ),
        "Your accommodations:",
    ]

    shown = 0
    for accom in owner_accommodations:
        if shown >= max_rows:
            break
        room_count = Room.objects.filter(accommodation=accom).count()
        status = str(getattr(accom, "approval_status", "") or "").strip().title() or "Unknown"
        company_type = str(getattr(accom, "company_type", "") or "Accommodation").strip().title()
        lines.append(
            f"- {accom.company_name} ({company_type}) | Status: {status} | Rooms: {room_count}"
        )
        shown += 1

    if len(owner_accommodations) > shown:
        lines.append(f"...and {len(owner_accommodations) - shown} more accommodation(s).")

    lines.append("Tip: Use Owner Hub to manage rooms and submit additional accommodations.")
    return "\n".join(lines)


def _build_owner_rooms_summary(user, *, max_rows=12):
    accepted_accommodations = list(
        Accomodation.objects.filter(owner=user, approval_status="accepted", is_active=True).order_by("company_name")
    )
    if not accepted_accommodations:
        return (
            "You don't have any accepted accommodation yet. "
            "Please wait for admin approval, then add rooms in Owner Hub."
        )

    rooms_qs = (
        Room.objects.select_related("accommodation")
        .filter(accommodation__in=accepted_accommodations)
        .order_by("accommodation__company_name", "room_name", "room_id")
    )
    total_rooms = rooms_qs.count()
    if total_rooms <= 0:
        return (
            "Your accepted accommodation is ready, but no rooms are registered yet. "
            "Open Owner Hub and click Manage Rooms to add your first room."
        )

    available_slots = rooms_qs.aggregate(total=Sum("current_availability")).get("total") or 0
    lines = [
        (
            f"You currently have {total_rooms} room(s) across "
            f"{len(accepted_accommodations)} accepted accommodation(s). "
            f"Total available slots: {available_slots}."
        ),
        "Your rooms:",
    ]

    shown = 0
    for room in rooms_qs:
        if shown >= max_rows:
            break
        accom = getattr(room, "accommodation", None)
        accom_name = str(getattr(accom, "company_name", "") or "Accommodation").strip()
        room_name = str(getattr(room, "room_name", "") or f"Room {room.room_id}").strip()
        price = _to_decimal(getattr(room, "price_per_night", 0), default=Decimal("0"))
        capacity = _to_int(getattr(room, "person_limit", 0), default=0)
        current = _to_int(getattr(room, "current_availability", 0), default=0)
        status = str(getattr(room, "status", "") or "").strip().title() or "Unknown"
        lines.append(
            (
                f"- {accom_name} | {room_name} (Room {room.room_id}) | "
                f"PHP {price:.2f}/night | Capacity: {capacity} pax | "
                f"Available: {current} | Status: {status}"
            )
        )
        shown += 1

    if total_rooms > shown:
        lines.append(f"...and {total_rooms - shown} more room(s).")

    lines.append("Tip: Open Owner Hub > Manage Rooms to edit prices, pax, and availability.")
    return "\n".join(lines)


def _build_owner_performance_summary(user):
    accepted_accommodations = list(
        Accomodation.objects.filter(owner=user, approval_status="accepted", is_active=True)
    )
    if not accepted_accommodations:
        return (
            "You don't have any accepted accommodation yet, so performance metrics are not available.\n"
            "Once approved, add rooms and receive bookings to populate this summary."
        )

    rooms_qs = Room.objects.filter(accommodation__in=accepted_accommodations)
    total_rooms = rooms_qs.count()
    total_capacity = rooms_qs.aggregate(total=Sum("person_limit")).get("total") or 0
    total_available = rooms_qs.aggregate(total=Sum("current_availability")).get("total") or 0
    total_occupied = max(int(total_capacity) - int(total_available), 0)
    occupancy_pct = (float(total_occupied) / float(total_capacity) * 100.0) if total_capacity else 0.0

    booking_qs = AccommodationBooking.objects.filter(accommodation__in=accepted_accommodations)
    total_bookings = booking_qs.count()
    pending_count = booking_qs.filter(status="pending").count()
    confirmed_count = booking_qs.filter(status="confirmed").count()
    declined_count = booking_qs.filter(status="declined").count()
    cancelled_count = booking_qs.filter(status="cancelled").count()
    total_confirmed_revenue = booking_qs.filter(status="confirmed").aggregate(total=Sum("total_amount")).get("total") or Decimal("0")
    total_paid = booking_qs.aggregate(total=Sum("amount_paid")).get("total") or Decimal("0")

    lines = [
        "Owner Performance Snapshot",
        f"- Accepted accommodations: {len(accepted_accommodations)}",
        f"- Total rooms: {total_rooms}",
        f"- Estimated occupancy now: {occupancy_pct:.1f}% ({total_occupied} occupied capacity out of {total_capacity})",
        (
            f"- Booking counts: total {total_bookings}, pending {pending_count}, "
            f"confirmed {confirmed_count}, declined {declined_count}, cancelled {cancelled_count}"
        ),
        f"- Confirmed booking revenue (gross): PHP {Decimal(total_confirmed_revenue):.2f}",
        f"- Total amount paid (collected): PHP {Decimal(total_paid):.2f}",
        "Note: Occupancy is estimated from room capacity vs current availability.",
    ]
    return "\n".join(lines)


def _build_owner_direct_booking_flow_summary(user):
    active_qs = Accomodation.objects.filter(owner=user, is_active=True)
    if not active_qs.exists():
        return (
            "I couldn't find an active accommodation listing under your owner account yet.\n"
            "Register your accommodation first in Owner Hub."
        )

    accepted_qs = active_qs.filter(approval_status="accepted")
    pending_qs = active_qs.filter(approval_status="pending")
    declined_qs = active_qs.filter(approval_status="declined")
    accepted_rooms = Room.objects.filter(accommodation__in=accepted_qs)
    publicly_bookable_rooms = accepted_rooms.filter(
        status="AVAILABLE",
        current_availability__gte=1,
    )

    lines = [
        "Owner direct-booking workflow:",
        (
            f"- Listings: Accepted {accepted_qs.count()}, Pending {pending_qs.count()}, "
            f"Declined {declined_qs.count()}."
        ),
        (
            f"- Rooms under accepted listings: {accepted_rooms.count()} "
            f"(currently available: {publicly_bookable_rooms.count()})."
        ),
    ]
    if accepted_qs.exists() and publicly_bookable_rooms.exists():
        lines.append(
            "- Yes. Guests can book from your listing via the guest accommodation page and chatbot flow "
            "when your listing is accepted and room availability is set."
        )
    elif accepted_qs.exists() and not accepted_rooms.exists():
        lines.append(
            "- Guests cannot book yet because no rooms are registered under your accepted listing."
        )
    elif accepted_qs.exists():
        lines.append(
            "- Guests cannot book those rooms yet because they are not currently marked available."
        )
    else:
        lines.append(
            "- Guests cannot book your listing yet because there is no accepted listing currently visible."
        )
    lines.append(
        "Use Owner Hub and Manage Rooms to update listing status, room availability, and booking visibility."
    )
    return "\n".join(lines)


def _is_booking_count_command(message):
    text = str(message or "").strip().lower()
    if "booking" not in text:
        return False
    if any(token in text for token in ("all time", "all-time", "overall", "lifetime")):
        return True
    if any(token in text for token in ("how many", "count", "total", "number of", "pila", "ilan")):
        return True
    return bool(re.search(r"\bbookings?\b.*\b(today|this month|monthly|daily|now|all time|all-time|overall|lifetime)\b", text))


def _is_admin_pending_accommodations_command(message):
    return _contains_any_phrase(
        message,
        (
            "pending accommodations",
            "pending accommodation",
            "show pending hotels",
            "show pending inns",
            "accommodation approvals",
            "pending hotel registrations",
            "pending inns",
            "pending accom",
            "pakita pending accommodations",
        ),
    )


def _is_admin_pending_owner_accounts_command(message):
    return _contains_any_phrase(
        message,
        (
            "pending owner accounts",
            "pending owners",
            "owner account approvals",
            "pending accommodation owners",
            "accommodation owner approvals",
            "owner approvals",
            "pending owner",
            "pakita pending owners",
        ),
    )


def _is_admin_accommodation_bookings_command(message):
    return _contains_any_phrase(
        message,
        (
            "open accommodation bookings",
            "show accommodation bookings",
            "hotel bookings",
            "inn bookings",
            "room bookings",
            "accommodation booking list",
            "accommodation reservations",
            "hotel reservations",
            "open reservations",
            "pakita accommodation bookings",
        ),
    )


def _is_admin_tourism_manage_command(message):
    return _contains_any_phrase(
        message,
        (
            "open tourism information",
            "tourism information manage",
            "manage tourism information",
            "tourism management",
            "manage tourism spots",
            "open tourism spots",
            "pakita tourism information",
        ),
    )


def _is_admin_survey_results_command(message):
    return _contains_any_phrase(
        message,
        (
            "survey results",
            "open survey results",
            "show survey results",
            "sus results",
            "tam results",
            "rq4 results",
            "usability results",
            "acceptance results",
            "pakita survey",
        ),
    )


def _is_admin_map_command(message):
    return _contains_any_phrase(
        message,
        (
            "open map",
            "show map",
            "city map",
            "tourist map",
            "open city map",
            "open tourist map",
            "pakita map",
        ),
    )


def _is_admin_discounts_command(message):
    return _contains_any_phrase(
        message,
        (
            "open discounts",
            "show discounts",
            "manage discounts",
            "discount page",
            "discount management",
            "pakita discounts",
        ),
    )


def _is_admin_tour_list_command(message):
    return _contains_any_phrase(
        message,
        (
            "open tour list",
            "show tour list",
            "tour list",
            "manage tours",
            "tours page",
            "pakita tours",
        ),
    )


def _is_admin_activity_logs_command(message):
    return _contains_any_phrase(
        message,
        (
            "open activity logs",
            "show activity logs",
            "activity tracker",
            "user activity tracker",
            "open activity tracker",
            "pakita activity logs",
        ),
    )


def _is_admin_traveler_surveys_command(message):
    return _contains_any_phrase(
        message,
        (
            "open traveler surveys",
            "show traveler surveys",
            "traveler surveys",
            "tourism reports dashboard",
            "survey dashboard",
            "pakita traveler surveys",
        ),
    )


def _is_admin_tour_calendar_command(message):
    return _contains_any_phrase(
        message,
        (
            "open tour calendar",
            "show tour calendar",
            "tour calendar",
            "calendar page",
            "pakita calendar",
        ),
    )


def _is_employee_assigned_tours_command(message):
    return _contains_any_phrase(
        message,
        (
            "open assigned tours",
            "my assigned tours",
            "assigned tours",
            "show my assigned tours",
            "what tour am i assigned",
            "which tour am i assigned",
            "what tour package am i assigned",
            "tour package am i assigned",
            "assigned tour package",
            "assigned in",
            "assigned to",
            "ano ang assigned tour ko",
            "ano ang tour na assigned sa akin",
            "ano ang tour package na assigned sa akin",
            "ano yung tour package na assign sakin",
            "ano po yung tour package na assign sakin",
            "tour package na assign sakin",
            "unsa akong assigned tour",
            "unsa nga tour ang assigned nako",
            "unsa akong assigned tour package",
            "tasks for tours",
            "pakita assigned tours",
        ),
    )


def _build_employee_assigned_tours_summary(request, actor, *, max_rows=5):
    employee = actor.get("employee")
    if employee is None:
        employee_id = request.session.get("employee_id") if hasattr(request, "session") else None
        if employee_id:
            employee = Employee.objects.filter(emp_id=employee_id).first()
    if employee is None:
        return (
            "I can open your assigned tours page, but I could not resolve your employee profile in this session.\n"
            "Please open Assigned Tours from the dashboard."
        )

    assignments = list(
        TourAssignment.objects.select_related("schedule", "schedule__tour")
        .filter(employee=employee)
        .order_by("-assigned_date", "-id")
    )
    if not assignments:
        return (
            "You currently have no assigned tours.\n"
            "Once an admin assigns a tour schedule to your account, it will appear here."
        )

    now = timezone.now()
    active_count = 0
    upcoming_count = 0
    lines = [f"Assigned tours for {getattr(employee, 'first_name', 'Employee')}:"]
    for assignment in assignments:
        schedule = getattr(assignment, "schedule", None)
        if schedule is None:
            continue
        start_time = getattr(schedule, "start_time", None)
        end_time = getattr(schedule, "end_time", None)
        if start_time and end_time and start_time <= now <= end_time:
            active_count += 1
        elif start_time and start_time > now:
            upcoming_count += 1

    lines.append(f"- Total assigned: {len(assignments)} | Active now: {active_count} | Upcoming: {upcoming_count}")
    lines.append("Latest assignments:")

    shown = 0
    for assignment in assignments:
        if shown >= max_rows:
            break
        schedule = getattr(assignment, "schedule", None)
        if schedule is None:
            continue
        tour = getattr(schedule, "tour", None)
        tour_name = str(getattr(tour, "tour_name", "") or "Tour").strip()
        sched_id = str(getattr(schedule, "sched_id", "") or "").strip()
        start_time = getattr(schedule, "start_time", None)
        start_text = timezone.localtime(start_time).strftime("%b %d, %Y %I:%M %p") if start_time else "No start time"
        status = str(getattr(schedule, "status", "") or "").strip().lower() or "unknown"
        lines.append(f"- {tour_name} ({sched_id}) | {start_text} | Status: {status.title()}")
        shown += 1

    return "\n".join(lines)


def _is_employee_tour_calendar_command(message):
    return _contains_any_phrase(
        message,
        (
            "open tour calendar",
            "my tour calendar",
            "tour calendar",
            "show schedule",
            "show my schedule",
            "pakita calendar",
        ),
    )


def _is_employee_accommodations_command(message):
    return _contains_any_phrase(
        message,
        (
            "open accommodations",
            "employee accommodations",
            "accommodation list",
            "show accommodations",
            "list accommodations",
            "pakita accommodations",
        ),
    )


def _is_employee_profile_command(message):
    return _contains_any_phrase(
        message,
        (
            "open profile",
            "my profile",
            "employee profile",
            "show profile",
            "account profile",
            "pakita profile",
        ),
    )


def _is_employee_tour_list_command(message):
    return _contains_any_phrase(
        message,
        (
            "open tour list",
            "show tour list",
            "tour list",
            "list tours",
            "pakita tours",
        ),
    )


def _is_employee_create_tour_command(message):
    return _contains_any_phrase(
        message,
        (
            "open create tour",
            "create tour",
            "add new tour",
            "new tour",
            "open add tour",
            "pakita create tour",
        ),
    )


def _is_employee_map_command(message):
    return _contains_any_phrase(
        message,
        (
            "open map",
            "show map",
            "city map",
            "tourist map",
            "pakita map",
        ),
    )


def _detect_employee_support_topic(message):
    text = str(message or "").strip().lower()
    if not text:
        return ""
    compact_text = re.sub(r"[^a-z0-9\s]", " ", text)
    compact_text = re.sub(r"\s+", " ", compact_text).strip()

    if _contains_any_phrase(text, ("forgot my password", "forgot password", "reset password", "cannot access the employee dashboard", "dashboard not opening", "dashboard not opening", "nakalimutan ko password", "dili ma open dashboard")):
        return "employee_account_access_help"
    if _contains_any_phrase(text, ("record not showing", "not showing in the system", "hindi lumalabas ang tourist record", "wala nagpakita ang tourist record")):
        return "employee_record_visibility_issue"
    if _contains_any_phrase(text, ("monitoring dashboard", "go to monitoring dashboard", "monitoring panel", "monitoring dashboard sa employee")):
        return "employee_monitoring_dashboard_help"
    if _contains_any_phrase(text, ("can i update records from my account", "update records from my account", "pwede ba ako mag update ng records", "pwede ba ko mag update og records")):
        return "employee_record_update_help"

    if _contains_any_phrase(text, ("tourist records", "current tourists", "tourist arrivals", "tourist information records", "tourist by name", "tourist record", "records ng tourist", "listahan sa turista")):
        return "employee_tourist_monitoring"
    if _contains_any_phrase(text, ("all bookings in the system", "pending reservations", "bookings are confirmed", "cancelled reservations", "monitor accommodation bookings", "pending bookings", "monitor bookings", "bantayan ang bookings")):
        return "employee_booking_monitoring"
    if _contains_any_phrase(text, ("registered accommodations", "hotels or inns are active", "accommodation details", "accommodation records", "active hotels", "active inns", "mga active na hotel", "aktibong inns")):
        return "employee_accommodation_records"
    if _contains_any_phrase(text, ("update tourism destination details", "manage published attractions", "feedback or tourism concerns", "tourism concerns", "manage tourism info", "feedback sa turismo")):
        return "employee_workflow_destination_feedback"
    if _contains_any_phrase(text, ("tourism destination records", "destination is published", "published attractions", "tourism destination", "mga destination records", "published destinations")):
        return "employee_destination_records"
    if _contains_any_phrase(text, ("generate tourism reports", "booking summaries", "tourist statistics", "monitoring reports", "accommodation-related reports", "generate report", "accommodation reports", "gumawa ng report", "himo ug report")):
        return "employee_reports_support"
    if _contains_any_phrase(text, ("approve accommodation registrations", "review submitted accommodation listings", "review ang accommodation listing", "listing approval", "suriin ang listing approval", "review sa listing")):
        return "employee_workflow_listing_review"

    short_map = {
        "tourist records": "employee_tourist_monitoring",
        "pending bookings": "employee_booking_monitoring",
        "generate report": "employee_reports_support",
        "listing approval": "employee_workflow_listing_review",
        "dashboard not opening": "employee_account_access_help",
    }
    return short_map.get(compact_text, "")


def _build_employee_tourist_monitoring_summary(message):
    text = str(message or "").strip().lower()
    tourist_qs = Guest.objects.filter(is_active=True)
    total_tourists = tourist_qs.count()
    today = timezone.localdate()
    arrival_today = Pending.objects.count()
    active_bookings_today = TourBooking.objects.filter(booking_date__date=today).count()

    name_match = re.search(r"(?:name|named)\s+([a-zA-Z][a-zA-Z\s\-]{1,60})", text)
    if name_match:
        name_query = str(name_match.group(1) or "").strip()
        matches = tourist_qs.filter(
            Q(first_name__icontains=name_query)
            | Q(last_name__icontains=name_query)
            | Q(username__icontains=name_query)
        )[:5]
        if not matches:
            return (
                f"No tourist record matched '{name_query}'.\n"
                "Try searching by first name, last name, or username."
            )
        lines = [f"I found {len(matches)} tourist record(s) matching '{name_query}':"]
        for guest in matches:
            lines.append(f"- {guest.first_name} {guest.last_name} ({guest.username})")
        return "\n".join(lines)

    return (
        "Tourist monitoring snapshot:\n"
        f"- Active tourist records: {total_tourists}\n"
        f"- Tourist arrival entries today: {arrival_today}\n"
        f"- Tour booking records today: {active_bookings_today}\n"
        "For name search, ask: search tourist by name <name>."
    )


def _build_employee_booking_monitoring_summary():
    today = timezone.localdate()
    accom_qs = AccommodationBooking.objects.all()
    tour_rollup = _build_tour_booking_rollup(period="all_time", today=today)
    accom_counts = Counter(str(v or "").strip().lower() for v in accom_qs.values_list("status", flat=True))
    lines = [
        "Booking monitoring snapshot:",
        (
            f"- Accommodation bookings: total {accom_qs.count()}, "
            f"pending {accom_counts.get('pending', 0)}, confirmed {accom_counts.get('confirmed', 0)}, "
            f"declined {accom_counts.get('declined', 0)}, cancelled {accom_counts.get('cancelled', 0)}."
        ),
        (
            f"- Tour bookings: total {tour_rollup.get('total', 0)}, "
            f"pending {tour_rollup.get('pending', 0)}, active {tour_rollup.get('active', 0)}, "
            f"completed {tour_rollup.get('completed', 0)}, cancelled {tour_rollup.get('cancelled', 0)}."
        ),
        (
            f"- New accommodation bookings today ({today.isoformat()}): "
            f"{accom_qs.filter(booking_date__date=today).count()}"
        ),
    ]
    if tour_rollup.get("includes_pending_legacy") and int(tour_rollup.get("pending_legacy_count") or 0) > 0:
        lines.append(
            "Note: Tour totals include legacy pending-booking records to align with the booking report page."
        )
    return "\n".join(lines)


def _build_employee_accommodation_records_summary():
    accom_qs = Accomodation.objects.filter(is_active=True)
    total = accom_qs.count()
    accepted = accom_qs.filter(approval_status="accepted")
    pending = accom_qs.filter(approval_status="pending")
    declined = accom_qs.filter(approval_status="declined")
    rooms_total = Room.objects.filter(accommodation__in=accepted).count()
    return (
        "Accommodation records snapshot:\n"
        f"- Total active listings: {total}\n"
        f"- Accepted: {accepted.count()} | Pending: {pending.count()} | Declined: {declined.count()}\n"
        f"- Rooms under accepted listings: {rooms_total}"
    )


def _build_employee_destination_records_summary():
    all_dest = TourismInformation.objects.filter(is_active=True)
    published = all_dest.filter(publication_status="published")
    draft = all_dest.filter(publication_status="draft")
    archived = all_dest.filter(publication_status="archived")
    return (
        "Tourism destination records snapshot:\n"
        f"- Total active destination records: {all_dest.count()}\n"
        f"- Published: {published.count()} | Draft: {draft.count()} | Archived: {archived.count()}"
    )


def _build_employee_reports_support_summary():
    today = timezone.localdate()
    accom_bookings = AccommodationBooking.objects.all()
    tourism_records = TourismInformation.objects.published().count()
    tour_rollup = _build_tour_booking_rollup(period="all_time", today=today)
    lines = [
        "Monitoring and reports snapshot:",
        f"- Tour bookings total: {tour_rollup.get('total', 0)}",
        f"- Accommodation bookings total: {accom_bookings.count()}",
        f"- Published tourism destinations: {tourism_records}",
        (
            f"- Records updated today ({today.isoformat()}): "
            f"{TourismInformation.objects.filter(updated_at__date=today).count()}"
        ),
    ]
    if tour_rollup.get("includes_pending_legacy") and int(tour_rollup.get("pending_legacy_count") or 0) > 0:
        lines.append(
            "Note: Tour totals include legacy pending-booking records to align with the booking report page."
        )
    return "\n".join(lines)


def _is_guest_map_command(message):
    return _contains_any_phrase(
        message,
        (
            "open map",
            "show map",
            "city map",
            "tourist map",
            "bayawan map",
            "open city map",
            "open tourist map",
            "show bayawan map",
            "pakita map",
        ),
    )


def _extract_sched_id_from_message(message):
    text = str(message or "").strip()
    if not text:
        return ""
    match = re.search(r"(sched\d+)", text, flags=re.IGNORECASE)
    if not match:
        return ""
    return str(match.group(1) or "").strip()


def _extract_tour_selection_index(message):
    text = str(message or "").strip().lower()
    if not text:
        return 0

    ordinal_map = {
        "first": 1,
        "1st": 1,
        "second": 2,
        "2nd": 2,
        "third": 3,
        "3rd": 3,
        "fourth": 4,
        "4th": 4,
        "fifth": 5,
        "5th": 5,
    }
    for token, idx in ordinal_map.items():
        if re.search(rf"\b{re.escape(token)}\b", text):
            return idx

    # Accept "book #1", "book number 2", "book option 3", or plain "book 1".
    idx_match = re.search(
        r"\b(?:book|reserve|reservation)\b(?:\s+(?:#|number|no\.?|option|tour))?\s*(\d{1,2})\b",
        text,
        flags=re.IGNORECASE,
    )
    if idx_match:
        return _to_int(idx_match.group(1), default=0)

    return 0


def _extract_numeric_option_index(message):
    text = str(message or "").strip()
    if not re.fullmatch(r"\d{1,2}", text):
        return 0
    value = _to_int(text, default=0)
    return value if value > 0 else 0


def _extract_why_option_index(message):
    text = str(message or "").strip().lower()
    if not text:
        return 0
    if not ("why" in text or "explain" in text or "reason" in text):
        return 0
    match = re.search(r"\b(?:option|room|hotel|inn|#)\s*(\d{1,2})\b", text)
    if match:
        return _to_int(match.group(1), default=0)
    plain = re.search(r"\bwhy\s+(\d{1,2})\b", text)
    if plain:
        return _to_int(plain.group(1), default=0)
    return 0


def _extract_compare_top_n(message, default=3):
    text = str(message or "").strip().lower()
    if not text:
        return 0
    if not any(token in text for token in ("compare", "comparison")):
        return 0
    match = re.search(r"\btop\s*(\d{1,2})\b", text)
    if match:
        return max(2, min(_to_int(match.group(1), default=default), 5))
    num_match = re.search(r"\b(\d{1,2})\b", text)
    if num_match:
        return max(2, min(_to_int(num_match.group(1), default=default), 5))
    return default


def _merge_quick_replies(*reply_lists, limit=4):
    combined = []
    seen_values = set()
    for reply_list in reply_lists:
        if not isinstance(reply_list, list):
            continue
        for item in reply_list:
            if isinstance(item, dict):
                value = str(item.get("value") or "").strip()
                label = str(item.get("label") or value).strip()
                if not value:
                    continue
                dedupe_key = value.lower()
                if dedupe_key in seen_values:
                    continue
                seen_values.add(dedupe_key)
                combined.append({"label": label, "value": value})
            else:
                value = str(item or "").strip()
                if not value:
                    continue
                dedupe_key = value.lower()
                if dedupe_key in seen_values:
                    continue
                seen_values.add(dedupe_key)
                combined.append(value)
    return _sanitize_quick_replies(combined, limit=limit)


def _slot_quick_replies(slot_name):
    slot = str(slot_name or "").strip().lower()
    if slot == "company_type":
        return [
            {"label": "Hotel", "value": "hotel"},
            {"label": "Inn", "value": "inn"},
            {"label": "Either", "value": "either"},
        ]
    if slot == "location":
        return [
            {"label": "Bayawan", "value": "bayawan"},
            {"label": "Poblacion", "value": "poblacion"},
            {"label": "Villareal", "value": "villareal"},
            {"label": "Suba", "value": "suba"},
        ]
    if slot == "guests":
        return [
            {"label": "1 Guest", "value": "1 guest"},
            {"label": "2 Guests", "value": "2 guests"},
            {"label": "4 Guests", "value": "4 guests"},
        ]
    if slot == "budget":
        return [
            {"label": "PHP 1000", "value": "budget 1000"},
            {"label": "PHP 1500", "value": "budget 1500"},
            {"label": "PHP 2000", "value": "budget 2000"},
        ]
    if slot == "stay_details":
        return [
            {"label": "2 Nights", "value": "2 nights"},
            {"label": "3 Nights", "value": "3 nights"},
        ]
    return []


def _build_recommendation_assist_quick_replies(cached_rows):
    if not isinstance(cached_rows, list) or not cached_rows:
        return []
    rows = [row for row in cached_rows if isinstance(row, dict)]
    if not rows:
        return []
    replies = []
    first_rank = _to_int(rows[0].get("rank"), default=1)
    replies.append({"label": f"Why Option {first_rank}", "value": f"why option {first_rank}"})
    if len(rows) >= 2:
        top_n = min(3, len(rows))
        replies.append({"label": f"Compare Top {top_n}", "value": f"compare top {top_n}"})
    return replies


def _build_post_compare_quick_replies(cached_rows, top_n=3):
    rows = [row for row in (cached_rows or []) if isinstance(row, dict)]
    if not rows:
        return []
    n = max(2, min(_to_int(top_n, default=3), len(rows), 5))
    replies = []
    for row in rows[:n]:
        rank = _to_int(row.get("rank"), default=0)
        if rank > 0:
            replies.append({"label": f"Why Option {rank}", "value": f"why option {rank}"})
    first_rank = _to_int(rows[0].get("rank"), default=1)
    replies.append({"label": f"Book Option {first_rank}", "value": str(first_rank)})
    return replies


def _build_why_option_text(cached_rows, option_index):
    if not isinstance(cached_rows, list) or option_index <= 0:
        return ""
    selected = None
    for row in cached_rows:
        if not isinstance(row, dict):
            continue
        if _to_int(row.get("rank"), default=0) == option_index:
            selected = row
            break
    if not selected:
        return ""
    title = str(selected.get("title") or f"Option {option_index}").strip()
    subtitle = str(selected.get("subtitle") or "").strip()
    match_strength = str(selected.get("match_strength") or "").strip()
    reasons = selected.get("reasons") if isinstance(selected.get("reasons"), list) else []
    lines = [f"Why Option {option_index}: {title}"]
    if subtitle:
        lines.append(subtitle)
    if match_strength:
        lines.append(f"Match strength: {match_strength}")
    if reasons:
        lines.append("Primary matching reasons:")
        for reason in reasons[:4]:
            lines.append(f"- {str(reason)}")
    else:
        lines.append("Share your priority (budget, location, amenities, or guest count) for a more specific explanation.")
    return "\n".join(lines)


def _build_compare_options_text(cached_rows, top_n):
    if not isinstance(cached_rows, list) or len(cached_rows) < 2:
        return ""
    n = max(2, min(_to_int(top_n, default=3), 5))
    rows = [row for row in cached_rows if isinstance(row, dict)]
    if len(rows) < 2:
        return ""
    rows = rows[:n]
    lines = [f"Comparison of Top {len(rows)} options:"]
    for row in rows:
        rank = _to_int(row.get("rank"), default=0)
        title = str(row.get("title") or "").strip()
        subtitle = str(row.get("subtitle") or "").strip()
        match_strength = str(row.get("match_strength") or "").strip()
        reasons = row.get("reasons") if isinstance(row.get("reasons"), list) else []
        lines.append(f"{rank}. {title}")
        if subtitle:
            lines.append(f"   Details: {subtitle}")
        if match_strength:
            lines.append(f"   Match strength: {match_strength}")
        if reasons:
            lines.append(f"   Key reason: {str(reasons[0])}")
    lines.append("Next step: reply with 'why option <number>' for a detailed explanation.")
    return "\n".join(lines)


def _build_accommodation_selection_cache(items):
    if not isinstance(items, list):
        return []
    rows = []
    for idx, item in enumerate(items, 1):
        if not isinstance(item, dict):
            continue
        room_id = _to_int(item.get("room_id"), default=0)
        if room_id <= 0:
            continue
        rank = _to_int(item.get("rank"), default=idx)
        rows.append(
            {
                "rank": rank if rank > 0 else idx,
                "room_id": room_id,
                "title": str(item.get("title") or "").strip()[:120],
                "subtitle": str(item.get("subtitle") or "").strip()[:240],
                "match_strength": str(item.get("match_strength") or "").strip()[:24],
                "reasons": (
                    [str(reason).strip()[:160] for reason in item.get("reasons", []) if str(reason).strip()][:5]
                    if isinstance(item.get("reasons"), list)
                    else []
                ),
            }
        )
    return rows[:10]


def _resolve_accommodation_room_from_selection(selection_rows, selection_index):
    if not isinstance(selection_rows, list) or selection_index <= 0:
        return 0
    for row in selection_rows:
        if not isinstance(row, dict):
            continue
        if _to_int(row.get("rank"), default=0) == selection_index:
            return _to_int(row.get("room_id"), default=0)
    return 0


def _is_accommodation_detail_query(message):
    text = str(message or "").strip().lower()
    if not text:
        return False
    if any(token in text for token in ("why option", "compare top", "book option")):
        return False
    if any(token in text for token in ("why", "explain", "reason")):
        return False
    detail_markers = (
        "amenity",
        "amenities",
        "facility",
        "facilities",
        "details",
        "tell me more",
        "more about",
        "what does",
        "show details",
    )
    subject_markers = (
        "option",
        "room",
        "hotel",
        "inn",
        "this hotel",
        "this room",
    )
    return any(marker in text for marker in detail_markers) and any(
        marker in text for marker in subject_markers
    )


def _extract_detail_option_index(message):
    text = str(message or "").strip().lower()
    if not text:
        return 0
    option_match = re.search(r"\boption\s*(\d{1,2})\b", text)
    if option_match:
        return _to_int(option_match.group(1), default=0)
    ordinal_match = re.search(r"\b(first|second|third|fourth|fifth)\s+option\b", text)
    if ordinal_match:
        ordinal_map = {
            "first": 1,
            "second": 2,
            "third": 3,
            "fourth": 4,
            "fifth": 5,
        }
        return _to_int(ordinal_map.get(str(ordinal_match.group(1) or "").lower()), default=0)
    return 0


def _extract_detail_room_id(message):
    text = str(message or "").strip().lower()
    if not text:
        return 0
    room_match = re.search(r"\broom(?:\s*id)?\s*[:#-]?\s*(\d{1,6})\b", text)
    if room_match:
        return _to_int(room_match.group(1), default=0)
    return 0


def _normalize_amenities_for_display(raw_value):
    text = str(raw_value or "").strip()
    if not text:
        return []
    tokens = [
        str(item).strip()
        for item in re.split(r"[,\n;/|]+", text)
        if str(item).strip()
    ]
    deduped = []
    seen = set()
    for token in tokens:
        lowered = token.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        deduped.append(token)
    return deduped[:12]


def _extract_room_level_amenities_for_display(room):
    if room is None:
        return []
    details = getattr(room, "owner_details", None)
    if details is None:
        try:
            details = AuthoritativeRoomDetails.objects.filter(room=room).first()
        except Exception:
            details = None
    raw_value = getattr(details, "amenities", "") if details is not None else ""
    raw_text = str(raw_value or "").strip()
    if not raw_text:
        return []
    try:
        parsed = json.loads(raw_text)
    except Exception:
        parsed = None
    if isinstance(parsed, list):
        return _normalize_amenities_for_display(", ".join(str(item) for item in parsed))
    if isinstance(parsed, str):
        return _normalize_amenities_for_display(parsed)
    return _normalize_amenities_for_display(raw_text)


def _build_guest_room_detail_payload(message, cached_rows):
    text = str(message or "").strip().lower()
    option_index = _extract_detail_option_index(text)
    requested_room_id = _extract_detail_room_id(text)
    resolved_room_id = requested_room_id
    if resolved_room_id <= 0 and option_index > 0:
        resolved_room_id = _resolve_accommodation_room_from_selection(cached_rows, option_index)
    if resolved_room_id <= 0 and ("this hotel" in text or "this room" in text):
        first_row = cached_rows[0] if isinstance(cached_rows, list) and cached_rows else {}
        if isinstance(first_row, dict):
            resolved_room_id = _to_int(first_row.get("room_id"), default=0)

    if resolved_room_id <= 0:
        return {
            "fulfillmentText": (
                "I can show room details once you provide a room reference.\n"
                "Please send a Room ID (example: Room 112) or an option number from your latest recommendation list."
            ),
            "quick_replies": [
                "Show default hotel suggestions",
                "Recommend a hotel in Bayawan for 2 guests",
            ],
            "needs_clarification": True,
            "missing_slot": "room_reference",
        }

    room = (
        Room.objects.select_related("accommodation", "owner_details")
        .filter(
            room_id=resolved_room_id,
            accommodation__approval_status="accepted",
            accommodation__is_active=True,
        )
        .first()
    )
    if room is None:
        return {
            "fulfillmentText": (
                f"I couldn't find room {resolved_room_id} in the currently accepted hotel/inn listings.\n"
                "Please send another room reference or refresh recommendations."
            ),
            "quick_replies": [
                "Show default hotel suggestions",
                "Recommend a hotel in Bayawan for 2 guests",
            ],
        }

    accom = getattr(room, "accommodation", None)
    room_level_amenities = _extract_room_level_amenities_for_display(room)
    accommodation_level_amenities = _normalize_amenities_for_display(
        getattr(accom, "accommodation_amenities", "") if accom is not None else ""
    )
    lines = [
        (
            f"Room details: {getattr(accom, 'company_name', 'Accommodation')} - "
            f"{room.room_name} (Room {room.room_id})"
        ),
        f"Location: {getattr(accom, 'location', '') or 'Not specified'}",
        f"Type: {str(getattr(accom, 'company_type', '') or 'Accommodation').title()}",
        f"Rate: PHP {Decimal(room.price_per_night):.2f} per night",
        f"Capacity: up to {_to_int(room.person_limit, default=0)} guest(s)",
        f"Current availability slots: {_to_int(room.current_availability, default=0)}",
        f"Room status: {str(room.status or 'unknown').title()}",
    ]
    if room_level_amenities:
        lines.append(f"Amenities listed: {', '.join(room_level_amenities)}")
        if accommodation_level_amenities:
            lines.append("Additional property amenities may also be available at the accommodation level.")
    elif accommodation_level_amenities:
        lines.append(
            "Amenities listed (accommodation-level): "
            + ", ".join(accommodation_level_amenities)
        )
    else:
        lines.append(
            "Amenities information is currently limited for this room/property in the current records."
        )
    description = str(getattr(accom, "description", "") or "").strip()
    if description:
        lines.append(f"Description: {description}")

    quick_replies = [
        {"label": "Book This Room", "value": f"book room {room.room_id}"},
        {"label": "Why Option 1", "value": "why option 1"},
        {"label": "Show More Hotels/Inns", "value": "show default hotel suggestions"},
    ]
    return {
        "fulfillmentText": "\n".join(lines),
        "room_id": room.room_id,
        "quick_replies": quick_replies,
    }


def _is_guest_tour_booking_command(message):
    text = str(message or "").strip().lower()
    if not text:
        return False
    booking_terms = ("book", "booking", "reserve", "reservation")
    tour_terms = ("tour", "package", "schedule", "sched")
    return any(term in text for term in booking_terms) and any(term in text for term in tour_terms)


def _build_guest_tour_booking_link_payload(request, *, sched_id="", fallback_sched_ids=None, selection_index=0):
    resolved_sched_id = str(sched_id or "").strip()
    if not resolved_sched_id and isinstance(fallback_sched_ids, list):
        cleaned = [str(item).strip() for item in fallback_sched_ids if str(item).strip()]
        if selection_index > 0:
            pick = selection_index - 1
            if 0 <= pick < len(cleaned):
                resolved_sched_id = cleaned[pick]
            else:
                return {
                    "fulfillmentText": (
                        f"I couldn't find option #{selection_index} in your recent recommendations.\n"
                        f"Available options: {', '.join(cleaned[:5])}"
                    )
                }
        elif len(cleaned) == 1:
            resolved_sched_id = cleaned[0]
        elif len(cleaned) > 1:
            return {
                "fulfillmentText": (
                    "Please specify which schedule to book.\n"
                    f"Available recent options: {', '.join(cleaned[:5])}\n"
                    "Example: book tour sched00001"
                )
            }

    if not resolved_sched_id:
        return {
            "fulfillmentText": (
                "I can open tour booking for you. Please include a schedule ID.\n"
                "Example: book tour sched00001"
            )
        }

    schedule = (
        Tour_Schedule.objects.select_related("tour")
        .filter(
            sched_id__iexact=resolved_sched_id,
            tour__publication_status="published",
        )
        .first()
    )
    if schedule is None:
        return {
            "fulfillmentText": (
                f"I couldn't find published schedule {resolved_sched_id}. "
                "Please send a valid sched_id."
            )
        }

    booking_url = reverse("guest_book", kwargs={"tour_id": schedule.tour.tour_id})
    booking_url = f"{booking_url}?sched_id={schedule.sched_id}"
    if hasattr(request, "build_absolute_uri"):
        booking_url = request.build_absolute_uri(booking_url)

    return {
        "fulfillmentText": (
            f"I found {schedule.tour.tour_name} ({schedule.sched_id}). "
            "Click the button below to open booking in a new tab."
        ),
        "billing_link": booking_url,
        "billing_link_label": f"Open Tour Booking ({schedule.sched_id})",
        "open_in_new_tab": True,
    }


def _build_link_payload(request, *, text, route_name, label):
    link = reverse(route_name)
    if hasattr(request, "build_absolute_uri"):
        link = request.build_absolute_uri(link)
    return {
        "fulfillmentText": text,
        "billing_link": link,
        "billing_link_label": label,
        "open_in_new_tab": True,
    }


def _build_admin_pending_accommodations_summary():
    qs = Accomodation.objects.filter(is_active=True)
    pending = qs.filter(approval_status="pending")
    accepted = qs.filter(approval_status="accepted").count()
    declined = qs.filter(approval_status="declined").count()
    lines = [
        (
            f"Accommodation moderation summary: Pending {pending.count()}, "
            f"Accepted {accepted}, Declined {declined}."
        )
    ]
    sample = list(pending.order_by("-submitted_at", "company_name")[:6])
    if sample:
        lines.append("Latest pending accommodations:")
        for accom in sample:
            lines.append(f"- {accom.company_name} | {accom.company_type} | {accom.location}")
    return "\n".join(lines)


def _build_admin_pending_owner_accounts_summary():
    pending_group, _ = Group.objects.get_or_create(name="accommodation_owner_pending")
    approved_group, _ = Group.objects.get_or_create(name="accommodation_owner")
    declined_group, _ = Group.objects.get_or_create(name="accommodation_owner_declined")
    pending_count = pending_group.user_set.count()
    approved_count = approved_group.user_set.count()
    declined_count = declined_group.user_set.count()
    lines = [
        (
            "Accommodation owner account review summary: "
            f"Pending {pending_count}, Approved {approved_count}, Declined {declined_count}."
        )
    ]
    sample_pending = list(
        pending_group.user_set.order_by("date_joined", "username").values_list("username", flat=True)[:6]
    )
    if sample_pending:
        lines.append("Pending owner accounts:")
        for username in sample_pending:
            lines.append(f"- {username}")
    return "\n".join(lines)


def _detect_admin_support_topic(message):
    text = str(message or "").strip().lower()
    if not text:
        return ""
    compact_text = re.sub(r"[^a-z0-9\s]", " ", text)
    compact_text = re.sub(r"\s+", " ", compact_text).strip()

    if _contains_any_phrase(text, ("record not showing in the admin panel", "record not showing", "hindi lumalabas ang record sa admin panel", "dili makita sa admin panel ang record")):
        return "admin_record_visibility_issue"
    if _contains_any_phrase(text, ("reset a user account", "reset password", "reset a user password", "manage user accounts", "ireset ang user account", "manage users")):
        return "admin_user_account_management"
    if _contains_any_phrase(text, ("admin dashboard", "access the admin dashboard", "overall system summaries", "dashboard ng admin")):
        return "admin_reports_dashboard_support"

    if _contains_any_phrase(text, ("approve accommodation registrations", "review pending accommodation listings", "reject a submitted accommodation listing", "listings waiting for approval", "pending review", "pending listings", "i-approve ang accommodation listing", "approve listing", "aprubahan ang accommodation listing")):
        return "admin_approval_workflow"
    if _contains_any_phrase(text, ("publish a tourism destination", "unpublish a destination", "update destination details", "manage tourism attractions", "edit destination information after publishing", "manage tourism information")):
        return "admin_destination_management"
    if _contains_any_phrase(text, ("view all registered accommodations", "accommodation details in the system", "active or inactive", "manage accommodation records", "specific hotel or inn", "manage all accommodations")):
        return "admin_accommodation_records_management"
    if _contains_any_phrase(
        text,
        (
            "view all bookings in the system",
            "monitor confirmed and pending reservations",
            "check cancelled bookings",
            "booking summaries",
            "overall reservation activity",
            "check all bookings",
            "tour bookings",
            "tour booking",
            "in tour bookings",
            "tour package bookings",
            "tour reservations",
            "i mean tour packages",
        ),
    ):
        return "admin_booking_system_monitoring"
    if _contains_any_phrase(text, ("check employee and owner accounts", "manage system users", "update account roles or access", "i-manage ang users", "manage users", "user account management")):
        return "admin_user_account_management"
    if _contains_any_phrase(text, ("generate system reports", "tourism statistics and booking reports", "system summaries", "booking summary", "analytics and reports", "mga ulat ng system")):
        return "admin_reports_dashboard_support"
    if _contains_any_phrase(
        text,
        (
            "monitor chatbot activity",
            "chatbot activity",
            "show chatbot logs",
            "conversation logs",
            "chat metrics",
            "monitor chatbot usage",
            "chatbot usage",
            "chat logs",
            "pwede ba i monitor ang chatbot activity",
            "ma monitor ba ang chatbot activity",
            "ipakita ang chatbot logs",
        ),
    ):
        return "admin_chatbot_activity_monitoring"

    short = {
        "approve listing": "admin_approval_workflow",
        "pending accommodations": "admin_approval_workflow",
        "manage users": "admin_user_account_management",
        "admin dashboard": "admin_reports_dashboard_support",
        "booking summary": "admin_booking_system_monitoring",
        "chatbot activity": "admin_chatbot_activity_monitoring",
        "chat logs": "admin_chatbot_activity_monitoring",
    }
    return short.get(compact_text, "")


def _build_admin_approval_workflow_summary():
    accom_qs = Accomodation.objects.filter(is_active=True)
    pending_accom = accom_qs.filter(approval_status="pending")
    accepted_count = accom_qs.filter(approval_status="accepted").count()
    declined_count = accom_qs.filter(approval_status="declined").count()
    pending_owner_group, _ = Group.objects.get_or_create(name="accommodation_owner_pending")
    owner_pending_count = pending_owner_group.user_set.count()
    return (
        "Approval workflow snapshot:\n"
        f"- Accommodation listings: Pending {pending_accom.count()}, Accepted {accepted_count}, Declined {declined_count}\n"
        f"- Pending owner accounts: {owner_pending_count}\n"
        "Use Pending Accommodations and Pending Owner Accounts pages for review actions."
    )


def _build_admin_destination_management_summary():
    records = TourismInformation.objects.filter(is_active=True)
    return (
        "Destination/content management snapshot:\n"
        f"- Total active destination records: {records.count()}\n"
        f"- Published: {records.filter(publication_status='published').count()} | "
        f"Draft: {records.filter(publication_status='draft').count()} | "
        f"Archived: {records.filter(publication_status='archived').count()}\n"
        "Use Tourism Information management to publish/archive/update records."
    )


def _build_admin_accommodation_records_summary(message):
    text = str(message or "").strip().lower()
    qs = Accomodation.objects.filter(is_active=True)
    lines = [
        "Accommodation records snapshot:",
        f"- Total active listings: {qs.count()}",
        (
            f"- Accepted: {qs.filter(approval_status='accepted').count()} | "
            f"Pending: {qs.filter(approval_status='pending').count()} | "
            f"Declined: {qs.filter(approval_status='declined').count()}"
        ),
    ]
    search_match = re.search(r"(?:search|specific)\s+(?:for\s+)?([a-zA-Z][a-zA-Z\s\-]{2,60})", text)
    if search_match:
        needle = str(search_match.group(1) or "").strip()
        filtered = qs.filter(company_name__icontains=needle)[:5]
        if filtered:
            lines.append(f"Search results for '{needle}':")
            for accom in filtered:
                lines.append(
                    f"- {accom.company_name} | {accom.company_type} | {accom.location} | {str(accom.approval_status).title()}"
                )
        else:
            lines.append(f"No accommodation found for '{needle}'.")
    return "\n".join(lines)


def _build_admin_booking_system_summary():
    today = timezone.localdate()
    accom_qs = AccommodationBooking.objects.all()
    tour_qs = TourBooking.objects.all()
    accom_counts = Counter(str(v or "").strip().lower() for v in accom_qs.values_list("status", flat=True))
    tour_counts = Counter(str(v or "").strip().lower() for v in tour_qs.values_list("status", flat=True))
    return (
        "System-wide booking monitoring snapshot:\n"
        f"- Accommodation bookings: Total {accom_qs.count()}, Pending {accom_counts.get('pending', 0)}, "
        f"Confirmed {accom_counts.get('confirmed', 0)}, Declined {accom_counts.get('declined', 0)}, "
        f"Cancelled {accom_counts.get('cancelled', 0)}\n"
        f"- Tour bookings: Total {tour_qs.count()}, Pending {tour_counts.get('pending', 0)}, "
        f"Active {tour_counts.get('active', 0)}, Completed {tour_counts.get('completed', 0)}, "
        f"Cancelled {tour_counts.get('cancelled', 0)}\n"
        f"- Accommodation bookings today ({today.isoformat()}): {accom_qs.filter(booking_date__date=today).count()}"
    )


def _build_tour_booking_rollup(*, period="all_time", today=None):
    today = today or timezone.localdate()
    period = str(period or "all_time").strip().lower()
    include_pending_legacy = period == "all_time"

    tour_qs = TourBooking.objects.all()
    if period == "today":
        tour_qs = tour_qs.filter(booking_date__date=today)
    elif period == "month":
        tour_qs = tour_qs.filter(booking_date__year=today.year, booking_date__month=today.month)

    pending_qs = Pending.objects.all() if include_pending_legacy else Pending.objects.none()
    tour_counts = Counter(str(v or "").strip().lower() for v in tour_qs.values_list("status", flat=True))
    pending_counts = Counter(str(v or "").strip().lower() for v in pending_qs.values_list("status", flat=True))

    pending_count = tour_counts.get("pending", 0) + pending_counts.get("pending", 0)
    active_count = tour_counts.get("active", 0) + pending_counts.get("accepted", 0)
    completed_count = tour_counts.get("completed", 0) + pending_counts.get("completed", 0)
    cancelled_count = tour_counts.get("cancelled", 0) + pending_counts.get("cancelled", 0)
    declined_count = pending_counts.get("declined", 0)
    total_count = tour_qs.count() + pending_qs.count()

    tour_revenue_gross = (
        tour_qs.exclude(status="cancelled").aggregate(total=Sum("total_amount")).get("total") or Decimal("0")
    )
    tour_revenue_paid = (
        tour_qs.filter(payment_status="paid").aggregate(total=Sum("amount_paid")).get("total") or Decimal("0")
    )
    pending_accepted_gross = (
        pending_qs.filter(status__iexact="accepted")
        .aggregate(
            total=Sum(
                ExpressionWrapper(
                    F("total_guests") * F("sched_id__price"),
                    output_field=DecimalField(max_digits=12, decimal_places=2),
                )
            )
        )
        .get("total")
        or Decimal("0")
    )

    return {
        "total": total_count,
        "pending": pending_count,
        "active": active_count,
        "completed": completed_count,
        "cancelled": cancelled_count,
        "declined": declined_count,
        "revenue_gross": Decimal(tour_revenue_gross) + Decimal(pending_accepted_gross),
        "revenue_paid": Decimal(tour_revenue_paid),
        "includes_pending_legacy": bool(include_pending_legacy),
        "pending_legacy_count": pending_qs.count(),
    }


def _build_admin_user_account_summary():
    pending_owner_group, _ = Group.objects.get_or_create(name="accommodation_owner_pending")
    approved_owner_group, _ = Group.objects.get_or_create(name="accommodation_owner")
    return (
        "User/account management snapshot:\n"
        f"- Employee accounts: {Employee.objects.count()} (Accepted: {Employee.objects.filter(status='accepted').count()})\n"
        f"- Owner accounts: Pending {pending_owner_group.user_set.count()}, Approved {approved_owner_group.user_set.count()}\n"
        f"- Guest accounts: {Guest.objects.filter(is_active=True).count()}"
    )


def _build_admin_reports_dashboard_summary():
    today = timezone.localdate()
    accom_confirmed_total = (
        AccommodationBooking.objects.filter(status="confirmed").aggregate(total=Sum("total_amount")).get("total")
        or Decimal("0")
    )
    tour_paid_total = (
        TourBooking.objects.filter(payment_status="paid").aggregate(total=Sum("amount_paid")).get("total")
        or Decimal("0")
    )
    return (
        "Admin dashboard/report snapshot:\n"
        f"- Published tourism records: {TourismInformation.objects.published().count()}\n"
        f"- Accommodation bookings (total): {AccommodationBooking.objects.count()}\n"
        f"- Tour bookings (total): {TourBooking.objects.count()}\n"
        f"- Confirmed accommodation revenue (gross): PHP {Decimal(accom_confirmed_total):.2f}\n"
        f"- Paid tour revenue (collected): PHP {Decimal(tour_paid_total):.2f}\n"
        f"- Records updated today ({today.isoformat()}): {TourismInformation.objects.filter(updated_at__date=today).count()}"
    )


def _build_admin_chatbot_activity_summary():
    now = timezone.now()
    window_start = now - timedelta(days=7)
    total_chat_logs = ChatbotLog.objects.count()
    recent_chat_logs = ChatbotLog.objects.filter(created_at__gte=window_start).count()
    unique_chat_users = (
        ChatbotLog.objects.exclude(user__isnull=True)
        .values("user_id")
        .distinct()
        .count()
    )
    recent_reco_events = RecommendationEvent.objects.filter(event_time__gte=window_start).count()
    recent_metric_logs = SystemMetricLog.objects.filter(
        module="chat",
        logged_at__gte=window_start,
    ).count()
    recent_survey = UsabilitySurveyResponse.objects.filter(submitted_at__gte=window_start).count()

    return (
        "Chatbot activity snapshot (last 7 days):\n"
        f"- Chat messages logged: {recent_chat_logs} (all-time: {total_chat_logs})\n"
        f"- Unique authenticated chat users: {unique_chat_users}\n"
        f"- Recommendation/chat step events: {recent_reco_events}\n"
        f"- Chat runtime metric logs: {recent_metric_logs}\n"
        f"- Chat usability feedback entries: {recent_survey}\n"
        "Use Activity Logs for timeline monitoring and admin reports for broader system context."
    )


def _build_admin_record_visibility_diagnostic():
    inactive_accom = Accomodation.objects.filter(is_active=False).count()
    pending_accom = Accomodation.objects.filter(is_active=True, approval_status="pending").count()
    draft_dest = TourismInformation.objects.filter(is_active=True, publication_status="draft").count()
    archived_dest = TourismInformation.objects.filter(is_active=True, publication_status="archived").count()
    return (
        "Admin visibility diagnostic:\n"
        f"- Inactive accommodations: {inactive_accom}\n"
        f"- Pending accommodations: {pending_accom}\n"
        f"- Draft destinations: {draft_dest}\n"
        f"- Archived destinations: {archived_dest}\n"
        "If records are missing, verify status filters, approval status, and publication state."
    )


def _build_booking_count_summary(actor, *, user=None, message=""):
    role = str(actor.get("role") or "").strip().lower()
    text = str(message or "").strip().lower()
    today = timezone.localdate()
    period_label = "all time"

    if "today" in text:
        qs = qs.filter(booking_date__date=today)
        period_label = f"today ({today.isoformat()})"
    elif "this month" in text or "monthly" in text:
        qs = qs.filter(booking_date__year=today.year, booking_date__month=today.month)
        period_label = f"this month ({today.year}-{today.month:02d})"

    if role in {"admin", "employee"}:
        tour_scope_markers = (
            "tour booking",
            "tour bookings",
            "tour package",
            "tour packages",
            "tour reservation",
            "tour reservations",
            " tour ",
            " tours",
            "packages",
        )
        accommodation_scope_markers = (
            "accommodation",
            "hotel",
            "inn",
            "room",
            "rooms",
        )
        has_tour_scope = any(marker in text for marker in tour_scope_markers)
        has_accommodation_scope = any(marker in text for marker in accommodation_scope_markers)

        if has_tour_scope and not has_accommodation_scope:
            period_key = "today" if "today" in text else ("month" if ("this month" in text or "monthly" in text) else "all_time")
            rollup = _build_tour_booking_rollup(period=period_key, today=today)
            lines = [
                (
                    f"Tour booking summary for {period_label}: Total {rollup['total']}, "
                    f"Pending {rollup['pending']}, Active {rollup['active']}, "
                    f"Completed {rollup['completed']}, Cancelled {rollup['cancelled']}."
                )
            ]
            if rollup["declined"]:
                lines.append(f"Declined: {rollup['declined']}.")
            if "revenue" in text or "income" in text:
                lines.append(
                    f"Tour revenue for {period_label}: Gross PHP {Decimal(rollup['revenue_gross']):.2f}, "
                    f"Paid PHP {Decimal(rollup['revenue_paid']):.2f}."
                )
            if rollup["includes_pending_legacy"] and rollup["pending_legacy_count"] > 0:
                lines.append(
                    "Note: Includes legacy tour booking records from the pending-booking module to match booking report totals."
                )
            return "\n".join(lines)

        if has_tour_scope and has_accommodation_scope:
            accom_qs = AccommodationBooking.objects.all()
            period_key = "today" if "today" in text else ("month" if ("this month" in text or "monthly" in text) else "all_time")
            rollup = _build_tour_booking_rollup(period=period_key, today=today)
            if "today" in text:
                accom_qs = accom_qs.filter(booking_date__date=today)
            elif "this month" in text or "monthly" in text:
                accom_qs = accom_qs.filter(booking_date__year=today.year, booking_date__month=today.month)
            accom_status_counts = Counter(str(s or "").strip().lower() for s in accom_qs.values_list("status", flat=True))
            lines = [
                (
                f"Booking summary for {period_label}:\n"
                f"- Accommodation: Total {accom_qs.count()}, Pending {accom_status_counts.get('pending', 0)}, "
                f"Confirmed {accom_status_counts.get('confirmed', 0)}, Declined {accom_status_counts.get('declined', 0)}, "
                f"Cancelled {accom_status_counts.get('cancelled', 0)}\n"
                f"- Tour: Total {rollup['total']}, Pending {rollup['pending']}, Active {rollup['active']}, "
                f"Completed {rollup['completed']}, Cancelled {rollup['cancelled']}."
                )
            ]
            if "revenue" in text or "income" in text:
                accom_revenue = accom_qs.filter(status="confirmed").aggregate(total=Sum("total_amount")).get("total") or Decimal("0")
                lines.append(
                    f"Revenue for {period_label}: Accommodation gross PHP {Decimal(accom_revenue):.2f}, "
                    f"Tour gross PHP {Decimal(rollup['revenue_gross']):.2f}, Tour paid PHP {Decimal(rollup['revenue_paid']):.2f}."
                )
            if rollup["includes_pending_legacy"] and rollup["pending_legacy_count"] > 0:
                lines.append(
                    "Note: Tour totals include legacy pending-booking records to align with the tour booking report page."
                )
            return "\n".join(lines)

        if not has_tour_scope and not has_accommodation_scope:
            accom_qs = AccommodationBooking.objects.all()
            period_key = "today" if "today" in text else ("month" if ("this month" in text or "monthly" in text) else "all_time")
            rollup = _build_tour_booking_rollup(period=period_key, today=today)
            if "today" in text:
                accom_qs = accom_qs.filter(booking_date__date=today)
            elif "this month" in text or "monthly" in text:
                accom_qs = accom_qs.filter(booking_date__year=today.year, booking_date__month=today.month)
            accom_status_counts = Counter(str(s or "").strip().lower() for s in accom_qs.values_list("status", flat=True))
            lines = [
                (
                f"Booking summary for {period_label}:\n"
                f"- Accommodation: Total {accom_qs.count()}, Pending {accom_status_counts.get('pending', 0)}, "
                f"Confirmed {accom_status_counts.get('confirmed', 0)}, Declined {accom_status_counts.get('declined', 0)}, "
                f"Cancelled {accom_status_counts.get('cancelled', 0)}\n"
                f"- Tour: Total {rollup['total']}, Pending {rollup['pending']}, Active {rollup['active']}, "
                f"Completed {rollup['completed']}, Cancelled {rollup['cancelled']}."
                )
            ]
            if "revenue" in text or "income" in text:
                accom_revenue = accom_qs.filter(status="confirmed").aggregate(total=Sum("total_amount")).get("total") or Decimal("0")
                lines.append(
                    f"Revenue for {period_label}: Accommodation gross PHP {Decimal(accom_revenue):.2f}, "
                    f"Tour gross PHP {Decimal(rollup['revenue_gross']):.2f}, Tour paid PHP {Decimal(rollup['revenue_paid']):.2f}."
                )
            if rollup["includes_pending_legacy"] and rollup["pending_legacy_count"] > 0:
                lines.append(
                    "Note: Tour totals include legacy pending-booking records to align with the tour booking report page."
                )
            return "\n".join(lines)

    qs = AccommodationBooking.objects.all()
    if role == "guest" and user is not None:
        qs = qs.filter(guest=user)
    elif role == "owner" and user is not None:
        qs = qs.filter(accommodation__owner=user)

    total = qs.count()
    status_counts = Counter(str(s or "").strip().lower() for s in qs.values_list("status", flat=True))
    return (
        f"Booking summary for {period_label}: Total {total}, "
        f"Pending {status_counts.get('pending', 0)}, "
        f"Confirmed {status_counts.get('confirmed', 0)}, "
        f"Declined {status_counts.get('declined', 0)}, "
        f"Cancelled {status_counts.get('cancelled', 0)}."
    )


def _is_default_accommodation_suggestions_command(message):
    text = (message or "").strip().lower()
    if not text:
        return False
    phrases = [
        "show default hotel suggestions",
        "show available hotels",
        "show available inns",
        "show available accommodations",
        "show available hotels and inns",
        "show available hotels/inns",
        "show hotels and inns",
        "show hotels near me",
        "default hotel suggestions",
        "show hotel suggestions",
        "show inn suggestions",
        "show accommodation suggestions",
        "suggest hotels",
        "suggest inns",
    ]
    return any(phrase in text for phrase in phrases)


def _openai_extract_intent_and_params(message):
    # Legacy compatibility wrapper kept to avoid breaking imports/tests.
    # OpenAI is no longer used for intent parsing.
    parsed = _classify_intent_and_extract_params(message)
    parsed["source"] = "legacy_wrapper_no_openai_parse"
    return parsed


def _fallback_nlg_paraphrase(reply):
    text = str(reply or "").strip()
    if not text:
        return text
    lines = [line.strip() for line in text.splitlines() if str(line).strip()]
    if not lines:
        return text
    if len(lines) == 1:
        return lines[0]
    normalized = []
    for line in lines:
        normalized.append(re.sub(r"\s{2,}", " ", line))
    merged = "\n".join(normalized)
    if len(lines) == 1:
        return merged
    # Keep facts unchanged but improve readability for template-heavy text.
    if not merged.endswith(".") and not merged.endswith("?"):
        merged = f"{merged}."
    return merged


def _extract_critical_facts_for_nlg_guardrails(text):
    raw = str(text or "")
    lowered = raw.lower()

    urls = re.findall(r"https?://[^\s)>\]}]+", raw, flags=re.IGNORECASE)
    iso_dates = re.findall(r"\b\d{4}-\d{2}-\d{2}\b", raw)
    money_tokens = re.findall(r"(?:php|₱)\s*([0-9][0-9,]*(?:\.[0-9]{1,2})?)", raw, flags=re.IGNORECASE)
    booking_ids = re.findall(r"\bbooking\s*id\s*[:#-]?\s*([A-Za-z0-9\-]+)\b", raw, flags=re.IGNORECASE)
    room_ids = re.findall(r"\broom\s*id\s*[:#-]?\s*(\d+)\b", raw, flags=re.IGNORECASE)
    quantity_pairs = re.findall(
        r"\b(\d+)\s*(guest|guests|night|nights|room|rooms|booking|bookings|pax)\b",
        lowered,
        flags=re.IGNORECASE,
    )

    normalized_money = []
    for token in money_tokens:
        cleaned = str(token or "").replace(",", "").strip()
        if not cleaned:
            continue
        try:
            normalized_money.append(f"{float(cleaned):.2f}")
        except Exception:
            continue

    normalized_urls = sorted({str(url).strip() for url in urls if str(url).strip()})
    normalized_dates = sorted({str(val).strip() for val in iso_dates if str(val).strip()})
    normalized_booking_ids = sorted({str(val).strip().lower() for val in booking_ids if str(val).strip()})
    normalized_room_ids = sorted({str(val).strip() for val in room_ids if str(val).strip()})
    normalized_quantities = sorted(
        {
            f"{str(num).strip()}:{str(unit).strip().lower()}"
            for num, unit in quantity_pairs
            if str(num).strip() and str(unit).strip()
        }
    )
    normalized_money = sorted(set(normalized_money))

    return {
        "urls": normalized_urls,
        "iso_dates": normalized_dates,
        "amounts": normalized_money,
        "booking_ids": normalized_booking_ids,
        "room_ids": normalized_room_ids,
        "quantities": normalized_quantities,
    }


def _guardrails_validate_nlg_output(backend_reply, candidate_reply):
    backend_facts = _extract_critical_facts_for_nlg_guardrails(backend_reply)
    candidate_facts = _extract_critical_facts_for_nlg_guardrails(candidate_reply)
    reasons = []
    for key in ("urls", "iso_dates", "amounts", "booking_ids", "room_ids", "quantities"):
        expected_values = backend_facts.get(key) or []
        actual_values = set(candidate_facts.get(key) or [])
        for expected in expected_values:
            if expected not in actual_values:
                reasons.append(f"missing_{key}:{expected}")
        # Block injected factual values in structured replies.
        if key in ("amounts", "booking_ids", "room_ids", "quantities") and expected_values:
            expected_set = set(expected_values)
            for actual in actual_values:
                if actual not in expected_set:
                    reasons.append(f"unexpected_{key}:{actual}")
    return (len(reasons) == 0), reasons


def _apply_nlg_output_guardrails(*, request, provider_source, backend_reply, candidate_reply):
    candidate_text = str(candidate_reply or "").strip()
    backend_text = str(backend_reply or "").strip()
    if not candidate_text or not backend_text:
        return candidate_text, provider_source

    is_valid, reasons = _guardrails_validate_nlg_output(backend_text, candidate_text)
    if is_valid:
        return candidate_text, provider_source

    _safe_log_chat_runtime_event(
        request,
        event_key="output_guardrail_triggered",
        detail=";".join(reasons[:6]),
    )
    fallback_source = f"{provider_source}_guardrail_fallback"
    return backend_text, fallback_source


def generate_final_ai_response(*, request, intent, user_message, backend_reply):
    reply = str(backend_reply or "").strip()
    if not reply:
        return reply, "empty_backend_reply"

    # Preserve structured booking/slot templates exactly to avoid key-value drift.
    if any(
        marker in reply.lower()
        for marker in (
            "great. here are the details i have so far:",
            "recorded details:",
            "thank you. i have recorded the following details:",
            "details received:",
            "booking receipt / summary",
            "booking summary (draft",
            "booking draft (not yet saved)",
        )
    ):
        return reply, "backend_structured_template"

    openai_api_key = str(os.getenv("OPENAI_API_KEY", "")).strip()
    gemini_api_key = str(os.getenv("GEMINI_API_KEY", "")).strip()
    if getattr(settings, "TESTING", False) and not openai_api_key:
        # Keep tests deterministic by skipping live Gemini rewrites unless
        # OpenAI is explicitly enabled by a test case.
        gemini_api_key = ""
    nlg_enabled = str(
        os.getenv("CHATBOT_LLM_NLG_ENABLED", os.getenv("CHATBOT_OPENAI_NLG_ENABLED", "1"))
    ).strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )
    if not nlg_enabled:
        return _fallback_nlg_paraphrase(reply), "llm_nlg_disabled_paraphrase"

    user_id = ""
    user = getattr(request, "user", None)
    if user and getattr(user, "is_authenticated", False):
        user_id = str(getattr(user, "pk", "") or "")

    # LLM receives sanitized backend context only.
    nlg_payload = {
        "intent": str(intent or ""),
        "user_message": str(user_message or "")[:500],
        "backend_reply": reply[:3500],
        "user_id": user_id[:40],
    }
    system_prompt = (
        "You are a tourism reservation assistant NLG layer.\n"
        "Rewrite the backend reply into clear, professional, and formal language.\n"
        "Use polite and concise phrasing suitable for customer support.\n"
        "Do not add new facts, prices, dates, IDs, links, or policy claims.\n"
        "Keep all booking/payment constraints exactly as provided.\n"
        "Return plain text only."
    )

    if openai_api_key and OpenAI is not None:
        model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        try:
            client = OpenAI(api_key=openai_api_key)
            completion = client.chat.completions.create(
                model=model,
                temperature=0.2,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": json.dumps(nlg_payload, ensure_ascii=True)},
                ],
            )
            phrased = str(completion.choices[0].message.content or "").strip()
            if phrased:
                return _apply_nlg_output_guardrails(
                    request=request,
                    provider_source="openai_nlg",
                    backend_reply=reply,
                    candidate_reply=phrased,
                )
        except Exception:
            pass

    if gemini_api_key and genai is not None:
        gemini_model = str(os.getenv("GEMINI_MODEL", "gemini-1.5-flash") or "").strip() or "gemini-1.5-flash"
        gemini_client = None
        try:
            gemini_client = genai.Client(api_key=gemini_api_key)
        except Exception:
            gemini_client = None
        if gemini_client is None:
            _safe_log_chat_runtime_event(
                request,
                event_key="gemini_failure_fallback",
                detail="gemini_nlg_unavailable",
            )
            return reply, "gemini_nlg_unavailable"

        last_error = ""
        max_attempts = max(1, min(_to_int(os.getenv("CHATBOT_GEMINI_NLG_RETRY", "2"), default=2), 3))
        for attempt in range(max_attempts):
            try:
                prompt = (
                    f"{system_prompt}\n\n"
                    "Input JSON:\n"
                    f"{json.dumps(nlg_payload, ensure_ascii=True)}"
                )
                gemini_response = gemini_client.models.generate_content(
                    model=gemini_model,
                    contents=prompt,
                )
                phrased = str(getattr(gemini_response, "text", "") or "").strip()
                if phrased:
                    source = "gemini_nlg"
                    if attempt == 0:
                        source = "gemini_nlg"
                    else:
                        source = "gemini_nlg_retry"
                    return _apply_nlg_output_guardrails(
                        request=request,
                        provider_source=source,
                        backend_reply=reply,
                        candidate_reply=phrased,
                    )
            except Exception as exc:
                last_error = str(exc)
                continue

        fallback_text = _fallback_nlg_paraphrase(reply)
        if fallback_text and fallback_text != reply:
            _safe_log_chat_runtime_event(
                request,
                event_key="gemini_failure_fallback",
                detail="gemini_nlg_fallback_paraphrase",
            )
            return fallback_text, "gemini_nlg_fallback_paraphrase"
        if last_error:
            _safe_log_chat_runtime_event(
                request,
                event_key="gemini_failure_fallback",
                detail="gemini_nlg_error",
            )
            return reply, "gemini_nlg_error"
        _safe_log_chat_runtime_event(
            request,
            event_key="gemini_failure_fallback",
            detail="gemini_nlg_empty",
        )
        return reply, "gemini_nlg_empty"

    if openai_api_key and OpenAI is not None:
        return _fallback_nlg_paraphrase(reply), "openai_nlg_error_paraphrase"
    if gemini_api_key and genai is None:
        _safe_log_chat_runtime_event(
            request,
            event_key="gemini_failure_fallback",
            detail="gemini_nlg_unavailable_paraphrase",
        )
        return _fallback_nlg_paraphrase(reply), "gemini_nlg_unavailable_paraphrase"
    if gemini_api_key:
        _safe_log_chat_runtime_event(
            request,
            event_key="gemini_failure_fallback",
            detail="gemini_nlg_error_paraphrase",
        )
        return _fallback_nlg_paraphrase(reply), "gemini_nlg_error_paraphrase"
    if openai_api_key:
        return _fallback_nlg_paraphrase(reply), "openai_nlg_error_paraphrase"
    return _fallback_nlg_paraphrase(reply), "llm_nlg_unavailable_paraphrase"

def _openai_generate_final_response(*, request, intent, user_message, backend_reply):
    """
    Backward-compatible alias.
    Deprecated naming retained to avoid breaking older imports/tests.
    """
    return generate_final_ai_response(
        request=request,
        intent=intent,
        user_message=user_message,
        backend_reply=backend_reply,
    )


@csrf_exempt
def ai_chat(request):
    start_time = time.perf_counter()

    if request.method != "POST":
        return _chat_json_response(request, start_time, {"status": "ok"})

    try:
        payload = json.loads(request.body or "{}")
    except json.JSONDecodeError:
        return _chat_json_response(
            request,
            start_time,
            {"fulfillmentText": "Invalid JSON payload."},
            status=400,
            error_message="invalid_json_payload",
        )

    actor = _resolve_chat_actor(request)
    user = actor.get("user")
    if not actor.get("is_allowed"):
        return _chat_json_response(
            request,
            start_time,
            {
                "fulfillmentText": "Please log in first to use the chatbot.",
                "error_code": "chat_requires_login",
            },
            status=401,
            error_message="chat_requires_login",
        )

    raw_message = str(payload.get("message", "")).strip()
    translated_message, detected_language = translate_to_english(raw_message)
    message = str(translated_message or raw_message).strip()
    init_suggestions = bool(payload.get("init_suggestions"))
    if not message:
        return _chat_json_response(
            request,
            start_time,
            {"fulfillmentText": "Please send a message in this format: {\"message\": \"...\"}."},
            status=400,
            error_message="missing_message",
        )
    request._chatbot_log_context = {
        "user_message": raw_message,
        "resolved_intent": "",
        "resolved_params": {},
        "intent_classifier": {},
        "response_nlg_source": "",
        "fallback_used": False,
        "provenance": {
            "chat_role": actor.get("role", ""),
            "session_id": (request.session.session_key or "") if hasattr(request, "session") else "",
            "detected_language": detected_language,
            "input_translated_to_english": bool(
                str(raw_message or "").strip()
                and str(message or "").strip()
                and str(raw_message).strip() != str(message).strip()
            ),
            "response_translated_to_user_language": False,
        },
    }
    admin_topic_hint = _detect_admin_support_topic(message) if actor.get("role") == "admin" else ""
    employee_topic_hint = _detect_employee_support_topic(message) if actor.get("role") == "employee" else ""

    if _is_help_or_greeting_command(message):
        request._chatbot_log_context["resolved_intent"] = "role_help"
        help_payload = _build_role_help_payload(actor)
        return _chat_json_response(request, start_time, help_payload)

    if actor.get("role") == "guest":
        current_state = _load_chat_state(request)
        fallback_sched_ids = (
            current_state.get("last_tour_recommendation_sched_ids")
            if isinstance(current_state.get("last_tour_recommendation_sched_ids"), list)
            else []
        )
        selection_index = _extract_tour_selection_index(message)
        is_tour_booking_shortcut = (
            selection_index > 0
            and bool(re.search(r"\b(book|reserve|reservation)\b", str(message or "").lower()))
        )
        if _is_guest_tour_booking_command(message) or is_tour_booking_shortcut:
            request._chatbot_log_context["resolved_intent"] = "book_tour_via_link"
            payload = _build_guest_tour_booking_link_payload(
                request,
                sched_id=_extract_sched_id_from_message(message),
                fallback_sched_ids=fallback_sched_ids,
                selection_index=selection_index,
            )
            return _chat_json_response(request, start_time, payload)

    if (
        actor.get("role") in {"admin", "employee"}
        and _is_open_dashboard_command(message)
        and not (actor.get("role") == "admin" and admin_topic_hint)
        and not (actor.get("role") == "employee" and employee_topic_hint)
    ):
        request._chatbot_log_context["resolved_intent"] = "open_dashboard"
        target_name = "admin_app:admin_dashboard" if actor.get("role") == "admin" else "admin_app:employee_dashboard"
        target_url = reverse(target_name)
        if hasattr(request, "build_absolute_uri"):
            target_url = request.build_absolute_uri(target_url)
        return _chat_json_response(
            request,
            start_time,
            {
                "fulfillmentText": "Opening your dashboard now.",
                "redirect_url": target_url,
            },
        )
    if actor.get("role") == "owner" and _is_open_dashboard_command(message):
        request._chatbot_log_context["resolved_intent"] = "open_owner_dashboard"
        return _chat_json_response(
            request,
            start_time,
            _build_link_payload(
                request,
                text="I found your accommodation dashboard. Click the button below to open it in a new tab.",
                route_name="admin_app:accommodation_dashboard",
                label="Open Accommodation Dashboard",
            ),
        )

    if _is_booking_count_command(message):
        request._chatbot_log_context["resolved_intent"] = "booking_count_summary"
        summary = _build_booking_count_summary(actor, user=user, message=message)
        payload = {"fulfillmentText": summary}
        role = str(actor.get("role") or "").strip().lower()
        if role == "owner":
            payload.update(
                _build_link_payload(
                    request,
                    text=summary,
                    route_name="admin_app:owner_accommodation_bookings",
                    label="View Owner Accommodation Bookings",
                )
            )
            payload["quick_replies"] = [
                "How many bookings do we have today?",
                "How many bookings do we have this month?",
                "Open reports and analytics",
            ]
        elif role == "guest":
            payload.update(
                _build_link_payload(
                    request,
                    text=summary,
                    route_name="my_accommodation_bookings",
                    label="View My Hotel/Inn Bookings",
                )
            )
        elif role == "admin":
            admin_message_text = str(message or "").strip().lower()
            admin_tour_scope = any(
                marker in admin_message_text
                for marker in ("tour booking", "tour bookings", "tour package", "tour packages", "tour reservation", "tour reservations", " tour ", " tours")
            )
            admin_accommodation_scope = any(
                marker in admin_message_text
                for marker in ("accommodation", "hotel", "inn", "room", "rooms")
            )
            admin_generic_scope = not admin_tour_scope and not admin_accommodation_scope
            payload.update(
                _build_link_payload(
                    request,
                    text=summary,
                    route_name=(
                        "tour_app:pending_view"
                        if admin_tour_scope
                        else ("admin_app:admin_dashboard" if admin_generic_scope else "admin_app:accommodation_bookings")
                    ),
                    label=(
                        "Open Tour Bookings"
                        if admin_tour_scope
                        else ("Open Booking Monitoring" if admin_generic_scope else "Open Accommodation Bookings")
                    ),
                )
            )
        return _chat_json_response(request, start_time, payload)

    if actor.get("role") == "owner" and _is_owner_hub_command(message):
        request._chatbot_log_context["resolved_intent"] = "open_owner_hub"
        return _chat_json_response(
            request,
            start_time,
            _build_link_payload(
                request,
                text="I found your Owner Hub. Click the button below to open it in a new tab.",
                route_name="admin_app:owner_hub",
                label="Open Owner Hub",
            ),
        )

    if actor.get("role") == "owner" and _is_owner_reports_analytics_command(message):
        request._chatbot_log_context["resolved_intent"] = "open_owner_reports_analytics"
        return _chat_json_response(
            request,
            start_time,
            _build_link_payload(
                request,
                text="I found your Reports & Analytics page. Click the button below to open it in a new tab.",
                route_name="accom_app:owner_reports_analytics",
                label="Open Reports & Analytics",
            ),
        )

    if actor.get("role") == "owner" and _is_owner_register_accommodation_command(message):
        request._chatbot_log_context["resolved_intent"] = "open_owner_accommodation_register"
        return _chat_json_response(
            request,
            start_time,
            _build_link_payload(
                request,
                text="I found the accommodation registration page. Click the button below to open it in a new tab.",
                route_name="admin_app:accommodation_register",
                label="Register New Accommodation",
            ),
        )

    if (
        actor.get("role") == "owner"
        and _is_owner_accommodation_overview_command(message)
        and not _detect_owner_support_topic(message)
    ):
        request._chatbot_log_context["resolved_intent"] = "owner_accommodation_overview"
        owner_accom_reply = _build_owner_accommodations_summary(user)
        return _chat_json_response(
            request,
            start_time,
            {
                "fulfillmentText": owner_accom_reply,
                "quick_replies": ["Show my rooms", "Show my bookings", "Open Owner Hub"],
            },
        )

    if actor.get("role") == "owner" and _is_owner_performance_summary_command(message):
        request._chatbot_log_context["resolved_intent"] = "owner_performance_summary"
        owner_summary_reply = _build_owner_performance_summary(user)
        return _chat_json_response(
            request,
            start_time,
            {
                "fulfillmentText": owner_summary_reply,
                "quick_replies": ["Show my rooms", "Show my bookings", "Open Owner Hub"],
            },
        )

    if actor.get("role") == "admin" and admin_topic_hint:
        request._chatbot_log_context["resolved_intent"] = admin_topic_hint
        short_admin = {"approve listing", "pending accommodations", "manage users", "admin dashboard", "booking summary"}

        if admin_topic_hint == "admin_approval_workflow":
            payload = _build_link_payload(
                request,
                text=_build_admin_approval_workflow_summary(),
                route_name="admin_app:pending_accommodation",
                label="Open Pending Accommodations",
            )
            if str(message or "").strip().lower() in short_admin:
                payload["needs_clarification"] = True
                payload["missing_slot"] = "approval_scope"
                payload["quick_replies"] = [
                    "Show pending accommodations",
                    "Show pending owner accounts",
                    "Open admin dashboard",
                ]
            return _chat_json_response(request, start_time, payload)

        if admin_topic_hint == "admin_destination_management":
            return _chat_json_response(
                request,
                start_time,
                _build_link_payload(
                    request,
                    text=_build_admin_destination_management_summary(),
                    route_name="admin_app:tourism_information_manage",
                    label="Open Tourism Information",
                ),
            )

        if admin_topic_hint == "admin_accommodation_records_management":
            return _chat_json_response(
                request,
                start_time,
                _build_link_payload(
                    request,
                    text=_build_admin_accommodation_records_summary(message),
                    route_name="admin_app:accommodation_bookings",
                    label="Open Accommodation Records",
                ),
            )

        if admin_topic_hint == "admin_booking_system_monitoring":
            payload = _build_link_payload(
                request,
                text=_build_admin_booking_system_summary(),
                route_name="admin_app:accommodation_bookings",
                label="Open Booking Monitoring",
            )
            if str(message or "").strip().lower() in short_admin:
                payload["needs_clarification"] = True
                payload["missing_slot"] = "booking_summary_scope"
                payload["quick_replies"] = [
                    "Show pending reservations",
                    "Show cancelled bookings",
                    "Open admin dashboard",
                ]
            return _chat_json_response(request, start_time, payload)

        if admin_topic_hint == "admin_user_account_management":
            payload = _build_link_payload(
                request,
                text=_build_admin_user_account_summary(),
                route_name="admin_app:pending_accommodation_owners",
                label="Open Owner/Account Review",
            )
            if str(message or "").strip().lower() in short_admin:
                payload["needs_clarification"] = True
                payload["missing_slot"] = "account_scope"
                payload["quick_replies"] = [
                    "Show pending owner accounts",
                    "Show employee accounts",
                    "Reset password help",
                ]
            return _chat_json_response(request, start_time, payload)

        if admin_topic_hint == "admin_reports_dashboard_support":
            payload = _build_link_payload(
                request,
                text=_build_admin_reports_dashboard_summary(),
                route_name="admin_app:admin_dashboard",
                label="Open Admin Dashboard",
            )
            if str(message or "").strip().lower() in short_admin:
                payload["needs_clarification"] = True
                payload["missing_slot"] = "report_scope"
                payload["quick_replies"] = [
                    "Show booking summary",
                    "Show tourism statistics",
                    "Open admin dashboard",
                ]
            return _chat_json_response(request, start_time, payload)

        if admin_topic_hint == "admin_chatbot_activity_monitoring":
            return _chat_json_response(
                request,
                start_time,
                _build_link_payload(
                    request,
                    text=_build_admin_chatbot_activity_summary(),
                    route_name="admin_app:activity_tracker",
                    label="Open Activity Logs",
                ),
            )

        if admin_topic_hint == "admin_record_visibility_issue":
            return _chat_json_response(
                request,
                start_time,
                _build_link_payload(
                    request,
                    text=_build_admin_record_visibility_diagnostic(),
                    route_name="admin_app:admin_dashboard",
                    label="Open Admin Dashboard",
                ),
            )

    if actor.get("role") == "admin" and _is_admin_pending_accommodations_command(message):
        request._chatbot_log_context["resolved_intent"] = "admin_pending_accommodations"
        summary = _build_admin_pending_accommodations_summary()
        return _chat_json_response(
            request,
            start_time,
            _build_link_payload(
                request,
                text=summary,
                route_name="admin_app:pending_accommodation",
                label="Open Pending Accommodations",
            ),
        )

    if actor.get("role") == "admin" and _is_admin_pending_owner_accounts_command(message):
        request._chatbot_log_context["resolved_intent"] = "admin_pending_owner_accounts"
        summary = _build_admin_pending_owner_accounts_summary()
        return _chat_json_response(
            request,
            start_time,
            _build_link_payload(
                request,
                text=summary,
                route_name="admin_app:pending_accommodation_owners",
                label="Open Pending Owner Accounts",
            ),
        )

    if actor.get("role") == "admin" and _is_admin_accommodation_bookings_command(message):
        request._chatbot_log_context["resolved_intent"] = "admin_accommodation_bookings"
        return _chat_json_response(
            request,
            start_time,
            _build_link_payload(
                request,
                text="I found the Accommodation Bookings page. Click the button below to open it in a new tab.",
                route_name="admin_app:accommodation_bookings",
                label="Open Accommodation Bookings",
            ),
        )

    if actor.get("role") == "admin" and _is_admin_tourism_manage_command(message):
        request._chatbot_log_context["resolved_intent"] = "admin_tourism_information_manage"
        return _chat_json_response(
            request,
            start_time,
            _build_link_payload(
                request,
                text="I found the Tourism Information management page. Click the button below to open it in a new tab.",
                route_name="admin_app:tourism_information_manage",
                label="Open Tourism Information",
            ),
        )

    if actor.get("role") == "admin" and _is_admin_survey_results_command(message):
        request._chatbot_log_context["resolved_intent"] = "admin_survey_results"
        return _chat_json_response(
            request,
            start_time,
            _build_link_payload(
                request,
                text="I found the survey results dashboard. Click the button below to open it in a new tab.",
                route_name="admin_app:survey_results_dashboard",
                label="Open Survey Results",
            ),
        )

    if actor.get("role") == "admin" and _is_admin_traveler_surveys_command(message):
        request._chatbot_log_context["resolved_intent"] = "admin_traveler_surveys"
        return _chat_json_response(
            request,
            start_time,
            _build_link_payload(
                request,
                text="I found the Traveler Surveys page. Click the button below to open it in a new tab.",
                route_name="admin_app:survey_results_dashboard",
                label="Open Traveler Surveys",
            ),
        )

    if actor.get("role") == "admin" and _is_admin_tour_calendar_command(message):
        request._chatbot_log_context["resolved_intent"] = "admin_tour_calendar"
        return _chat_json_response(
            request,
            start_time,
            _build_link_payload(
                request,
                text="I found the Tour Calendar page. Click the button below to open it in a new tab.",
                route_name="admin_app:tour_calendar",
                label="Open Tour Calendar",
            ),
        )

    if actor.get("role") == "admin" and _is_admin_tour_list_command(message):
        request._chatbot_log_context["resolved_intent"] = "admin_tour_list"
        return _chat_json_response(
            request,
            start_time,
            _build_link_payload(
                request,
                text="I found the Tour List page. Click the button below to open it in a new tab.",
                route_name="tour_app:home",
                label="Open Tour List",
            ),
        )

    if actor.get("role") == "admin" and _is_admin_activity_logs_command(message):
        request._chatbot_log_context["resolved_intent"] = "admin_activity_logs"
        return _chat_json_response(
            request,
            start_time,
            _build_link_payload(
                request,
                text="I found the Activity Logs page. Click the button below to open it in a new tab.",
                route_name="admin_app:activity_tracker",
                label="Open Activity Logs",
            ),
        )

    if actor.get("role") == "admin" and _is_admin_map_command(message):
        request._chatbot_log_context["resolved_intent"] = "admin_map"
        return _chat_json_response(
            request,
            start_time,
            _build_link_payload(
                request,
                text="I found the Map page. Click the button below to open it in a new tab.",
                route_name="admin_app:map",
                label="Open Map",
            ),
        )

    if actor.get("role") == "admin" and _is_admin_discounts_command(message):
        request._chatbot_log_context["resolved_intent"] = "admin_discounts"
        return _chat_json_response(
            request,
            start_time,
            _build_link_payload(
                request,
                text="I found the Discounts page. Click the button below to open it in a new tab.",
                route_name="tour_app:admission_rate",
                label="Open Discounts",
            ),
        )

    if actor.get("role") == "employee" and _is_employee_assigned_tours_command(message) and not employee_topic_hint:
        request._chatbot_log_context["resolved_intent"] = "employee_assigned_tours"
        return _chat_json_response(
            request,
            start_time,
            _build_link_payload(
                request,
                text=_build_employee_assigned_tours_summary(request, actor),
                route_name="admin_app:employee_assigned_tours",
                label="Open Assigned Tours",
            ),
        )

    if actor.get("role") == "employee" and _is_employee_tour_calendar_command(message) and not employee_topic_hint:
        request._chatbot_log_context["resolved_intent"] = "employee_tour_calendar"
        return _chat_json_response(
            request,
            start_time,
            _build_link_payload(
                request,
                text="I found your tour calendar. Click the button below to open it in a new tab.",
                route_name="admin_app:employee_tour_calendar",
                label="Open Tour Calendar",
            ),
        )

    if actor.get("role") == "employee" and _is_employee_accommodations_command(message) and not employee_topic_hint:
        request._chatbot_log_context["resolved_intent"] = "employee_accommodations"
        return _chat_json_response(
            request,
            start_time,
            _build_link_payload(
                request,
                text="I found the accommodations page for employee view. Click the button below to open it in a new tab.",
                route_name="admin_app:employee_accommodations",
                label="Open Accommodations",
            ),
        )

    if actor.get("role") == "employee" and _is_employee_profile_command(message) and not employee_topic_hint:
        request._chatbot_log_context["resolved_intent"] = "employee_profile"
        return _chat_json_response(
            request,
            start_time,
            _build_link_payload(
                request,
                text="I found your profile page. Click the button below to open it in a new tab.",
                route_name="admin_app:employee_profile",
                label="Open Profile",
            ),
        )

    if actor.get("role") == "employee" and _is_employee_tour_list_command(message) and not employee_topic_hint:
        request._chatbot_log_context["resolved_intent"] = "employee_tour_list"
        return _chat_json_response(
            request,
            start_time,
            _build_link_payload(
                request,
                text="I found the Tour List page. Click the button below to open it in a new tab.",
                route_name="tour_app:home",
                label="Open Tour List",
            ),
        )

    if actor.get("role") == "employee" and _is_employee_create_tour_command(message) and not employee_topic_hint:
        request._chatbot_log_context["resolved_intent"] = "employee_create_tour"
        return _chat_json_response(
            request,
            start_time,
            _build_link_payload(
                request,
                text="I found the Create Tour page. Click the button below to open it in a new tab.",
                route_name="tour_app:add_tour",
                label="Open Create Tour",
            ),
        )

    if actor.get("role") == "employee" and _is_employee_map_command(message) and not employee_topic_hint:
        request._chatbot_log_context["resolved_intent"] = "employee_map"
        return _chat_json_response(
            request,
            start_time,
            _build_link_payload(
                request,
                text="I found the Map page. Click the button below to open it in a new tab.",
                route_name="admin_app:map",
                label="Open Map",
            ),
        )

    if actor.get("role") == "employee":
        employee_topic = employee_topic_hint
        if employee_topic:
            request._chatbot_log_context["resolved_intent"] = employee_topic
            short_inputs = {"tourist records", "pending bookings", "generate report", "listing approval", "dashboard not opening"}

            if employee_topic == "employee_account_access_help":
                payload = _build_link_payload(
                    request,
                    text=(
                        "If you forgot your password or cannot open the employee dashboard, use the password recovery page first.\n"
                        "After reset, sign in again and open the Employee Dashboard."
                    ),
                    route_name="admin_app:forgot_password",
                    label="Open Password Recovery",
                )
                if str(message or "").strip().lower() in short_inputs:
                    payload["needs_clarification"] = True
                    payload["missing_slot"] = "employee_access_issue"
                    payload["quick_replies"] = [
                        "Open employee dashboard",
                        "Reset password",
                        "Record not showing in system",
                    ]
                return _chat_json_response(request, start_time, payload)

            if employee_topic == "employee_tourist_monitoring":
                payload = _build_link_payload(
                    request,
                    text=_build_employee_tourist_monitoring_summary(message),
                    route_name="admin_app:employee_dashboard",
                    label="Open Employee Dashboard",
                )
                if str(message or "").strip().lower() in short_inputs:
                    payload["needs_clarification"] = True
                    payload["missing_slot"] = "tourist_monitoring_scope"
                    payload["quick_replies"] = [
                        "Search tourist by name",
                        "Show booking summaries",
                        "Open monitoring dashboard",
                    ]
                return _chat_json_response(request, start_time, payload)

            if employee_topic == "employee_booking_monitoring":
                payload = _build_link_payload(
                    request,
                    text=_build_employee_booking_monitoring_summary(),
                    route_name="admin_app:employee_dashboard",
                    label="Open Monitoring Dashboard",
                )
                if str(message or "").strip().lower() in short_inputs:
                    payload["needs_clarification"] = True
                    payload["missing_slot"] = "booking_status_scope"
                    payload["quick_replies"] = [
                        "Show pending reservations",
                        "Show confirmed bookings",
                        "Show cancelled reservations",
                    ]
                return _chat_json_response(request, start_time, payload)

            if employee_topic == "employee_accommodation_records":
                return _chat_json_response(
                    request,
                    start_time,
                    _build_link_payload(
                        request,
                        text=_build_employee_accommodation_records_summary(),
                        route_name="admin_app:employee_accommodations",
                        label="Open Employee Accommodations",
                    ),
                )

            if employee_topic == "employee_destination_records":
                return _chat_json_response(
                    request,
                    start_time,
                    _build_link_payload(
                        request,
                        text=_build_employee_destination_records_summary(),
                        route_name="admin_app:employee_dashboard",
                        label="Open Monitoring Dashboard",
                    ),
                )

            if employee_topic == "employee_reports_support":
                payload = _build_link_payload(
                    request,
                    text=_build_employee_reports_support_summary(),
                    route_name="admin_app:employee_dashboard",
                    label="Open Monitoring Dashboard",
                )
                if str(message or "").strip().lower() in short_inputs:
                    payload["needs_clarification"] = True
                    payload["missing_slot"] = "report_type"
                    payload["quick_replies"] = [
                        "Show booking summaries",
                        "Show tourist statistics",
                        "Show accommodation reports",
                    ]
                return _chat_json_response(request, start_time, payload)

            if employee_topic == "employee_workflow_listing_review":
                payload = _build_link_payload(
                    request,
                    text=(
                        "Accommodation approval decisions are admin-controlled.\n"
                        "As employee, you can review listing records and flag issues for admin action."
                    ),
                    route_name="admin_app:employee_accommodations",
                    label="Open Employee Accommodations",
                )
                if str(message or "").strip().lower() in short_inputs:
                    payload["needs_clarification"] = True
                    payload["missing_slot"] = "listing_review_scope"
                    payload["quick_replies"] = [
                        "Review submitted listings",
                        "Open monitoring dashboard",
                        "How to flag listing issue",
                    ]
                return _chat_json_response(request, start_time, payload)

            if employee_topic == "employee_workflow_destination_feedback":
                return _chat_json_response(
                    request,
                    start_time,
                    _build_link_payload(
                        request,
                        text=(
                            "You can manage destination record updates from staff workflows and escalate publishing changes for approval when needed.\n"
                            "For tourism concerns/feedback, use your notifications and monitoring workflow."
                        ),
                        route_name="admin_app:employee_notifications",
                        label="Open Employee Notifications",
                    ),
                )

            if employee_topic == "employee_record_visibility_issue":
                return _chat_json_response(
                    request,
                    start_time,
                    _build_link_payload(
                        request,
                        text=(
                            "Record visibility check:\n"
                            f"{_build_employee_tourist_monitoring_summary(message)}\n"
                            f"{_build_employee_accommodation_records_summary()}\n"
                            "If records are still missing, verify status filters, publication state, and approval status."
                        ),
                        route_name="admin_app:employee_dashboard",
                        label="Open Monitoring Dashboard",
                    ),
                )

            if employee_topic == "employee_monitoring_dashboard_help":
                return _chat_json_response(
                    request,
                    start_time,
                    _build_link_payload(
                        request,
                        text="I found the monitoring dashboard for employee operations.",
                        route_name="admin_app:employee_dashboard",
                        label="Open Monitoring Dashboard",
                    ),
                )

            if employee_topic == "employee_record_update_help":
                return _chat_json_response(
                    request,
                    start_time,
                    _build_link_payload(
                        request,
                        text=(
                            "You can update records that are allowed by your employee role in staff workflows.\n"
                            "Some approval/publish actions remain admin-controlled."
                        ),
                        route_name="admin_app:employee_dashboard",
                        label="Open Employee Dashboard",
                    ),
                )

    if actor.get("role") == "guest" and _is_guest_map_command(message):
        request._chatbot_log_context["resolved_intent"] = "guest_map"
        return _chat_json_response(
            request,
            start_time,
            _build_link_payload(
                request,
                text="I found the Bayawan City Map page. Click the button below to open it in a new tab.",
                route_name="map",
                label="Open Map",
            ),
        )

    if actor.get("role") == "guest" and _is_guest_password_help_command(message):
        request._chatbot_log_context["resolved_intent"] = "guest_password_help"
        return _chat_json_response(
            request,
            start_time,
            _build_link_payload(
                request,
                text=(
                    "If you forgot your password, open the Guest Login page and use the password reset option.\n"
                    "If reset is unavailable, please contact the Tourism Office for manual account recovery."
                ),
                route_name="login",
                label="Open Guest Login",
            ),
        )

    if actor.get("role") == "guest" and _is_guest_booking_cancel_support_command(message):
        request._chatbot_log_context["resolved_intent"] = "guest_booking_cancel_support"
        return _chat_json_response(
            request,
            start_time,
            _build_link_payload(
                request,
                text=(
                    "To cancel a booking, open your Hotel/Inn Bookings page, select the booking, then click Cancel.\n"
                    "Only pending or confirmed bookings can be cancelled."
                ),
                route_name="my_accommodation_bookings",
                label="Open My Hotel/Inn Bookings",
            ),
        )

    if actor.get("role") == "guest" and _is_guest_booking_change_date_command(message):
        request._chatbot_log_context["resolved_intent"] = "guest_booking_change_date_support"
        return _chat_json_response(
            request,
            start_time,
            _build_link_payload(
                request,
                text=(
                    "Direct date change is not yet enabled in this prototype.\n"
                    "Please cancel the current booking (if pending/confirmed), then create a new booking with your new dates."
                ),
                route_name="my_accommodation_bookings",
                label="Open My Hotel/Inn Bookings",
            ),
        )

    if actor.get("role") == "guest" and _is_guest_payment_methods_command(message):
        request._chatbot_log_context["resolved_intent"] = "guest_payment_methods"
        methods = [
            str(label or "").strip()
            for value, label in Billing.PAYMENT_METHOD_CHOICES
            if str(value or "").strip()
        ]
        methods_text = ", ".join(methods) if methods else "Cash, GCash, Bank Transfer, Card"
        return _chat_json_response(
            request,
            start_time,
            {
                "fulfillmentText": (
                    f"Supported payment methods in the booking records are: {methods_text}.\n"
                    "Final payment handling is processed through the LGU-linked billing/payment flow."
                )
            },
        )

    if actor.get("role") == "guest" and _is_guest_down_payment_command(message):
        request._chatbot_log_context["resolved_intent"] = "guest_down_payment_policy"
        return _chat_json_response(
            request,
            start_time,
            {
                "fulfillmentText": (
                    "Down payment policy may vary per accommodation and LGU payment process.\n"
                    "This system computes your total and then routes payment through the LGU-linked billing flow."
                ),
            },
        )

    if actor.get("role") == "guest" and _is_guest_room_availability_command(message):
        request._chatbot_log_context["resolved_intent"] = "guest_room_availability_check"
        parsed_avail = _extract_params_with_confidence(message)
        avail_params = (
            parsed_avail.get("params") if isinstance(parsed_avail.get("params"), dict) else {}
        )
        avail_text = _build_guest_room_availability_summary(message, avail_params)
        return _chat_json_response(
            request,
            start_time,
            {
                "fulfillmentText": avail_text,
                "quick_replies": [
                    "Recommend a hotel in Bayawan for 2 guests",
                    "Show default hotel suggestions",
                    "View my accommodation bookings",
                ],
            },
        )

    if actor.get("role") == "guest" and _is_guest_booking_requirements_command(message):
        request._chatbot_log_context["resolved_intent"] = "guest_booking_requirements_help"
        response = {
            "fulfillmentText": (
                "Great question. Booking has 2 simple steps:\n"
                "1) Choose a room first from available hotels/inns.\n"
                "2) Send booking details: check-in date, check-out date, and number of guests.\n\n"
                "Required details:\n"
                "- Room reference (Room ID or selected option)\n"
                "- Check-in date\n"
                "- Check-out date\n"
                "- Number of guests\n\n"
                "Optional details for better matching:\n"
                "- Budget\n"
                "- Preferred location\n"
                "- Amenities (Wi-Fi, aircon, etc.)\n\n"
                "After confirmation, I will generate your booking summary and LGU payment link."
            ),
            "quick_replies": [
                "Show available hotels and inns",
                "Recommend a hotel in Bayawan for 2 guests under 1500",
                "View my accommodation bookings",
            ],
        }
        return _chat_json_response(
            request,
            start_time,
            response,
        )

    if _is_reset_command(message):
        _clear_chat_state(request)
        request._chatbot_log_context["resolved_intent"] = "reset_command"
        return _chat_json_response(
            request,
            start_time,
            {
                "fulfillmentText": (
                    "Conversation context cleared. You can start a new hotel/inn request anytime."
                )
            },
        )

    if _is_default_accommodation_suggestions_command(message):
        request._chatbot_log_context["resolved_intent"] = "get_accommodation_recommendation_init"
        if actor.get("role") != "guest":
            help_payload = _build_role_help_payload(actor)
            return _chat_json_response(request, start_time, help_payload)
        suggestions_payload = _get_default_accommodation_suggestions(limit=3)
        default_reply = None
        recommendation_trace = []
        if isinstance(suggestions_payload, tuple):
            default_reply, recommendation_trace = suggestions_payload
        else:
            default_reply = suggestions_payload

        response = {"fulfillmentText": default_reply}
        if recommendation_trace:
            response["recommendation_trace"] = recommendation_trace
            response["quick_replies"] = _merge_quick_replies(
                response.get("quick_replies") if isinstance(response.get("quick_replies"), list) else [],
                _build_recommendation_assist_quick_replies(
                    _build_accommodation_selection_cache(recommendation_trace)
                ),
                limit=4,
            )
            _inject_recommendation_context(
                response,
                params={},
                default_summary="based on currently available hotels/inns (no preference filter yet)",
            )

        _save_chat_state(
            request,
            {
                "pending_intent": "get_accommodation_recommendation",
                "params": {},
                "missing_slot": "",
                "last_accommodation_recommendations": _build_accommodation_selection_cache(recommendation_trace),
            },
        )
        return _chat_json_response(request, start_time, response)

    if _is_my_accommodation_booking_status_command(message):
        request._chatbot_log_context["resolved_intent"] = "view_my_accommodation_bookings"
        response_payload = {}
        if actor.get("role") == "owner":
            my_bookings_url = reverse("admin_app:owner_accommodation_bookings")
            booking_label = "View My Accommodation Bookings (Owner)"
            booking_reply = "I found your accommodation-owner bookings page. Click the button below to open it in a new tab."
        elif actor.get("role") == "admin":
            if _is_accommodation_bookings_page_command(message):
                my_bookings_url = reverse("admin_app:accommodation_bookings")
                booking_label = "Open Accommodation Bookings"
                booking_reply = "I found the Accommodation Bookings page. Click the button below to open it in a new tab."
            else:
                my_bookings_url = reverse("admin_app:admin_dashboard")
                booking_label = "Open Admin Dashboard"
                booking_reply = "I found your admin dashboard. Click the button below to open it in a new tab."
        elif actor.get("role") == "employee":
            my_bookings_url = reverse("admin_app:employee_dashboard")
            booking_label = "Open Employee Dashboard"
            booking_reply = "I found your employee dashboard. Click the button below to open it in a new tab."
        else:
            my_bookings_url = reverse("my_accommodation_bookings")
            booking_label = "View My Hotel/Inn Bookings"
            booking_reply = "I found your hotel/inn bookings page. Click the button below to open it in a new tab."
        if hasattr(request, "build_absolute_uri"):
            my_bookings_url = request.build_absolute_uri(my_bookings_url)
        response_payload = {
            "fulfillmentText": booking_reply,
            "billing_link": my_bookings_url,
            "billing_link_label": booking_label,
            "open_in_new_tab": True,
        }
        return _chat_json_response(
            request,
            start_time,
            response_payload,
        )

    if (
        actor.get("role") == "owner"
        and _is_owner_room_overview_command(message)
        and not _detect_owner_support_topic(message)
    ):
        request._chatbot_log_context["resolved_intent"] = "owner_room_overview"
        owner_room_reply = _build_owner_rooms_summary(user)
        return _chat_json_response(
            request,
            start_time,
            {"fulfillmentText": owner_room_reply},
        )

    if actor.get("role") == "owner":
        owner_topic = _detect_owner_support_topic(message)
        if owner_topic:
            request._chatbot_log_context["resolved_intent"] = owner_topic
            if owner_topic == "owner_password_help":
                return _chat_json_response(
                    request,
                    start_time,
                    _build_link_payload(
                        request,
                        text=(
                            "If you forgot your owner password, open the admin/owner password recovery page.\n"
                            "After reset, log in again and continue in Owner Hub."
                        ),
                        route_name="admin_app:forgot_password",
                        label="Open Password Recovery",
                    ),
                )
            if owner_topic in {"owner_register_listing", "owner_listing_requirements"}:
                return _chat_json_response(
                    request,
                    start_time,
                    _build_link_payload(
                        request,
                        text=(
                            "To register your accommodation, open the registration form and submit complete business/profile details.\n"
                            "Required fields are validated directly in the form before submission."
                        ),
                        route_name="admin_app:accommodation_register",
                        label="Open Accommodation Registration",
                    ),
                )
            if owner_topic == "owner_listing_update":
                return _chat_json_response(
                    request,
                    start_time,
                    _build_link_payload(
                        request,
                        text=(
                            "Yes, you can update your listing details after registration.\n"
                            "Use Owner Hub to edit company information and listing details."
                        ),
                        route_name="admin_app:owner_hub",
                        label="Open Owner Hub",
                    ),
                )
            if owner_topic in {
                "owner_add_room",
                "owner_update_room_price",
                "owner_update_room_capacity",
                "owner_mark_room_unavailable",
                "owner_edit_room_amenities",
                "owner_update_availability",
                "owner_block_dates",
                "owner_room_still_available_issue",
                "owner_room_update_issue",
            }:
                detail_map = {
                    "owner_add_room": "To add a new room, open Manage Rooms and click Add Room.",
                    "owner_update_room_price": "To update room price, open Manage Rooms then edit the room pricing fields.",
                    "owner_update_room_capacity": "To update room capacity, open Manage Rooms then edit the room guest capacity.",
                    "owner_mark_room_unavailable": "To mark a room unavailable or under maintenance, open Manage Rooms and set room status to UNAVAILABLE.",
                    "owner_edit_room_amenities": "Room amenities are managed through your room/listing details in Owner Hub and Manage Rooms.",
                    "owner_update_availability": "To update room availability, edit room status/availability values in Manage Rooms.",
                    "owner_block_dates": (
                        "Date-blocking per room is handled via room availability/status workflow in this version.\n"
                        "Set room as UNAVAILABLE for maintenance windows and reopen after."
                    ),
                    "owner_room_still_available_issue": (
                        "If a room still appears available, verify room status, current availability, and accepted listing status.\n"
                        "Then refresh Manage Rooms and save updates."
                    ),
                    "owner_room_update_issue": (
                        "If room update is failing, verify your listing is accepted and the room belongs to your account.\n"
                        "Then retry from Manage Rooms."
                    ),
                }
                short_owner_texts = {"add room", "update price", "view bookings", "listing not showing", "room unavailable how"}
                payload = _build_owner_manage_rooms_link_payload(
                    request,
                    text=str(detail_map.get(owner_topic) or "Open Manage Rooms to continue."),
                    label="Open Manage Rooms",
                )
                if str(message or "").strip().lower() in short_owner_texts:
                    payload["needs_clarification"] = True
                    payload["missing_slot"] = "owner_room_action"
                    payload["quick_replies"] = [
                        "Update room price",
                        "Change room capacity",
                        "Mark room unavailable",
                        "View bookings",
                    ]
                return _chat_json_response(request, start_time, payload)
            if owner_topic in {
                "owner_view_bookings",
                "owner_booking_guest_details",
                "owner_confirm_reservation",
                "owner_pending_reservations",
                "owner_cancelled_reservations",
            }:
                booking_text = _build_owner_booking_payment_summary(user)
                if owner_topic == "owner_confirm_reservation":
                    booking_text = (
                        f"{booking_text}\n\n"
                        "To confirm a reservation, open Owner Bookings and update the booking status to Confirmed."
                    )
                if owner_topic == "owner_booking_guest_details":
                    booking_text = (
                        f"{booking_text}\n\n"
                        "Guest reservation details are available inside each booking record."
                    )
                payload = _build_link_payload(
                    request,
                    text=booking_text,
                    route_name="admin_app:owner_accommodation_bookings",
                    label="Open Owner Bookings",
                )
                if str(message or "").strip().lower() == "view bookings":
                    payload["needs_clarification"] = True
                    payload["missing_slot"] = "owner_booking_filter"
                    payload["quick_replies"] = [
                        "Show pending reservations",
                        "Show payment status",
                        "Show cancelled reservations",
                    ]
                return _chat_json_response(
                    request,
                    start_time,
                    payload,
                )
            if owner_topic in {"owner_payment_status", "owner_billing_details", "owner_transactions"}:
                billing_text = (
                    _build_owner_billing_details_summary(user)
                    if owner_topic != "owner_payment_status"
                    else _build_owner_booking_payment_summary(user)
                )
                return _chat_json_response(
                    request,
                    start_time,
                    _build_link_payload(
                        request,
                        text=billing_text,
                        route_name="admin_app:owner_accommodation_bookings",
                        label="Open Booking Transactions",
                    ),
                )
            if owner_topic == "owner_direct_booking_flow":
                return _chat_json_response(
                    request,
                    start_time,
                    _build_link_payload(
                        request,
                        text=_build_owner_direct_booking_flow_summary(user),
                        route_name="admin_app:owner_hub",
                        label="Open Owner Hub",
                    ),
                )
            if owner_topic == "owner_listing_visibility":
                payload = _build_link_payload(
                    request,
                    text=_build_owner_listing_visibility_diagnostic(user),
                    route_name="admin_app:owner_hub",
                    label="Open Owner Hub",
                )
                if str(message or "").strip().lower() == "listing not showing":
                    payload["needs_clarification"] = True
                    payload["missing_slot"] = "listing_status_context"
                    payload["quick_replies"] = [
                        "Show my accommodations",
                        "Show my rooms",
                        "Open Owner Hub",
                    ]
                return _chat_json_response(
                    request,
                    start_time,
                    payload,
                )

    chat_state = _load_chat_state(request)
    pending_booking = (
        chat_state.get("pending_booking") if isinstance(chat_state.get("pending_booking"), dict) else {}
    )
    pending_booking_params = (
        pending_booking.get("params") if isinstance(pending_booking.get("params"), dict) else {}
    )
    pending_booking_created_at = _to_int(pending_booking.get("created_at"), default=0)
    state_intent_hint = str(chat_state.get("pending_intent") or "").strip().lower()
    state_params_hint = chat_state.get("params") if isinstance(chat_state.get("params"), dict) else {}
    cached_accommodation_rows = (
        chat_state.get("last_accommodation_recommendations")
        if isinstance(chat_state.get("last_accommodation_recommendations"), list)
        else []
    )

    if actor.get("role") == "guest" and _is_accommodation_detail_query(message):
        request._chatbot_log_context["resolved_intent"] = "get_accommodation_details"
        detail_payload = _build_guest_room_detail_payload(message, cached_accommodation_rows)
        return _chat_json_response(request, start_time, detail_payload)

    if actor.get("role") == "guest" and _is_forget_preferences_command(message):
        _clear_saved_chat_preferences(request)
        return _chat_json_response(
            request,
            start_time,
            {
                "fulfillmentText": (
                    "Done. I cleared your saved accommodation preferences for this account."
                ),
                "quick_replies": [
                    "Show default hotel suggestions",
                    "Recommend a hotel in Bayawan",
                ],
            },
        )

    if actor.get("role") == "guest" and _is_remember_preferences_command(message):
        parsed_pref = _extract_params_with_confidence(message)
        parsed_pref_params = (
            parsed_pref.get("params") if isinstance(parsed_pref.get("params"), dict) else {}
        )
        merged_pref_params = dict(state_params_hint)
        merged_pref_params.update(parsed_pref_params)
        payload = _extract_memory_preference_payload(merged_pref_params)
        if not payload:
            return _chat_json_response(
                request,
                start_time,
                {
                    "fulfillmentText": (
                        "I can remember your preferences, but I need at least one detail first "
                        "(type, location, budget, guests, or preference tags)."
                    )
                },
            )
        _save_saved_chat_preferences(request, payload)
        summary_parts = []
        if str(payload.get("company_type") or "").strip():
            summary_parts.append(f"type: {payload.get('company_type')}")
        if str(payload.get("location") or "").strip():
            summary_parts.append(f"location: {payload.get('location')}")
        if _to_int(payload.get("budget"), default=0) > 0:
            summary_parts.append(f"budget: PHP {_to_int(payload.get('budget'), default=0)}")
        if _to_int(payload.get("guests"), default=0) > 0:
            summary_parts.append(f"guests: {_to_int(payload.get('guests'), default=0)}")
        if isinstance(payload.get("preference_tags"), list) and payload.get("preference_tags"):
            summary_parts.append(f"preferences: {', '.join(str(v) for v in payload.get('preference_tags')[:4])}")
        summary_text = "; ".join(summary_parts) if summary_parts else "basic preferences saved"
        return _chat_json_response(
            request,
            start_time,
            {
                "fulfillmentText": f"Saved. I will reuse these defaults in your next requests: {summary_text}.",
                "quick_replies": [
                    "Show default hotel suggestions",
                    "Recommend a hotel",
                    "Forget my preferences",
                ],
            },
        )

    if pending_booking_params:
        request._chatbot_log_context["resolved_intent"] = "book_accommodation_confirmation"
        request._chatbot_log_context["resolved_params"] = pending_booking_params
        now_epoch = int(time.time())
        if (
            pending_booking_created_at > 0
            and (now_epoch - pending_booking_created_at) > _PENDING_BOOKING_TTL_SECONDS
        ):
            next_state = dict(chat_state)
            next_state.pop("pending_booking", None)
            _save_chat_state(request, next_state)
            return _chat_json_response(
                request,
                start_time,
                {
                    "fulfillmentText": (
                        "Your pending booking confirmation expired after 10 minutes. "
                        "Please send your booking details again."
                    )
                },
            )

        if _is_booking_confirmation_accept(message):
            booking_result = _book_accommodation_from_chat(request, pending_booking_params, commit=True)
            _safe_log_chat_booking_linkage(request, message, pending_booking_params, booking_result)
            next_state = dict(chat_state)
            next_state.pop("pending_booking", None)
            _save_chat_state(request, next_state)
            request._chatbot_log_context["provenance"] = {
                "booking_result_status": booking_result.get("booking_status", ""),
                "booking_id": booking_result.get("booking_id"),
            }
            response = {"fulfillmentText": booking_result["reply"]}
            if booking_result.get("billing_link"):
                response["billing_link"] = booking_result["billing_link"]
                if booking_result.get("billing_link_label"):
                    response["billing_link_label"] = booking_result["billing_link_label"]
            if booking_result.get("booking_id") is not None:
                response["booking_id"] = booking_result["booking_id"]
            if booking_result.get("receipt_text"):
                response["receipt_text"] = booking_result["receipt_text"]
                response["receipt_filename"] = booking_result.get("receipt_filename") or "ibayaw_booking_receipt.png"
            if isinstance(booking_result.get("quick_replies"), list):
                response["quick_replies"] = _sanitize_quick_replies(
                    booking_result.get("quick_replies"),
                    limit=4,
                )
            if booking_result.get("booking_id"):
                response["show_feedback_prompt"] = True
            return _chat_json_response(request, start_time, response)

        if _is_booking_confirmation_decline(message):
            next_state = dict(chat_state)
            next_state.pop("pending_booking", None)
            _save_chat_state(request, next_state)
            return _chat_json_response(
                request,
                start_time,
                {"fulfillmentText": "Okay, I cancelled that draft booking. You can send new booking details anytime."},
            )

        return _chat_json_response(
            request,
            start_time,
            {
                "fulfillmentText": "Please reply YES to confirm the booking, or NO to cancel it.",
                "quick_replies": ["Yes", "No"],
            },
        )

    # Compact numeric selection support for accommodation recommendations:
    # if the user replies with just "1", "2", "3", map it to the last shown room.
    if actor.get("role") == "guest" and state_intent_hint in ("get_accommodation_recommendation", "gethotelrecommendation"):
        cached_rows = (
            chat_state.get("last_accommodation_recommendations")
            if isinstance(chat_state.get("last_accommodation_recommendations"), list)
            else []
        )
        why_option_index = _extract_why_option_index(message)
        if why_option_index > 0 and cached_rows:
            why_text = _build_why_option_text(cached_rows, why_option_index)
            if why_text:
                _safe_log_chat_step_event(
                    request,
                    event_type="click",
                    item_ref=f"chat:why_option:{why_option_index}",
                )
                return _chat_json_response(
                    request,
                    start_time,
                    {
                        "fulfillmentText": why_text,
                        "quick_replies": _merge_quick_replies(
                            _build_recommendation_assist_quick_replies(cached_rows),
                            [{"label": f"Book Option {why_option_index}", "value": str(why_option_index)}],
                            limit=4,
                        ),
                    },
                )

        compare_top_n = _extract_compare_top_n(message, default=3)
        if compare_top_n > 0 and cached_rows:
            compare_text = _build_compare_options_text(cached_rows, compare_top_n)
            if compare_text:
                _safe_log_chat_step_event(
                    request,
                    event_type="click",
                    item_ref=f"chat:compare_top:{compare_top_n}",
                )
                return _chat_json_response(
                    request,
                    start_time,
                    {
                        "fulfillmentText": compare_text,
                        "quick_replies": _sanitize_quick_replies(
                            _build_post_compare_quick_replies(cached_rows, compare_top_n),
                            limit=4,
                        ),
                    },
                )

        option_index = _extract_numeric_option_index(message)
        if option_index > 0 and cached_rows:
            selected_room_id = _resolve_accommodation_room_from_selection(cached_rows, option_index)
            if selected_room_id > 0:
                selected_params = dict(state_params_hint)
                selected_params["room_id"] = selected_room_id
                selected_params["_personalization_opt_out"] = bool(
                    selected_params.get("_personalization_opt_out", False)
                )
                next_state = dict(chat_state)
                next_state["pending_intent"] = "book_accommodation"
                next_state["params"] = selected_params
                next_state["missing_slot"] = ""
                _save_chat_state(request, next_state)

                booking_result = _book_accommodation_from_chat(request, selected_params, commit=False)
                response = {"fulfillmentText": booking_result.get("reply", "")}
                if booking_result.get("billing_link"):
                    response["billing_link"] = booking_result["billing_link"]
                    if booking_result.get("billing_link_label"):
                        response["billing_link_label"] = booking_result["billing_link_label"]
                if booking_result.get("booking_id") is not None:
                    response["booking_id"] = booking_result["booking_id"]
                if booking_result.get("receipt_text"):
                    response["receipt_text"] = booking_result["receipt_text"]
                    response["receipt_filename"] = booking_result.get("receipt_filename") or "ibayaw_booking_receipt.png"
                if isinstance(booking_result.get("quick_replies"), list):
                    response["quick_replies"] = _sanitize_quick_replies(
                        booking_result.get("quick_replies"),
                        limit=4,
                    )
                return _chat_json_response(request, start_time, response)

    pending_budget_offer = (
        chat_state.get("pending_budget_offer")
        if isinstance(chat_state.get("pending_budget_offer"), dict)
        else {}
    )
    offered_budget = _to_int(pending_budget_offer.get("suggested_budget"), default=0)
    if pending_budget_offer and state_intent_hint in ("get_accommodation_recommendation", "gethotelrecommendation"):
        request._chatbot_log_context["resolved_intent"] = "get_accommodation_recommendation"
        parsed_offer_reply = _extract_params_with_confidence(message)
        reply_params = parsed_offer_reply.get("params") if isinstance(parsed_offer_reply.get("params"), dict) else {}
        custom_budget = _to_int(reply_params.get("budget"), default=0)

        if _is_personalization_accept_message(message):
            if offered_budget > 0:
                accepted_params = dict(state_params_hint)
                accepted_params["budget"] = offered_budget
                reply, logged_recommended_items, accommodation_meta = _safe_get_accommodation_recommendations(accepted_params)
                next_state = dict(chat_state)
                next_state["pending_intent"] = "get_accommodation_recommendation"
                next_state["params"] = accepted_params
                next_state["missing_slot"] = ""
                next_state.pop("pending_budget_offer", None)
                next_state["last_accommodation_recommendations"] = _build_accommodation_selection_cache(
                    logged_recommended_items
                )
                if (
                    isinstance(accommodation_meta, dict)
                    and "budget_too_low" in (accommodation_meta.get("no_match_reasons") or [])
                    and _to_int(accommodation_meta.get("suggested_budget_min"), default=0) > 0
                ):
                    next_state["pending_budget_offer"] = {
                        "suggested_budget": _to_int(accommodation_meta.get("suggested_budget_min"), default=0)
                    }
                _save_chat_state(request, next_state)

                response = {"fulfillmentText": reply}
                if logged_recommended_items:
                    response["recommendation_trace"] = logged_recommended_items
                if isinstance(accommodation_meta, dict):
                    if isinstance(accommodation_meta.get("no_match_reasons"), list):
                        response["no_match_reasons"] = [str(v) for v in accommodation_meta.get("no_match_reasons")]
                    if accommodation_meta.get("suggested_budget_min") not in (None, ""):
                        response["suggested_budget_min"] = accommodation_meta.get("suggested_budget_min")
                    if accommodation_meta.get("fallback_applied"):
                        response["recommendation_fallback"] = str(accommodation_meta.get("fallback_applied"))
                    if isinstance(accommodation_meta.get("quick_replies"), list):
                        response["quick_replies"] = _sanitize_quick_replies(
                            accommodation_meta.get("quick_replies"),
                            limit=4,
                        )
                if logged_recommended_items:
                    response["quick_replies"] = _merge_quick_replies(
                        response.get("quick_replies") if isinstance(response.get("quick_replies"), list) else [],
                        _build_recommendation_assist_quick_replies(
                            _build_accommodation_selection_cache(logged_recommended_items)
                        ),
                        limit=4,
                    )
                _inject_recommendation_context(response, accepted_params)
                return _chat_json_response(request, start_time, response)

        if _is_personalization_decline_message(message):
            next_state = dict(chat_state)
            next_state.pop("pending_budget_offer", None)
            _save_chat_state(request, next_state)
            return _chat_json_response(
                request,
                start_time,
                {"fulfillmentText": "Okay, no problem. Please share your preferred budget per night in PHP."},
            )

        if custom_budget > 0:
            accepted_params = dict(state_params_hint)
            accepted_params["budget"] = custom_budget
            reply, logged_recommended_items, accommodation_meta = _safe_get_accommodation_recommendations(accepted_params)
            next_state = dict(chat_state)
            next_state["pending_intent"] = "get_accommodation_recommendation"
            next_state["params"] = accepted_params
            next_state["missing_slot"] = ""
            next_state.pop("pending_budget_offer", None)
            next_state["last_accommodation_recommendations"] = _build_accommodation_selection_cache(
                logged_recommended_items
            )
            if (
                isinstance(accommodation_meta, dict)
                and "budget_too_low" in (accommodation_meta.get("no_match_reasons") or [])
                and _to_int(accommodation_meta.get("suggested_budget_min"), default=0) > 0
            ):
                next_state["pending_budget_offer"] = {
                    "suggested_budget": _to_int(accommodation_meta.get("suggested_budget_min"), default=0)
                }
            _save_chat_state(request, next_state)

            response = {"fulfillmentText": reply}
            if logged_recommended_items:
                response["recommendation_trace"] = logged_recommended_items
            if isinstance(accommodation_meta, dict):
                if isinstance(accommodation_meta.get("no_match_reasons"), list):
                    response["no_match_reasons"] = [str(v) for v in accommodation_meta.get("no_match_reasons")]
                if accommodation_meta.get("suggested_budget_min") not in (None, ""):
                    response["suggested_budget_min"] = accommodation_meta.get("suggested_budget_min")
                if accommodation_meta.get("fallback_applied"):
                    response["recommendation_fallback"] = str(accommodation_meta.get("fallback_applied"))
                if isinstance(accommodation_meta.get("quick_replies"), list):
                    response["quick_replies"] = _sanitize_quick_replies(
                        accommodation_meta.get("quick_replies"),
                        limit=4,
                    )
            if logged_recommended_items:
                response["quick_replies"] = _merge_quick_replies(
                    response.get("quick_replies") if isinstance(response.get("quick_replies"), list) else [],
                    _build_recommendation_assist_quick_replies(
                        _build_accommodation_selection_cache(logged_recommended_items)
                    ),
                    limit=4,
                )
            _inject_recommendation_context(response, accepted_params)
            return _chat_json_response(request, start_time, response)

        # Preserve non-budget slot updates (like new dates/guests) while waiting for
        # explicit budget confirmation so the user doesn't need to retype them.
        non_budget_updates = {}
        for key in (
            "check_in",
            "check_out",
            "nights",
            "guests",
            "adults",
            "children",
            "location",
            "company_type",
            "preference_tags",
            "prefer_low_price",
            "amenities",
            "broaden_location",
            "broaden_company_type",
        ):
            if key in reply_params and reply_params.get(key) not in ("", None):
                non_budget_updates[key] = reply_params.get(key)
        if non_budget_updates:
            updated_state_params = dict(state_params_hint)
            updated_state_params.update(non_budget_updates)
            next_state = dict(chat_state)
            next_state["params"] = updated_state_params
            _save_chat_state(request, next_state)

        return _chat_json_response(
            request,
            start_time,
            {
                "fulfillmentText": (
                    f"I saved your dates/filters. I still need budget confirmation first.\n"
                    f"Reply YES to use PHP {offered_budget} budget, "
                    "reply NO to skip, or send a different budget amount."
                )
            },
        )

    parsed = _classify_intent_and_extract_params(message)
    intent = str(parsed["intent"]).strip().lower()
    params = parsed["params"] if isinstance(parsed["params"], dict) else {}
    parse_confidence = float(parsed.get("confidence", 1.0) or 1.0)
    parse_source = str(parsed.get("source") or "").strip().lower()
    needs_parser_clarification = bool(parsed.get("needs_clarification"))
    parser_clarification_question = str(parsed.get("clarification_question") or "").strip()
    parser_clarification_field = str(parsed.get("clarification_field") or "").strip()
    parser_clarification_options = (
        parsed.get("clarification_options")
        if isinstance(parsed.get("clarification_options"), list)
        else []
    )
    intent_classifier = parsed.get("intent_classifier") if isinstance(parsed.get("intent_classifier"), dict) else {}
    request._chatbot_log_context["resolved_intent"] = intent
    request._chatbot_log_context["resolved_params"] = params
    request._chatbot_log_context["intent_classifier"] = intent_classifier
    request._chatbot_log_context["fallback_used"] = str(parsed.get("source") or "") in (
        "heuristic_intent_fallback",
        "text_cnn_unavailable",
        "text_cnn_low_confidence",
        "text_cnn_incompatible_label_space",
    )
    state_intent = str(chat_state.get("pending_intent") or "").strip().lower()
    state_params = chat_state.get("params") if isinstance(chat_state.get("params"), dict) else {}
    state_missing_slot = str(chat_state.get("missing_slot") or "").strip().lower()
    state_default_offer = (
        chat_state.get("default_offer") if isinstance(chat_state.get("default_offer"), dict) else {}
    )
    state_offer_defaults = (
        state_default_offer.get("defaults")
        if isinstance(state_default_offer.get("defaults"), dict)
        else {}
    )
    state_offer_prompt = str(state_default_offer.get("prompt") or "").strip()
    state_skip_default_offer = bool(chat_state.get("skip_default_offer")) or bool(
        state_params.get("_personalization_opt_out", False)
    )

    accommodation_intents = ("get_accommodation_recommendation", "gethotelrecommendation")
    booking_intents = ("book_accommodation", "bookhotel", "book_hotel", "reserve_accommodation")
    continuation_intents = accommodation_intents + booking_intents + ("get_recommendation",)
    continuing_accommodation_flow = (
        state_intent in accommodation_intents
        and (
            intent in accommodation_intents
            or (
                intent == "get_recommendation"
                and not _looks_like_tour_request(message)
                and (
                    _looks_like_slot_update(params)
                    or state_missing_slot in ("company_type", "location", "budget", "guests", "stay_details", "accommodation_details")
                )
            )
        )
    )
    if continuing_accommodation_flow:
        intent = state_intent

    continuing_booking_flow = (
        state_intent in booking_intents
        and (
            intent in booking_intents
            or _looks_like_slot_update(params)
            or state_missing_slot in ("guests", "stay_details", "accommodation_details", "date_range")
        )
    )
    if continuing_booking_flow:
        intent = state_intent

    if (
        intent == "get_recommendation"
        and not continuing_accommodation_flow
        and not _looks_like_tour_request(message)
        and _is_out_of_scope_message(message)
    ):
        intent = "out_of_scope"

    compact_num_match = re.fullmatch(r"\s*(\d+)\s*", message)
    if compact_num_match and state_missing_slot in ("guests", "stay_details", "accommodation_details"):
        # Avoid treating compact numeric replies as budget when we are collecting
        # guests or stay details in a running accommodation flow.
        existing_budget = _to_int(state_params.get("budget"), default=0)
        parsed_budget = _to_int(params.get("budget"), default=0)
        if existing_budget > 0 and parsed_budget > 0 and parsed_budget != existing_budget:
            params.pop("budget", None)

    merged_params = dict(state_params)
    merged_params.update(params)
    params = merged_params
    guest_count_update_note = ""
    previous_guests = _to_int(state_params.get("guests"), default=0)
    current_guests = _to_int(params.get("guests"), default=0)
    if (
        state_intent in accommodation_intents + booking_intents
        and current_guests > 0
        and _message_mentions_guest_count(message)
    ):
        if previous_guests <= 0:
            guest_count_update_note = f"Noted. I updated your guest count to {current_guests}."
        elif previous_guests != current_guests:
            guest_count_update_note = (
                f"Noted. I updated your guest count from {previous_guests} to {current_guests}."
            )

    # User explicitly removed budget constraint; keep accommodation flow and clear budget state.
    if _to_bool(params.get("clear_budget"), default=False):
        params["budget"] = 0
        params.pop("pending_budget_offer", None)

    # If a new plain location was provided without a resolved map anchor, clear prior anchor context.
    if _to_bool(params.get("clear_location_anchor"), default=False) and not str(params.get("location_anchor") or "").strip():
        params.pop("location_anchor", None)
        params.pop("location_anchor_source", None)
        params.pop("location_scope_note", None)
    if actor.get("role") == "guest":
        saved_prefs = _load_saved_chat_preferences(request)
        params = _apply_saved_preferences_to_params(params, saved_prefs)
        if (
            intent == "get_recommendation"
            and isinstance(saved_prefs, dict)
            and bool(saved_prefs)
            and not _looks_like_tour_request(message)
            and not _is_out_of_scope_message(message)
            and not _contains_any_phrase(
                message,
                (
                    "tourism information",
                    "tourist spot",
                    "attraction",
                    "landmark",
                    "history",
                ),
            )
        ):
            intent = "get_accommodation_recommendation"
    lowered_message = str(message or "").strip().lower()
    if (
        state_intent in accommodation_intents
        and any(token in lowered_message for token in ("cheaper", "lower price", "less expensive", "mas mura", "barato"))
    ):
        existing_budget = _to_int(state_params.get("budget"), default=0)
        if existing_budget > 0:
            params["budget"] = max(500, int(existing_budget * 0.8))
        params["prefer_low_price"] = True
    if (
        state_intent in accommodation_intents + booking_intents
        and _to_int(state_params.get("guests"), default=0) > 0
        and _to_int(params.get("guests"), default=0) == 1
        and not _message_mentions_guest_count(message)
    ):
        # Keep previously provided guest count; don't overwrite with parser fallback defaults.
        params["guests"] = _to_int(state_params.get("guests"), default=0)
    request._chatbot_log_context["resolved_intent"] = intent
    request._chatbot_log_context["resolved_params"] = params

    # Context-aware slot repair for short follow-up replies while collecting
    # accommodation details (e.g., user replies with just "suba" or "inn").
    compact_text = str(message or "").strip()
    compact_lower = compact_text.lower()
    if state_intent in accommodation_intents:
        if state_missing_slot == "company_type" and "company_type" not in params:
            if compact_lower in ("hotel", "inn", "either", "hotel or inn"):
                params["company_type"] = "either" if compact_lower == "hotel or inn" else compact_lower
                intent = state_intent
        if state_missing_slot == "location" and not str(params.get("location") or "").strip():
            simple_location = re.fullmatch(r"[a-zA-Z][a-zA-Z\\s\\-]{1,60}", compact_text)
            if simple_location and compact_lower not in ("yes", "no", "hotel", "inn", "either"):
                params["location"] = compact_text
                intent = state_intent

    # Context-aware compact answers:
    # - If we are explicitly waiting for guests, allow plain integer replies like "2".
    # - If we are waiting for stay details, allow plain integer replies as nights.
    if compact_num_match:
        compact_num = _to_int(compact_num_match.group(1), default=0)
        if state_missing_slot == "guests" and _to_int(params.get("guests"), default=0) <= 0 and compact_num > 0:
            params["guests"] = compact_num
            params.pop("budget", None)
        elif state_missing_slot == "budget" and _to_int(params.get("budget"), default=0) <= 0 and compact_num > 0:
            params["budget"] = compact_num
        elif (
            state_missing_slot == "stay_details"
            and not _has_stay_details(params)
            and compact_num > 0
        ):
            params["nights"] = compact_num
        elif (
            state_missing_slot == "accommodation_details"
            and compact_num > 0
            and _to_int(params.get("guests"), default=0) <= 0
            and _to_int(params.get("budget"), default=0) <= 0
        ):
            params["guests"] = compact_num

    if state_intent in accommodation_intents and state_offer_defaults and not state_skip_default_offer:
        if _is_personalization_decline_message(message):
            declined_params = dict(params)
            declined_params["_personalization_opt_out"] = True
            missing_slot, question = _next_accommodation_clarifying_question(declined_params)
            next_state = {
                "pending_intent": state_intent,
                "params": declined_params,
                "skip_default_offer": True,
            }
            if missing_slot:
                next_state["missing_slot"] = missing_slot
            company_type = str(declined_params.get("company_type") or "").strip().lower()
            location = str(declined_params.get("location") or "").strip()
            preference_tags = (
                declined_params.get("preference_tags")
                if isinstance(declined_params.get("preference_tags"), list)
                else []
            )
            preview_key = f"{company_type}|{location.lower()}|{','.join(sorted(str(tag) for tag in preference_tags))}"
            if (
                missing_slot in ("stay_details", "budget", "accommodation_details")
                and state_missing_slot != "location"
                and company_type in ("hotel", "inn", "either")
                and (location or preference_tags)
                and str(chat_state.get("location_preview_for") or "") != preview_key
            ):
                preview_params = dict(declined_params)
                preview_reply, preview_items, preview_meta = _safe_get_accommodation_recommendations(preview_params)
                if (
                    "top hotel/inn recommendations" not in str(preview_reply or "").lower()
                    and company_type in ("hotel", "inn")
                ):
                    preview_any_type_params = dict(preview_params)
                    preview_any_type_params["company_type"] = "either"
                    preview_reply, preview_items, preview_meta = _safe_get_accommodation_recommendations(
                        preview_any_type_params
                    )
                if str(preview_reply or "").strip():
                    next_state["location_preview_for"] = preview_key
                    next_state["last_accommodation_recommendations"] = _build_accommodation_selection_cache(preview_items)
                    _save_chat_state(request, next_state)
                    prompt_suffix = (
                        f"To continue booking, {question}"
                        if question and "top hotel/inn recommendations" in str(preview_reply or "").lower()
                        else ""
                    )
                    response_text = str(preview_reply or "").strip()
                    if prompt_suffix:
                        response_text = f"{response_text}\n\n{prompt_suffix}"
                    response = {"fulfillmentText": response_text}
                    if preview_items:
                        response["recommendation_trace"] = preview_items
                    if isinstance(preview_meta, dict) and preview_meta.get("fallback_applied"):
                        response["recommendation_fallback"] = str(preview_meta.get("fallback_applied"))
                    if isinstance(preview_meta, dict) and isinstance(preview_meta.get("quick_replies"), list):
                        response["quick_replies"] = _sanitize_quick_replies(
                            preview_meta.get("quick_replies"),
                            limit=4,
                        )
                    if preview_items:
                        response["quick_replies"] = _merge_quick_replies(
                            response.get("quick_replies") if isinstance(response.get("quick_replies"), list) else [],
                            _build_recommendation_assist_quick_replies(
                                _build_accommodation_selection_cache(preview_items)
                            ),
                            limit=4,
                        )
                    return _chat_json_response(request, start_time, response)
            _save_chat_state(request, next_state)
            if question:
                return _chat_json_response(
                    request,
                    start_time,
                    {
                        "fulfillmentText": question,
                        "quick_replies": _slot_quick_replies(missing_slot),
                        "needs_clarification": True,
                        "missing_slot": missing_slot,
                    },
                )
        elif _is_personalization_accept_message(message):
            accepted_params = dict(params)
            accepted_params.pop("_personalization_opt_out", None)
            for key, value in state_offer_defaults.items():
                if accepted_params.get(key) in ("", None):
                    accepted_params[key] = value
            params = accepted_params
            intent = state_intent
            _save_chat_state(
                request,
                {
                    "pending_intent": state_intent,
                    "params": params,
                    "skip_default_offer": False,
                },
            )
        elif not _looks_like_slot_update(params):
            return _chat_json_response(
                request,
                start_time,
                {"fulfillmentText": state_offer_prompt or PERSONALIZATION_PROMPT_DEFAULT},
            )

    if (
        not needs_parser_clarification
        and parse_source == "heuristic_intent_fallback"
        and parse_confidence < 0.45
        and intent == "get_recommendation"
        and not _looks_like_slot_update(params)
        and not _looks_like_tour_request(message)
    ):
        _save_chat_state(
            request,
            {
                "pending_intent": "get_accommodation_recommendation",
                "params": params,
                "missing_slot": "clarification",
            },
        )
        return _chat_json_response(
            request,
            start_time,
            {
                "fulfillmentText": (
                    "I want to make sure I understood your request correctly.\n"
                    "Do you want hotel/inn recommendations, tour recommendations, or tourism information?"
                ),
                "quick_replies": [
                    {"label": "Hotel/Inn Recommendations", "value": "recommend a hotel or inn"},
                    {"label": "Tour Recommendations", "value": "recommend a tour"},
                    {"label": "Tourism Information", "value": "show tourism information"},
                ],
                "needs_clarification": True,
                "confidence": parse_confidence,
            },
        )

    if needs_parser_clarification:
        if not parser_clarification_question:
            parser_clarification_question = "Could you clarify that so I can continue?"
        clarification_quick_replies = _merge_quick_replies(
            _sanitize_quick_replies(parser_clarification_options, limit=4),
            _slot_quick_replies(parser_clarification_field),
            limit=4,
        )
        _save_chat_state(
            request,
            {
                "pending_intent": state_intent or intent or "get_accommodation_recommendation",
                "params": params,
                "missing_slot": parser_clarification_field or "clarification",
            },
        )
        return _chat_json_response(
            request,
            start_time,
            {
                "fulfillmentText": parser_clarification_question,
                "quick_replies": clarification_quick_replies,
                "needs_clarification": True,
                "confidence": parse_confidence,
            },
        )
    cnn_prediction = None
    cnn_error = None
    logged_recommended_items = []
    accommodation_meta = None
    billing_actions = {}
    out_of_scope_quick_replies = []

    if init_suggestions:
        if actor.get("role") != "guest":
            request._chatbot_log_context["resolved_intent"] = "role_help_init"
            help_payload = _build_role_help_payload(actor)
            return _chat_json_response(request, start_time, help_payload)
        suggestions_payload = _get_default_accommodation_suggestions(limit=3)
        default_reply = None
        recommendation_trace = []
        if isinstance(suggestions_payload, tuple):
            default_reply, recommendation_trace = suggestions_payload
        else:
            default_reply = suggestions_payload
        response = {"fulfillmentText": default_reply}
        if recommendation_trace:
            response["recommendation_trace"] = recommendation_trace
        _save_chat_state(
            request,
            {
                "pending_intent": "get_accommodation_recommendation",
                "params": {},
                "missing_slot": "",
                "last_accommodation_recommendations": _build_accommodation_selection_cache(recommendation_trace),
            },
        )
        return _chat_json_response(request, start_time, response)

    if intent in ("get_recommendation", "gettourrecommendation"):
        if actor.get("role") in {"owner", "admin", "employee"}:
            _clear_chat_state(request)
            role_help = _build_role_help_payload(actor)
            role = str(actor.get("role") or "").strip().lower()
            if role == "admin":
                role_clarifier = (
                    "I can help with reports, approvals, system activity, and monitoring. "
                    "Could you clarify what you want to check?"
                )
            elif role == "owner":
                role_clarifier = (
                    "I can help you manage your accommodations, rooms, bookings, and reports. "
                    "What would you like to do?"
                )
            else:
                role_clarifier = (
                    "I can assist with tourism records, bookings, and monitoring. "
                    "Please specify your request."
                )
            if _looks_like_tour_request(message):
                reply = (
                    role_clarifier
                    + "\nTour recommendation cards are guest-focused in chat. "
                    "Please use your dashboard workflows for staff operations."
                )
            else:
                reply = (
                    role_clarifier + "\n" + str(role_help.get("fulfillmentText") or "")
                )
            out_of_scope_quick_replies = _sanitize_quick_replies(
                role_help.get("quick_replies") if isinstance(role_help.get("quick_replies"), list) else [],
                limit=4,
            )
            logged_recommended_items = []
        else:
            accommodation_like_request = bool(
                re.search(r"\b(hotel|inn|room|accommodation|stay)\b", str(message or "").lower())
            ) and not _looks_like_tour_request(message)
            if accommodation_like_request and _to_int(params.get("guests"), default=0) <= 0:
                missing_slot = "guests"
                question = _build_dynamic_accommodation_slot_question(missing_slot, params)
                _save_chat_state(
                    request,
                    {
                        "pending_intent": "get_accommodation_recommendation",
                        "params": params,
                        "missing_slot": missing_slot,
                    },
                )
                return _chat_json_response(
                    request,
                    start_time,
                    {
                        "fulfillmentText": question,
                        "quick_replies": _slot_quick_replies(missing_slot),
                        "needs_clarification": True,
                        "missing_slot": missing_slot,
                    },
                )
            _clear_chat_state(request)
            _safe_log_recommendation_event(request, intent)
            try:
                reply, logged_recommended_items = _get_recommendations(params)
            except Exception:
                reply = (
                    "I had trouble loading tour recommendations just now. "
                    "Please try again, or ask for tourism information while I reset."
                )
                logged_recommended_items = []
            rec_sched_ids = []
            for item in logged_recommended_items:
                if not isinstance(item, dict):
                    continue
                meta = item.get("meta") if isinstance(item.get("meta"), dict) else {}
                sched_id = str(meta.get("sched_id") or "").strip()
                if sched_id:
                    rec_sched_ids.append(sched_id)
            if rec_sched_ids:
                _save_chat_state(
                    request,
                    {
                        "last_tour_recommendation_sched_ids": rec_sched_ids[:5],
                    },
                )
    elif intent in ("get_tourism_information",):
        _clear_chat_state(request)
        reply = _get_tourism_information(params, message)
    elif intent in ("calculate_billing", "calculatetourbilling"):
        if actor.get("role") in {"owner", "admin", "employee"}:
            _clear_chat_state(request)
            reply = "Tour billing computation in chat is for guest bookings. Please use dashboard records for staff workflows."
        else:
            _clear_chat_state(request)
            reply = _calculate_billing(params)
    elif intent in ("get_accommodation_recommendation", "gethotelrecommendation"):
        if actor.get("role") in {"owner", "admin", "employee"}:
            _clear_chat_state(request)
            role_help = _build_role_help_payload(actor)
            reply = (
                "Accommodation recommendation cards are shown for guest booking flow only. "
                + str(role_help.get("fulfillmentText") or "")
            )
            logged_recommended_items = []
            accommodation_meta = None
        else:
            missing_slot, question = _next_accommodation_clarifying_question(params)
            if missing_slot:
                baseline = _infer_user_accommodation_baseline(user)
                defaults = _build_personalization_defaults(params, baseline)
                personalization_prompt = _build_personalization_offer_text(defaults, baseline)
                if (not state_skip_default_offer) and defaults and personalization_prompt:
                    _save_chat_state(
                        request,
                        {
                            "pending_intent": "get_accommodation_recommendation",
                            "params": params,
                            "missing_slot": missing_slot,
                            "skip_default_offer": False,
                            "default_offer": {
                                "defaults": defaults,
                                "prompt": personalization_prompt,
                            },
                        },
                    )
                    return _chat_json_response(
                        request,
                        start_time,
                        {
                            "fulfillmentText": personalization_prompt,
                        },
                    )

                company_type = str(params.get("company_type") or "").strip().lower()
                location = str(params.get("location") or "").strip()
                preference_tags = (
                    params.get("preference_tags")
                    if isinstance(params.get("preference_tags"), list)
                    else []
                )
                preview_key = f"{company_type}|{location.lower()}|{','.join(sorted(str(tag) for tag in preference_tags))}"
                if (
                    missing_slot in ("stay_details", "budget", "accommodation_details")
                    and state_missing_slot != "location"
                    and company_type in ("hotel", "inn", "either")
                    and (location or preference_tags)
                    and str(chat_state.get("location_preview_for") or "") != preview_key
                ):
                    preview_params = dict(params)
                    preview_reply, preview_items, preview_meta = _safe_get_accommodation_recommendations(preview_params)
                    if (
                        "top hotel/inn recommendations" not in str(preview_reply or "").lower()
                        and company_type in ("hotel", "inn")
                    ):
                        preview_any_type_params = dict(preview_params)
                        preview_any_type_params["company_type"] = "either"
                        preview_reply, preview_items, preview_meta = _safe_get_accommodation_recommendations(preview_any_type_params)
                    if "top hotel/inn recommendations" in str(preview_reply or "").lower():
                        next_missing_slot, next_question = _next_accommodation_clarifying_question(params)
                        next_state = {
                            "pending_intent": "get_accommodation_recommendation",
                            "params": params,
                            "missing_slot": next_missing_slot or "budget",
                            "skip_default_offer": state_skip_default_offer,
                            "location_preview_for": preview_key,
                            "last_accommodation_recommendations": _build_accommodation_selection_cache(preview_items),
                        }
                        _save_chat_state(request, next_state)
                        prompt_suffix = (
                            f"To continue booking, {next_question}"
                            if next_question else
                            "Tell me if you want to proceed with booking any room above."
                        )
                        response = {
                            "fulfillmentText": f"{preview_reply}\n\n{prompt_suffix}"
                        }
                        if preview_items:
                            response["recommendation_trace"] = preview_items
                        if isinstance(preview_meta, dict) and preview_meta.get("fallback_applied"):
                            response["recommendation_fallback"] = str(preview_meta.get("fallback_applied"))
                        if isinstance(preview_meta, dict) and isinstance(preview_meta.get("quick_replies"), list):
                            response["quick_replies"] = _sanitize_quick_replies(
                                preview_meta.get("quick_replies"),
                                limit=4,
                            )
                        if preview_items:
                            response["quick_replies"] = _merge_quick_replies(
                                response.get("quick_replies") if isinstance(response.get("quick_replies"), list) else [],
                                _build_recommendation_assist_quick_replies(
                                    _build_accommodation_selection_cache(preview_items)
                                ),
                                limit=4,
                            )
                        _inject_recommendation_context(response, preview_params)
                        return _chat_json_response(request, start_time, response)

                _save_chat_state(
                    request,
                    {
                        "pending_intent": "get_accommodation_recommendation",
                        "params": params,
                        "missing_slot": missing_slot,
                        "skip_default_offer": state_skip_default_offer,
                        "location_preview_for": chat_state.get("location_preview_for", ""),
                    },
                )
                return _chat_json_response(
                    request,
                    start_time,
                    {
                        "fulfillmentText": question,
                        "quick_replies": _slot_quick_replies(missing_slot),
                        "needs_clarification": True,
                        "missing_slot": missing_slot,
                    },
                )
            _safe_log_recommendation_event(request, intent)
            cnn_prediction, cnn_error = _predict_accommodation_class_from_text(message)
            if cnn_prediction and isinstance(params, dict):
                # Allow the recommender (or future logic) to use the predicted class.
                params.setdefault("predicted_accommodation_type", cnn_prediction["predicted_class"])
                params.setdefault("predicted_accommodation_confidence", float(cnn_prediction.get("confidence", 0.0)))
                try:
                    log_ctx = getattr(request, "_chatbot_log_context", None)
                    if isinstance(log_ctx, dict):
                        prov = log_ctx.get("provenance") if isinstance(log_ctx.get("provenance"), dict) else {}
                        runtime_models = prov.get("runtime_models") if isinstance(prov.get("runtime_models"), dict) else {}
                        runtime_models["accommodation_cnn_artifact_source"] = str(cnn_prediction.get("artifact_source") or "")[:80]
                        runtime_models["accommodation_cnn_predicted_class"] = str(cnn_prediction.get("predicted_class") or "")[:80]
                        runtime_models["accommodation_cnn_confidence"] = float(cnn_prediction.get("confidence", 0.0) or 0.0)
                        prov["runtime_models"] = runtime_models
                        log_ctx["provenance"] = prov
                except Exception:
                    pass
            reply, logged_recommended_items, accommodation_meta = _safe_get_accommodation_recommendations(params)
            next_state = {
                "pending_intent": "get_accommodation_recommendation",
                "params": params,
                "missing_slot": "",
                "skip_default_offer": False,
                "last_accommodation_recommendations": _build_accommodation_selection_cache(logged_recommended_items),
            }
            if (
                isinstance(accommodation_meta, dict)
                and "budget_too_low" in (accommodation_meta.get("no_match_reasons") or [])
                and _to_int(accommodation_meta.get("suggested_budget_min"), default=0) > 0
            ):
                next_state["pending_budget_offer"] = {
                    "suggested_budget": _to_int(accommodation_meta.get("suggested_budget_min"), default=0)
                }
            _save_chat_state(request, next_state)
            show_cnn_debug_in_chat = str(os.getenv("CHATBOT_SHOW_CNN_DEBUG", "0")).strip().lower() in (
                "1",
                "true",
                "yes",
                "on",
            )
            if (
                show_cnn_debug_in_chat
                and cnn_prediction
                and reply.startswith("I couldn't find a matching hotel or inn right now.")
            ):
                reply = f"{reply}\n\n{_format_cnn_prediction_for_chat(cnn_prediction)}"
    elif intent in ("calculate_accommodation_billing", "calculatehotelbilling"):
        if actor.get("role") in {"owner", "admin", "employee"}:
            _clear_chat_state(request)
            reply = "Accommodation billing in chat is for guest booking flow only."
        else:
            _clear_chat_state(request)
            reply = _calculate_accommodation_billing(params)
            room = _find_accommodation_room(params)
            if room is not None:
                billing_actions["quick_replies"] = [
                    {
                        "label": "Continue to Book",
                        "value": _build_book_from_billing_prompt(room, params),
                    },
                    {
                        "label": "Find Another Hotel",
                        "value": _build_find_another_accommodation_prompt(params),
                    },
                ]
    elif intent in ("book_accommodation", "bookhotel", "book_hotel", "reserve_accommodation"):
        if actor.get("role") in {"owner", "admin", "employee"}:
            _clear_chat_state(request)
            reply = "Room booking via chatbot is available for guest accounts. Use your role dashboard for management tasks."
        else:
            next_state = dict(chat_state)
            next_state.pop("pending_booking", None)
            _save_chat_state(request, next_state)
            booking_result = _book_accommodation_from_chat(request, params, commit=False)
            reply = booking_result["reply"]
            if guest_count_update_note:
                reply = f"{guest_count_update_note}\n\n{reply}"
            if booking_result.get("requires_confirmation"):
                latest_state = _load_chat_state(request)
                latest_state["pending_booking"] = {
                    "params": {
                        "room_id": booking_result.get("prepared_params", {}).get("room_id"),
                        "check_in": booking_result.get("prepared_params", {}).get("check_in"),
                        "check_out": booking_result.get("prepared_params", {}).get("check_out"),
                        "guests": booking_result.get("prepared_params", {}).get("guests"),
                    },
                    "created_at": int(time.time()),
                }
                _save_chat_state(request, latest_state)
            else:
                missing_slot = str(booking_result.get("missing_slot") or "").strip().lower()
                prepared_params = (
                    booking_result.get("prepared_params")
                    if isinstance(booking_result.get("prepared_params"), dict)
                    else {}
                )
                if missing_slot:
                    _safe_log_chat_runtime_event(
                        request,
                        event_key="clarification_triggered",
                        detail=missing_slot,
                    )
                    next_params = dict(params)
                    if booking_result.get("room_id") not in (None, ""):
                        next_params["room_id"] = booking_result.get("room_id")
                    for key in ("room_id", "check_in", "check_out", "guests"):
                        if prepared_params.get(key) not in (None, ""):
                            next_params[key] = prepared_params.get(key)
                    _save_chat_state(
                        request,
                        {
                            "pending_intent": "book_accommodation",
                            "params": next_params,
                            "missing_slot": missing_slot,
                        },
                    )
    else:
        if intent not in continuation_intents:
            _clear_chat_state(request)
        fallback_payload = _build_out_of_scope_payload(actor)
        reply = str(fallback_payload.get("fulfillmentText") or "").strip()
        if isinstance(fallback_payload.get("quick_replies"), list):
            out_of_scope_quick_replies = _sanitize_quick_replies(
                fallback_payload.get("quick_replies"),
                limit=4,
            )

    final_reply, nlg_source = generate_final_ai_response(
        request=request,
        intent=intent,
        user_message=raw_message,
        backend_reply=reply,
    )
    # Phase 2 UX consistency: keep outbound responses in English to avoid
    # unexpected language switching within the same conversation.
    translated_back = False

    request._chatbot_log_context["response_nlg_source"] = nlg_source
    response = {"fulfillmentText": final_reply}
    if 'booking_result' in locals():
        if booking_result.get("billing_link"):
            response["billing_link"] = booking_result["billing_link"]
            if booking_result.get("billing_link_label"):
                response["billing_link_label"] = booking_result["billing_link_label"]
        if booking_result.get("booking_id") is not None:
            response["booking_id"] = booking_result["booking_id"]
        if booking_result.get("receipt_text"):
            response["receipt_text"] = booking_result["receipt_text"]
            response["receipt_filename"] = booking_result.get("receipt_filename") or "ibayaw_booking_receipt.png"
        if isinstance(booking_result.get("quick_replies"), list):
            response["quick_replies"] = _sanitize_quick_replies(
                booking_result.get("quick_replies"),
                limit=4,
            )
        if booking_result.get("booking_id") and not booking_result.get("requires_confirmation"):
            response["show_feedback_prompt"] = True
    if isinstance(billing_actions, dict) and billing_actions:
        if isinstance(billing_actions.get("quick_replies"), list):
            response["quick_replies"] = _sanitize_quick_replies(
                billing_actions.get("quick_replies"),
                limit=4,
            )
    if out_of_scope_quick_replies:
        response["quick_replies"] = out_of_scope_quick_replies
    if cnn_prediction:
        response["cnn_prediction"] = cnn_prediction
    elif cnn_error and intent in ("get_accommodation_recommendation", "gethotelrecommendation"):
        response["cnn_prediction_error"] = cnn_error
    if isinstance(intent_classifier, dict):
        response["intent_classifier"] = {
            "source": str(intent_classifier.get("source") or ""),
            "confidence": float(intent_classifier.get("confidence", 0.0) or 0.0),
            "error": str(intent_classifier.get("error") or ""),
            "artifact_source": str(intent_classifier.get("artifact_source") or ""),
            "top_3": intent_classifier.get("top_3", []),
        }
    response["response_nlg_source"] = nlg_source
    if logged_recommended_items and intent in ("get_accommodation_recommendation", "gethotelrecommendation"):
        response["recommendation_trace"] = logged_recommended_items
        assist_qr = _build_recommendation_assist_quick_replies(
            _build_accommodation_selection_cache(logged_recommended_items)
        )
        response["quick_replies"] = _merge_quick_replies(
            response.get("quick_replies") if isinstance(response.get("quick_replies"), list) else [],
            assist_qr,
            limit=4,
        )
        _inject_recommendation_context(response, params if isinstance(params, dict) else {})
    if accommodation_meta and intent in ("get_accommodation_recommendation", "gethotelrecommendation"):
        if isinstance(accommodation_meta, dict):
            no_match_reasons = accommodation_meta.get("no_match_reasons")
            suggested_budget_min = accommodation_meta.get("suggested_budget_min")
            fallback_applied = accommodation_meta.get("fallback_applied")
            quick_replies = accommodation_meta.get("quick_replies")
            if isinstance(no_match_reasons, list):
                response["no_match_reasons"] = [str(item) for item in no_match_reasons]
            if suggested_budget_min not in (None, ""):
                response["suggested_budget_min"] = suggested_budget_min
            if fallback_applied:
                response["recommendation_fallback"] = str(fallback_applied)
            if isinstance(quick_replies, list):
                response["quick_replies"] = _sanitize_quick_replies(quick_replies, limit=4)
    _safe_log_step_events_from_response(request, intent=intent, response_payload=response)
    _safe_log_recommendation_result_with_metadata(
        request,
        intent,
        final_reply,
        params,
        cnn_prediction=cnn_prediction,
        message_text=message,
        recommended_items=logged_recommended_items,
    )
    request._chatbot_log_context["provenance"] = {
        "recommendation_fallback": response.get("recommendation_fallback", ""),
        "has_recommendation_trace": bool(response.get("recommendation_trace")),
        "status_code": 200,
        "detected_language": detected_language,
        "input_translated_to_english": bool(
            str(raw_message or "").strip() and str(message or "").strip() and str(raw_message).strip() != str(message).strip()
        ),
        "response_translated_to_user_language": translated_back,
    }
    return _chat_json_response(request, start_time, response)


@csrf_exempt
def openai_chat(request):
    """
    Backward-compatible alias.
    Deprecated naming retained to avoid breaking existing routes/imports.
    """
    return ai_chat(request)


@csrf_exempt
def chat_runtime_health(request):
    if request.method != "GET":
        return JsonResponse({"error": "Method not allowed."}, status=405)

    actor = _resolve_chat_actor(request)
    user = actor.get("user")
    if not user or not getattr(user, "is_authenticated", False):
        return JsonResponse(
            {
                "fulfillmentText": "Please log in first to use this endpoint.",
                "error_code": "chat_requires_login",
            },
            status=401,
        )

    openai_key_present = bool(str(os.getenv("OPENAI_API_KEY", "") or "").strip())
    gemini_key_present = bool(str(os.getenv("GEMINI_API_KEY", "") or "").strip())
    nlg_enabled = str(
        os.getenv("CHATBOT_LLM_NLG_ENABLED", os.getenv("CHATBOT_OPENAI_NLG_ENABLED", "1"))
    ).strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )
    provider = "none"
    if openai_key_present and OpenAI is not None:
        provider = "openai"
    elif gemini_key_present and genai is not None:
        provider = "gemini"
    intent_cnn_path, intent_cnn_source = _resolve_intent_text_cnn_model_path()
    accom_cnn_path, accom_cnn_source = _resolve_accommodation_text_cnn_model_path()
    dt_status = get_decision_tree_runtime_status(force_reload=False)
    return JsonResponse(
        {
            "status": "ok",
            "chatbot": {
                "translation": translation_runtime_health(),
                "nlg": {
                    "enabled": nlg_enabled,
                    "api_key_present": openai_key_present,
                    "gemini_api_key_present": gemini_key_present,
                    "client_available": OpenAI is not None,
                    "gemini_client_available": genai is not None,
                    "provider": provider,
                    "model": str(os.getenv("OPENAI_MODEL", "gpt-4o-mini") or "").strip() or "gpt-4o-mini",
                    "gemini_model": str(os.getenv("GEMINI_MODEL", "gemini-1.5-flash") or "").strip() or "gemini-1.5-flash",
                },
                "models": {
                    "intent_cnn_path": str(intent_cnn_path),
                    "intent_cnn_source": intent_cnn_source,
                    "intent_cnn_exists": bool(intent_cnn_path.exists()),
                    "accommodation_cnn_path": str(accom_cnn_path),
                    "accommodation_cnn_source": accom_cnn_source,
                    "accommodation_cnn_exists": bool(accom_cnn_path.exists()),
                    "decision_tree_path": str(dt_status.get("path") or ""),
                    "decision_tree_source": str(dt_status.get("source") or ""),
                    "decision_tree_fallback_used": bool(dt_status.get("fallback_used")),
                    "decision_tree_exists": bool(dt_status.get("file_exists")),
                },
            },
        }
    )


@csrf_exempt
def decision_tree_runtime_status(request):
    if request.method != "GET":
        return JsonResponse({"error": "Method not allowed."}, status=405)

    actor = _resolve_chat_actor(request)
    user = actor.get("user")
    if not user or not getattr(user, "is_authenticated", False):
        return JsonResponse(
            {
                "fulfillmentText": "Please log in first to use this endpoint.",
                "error_code": "chat_requires_login",
            },
            status=401,
        )

    if not (getattr(user, "is_staff", False) or getattr(user, "is_superuser", False)):
        return JsonResponse(
            {
                "error": "Forbidden.",
                "error_code": "chat_admin_required",
            },
            status=403,
        )

    status_payload = get_decision_tree_runtime_status(force_reload=True)
    return JsonResponse({"status": "ok", "decision_tree_runtime": status_payload})


@csrf_exempt
def log_recommendation_click(request):
    if request.method != "POST":
        return JsonResponse({"error": "Method not allowed."}, status=405)

    user = getattr(request, "user", None)
    if not user or not getattr(user, "is_authenticated", False):
        return JsonResponse(
            {
                "fulfillmentText": "Please log in first to use this tracking endpoint.",
                "error_code": "chat_requires_login",
            },
            status=401,
        )

    try:
        payload = json.loads(request.body or "{}")
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON payload."}, status=400)

    logged, item_ref = _safe_log_recommendation_click(request, payload)
    return JsonResponse({"status": "ok" if logged else "skipped", "item_ref": item_ref})


@csrf_exempt
def log_guest_funnel_event(request):
    if request.method != "POST":
        return JsonResponse({"error": "Method not allowed."}, status=405)

    try:
        payload = json.loads(request.body or "{}")
    except json.JSONDecodeError:
        payload = {}

    user = getattr(request, "user", None)
    if not user or not getattr(user, "is_authenticated", False):
        return JsonResponse({"status": "skipped", "reason": "not_authenticated"}, status=200)

    event_key = str((payload or {}).get("event_key") or "").strip().lower()
    config = _GUEST_FUNNEL_EVENT_MAP.get(event_key)
    if not config:
        return JsonResponse({"status": "skipped", "reason": "unknown_event_key"}, status=200)

    item_ref = str(config.get("item_ref") or "").strip()
    event_type = str(config.get("event_type") or "view").strip().lower()
    detail = str((payload or {}).get("detail") or "").strip()[:180]
    _safe_log_chat_step_event(
        request,
        event_type=event_type,
        item_ref=item_ref,
    )
    _safe_log_system_metric(
        endpoint=f"{getattr(request, 'path', '/api/chat/funnel-event/')}#event:{event_key}",
        response_time_ms=0,
        success_flag=True,
        status_code=200,
        error_message=detail,
        request=request,
    )
    return JsonResponse({"status": "ok", "event_key": event_key, "item_ref": item_ref, "event_type": event_type})


@csrf_exempt
def accommodation_booking_notifications(request):
    if request.method != "GET":
        return JsonResponse({"error": "Method not allowed."}, status=405)

    user = getattr(request, "user", None)
    if not user or not getattr(user, "is_authenticated", False):
        return JsonResponse(
            {
                "status": "skipped",
                "bookings": [],
                "reason": "not_authenticated",
            },
            status=200,
        )

    bookings = (
        AccommodationBooking.objects.filter(guest=user, status__in=("pending", "confirmed", "declined"))
        .select_related("accommodation", "room")
        .order_by("-last_updated")[:50]
    )
    rows = []
    for booking in bookings:
        rows.append(
            {
                "booking_id": booking.booking_id,
                "status": str(booking.status or "").lower(),
                "last_updated": booking.last_updated.isoformat() if booking.last_updated else "",
                "hotel_name": getattr(getattr(booking, "accommodation", None), "company_name", ""),
                "room_name": getattr(getattr(booking, "room", None), "room_name", ""),
            }
        )
    return JsonResponse({"status": "ok", "bookings": rows})


@csrf_exempt
def submit_usability_feedback(request):
    if request.method != "POST":
        return JsonResponse({"error": "Method not allowed."}, status=405)

    try:
        payload = json.loads(request.body or "{}")
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON payload."}, status=400)

    user = getattr(request, "user", None)
    if not user or not getattr(user, "is_authenticated", False):
        user = None

    data_source = _resolve_data_source(request=request, payload=payload)
    instrument = str(payload.get("instrument") or "").strip().lower()
    comment = str(payload.get("comment") or "").strip()[:1000]
    batch_id = str(payload.get("survey_batch_id") or "").strip()[:64]
    if not batch_id:
        batch_id = f"survey-{uuid.uuid4().hex}"

    response_items = _normalize_survey_response_items(payload.get("responses"))
    if response_items:
        if instrument == "sus_tam_full":
            submitted_codes = {item["statement_code"] for item in response_items}
            if not _FULL_SURVEY_CODES.issubset(submitted_codes):
                missing = sorted(list(_FULL_SURVEY_CODES.difference(submitted_codes)))
                return JsonResponse(
                    {
                        "error": "Missing required survey items for sus_tam_full.",
                        "missing_codes": missing,
                    },
                    status=400,
                )
        elif instrument == "difficulty_full":
            submitted_codes = {item["statement_code"] for item in response_items}
            required = set(_DIFFICULTY_CODES)
            if not required.issubset(submitted_codes):
                missing = sorted(list(required.difference(submitted_codes)))
                return JsonResponse(
                    {
                        "error": "Missing required survey items for difficulty_full.",
                        "missing_codes": missing,
                    },
                    status=400,
                )
        elif instrument == "sus_tam_difficulty_full":
            submitted_codes = {item["statement_code"] for item in response_items}
            required = set(_FULL_SURVEY_CODES).union(_DIFFICULTY_CODES)
            if not required.issubset(submitted_codes):
                missing = sorted(list(required.difference(submitted_codes)))
                return JsonResponse(
                    {
                        "error": "Missing required survey items for sus_tam_difficulty_full.",
                        "missing_codes": missing,
                    },
                    status=400,
                )

        try:
            with transaction.atomic():
                for idx, item in enumerate(response_items):
                    row_comment = item.get("comment") or (comment if idx == 0 else "")
                    UsabilitySurveyResponse.objects.create(
                        user=user,
                        statement_code=item["statement_code"],
                        likert_score=item["likert_score"],
                        comment=row_comment,
                        survey_batch_id=batch_id,
                        data_source=data_source,
                    )
        except Exception:
            return JsonResponse({"status": "error"}, status=500)

        return JsonResponse(
            {
                "status": "ok",
                "saved_count": len(response_items),
                "survey_batch_id": batch_id,
            }
        )

    statement_code = str(payload.get("statement_code") or "CHAT_UX_QUICK").strip().upper()[:30]
    likert_score = _to_int(payload.get("likert_score"), default=0)
    if likert_score < 1 or likert_score > 5:
        return JsonResponse({"error": "likert_score must be between 1 and 5."}, status=400)

    try:
        UsabilitySurveyResponse.objects.create(
            user=user,
            statement_code=statement_code or "CHAT_UX_QUICK",
            likert_score=likert_score,
            comment=comment,
            survey_batch_id=batch_id,
            data_source=data_source,
        )
    except Exception:
        return JsonResponse({"status": "error"}, status=500)

    return JsonResponse({"status": "ok", "saved_count": 1, "survey_batch_id": batch_id})


@csrf_exempt
def text_cnn_predict(request):
    if request.method != "POST":
        return JsonResponse(
            {
                "status": "ok",
                "usage": "POST JSON: {\"message\": \"need a cheap inn near terminal\"}",
            }
        )

    try:
        payload = json.loads(request.body or "{}")
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON payload."}, status=400)

    message = str(payload.get("message", "")).strip()
    if not message:
        return JsonResponse({"error": "Missing 'message'."}, status=400)

    prediction, err = _predict_accommodation_class_from_text(message)
    if err:
        return JsonResponse({"error": err}, status=500)
    return JsonResponse(prediction)

