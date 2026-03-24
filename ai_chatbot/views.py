import json
import os
import re
import time
import uuid
from collections import Counter
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from urllib.parse import urlencode

from django.conf import settings
from django.contrib.auth.models import Group
from django.db import transaction
from django.db.models import F, Q, Sum
from django.http import JsonResponse
from django.urls import reverse
from django.utils import timezone
from django.views.decorators.csrf import csrf_exempt

try:
    from openai import OpenAI
except ModuleNotFoundError:
    OpenAI = None

try:
    import numpy as np
    import pandas as pd
    import tensorflow as tf
except ModuleNotFoundError:
    np = None
    pd = None
    tf = None

from tour_app.models import Admission_Rates, Tour_Schedule
from admin_app.models import Accomodation, Employee, Room, TourismInformation
from guest_app.models import AccommodationBooking
from guest_app.booking_integrity import create_accommodation_booking_with_integrity
from .recommenders import (
    recommend_tours,
    recommend_accommodations_with_diagnostics,
    calculate_accommodation_billing,
)
from .llm_translation import (
    translate_to_english,
    translate_to_user_language,
)
from .models import (
    ChatbotLog,
    RecommendationEvent,
    RecommendationResult,
    SystemMetricLog,
    UsabilitySurveyResponse,
)


_TEXT_CNN_MODEL_CACHE = None
_TEXT_CNN_MODEL_PATH_CACHE = None
_CHAT_STATE_SESSION_KEY = "ai_chatbot_state"
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
_SUS_CODES = [f"SUS_Q{i}" for i in range(1, 11)]
_TAM_CODES = [f"PU_Q{i}" for i in range(1, 5)] + [f"PEU_Q{i}" for i in range(1, 5)]
_DIFFICULTY_CODES = ["DIFF_DISCOVER", "DIFF_MATCH", "DIFF_PLAN", "DIFF_BOOKPAY"]
_FULL_SURVEY_CODES = set(_SUS_CODES + _TAM_CODES)


def _to_bool_env(value, default=False):
    raw = str(value or "").strip().lower()
    if not raw:
        return bool(default)
    if raw in ("1", "true", "yes", "on", "y"):
        return True
    if raw in ("0", "false", "no", "off", "n"):
        return False
    return bool(default)


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
    if not resolved_path.exists():
        return None, f"model_not_found:{resolved_path}"

    if _TEXT_CNN_MODEL_CACHE is not None and _TEXT_CNN_MODEL_PATH_CACHE == str(resolved_path):
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
        return _TEXT_CNN_MODEL_CACHE, None
    except Exception as exc:
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

        if mode_counts:
            context_payload["hybrid_mode_counts"] = dict(mode_counts)
            context_payload["top_mode"] = mode_counts.most_common(1)[0][0]
        if dt_scores:
            context_payload["decision_tree_score_avg"] = round(sum(dt_scores) / len(dt_scores), 6)
            context_payload["decision_tree_score_count"] = len(dt_scores)

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
    response = JsonResponse(payload, status=status)
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
            response_payload = payload if isinstance(payload, dict) else {}
            bot_text = str(response_payload.get("fulfillmentText") or "")
            _safe_log_chatbot_interaction(
                request,
                user_message=context.get("user_message", ""),
                resolved_intent=context.get("resolved_intent", ""),
                resolved_params=context.get("resolved_params", {}),
                bot_response=bot_text,
                intent_classifier=context.get("intent_classifier", {}),
                response_nlg_source=(
                    response_payload.get("response_nlg_source")
                    or context.get("response_nlg_source", "")
                ),
                fallback_used=bool(
                    context.get("fallback_used")
                    or response_payload.get("recommendation_fallback")
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
    session = getattr(request, "session", None)
    if session is None:
        return {}

    state = session.get(_CHAT_STATE_SESSION_KEY, {})
    if not isinstance(state, dict):
        session.pop(_CHAT_STATE_SESSION_KEY, None)
        return {}

    last_updated_epoch = _to_int(state.get("last_updated_epoch"), default=0)
    if last_updated_epoch <= 0:
        session.pop(_CHAT_STATE_SESSION_KEY, None)
        return {}

    if int(time.time()) - last_updated_epoch > _CHAT_STATE_TTL_SECONDS:
        session.pop(_CHAT_STATE_SESSION_KEY, None)
        return {}

    return state


def _save_chat_state(request, state):
    session = getattr(request, "session", None)
    if session is None:
        return

    payload = state if isinstance(state, dict) else {}
    payload["last_updated_epoch"] = int(time.time())
    session[_CHAT_STATE_SESSION_KEY] = payload
    session.modified = True


def _clear_chat_state(request):
    session = getattr(request, "session", None)
    if session is None:
        return
    if _CHAT_STATE_SESSION_KEY in session:
        session.pop(_CHAT_STATE_SESSION_KEY, None)
        session.modified = True


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
    if not location:
        missing_fields.append(("location", "Preferred area/location in Bayawan"))

    budget = _to_int(params.get("budget"), default=0)
    if budget <= 0:
        missing_fields.append(("budget", "Budget per night in PHP"))

    guests = _to_int(params.get("guests"), default=0)
    if guests <= 0:
        missing_fields.append(("guests", "Number of guests"))

    if not _has_stay_details(params):
        missing_fields.append(
            ("stay_details", "Stay details: check-in/check-out dates (YYYY-MM-DD) or number of nights")
        )

    if len(missing_fields) == 1:
        field = missing_fields[0][0]
        if field == "company_type":
            return "company_type", "Would you like a hotel, an inn, or either?"
        if field == "location":
            return "location", "Which area in Bayawan should I prioritize?"
        if field == "budget":
            return "budget", "What is your budget per night in PHP?"
        if field == "guests":
            return "guests", "How many guests will stay?"
        if field == "stay_details":
            return "stay_details", "Please share check-in/check-out dates (YYYY-MM-DD) or tell me the number of nights."

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

        missing_prompts = []
        for field, _label in missing_fields:
            if field == "company_type":
                missing_prompts.append("Do you prefer a hotel, an inn, or either?")
            elif field == "location":
                missing_prompts.append("Which area in Bayawan should I prioritize?")
            elif field == "budget":
                missing_prompts.append("What is your budget per night in PHP?")
            elif field == "guests":
                missing_prompts.append("How many guests will stay?")
            elif field == "stay_details":
                missing_prompts.append(
                    "Please share check-in/check-out dates (YYYY-MM-DD) or tell me number of nights."
                )

        prefix = "I can proceed once I get a few more details."
        if acknowledged_parts:
            prefix = (
                "I understood: " + "; ".join(acknowledged_parts) + ". "
                "I just need a few more details."
            )

        if len(missing_prompts) <= 2:
            follow_up = " ".join(missing_prompts)
        else:
            follow_up = "\n".join(f"- {item}" for item in missing_prompts)

        return (
            "accommodation_details",
            (
                f"{prefix}\n"
                f"{follow_up}\n\n"
                "You can send them in one message, for example:\n"
                "hotel in bayawan, budget 1500, 2 guests, 2026-03-10 to 2026-03-12"
            ),
        )

    return None, ""


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
        "broaden_location",
        "broaden_company_type",
    }
    return any(key in params for key in slot_keys)


def _looks_like_tour_request(message):
    text = str(message or "").strip().lower()
    if not text:
        return False
    tour_keywords = ("tour", "itinerary", "schedule", "destination", "package")
    return any(keyword in text for keyword in tour_keywords)


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
    return "different budget" in text


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
    return text in accept_phrases


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

    sample_size = _to_int(baseline.get("sample_size"), default=0)
    basis_text = "Based on your past bookings"
    if sample_size > 0:
        basis_text = f"Based on your {sample_size} past booking(s)"
    defaults_text = ", ".join(parts)
    return f"{basis_text}, I can use defaults {defaults_text}. Want similar options? Reply yes or no."


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
    return total if total > 0 else 1


def _get_recommendations(params):
    results = recommend_tours(params, limit=3)
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
        known_text = f"I understood: {'; '.join(known_bits)}.\n" if known_bits else ""
        return (
            f"{known_text}"
            "I couldn’t find a strong tour match yet. "
            "Try increasing budget, changing tour type, or sharing preferred destination."
        ), []

    lines = ["Top recommendations for you (CNN + Decision Tree):"]
    items_payload = []
    for idx, item in enumerate(results, 1):
        lines.append(f"{idx}. {item.title} | {item.subtitle}")
        items_payload.append(
            {
                "rank": idx,
                "title": item.title,
                "subtitle": item.subtitle,
                "score": round(float(item.score), 6),
                "meta": item.meta if isinstance(item.meta, dict) else {},
            }
        )
    return "\n".join(lines), items_payload


def _get_accommodation_recommendations(params):
    results, diagnostics = recommend_accommodations_with_diagnostics(params, limit=3)
    if not results:
        no_match_reasons = diagnostics.get("no_match_reasons") or []
        suggested_budget_min = diagnostics.get("suggested_budget_min")
        location = str(params.get("location") or "").strip()
        budget = _to_decimal(params.get("budget"), default=Decimal("0"))
        guests = _to_int(params.get("guests"), default=0)
        company_type = str(params.get("company_type") or "").strip().lower()

        known_bits = []
        if company_type in ("hotel", "inn", "either"):
            known_bits.append(f"type: {company_type}")
        if location:
            known_bits.append(f"location: {location}")
        if budget > 0:
            known_bits.append(f"budget: PHP {budget:.2f}")
        if guests > 0:
            known_bits.append(f"guests: {guests}")
        known_text = f"I understood: {'; '.join(known_bits)}.\n" if known_bits else ""

        lines = [
            f"{known_text}I couldn’t find a strong hotel/inn match yet."
        ]
        quick_replies = []
        if "budget_too_low" in no_match_reasons and suggested_budget_min:
            min_budget_value = Decimal(str(suggested_budget_min))
            quick_replies.append(f"budget {int(min_budget_value)}")
            if location and budget > 0:
                lines.append(
                    f"No rooms in {location.title()} are available under PHP {budget:.2f}. "
                    f"Cheapest matching option starts at PHP {min_budget_value:.2f}. "
                    f"Do you want to use PHP {min_budget_value:.0f} as your budget?"
                )
            else:
                lines.append(
                    f"Cheapest matching option starts at PHP {min_budget_value:.2f}. "
                    f"Do you want to use PHP {min_budget_value:.0f} as your budget?"
                )
        elif "location_too_narrow" in no_match_reasons:
            lines.append(
                "I found options outside your current location filter. "
                "If you want, send: broaden location"
            )
            quick_replies.append("broaden location")
        elif "type_too_narrow" in no_match_reasons:
            lines.append(
                "I found options under a different accommodation type. "
                "If you want, send: include both hotel and inn"
            )
            quick_replies.append("include both hotel and inn")

        if not quick_replies:
            quick_replies.append("show default hotel suggestions")

        return "\n\n".join(lines), [], {
            "no_match_reasons": no_match_reasons,
            "suggested_budget_min": suggested_budget_min,
            "fallback_applied": diagnostics.get("fallback_applied", "none"),
            "quick_replies": quick_replies,
        }

    lines = ["Top hotel/inn recommendations for you (CNN + Decision Tree):"]
    items_payload = []
    for idx, item in enumerate(results, 1):
        item_meta = item.meta if isinstance(item.meta, dict) else {}
        room_id = item_meta.get("room_id")
        room_id_label = f" | Room ID: {room_id}" if room_id not in (None, "") else ""
        trace = item_meta.get("trace") if isinstance(item_meta.get("trace"), dict) else {}
        reasons = trace.get("reasons") if isinstance(trace.get("reasons"), list) else []
        match_score = trace.get("match_score")
        match_strength = str(trace.get("match_strength") or "").strip()
        lines.append(f"{idx}. {item.title}{room_id_label} | {item.subtitle}")
        if reasons:
            lines.append(f"   Reasons: {'; '.join(str(r) for r in reasons)}")
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
    return "\n".join(lines), items_payload, {
        "fallback_applied": diagnostics.get("fallback_applied", "none"),
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

    lines = ["Here are some suggested stays:"]
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
                    "Perfect for quick stay suggestions",
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
    room_id = params.get("room_id")
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

    guests = max(_resolve_guests(params), 1)

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
    guests = max(_resolve_guests(params), 1)
    check_in = _normalize_iso_date(params.get("check_in"))
    check_out = _normalize_iso_date(params.get("check_out"))

    if check_in and check_out:
        return f"book room {room_id} for {guests} guests from {check_in} to {check_out}"
    return f"book room {room_id} for {guests} guests"


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
    num_guests = max(_resolve_guests(params), 1)

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
        return {
            "reply": (
                "I can prepare your booking now. I just need the missing stay date details.\n"
                f"I understood: {'; '.join(known_bits)}.\n"
                f"{ask_line}"
            ),
            "room_id": room.room_id,
            "accom_id": getattr(room.accommodation, "accom_id", None),
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
        }

    if check_out_dt <= check_in_dt:
        return {
            "reply": (
                f"I received check-in {check_in_dt.isoformat()} and check-out {check_out_dt.isoformat()}.\n"
                "Check-out must be later than check-in. Please send updated dates."
            ),
            "room_id": room.room_id,
            "accom_id": getattr(room.accommodation, "accom_id", None),
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
                "Confirm booking to generate LGU billing reference/link? (Yes/No)"
            )
            if not commit
            else "Billing / Payment Link: use the button/link provided in the chat."
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
            if booking and commit
            else "Proceed to LGU Payment"
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
            (["view my accommodation bookings"] if booking and commit else ["Yes", "No"])
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
    numeric_only = re.fullmatch(
        r"\s*([0-9][0-9,]*(?:\.[0-9]+)?k?)\s*",
        text,
        flags=re.IGNORECASE,
    )
    known_location_map = {
        "terminal": "terminal area",
        "terminal area": "terminal area",
        "poblacion": "poblacion",
        "bayawan": "bayawan",
        "bayawan city": "bayawan city",
        "suba": "suba",
    }

    def _set_clarification(field, question, penalty=0.3):
        nonlocal confidence, needs_clarification, clarification_question, clarification_field
        needs_clarification = True
        confidence = max(0.0, confidence - float(penalty))
        if not clarification_question:
            clarification_question = question
            clarification_field = field

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

    # Extract guest count from "<n> guest(s)/people/person/pax".
    guest_match = re.search(r"(\d+)\s*(guest|guests|people|person|pax)", text)
    if guest_match:
        params["guests"] = int(guest_match.group(1))

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
    else:
        # Numeric-only message fallback, e.g. "1500" or "1.5k"
        if numeric_only:
            budget_value = _parse_compact_number(numeric_only.group(1))
            if budget_value is not None:
                params["budget"] = budget_value

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
        r"\b(in|near|around)\s+([a-z\s]+?)(?=\s+(?:for|under|below|budget|with|from)\b|$)",
        text,
    )
    if loc_match:
        loc_prefix = str(loc_match.group(1) or "").strip()
        raw_location = " ".join(loc_match.group(2).split()).strip()
        if raw_location:
            normalized_location = known_location_map.get(raw_location)
            if normalized_location:
                params["location"] = normalized_location
            elif loc_prefix in ("near", "around"):
                _set_clarification(
                    "location",
                    "I couldn't map that place. Please specify the barangay or area in Bayawan.",
                    penalty=0.35,
                )
            else:
                params["location"] = raw_location

    # Capture common location mentions even without "in/near/around".
    for known_location in known_location_map:
        if known_location in text and "location" not in params:
            params["location"] = known_location_map.get(known_location, known_location)
            break

    # Extract room_id like Room0001 or just room 12
    room_match = re.search(r"\broom\s*(\d+)\b", text)
    if room_match:
        params["room_id"] = room_match.group(1)

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

    # Extract accommodation/hotel name from "at <name>" or "hotel <name>".
    at_match = re.search(r"\bat\s+([a-z0-9\s\-&]+)", text)
    if at_match and "check-in" not in at_match.group(1):
        params.setdefault("accom_name", at_match.group(1).strip())

    hotel_name_match = re.search(r"\b(?:hotel|inn)\s+([a-z0-9\s\-&]+)", text)
    if hotel_name_match:
        params.setdefault("accom_name", hotel_name_match.group(1).strip())

    # Respect explicit accommodation type words in the user's prompt.
    if "inn" in text and "hotel" not in text:
        params.setdefault("company_type", "inn")
    elif "hotel" in text and "inn" not in text:
        params.setdefault("company_type", "hotel")
    elif "hotel" in text and "inn" in text:
        params.setdefault("company_type", "either")

    # Basic keyword preference extraction.
    for keyword in ["river", "mountain", "sea", "sunset", "forest"]:
        if keyword in text:
            params["preference"] = keyword
            break

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
    }


def _extract_params_from_message(message):
    parsed = _extract_params_with_confidence(message)
    return parsed.get("params", {})


def _intent_from_message(message):
    text = (message or "").lower()
    billing_keywords = [
        "bill",
        "billing",
        "total",
        "price",
        "cost",
        "how much",
        "amount due",
    ]
    booking_keywords = [
        "book",
        "booking",
        "reserve",
        "reservation",
    ]
    accommodation_keywords = ["hotel", "inn", "room", "accommodation"]
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
    ]
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
    ]
    return any(phrase in text for phrase in my_accommodation_booking_phrases)


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
    text = str(message or "").strip().lower()
    if not text:
        return False
    return any(str(phrase).strip().lower() in text for phrase in phrases)


def _build_role_help_payload(actor):
    role = str(actor.get("role") or "").strip().lower()
    if role == "owner":
        return {
            "fulfillmentText": (
                "Owner assistant mode is active. I can help you check your business side.\n"
                "- Show my accommodations\n"
                "- Show my rooms\n"
                "- Show my bookings\n"
                "- Open Owner Hub\n"
                "- Show tourism information about <place>\n\n"
                "For registration/edits, use Owner Hub."
            ),
            "quick_replies": [
                "Show my accommodations",
                "Show my rooms",
                "Show my bookings",
                "Open Owner Hub",
            ],
        }
    if role == "admin":
        return {
            "fulfillmentText": (
                "Admin assistant mode is active.\n"
                "I can answer tourism information, show moderation summaries, and help with quick navigation.\n"
                "Use the Admin Dashboard for approvals, encoding, and reports."
            ),
            "quick_replies": [
                "Open dashboard",
                "Show pending accommodations",
                "Show pending owner accounts",
                "Open accommodation bookings",
            ],
        }
    if role == "employee":
        return {
            "fulfillmentText": (
                "Employee assistant mode is active.\n"
                "I can answer tourism information and guide role-based navigation.\n"
                "Use the Employee Dashboard for operational tasks."
            ),
            "quick_replies": [
                "Open dashboard",
                "Open assigned tours",
                "Open tour calendar",
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
            "Show tourism information in Bayawan",
            "Show my bookings",
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
        ),
    )


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


def _is_booking_count_command(message):
    text = str(message or "").strip().lower()
    if "booking" not in text:
        return False
    if any(token in text for token in ("how many", "count", "total", "number of", "pila", "ilan")):
        return True
    return bool(re.search(r"\bbookings?\b.*\b(today|this month|monthly|daily|now)\b", text))


def _is_admin_pending_accommodations_command(message):
    return _contains_any_phrase(
        message,
        (
            "pending accommodations",
            "pending accommodation",
            "show pending hotels",
            "show pending inns",
            "accommodation approvals",
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
        ),
    )


def _is_employee_assigned_tours_command(message):
    return _contains_any_phrase(
        message,
        (
            "open assigned tours",
            "my assigned tours",
            "assigned tours",
        ),
    )


def _is_employee_tour_calendar_command(message):
    return _contains_any_phrase(
        message,
        (
            "open tour calendar",
            "my tour calendar",
            "tour calendar",
        ),
    )


def _is_employee_accommodations_command(message):
    return _contains_any_phrase(
        message,
        (
            "open accommodations",
            "employee accommodations",
            "accommodation list",
        ),
    )


def _is_employee_profile_command(message):
    return _contains_any_phrase(
        message,
        (
            "open profile",
            "my profile",
            "employee profile",
        ),
    )


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
    return (
        "Accommodation owner account review summary: "
        f"Pending {pending_count}, Approved {approved_count}, Declined {declined_count}."
    )


def _build_booking_count_summary(actor, *, user=None, message=""):
    role = str(actor.get("role") or "").strip().lower()
    text = str(message or "").strip().lower()
    today = timezone.localdate()
    qs = AccommodationBooking.objects.all()
    period_label = "all time"

    if "today" in text:
        qs = qs.filter(booking_date__date=today)
        period_label = f"today ({today.isoformat()})"
    elif "this month" in text or "monthly" in text:
        qs = qs.filter(booking_date__year=today.year, booking_date__month=today.month)
        period_label = f"this month ({today.year}-{today.month:02d})"

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


def _openai_generate_final_response(*, request, intent, user_message, backend_reply):
    reply = str(backend_reply or "").strip()
    if not reply:
        return reply, "empty_backend_reply"

    api_key = str(os.getenv("OPENAI_API_KEY", "")).strip()
    nlg_enabled = str(os.getenv("CHATBOT_OPENAI_NLG_ENABLED", "1")).strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )
    if not nlg_enabled:
        return reply, "openai_nlg_disabled"
    if not api_key or OpenAI is None:
        return reply, "openai_nlg_unavailable"

    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    user_id = ""
    user = getattr(request, "user", None)
    if user and getattr(user, "is_authenticated", False):
        user_id = str(getattr(user, "pk", "") or "")

    # OpenAI receives sanitized backend context only.
    nlg_payload = {
        "intent": str(intent or ""),
        "user_message": str(user_message or "")[:500],
        "backend_reply": reply[:3500],
        "user_id": user_id[:40],
    }
    system_prompt = (
        "You are a tourism reservation assistant NLG layer.\n"
        "Rewrite the backend reply into clear natural language.\n"
        "Do not add new facts, prices, dates, IDs, links, or policy claims.\n"
        "Keep all booking/payment constraints exactly as provided.\n"
        "Return plain text only."
    )

    try:
        client = OpenAI(api_key=api_key)
        completion = client.chat.completions.create(
            model=model,
            temperature=0.2,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(nlg_payload, ensure_ascii=True)},
            ],
        )
        phrased = str(completion.choices[0].message.content or "").strip()
        if not phrased:
            return reply, "openai_nlg_empty"
        return phrased, "openai_nlg"
    except Exception:
        return reply, "openai_nlg_error"


@csrf_exempt
def openai_chat(request):
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
        "provenance": {"chat_role": actor.get("role", "")},
    }

    if _is_help_or_greeting_command(message):
        request._chatbot_log_context["resolved_intent"] = "role_help"
        help_payload = _build_role_help_payload(actor)
        return _chat_json_response(request, start_time, help_payload)

    if actor.get("role") in {"admin", "employee"} and _is_open_dashboard_command(message):
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
            payload.update(
                _build_link_payload(
                    request,
                    text=summary,
                    route_name="admin_app:accommodation_bookings",
                    label="Open Accommodation Bookings",
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

    if actor.get("role") == "owner" and _is_owner_accommodation_overview_command(message):
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

    if actor.get("role") == "employee" and _is_employee_assigned_tours_command(message):
        request._chatbot_log_context["resolved_intent"] = "employee_assigned_tours"
        return _chat_json_response(
            request,
            start_time,
            _build_link_payload(
                request,
                text="I found your assigned tours page. Click the button below to open it in a new tab.",
                route_name="admin_app:employee_assigned_tours",
                label="Open Assigned Tours",
            ),
        )

    if actor.get("role") == "employee" and _is_employee_tour_calendar_command(message):
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

    if actor.get("role") == "employee" and _is_employee_accommodations_command(message):
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

    if actor.get("role") == "employee" and _is_employee_profile_command(message):
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

        _save_chat_state(
            request,
            {
                "pending_intent": "get_accommodation_recommendation",
                "params": {},
                "missing_slot": "",
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

    if actor.get("role") == "owner" and _is_owner_room_overview_command(message):
        request._chatbot_log_context["resolved_intent"] = "owner_room_overview"
        owner_room_reply = _build_owner_rooms_summary(user)
        return _chat_json_response(
            request,
            start_time,
            {"fulfillmentText": owner_room_reply},
        )

    chat_state = _load_chat_state(request)
    pending_booking = (
        chat_state.get("pending_booking") if isinstance(chat_state.get("pending_booking"), dict) else {}
    )
    pending_booking_params = (
        pending_booking.get("params") if isinstance(pending_booking.get("params"), dict) else {}
    )
    pending_booking_created_at = _to_int(pending_booking.get("created_at"), default=0)

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

    pending_budget_offer = (
        chat_state.get("pending_budget_offer")
        if isinstance(chat_state.get("pending_budget_offer"), dict)
        else {}
    )
    state_intent_hint = str(chat_state.get("pending_intent") or "").strip().lower()
    state_params_hint = chat_state.get("params") if isinstance(chat_state.get("params"), dict) else {}
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
                reply, logged_recommended_items, accommodation_meta = _get_accommodation_recommendations(accepted_params)
                next_state = dict(chat_state)
                next_state["pending_intent"] = "get_accommodation_recommendation"
                next_state["params"] = accepted_params
                next_state["missing_slot"] = ""
                next_state.pop("pending_budget_offer", None)
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
            reply, logged_recommended_items, accommodation_meta = _get_accommodation_recommendations(accepted_params)
            next_state = dict(chat_state)
            next_state["pending_intent"] = "get_accommodation_recommendation"
            next_state["params"] = accepted_params
            next_state["missing_slot"] = ""
            next_state.pop("pending_budget_offer", None)
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
    needs_parser_clarification = bool(parsed.get("needs_clarification"))
    parser_clarification_question = str(parsed.get("clarification_question") or "").strip()
    parser_clarification_field = str(parsed.get("clarification_field") or "").strip()
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

    accommodation_intents = ("get_accommodation_recommendation", "gethotelrecommendation")
    continuation_intents = accommodation_intents + ("get_recommendation",)
    continuing_accommodation_flow = (
        state_intent in accommodation_intents
        and (
            intent in accommodation_intents
            or (
                intent == "get_recommendation"
                and _looks_like_slot_update(params)
                and not _looks_like_tour_request(message)
            )
        )
    )
    if continuing_accommodation_flow:
        intent = state_intent

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

    if state_intent in accommodation_intents and state_offer_defaults:
        if _is_personalization_decline_message(message):
            next_state = {
                "pending_intent": state_intent,
                "params": params,
            }
            missing_slot, question = _next_accommodation_clarifying_question(params)
            if missing_slot:
                next_state["missing_slot"] = missing_slot
            _save_chat_state(request, next_state)
            if question:
                return _chat_json_response(request, start_time, {"fulfillmentText": question})
        elif _is_personalization_accept_message(message):
            accepted_params = dict(params)
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
                },
            )
        elif not _looks_like_slot_update(params):
            return _chat_json_response(
                request,
                start_time,
                {"fulfillmentText": state_offer_prompt or "Want me to use your previous booking preferences? Reply yes or no."},
            )

    if needs_parser_clarification:
        if not parser_clarification_question:
            parser_clarification_question = "Could you clarify that so I can continue?"
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
                "needs_clarification": True,
                "confidence": parse_confidence,
            },
        )
    cnn_prediction = None
    cnn_error = None
    logged_recommended_items = []
    accommodation_meta = None
    billing_actions = {}

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
        return _chat_json_response(request, start_time, response)

    if intent in ("get_recommendation", "gettourrecommendation"):
        if actor.get("role") in {"owner", "admin", "employee"}:
            _clear_chat_state(request)
            role_help = _build_role_help_payload(actor)
            reply = (
                "Tour package recommendations are currently guest-focused. "
                + str(role_help.get("fulfillmentText") or "")
            )
            logged_recommended_items = []
        else:
            _clear_chat_state(request)
            _safe_log_recommendation_event(request, intent)
            reply, logged_recommended_items = _get_recommendations(params)
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
                if defaults and personalization_prompt:
                    _save_chat_state(
                        request,
                        {
                            "pending_intent": "get_accommodation_recommendation",
                            "params": params,
                            "missing_slot": missing_slot,
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
                preview_key = f"{company_type}|{location.lower()}"
                if (
                    missing_slot in ("budget", "accommodation_details")
                    and company_type in ("hotel", "inn", "either")
                    and location
                    and str(chat_state.get("location_preview_for") or "") != preview_key
                ):
                    preview_params = dict(params)
                    preview_params.setdefault("guests", 1)
                    preview_reply, preview_items, preview_meta = _get_accommodation_recommendations(preview_params)
                    if (
                        "top hotel/inn recommendations" not in str(preview_reply or "").lower()
                        and company_type in ("hotel", "inn")
                    ):
                        preview_any_type_params = dict(preview_params)
                        preview_any_type_params["company_type"] = "either"
                        preview_reply, preview_items, preview_meta = _get_accommodation_recommendations(preview_any_type_params)
                    if "top hotel/inn recommendations" in str(preview_reply or "").lower():
                        next_missing_slot, _ = _next_accommodation_clarifying_question(params)
                        next_state = {
                            "pending_intent": "get_accommodation_recommendation",
                            "params": params,
                            "missing_slot": next_missing_slot or "budget",
                            "location_preview_for": preview_key,
                        }
                        _save_chat_state(request, next_state)
                        missing_prompts = []
                        if _to_int(params.get("budget"), default=0) <= 0:
                            missing_prompts.append("budget per night in PHP")
                        if _to_int(params.get("guests"), default=0) <= 0:
                            missing_prompts.append("number of guests")
                        if not _has_stay_details(params):
                            missing_prompts.append("check-in/check-out dates or number of nights")
                        prompt_suffix = (
                            "To continue booking, please share your "
                            + ", ".join(missing_prompts)
                            + "."
                        ) if missing_prompts else "Tell me if you want to proceed with booking any room above."
                        response = {
                            "fulfillmentText": f"{preview_reply}\n\n{prompt_suffix}"
                        }
                        if preview_items:
                            response["recommendation_trace"] = preview_items
                        if isinstance(preview_meta, dict) and preview_meta.get("fallback_applied"):
                            response["recommendation_fallback"] = str(preview_meta.get("fallback_applied"))
                        return _chat_json_response(request, start_time, response)

                _save_chat_state(
                    request,
                    {
                        "pending_intent": "get_accommodation_recommendation",
                        "params": params,
                        "missing_slot": missing_slot,
                        "location_preview_for": chat_state.get("location_preview_for", ""),
                    },
                )
                return _chat_json_response(
                    request,
                    start_time,
                    {"fulfillmentText": question},
                )
            _safe_log_recommendation_event(request, intent)
            cnn_prediction, cnn_error = _predict_accommodation_class_from_text(message)
            if cnn_prediction and isinstance(params, dict):
                # Allow the recommender (or future logic) to use the predicted class.
                params.setdefault("predicted_accommodation_type", cnn_prediction["predicted_class"])
                params.setdefault("predicted_accommodation_confidence", float(cnn_prediction.get("confidence", 0.0)))
            reply, logged_recommended_items, accommodation_meta = _get_accommodation_recommendations(params)
            next_state = {
                "pending_intent": "get_accommodation_recommendation",
                "params": params,
                "missing_slot": "",
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
        if intent not in continuation_intents:
            _clear_chat_state(request)
        reply = (
            "I can help with tourism information, tour and accommodation recommendations, and billing. "
            "Try: 'show tourism information about Bayawan attractions', "
            "'recommend a tour for 2 guests under 1500', "
            "'recommend a hotel in Bayawan for 2 guests under 2000', "
            "'book room 12 for 2 guests from 2026-03-10 to 2026-03-12', "
            "or 'calculate hotel bill for room 12 for 2 nights'."
        )

    final_reply, nlg_source = _openai_generate_final_response(
        request=request,
        intent=intent,
        user_message=raw_message,
        backend_reply=reply,
    )
    translated_back = False
    if detected_language not in ("en", "english"):
        translated_final_reply = translate_to_user_language(final_reply, detected_language)
        if str(translated_final_reply or "").strip():
            final_reply = str(translated_final_reply).strip()
            translated_back = True
            nlg_source = f"{nlg_source}|gemini_back_translate"

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
def accommodation_booking_notifications(request):
    if request.method != "GET":
        return JsonResponse({"error": "Method not allowed."}, status=405)

    user = getattr(request, "user", None)
    if not user or not getattr(user, "is_authenticated", False):
        return JsonResponse(
            {
                "fulfillmentText": "Please log in first to use this endpoint.",
                "error_code": "chat_requires_login",
            },
            status=401,
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
