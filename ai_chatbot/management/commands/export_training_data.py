import csv
import hashlib
from collections import defaultdict
from datetime import timedelta
from pathlib import Path

from django.core.management.base import BaseCommand
from django.utils import timezone

from admin_app.models import Room
from ai_chatbot.models import RecommendationResult


TEXT_HEADERS = [
    "message_id",
    "message_text",
    "label_intent",
    "label_accommodation_type",
    "label_budget_level",
    "label_location_signal",
    "source",
    "timestamp",
    "split",
]

RECO_HEADERS = [
    "record_id",
    "session_id",
    "user_id_hash",
    "query_id",
    "requested_guests",
    "requested_budget",
    "requested_location",
    "requested_accommodation_type",
    "room_id",
    "accom_id",
    "room_price_per_night",
    "room_capacity",
    "room_available",
    "accom_location",
    "company_type",
    "nights_requested",
    "cnn_intent_label",
    "cnn_confidence",
    "shown_rank",
    "timestamp",
    "was_clicked",
    "was_booked",
    "was_selected",
    "relevance_label",
]

ACCOM_INTENTS = {
    "get_accommodation_recommendation",
    "gethotelrecommendation",
}

BOOKING_INTENTS = {
    "book_accommodation",
    "bookhotel",
    "book_hotel",
    "reserve_accommodation",
}


def _to_iso(dt):
    if not dt:
        return ""
    try:
        return dt.isoformat()
    except Exception:
        return str(dt)


def _safe_int(value, default=""):
    try:
        if value in ("", None):
            return default
        return int(value)
    except Exception:
        try:
            return int(float(value))
        except Exception:
            return default


def _safe_float(value, default=""):
    try:
        if value in ("", None):
            return default
        return float(value)
    except Exception:
        return default


def _budget_level(budget):
    value = _safe_float(budget, default=None)
    if value is None:
        return "unspecified"
    if value <= 1500:
        return "low"
    if value <= 3000:
        return "mid"
    return "high"


def _location_signal(params):
    location = str((params or {}).get("location") or "").strip()
    return "with_location" if location else "without_location"


def _hash_user_id(user_id):
    if not user_id:
        return ""
    digest = hashlib.sha256(f"user:{user_id}".encode("utf-8")).hexdigest()[:10].upper()
    return f"UHASH_{digest}"


def _extract_nights(params):
    params = params or {}
    nights = _safe_int(params.get("nights"), default=None)
    if nights:
        return nights
    check_in = str(params.get("check_in") or "").strip()
    check_out = str(params.get("check_out") or "").strip()
    if check_in and check_out:
        try:
            from datetime import datetime

            ci = datetime.strptime(check_in, "%Y-%m-%d").date()
            co = datetime.strptime(check_out, "%Y-%m-%d").date()
            return max((co - ci).days, 1)
        except Exception:
            return ""
    return ""


class Command(BaseCommand):
    help = "Export chatbot/recommendation logs into Text-CNN and Decision Tree CSV training templates."

    def add_arguments(self, parser):
        parser.add_argument(
            "--output-dir",
            default="thesis_data_templates",
            help="Directory where CSV exports will be written (default: thesis_data_templates)",
        )
        parser.add_argument(
            "--since-days",
            type=int,
            default=None,
            help="Only export RecommendationResult records from the last N days",
        )
        parser.add_argument(
            "--suffix",
            default="",
            help="Optional filename suffix, e.g. pilot_week1",
        )

    def handle(self, *args, **options):
        output_dir = Path(options["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)

        qs = RecommendationResult.objects.select_related("user").order_by("generated_at")
        since_days = options.get("since_days")
        if since_days is not None:
            cutoff = timezone.now() - timedelta(days=max(int(since_days), 0))
            qs = qs.filter(generated_at__gte=cutoff)

        results = list(qs)
        if not results:
            self.stdout.write(self.style.WARNING("No RecommendationResult records found to export."))

        suffix = str(options.get("suffix") or "").strip()
        suffix_part = f"_{suffix}" if suffix else ""

        text_path = output_dir / f"text_cnn_messages_export{suffix_part}.csv"
        reco_path = output_dir / f"accommodation_reco_training_export{suffix_part}.csv"

        booking_outcomes = self._build_booking_outcome_index(results)
        text_rows = self._build_text_rows(results)
        reco_rows = self._build_reco_rows(results, booking_outcomes)

        self._write_csv(text_path, TEXT_HEADERS, text_rows)
        self._write_csv(reco_path, RECO_HEADERS, reco_rows)

        self.stdout.write(self.style.SUCCESS(f"Exported Text-CNN rows: {len(text_rows)} -> {text_path}"))
        self.stdout.write(self.style.SUCCESS(f"Exported Decision Tree rows: {len(reco_rows)} -> {reco_path}"))

    def _write_csv(self, path, headers, rows):
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            for row in rows:
                writer.writerow({h: row.get(h, "") for h in headers})

    def _build_booking_outcome_index(self, results):
        index = defaultdict(list)
        for rr in results:
            ctx = rr.context_json if isinstance(rr.context_json, dict) else {}
            intent = str(ctx.get("intent") or "").strip().lower()
            if intent not in BOOKING_INTENTS:
                continue

            outcome = ctx.get("booking_outcome") if isinstance(ctx.get("booking_outcome"), dict) else {}
            room_id = outcome.get("room_id")
            session_id = str(ctx.get("session_id") or "").strip()
            if room_id in ("", None) or not session_id:
                continue

            index[(session_id, str(room_id))].append(
                {
                    "generated_at": rr.generated_at,
                    "booking_id": outcome.get("booking_id"),
                    "booking_status": outcome.get("booking_status") or "",
                    "billing_link": outcome.get("billing_link") or "",
                    "accom_id": outcome.get("accom_id"),
                }
            )
        return index

    def _build_text_rows(self, results):
        rows = []
        seen = set()
        for i, rr in enumerate(results, start=1):
            ctx = rr.context_json if isinstance(rr.context_json, dict) else {}
            params = ctx.get("params") if isinstance(ctx.get("params"), dict) else {}
            message_text = str(ctx.get("message_text") or "").strip()
            if not message_text:
                continue

            key = (message_text, _to_iso(rr.generated_at), str(ctx.get("intent") or "").strip().lower())
            if key in seen:
                continue
            seen.add(key)

            intent = str(ctx.get("intent") or "").strip().lower()
            company_type = str(params.get("company_type") or "").strip().lower()
            if company_type not in {"hotel", "inn", "either", "unknown"}:
                company_type = "unknown"

            rows.append(
                {
                    "message_id": f"MSG{i:06d}",
                    "message_text": message_text,
                    "label_intent": intent or "",
                    "label_accommodation_type": company_type,
                    "label_budget_level": _budget_level(params.get("budget")),
                    "label_location_signal": _location_signal(params),
                    "source": "chatbot_log",
                    "timestamp": _to_iso(rr.generated_at),
                    "split": "",
                }
            )
        return rows

    def _build_reco_rows(self, results, booking_outcomes):
        rec_items = []
        room_ids = set()

        for rr in results:
            ctx = rr.context_json if isinstance(rr.context_json, dict) else {}
            intent = str(ctx.get("intent") or "").strip().lower()
            if intent not in ACCOM_INTENTS:
                continue
            if not isinstance(rr.recommended_items_json, list):
                continue

            params = ctx.get("params") if isinstance(ctx.get("params"), dict) else {}
            session_id = str(ctx.get("session_id") or "").strip()
            cnn_pred = ctx.get("cnn_prediction") if isinstance(ctx.get("cnn_prediction"), dict) else {}

            for item in rr.recommended_items_json:
                if not isinstance(item, dict):
                    continue
                room_id = item.get("room_id")
                accom_id = item.get("accom_id")
                if room_id in ("", None):
                    continue
                room_ids.add(int(room_id))
                rec_items.append(
                    {
                        "rr": rr,
                        "ctx": ctx,
                        "params": params,
                        "session_id": session_id,
                        "cnn_pred": cnn_pred,
                        "item": item,
                        "room_id": int(room_id),
                        "accom_id": accom_id,
                    }
                )

        room_map = {
            room.room_id: room
            for room in Room.objects.select_related("accommodation").filter(room_id__in=room_ids)
        }

        rows = []
        for i, payload in enumerate(rec_items, start=1):
            rr = payload["rr"]
            params = payload["params"]
            item = payload["item"]
            room_id = payload["room_id"]
            room = room_map.get(room_id)
            accom = getattr(room, "accommodation", None)

            requested_accom_type = str(params.get("company_type") or "").strip().lower()
            if not requested_accom_type:
                requested_accom_type = str(params.get("predicted_accommodation_type") or "").strip().lower()
            if not requested_accom_type:
                requested_accom_type = "either"

            session_id = payload["session_id"]
            outcome = self._match_booking_outcome(
                booking_outcomes,
                session_id=session_id,
                room_id=room_id,
                recommendation_time=rr.generated_at,
            )

            was_booked = 1 if outcome and outcome.get("booking_id") else 0
            was_selected = 1 if was_booked else 0
            was_clicked = 1 if was_selected else 0

            room_available = ""
            if room is not None:
                room_available = 1 if (
                    str(getattr(room, "status", "")).upper() == "AVAILABLE"
                    and _safe_int(getattr(room, "current_availability", None), default=0) > 0
                ) else 0

            relevance_label = ""
            if was_selected or was_booked:
                relevance_label = "relevant"
            elif room_available == 0:
                relevance_label = "not_relevant"

            cnn_conf = ""
            if payload["cnn_pred"]:
                cnn_conf = _safe_float(payload["cnn_pred"].get("confidence"), default="")

            rows.append(
                {
                    "record_id": f"REC{i:06d}",
                    "session_id": session_id,
                    "user_id_hash": _hash_user_id(getattr(rr, "user_id", None)),
                    "query_id": f"RR{rr.result_id}",
                    "requested_guests": _safe_int(params.get("guests"), default=""),
                    "requested_budget": params.get("budget", ""),
                    "requested_location": params.get("location", ""),
                    "requested_accommodation_type": requested_accom_type,
                    "room_id": room_id,
                    "accom_id": payload["accom_id"] or getattr(accom, "accom_id", ""),
                    "room_price_per_night": getattr(room, "price_per_night", "") if room else "",
                    "room_capacity": getattr(room, "person_limit", "") if room else "",
                    "room_available": room_available,
                    "accom_location": getattr(accom, "location", "") if accom else "",
                    "company_type": getattr(accom, "company_type", "") if accom else "",
                    "nights_requested": _extract_nights(params),
                    "cnn_intent_label": "get_accommodation_recommendation",
                    "cnn_confidence": cnn_conf,
                    "shown_rank": _safe_int(item.get("rank"), default=""),
                    "timestamp": _to_iso(rr.generated_at),
                    "was_clicked": was_clicked,
                    "was_booked": was_booked,
                    "was_selected": was_selected,
                    "relevance_label": relevance_label,
                }
            )
        return rows

    def _match_booking_outcome(self, booking_outcomes, *, session_id, room_id, recommendation_time):
        if not session_id:
            return None
        candidates = booking_outcomes.get((session_id, str(room_id))) or []
        if not candidates:
            return None
        for item in candidates:
            outcome_time = item.get("generated_at")
            if outcome_time is None or recommendation_time is None:
                return item
            if outcome_time >= recommendation_time:
                return item
        return candidates[0]

