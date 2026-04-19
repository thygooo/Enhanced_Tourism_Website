import csv
import json
from datetime import timedelta
from pathlib import Path

from django.core.management.base import BaseCommand
from django.utils import timezone

from ai_chatbot.models import ChatbotLog, RecommendationEvent, RecommendationResult


def _normalize_source(source):
    value = str(source or "all").strip().lower()
    allowed = {"all", "unlabeled", "demo_seeded", "pilot_test", "real_world"}
    return value if value in allowed else "all"


def _extract_session_id(provenance_json):
    if not isinstance(provenance_json, dict):
        return ""
    session_id = str(provenance_json.get("session_id") or "").strip()
    if session_id:
        return session_id
    extra = provenance_json.get("extra") if isinstance(provenance_json.get("extra"), dict) else {}
    return str(extra.get("session_id") or "").strip()


class Command(BaseCommand):
    help = "Export chatbot conversation evidence (CSV + JSON) for thesis/demo use."

    def add_arguments(self, parser):
        parser.add_argument("--days", type=int, default=30)
        parser.add_argument(
            "--source",
            type=str,
            default="all",
            choices=["all", "unlabeled", "demo_seeded", "pilot_test", "real_world"],
        )
        parser.add_argument(
            "--out-csv",
            type=str,
            default="thesis_data_templates/chatbot_conversation_evidence.csv",
        )
        parser.add_argument(
            "--out-json",
            type=str,
            default="thesis_data_templates/chatbot_conversation_evidence.json",
        )
        parser.add_argument(
            "--out-kpi-csv",
            type=str,
            default="thesis_data_templates/chatbot_quality_kpis.csv",
        )

    def handle(self, *args, **options):
        days = max(int(options.get("days") or 30), 1)
        source = _normalize_source(options.get("source"))
        out_csv = Path(str(options.get("out_csv")))
        out_json = Path(str(options.get("out_json")))
        out_kpi_csv = Path(str(options.get("out_kpi_csv")))
        since = timezone.now() - timedelta(days=days)

        logs_qs = ChatbotLog.objects.filter(created_at__gte=since).order_by("created_at")
        events_qs = RecommendationEvent.objects.filter(event_time__gte=since).order_by("event_time")
        results_qs = RecommendationResult.objects.filter(generated_at__gte=since).order_by("generated_at")

        if source != "all":
            logs_qs = logs_qs.filter(data_source=source)
            events_qs = events_qs.filter(data_source=source)
            results_qs = results_qs.filter(data_source=source)

        events_by_session = {}
        for event in events_qs.iterator():
            key = str(event.session_id or "").strip()
            if not key:
                continue
            events_by_session.setdefault(key, []).append(event)

        results_by_session = {}
        for result in results_qs.iterator():
            ctx = result.context_json if isinstance(result.context_json, dict) else {}
            key = str(ctx.get("session_id") or "").strip()
            if not key:
                continue
            results_by_session.setdefault(key, []).append(result)

        rows = []
        for row in logs_qs.iterator():
            session_id = _extract_session_id(row.provenance_json)
            matching_events = events_by_session.get(session_id, []) if session_id else []
            matching_results = results_by_session.get(session_id, []) if session_id else []

            recommendation_trace_count = 0
            if matching_results:
                latest = matching_results[-1]
                items = latest.recommended_items_json if isinstance(latest.recommended_items_json, list) else []
                recommendation_trace_count = len(items)

            row_payload = {
                "created_at": row.created_at.isoformat() if row.created_at else "",
                "log_id": row.log_id,
                "user_id": row.user_id or "",
                "session_id": session_id,
                "resolved_intent": row.resolved_intent,
                "intent_classifier_source": row.intent_classifier_source,
                "response_nlg_source": row.response_nlg_source,
                "fallback_used": int(bool(row.fallback_used)),
                "user_message": row.user_message,
                "bot_response": row.bot_response,
                "params_json": json.dumps(row.resolved_params_json, ensure_ascii=False),
                "step_event_count_in_session": len(matching_events),
                "step_event_click_count_in_session": sum(1 for ev in matching_events if ev.event_type == "click"),
                "step_event_book_count_in_session": sum(1 for ev in matching_events if ev.event_type == "book"),
                "recommendation_result_count_in_session": len(matching_results),
                "latest_recommendation_trace_count": recommendation_trace_count,
            }
            rows.append(row_payload)

        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with out_csv.open("w", newline="", encoding="utf-8") as handle:
            fieldnames = [
                "created_at",
                "log_id",
                "user_id",
                "session_id",
                "resolved_intent",
                "intent_classifier_source",
                "response_nlg_source",
                "fallback_used",
                "user_message",
                "bot_response",
                "params_json",
                "step_event_count_in_session",
                "step_event_click_count_in_session",
                "step_event_book_count_in_session",
                "recommendation_result_count_in_session",
                "latest_recommendation_trace_count",
            ]
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        total = len(rows)
        nlg_success = sum(
            1 for r in rows
            if str(r.get("response_nlg_source") or "").strip().lower() in {"openai_nlg", "gemini_nlg", "gemini_nlg_retry"}
        )
        nlg_source_present = sum(1 for r in rows if str(r.get("response_nlg_source") or "").strip())
        fallback_count = sum(1 for r in rows if int(r.get("fallback_used") or 0) == 1)
        clarification_count = sum(
            1 for r in rows
            if str(r.get("resolved_intent") or "").strip().lower() in {"clarification", "role_help", "out_of_scope"}
            or "clarify" in str(r.get("bot_response") or "").lower()
        )
        accommodation_reco_count = sum(
            1 for r in rows
            if str(r.get("resolved_intent") or "").strip().lower() in {"get_accommodation_recommendation", "gethotelrecommendation"}
        )
        booking_count = sum(
            1 for r in rows
            if str(r.get("resolved_intent") or "").strip().lower() in {"book_accommodation", "bookhotel", "book_hotel", "reserve_accommodation"}
        )

        def _rate(numer, denom):
            return round((float(numer) / float(denom) * 100.0), 4) if denom else 0.0

        kpi_rows = [
            {"kpi": "total_chat_logs", "value": total, "formula": "count(rows)"},
            {"kpi": "fallback_rate_pct", "value": _rate(fallback_count, total), "formula": "fallback_used_count / total_logs * 100"},
            {"kpi": "nlg_source_present_rate_pct", "value": _rate(nlg_source_present, total), "formula": "logs_with_response_nlg_source / total_logs * 100"},
            {"kpi": "nlg_success_rate_pct", "value": _rate(nlg_success, total), "formula": "logs_with_success_nlg_source / total_logs * 100"},
            {"kpi": "clarification_rate_pct", "value": _rate(clarification_count, total), "formula": "clarification_like_logs / total_logs * 100"},
            {"kpi": "accommodation_recommendation_intent_share_pct", "value": _rate(accommodation_reco_count, total), "formula": "accommodation_recommendation_logs / total_logs * 100"},
            {"kpi": "booking_intent_share_pct", "value": _rate(booking_count, total), "formula": "booking_logs / total_logs * 100"},
        ]

        out_kpi_csv.parent.mkdir(parents=True, exist_ok=True)
        with out_kpi_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["kpi", "value", "formula"])
            writer.writeheader()
            writer.writerows(kpi_rows)

        bundle = {
            "generated_at": timezone.now().isoformat(),
            "days": days,
            "source": source,
            "rows_exported": len(rows),
            "csv_path": str(out_csv),
            "kpi_csv_path": str(out_kpi_csv),
            "kpi_summary": {row["kpi"]: row["value"] for row in kpi_rows},
            "records": rows,
        }
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(bundle, indent=2, ensure_ascii=False), encoding="utf-8")

        self.stdout.write(self.style.SUCCESS("Chatbot conversation evidence export complete."))
        self.stdout.write(f"- rows_exported: {len(rows)}")
        self.stdout.write(f"- csv: {out_csv}")
        self.stdout.write(f"- kpi_csv: {out_kpi_csv}")
        self.stdout.write(f"- json: {out_json}")
