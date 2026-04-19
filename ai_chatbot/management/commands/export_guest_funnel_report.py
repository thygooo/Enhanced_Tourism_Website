import json
from pathlib import Path

import pandas as pd
from django.core.management.base import BaseCommand
from django.utils import timezone

from ai_chatbot.models import RecommendationEvent


FUNNEL_STAGES = [
    ("chatbot_opened", ["chat:funnel_chatbot_opened"]),
    ("quick_start_clicked", ["chat:funnel_quick_start_clicked"]),
    (
        "recommendation_shown",
        ["chat:funnel_recommendation_shown", "chat:accommodation_recommendation_rendered"],
    ),
    ("recommendation_card_clicked", ["chat:funnel_recommendation_card_clicked"]),
    ("book_button_clicked", ["chat:funnel_book_button_clicked"]),
    ("booking_flow_started", ["chat:funnel_booking_flow_started", "chat:accommodation_booking_draft_or_pending"]),
    ("billing_link_shown", ["chat:funnel_billing_link_shown", "chat:lgu_payment_handoff_ready"]),
    ("billing_link_clicked", ["chat:funnel_billing_link_clicked", "chat:billing_link_click"]),
    ("booking_completed", ["chat:funnel_booking_completed", "chat:accommodation_booking_confirmed"]),
]


def _safe_div(num, den):
    if not den:
        return 0.0
    return float(num) / float(den)


class Command(BaseCommand):
    help = "Export guest chatbot funnel events for RQ3.3/RQ3.4 evidence."

    def add_arguments(self, parser):
        parser.add_argument("--days", type=int, default=30, help="Window size in days (default: 30).")
        parser.add_argument(
            "--out-dir",
            default="thesis_data_templates",
            help="Output directory for CSV/JSON report.",
        )
        parser.add_argument(
            "--label",
            default="",
            help="Optional suffix label for file names (e.g., before_ux, after_ux).",
        )

    def handle(self, *args, **options):
        days = max(int(options.get("days") or 30), 1)
        label = str(options.get("label") or "").strip().lower().replace(" ", "_")
        out_dir = Path(str(options.get("out_dir") or "thesis_data_templates")).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)

        since = timezone.now() - timezone.timedelta(days=days)
        base_qs = RecommendationEvent.objects.filter(event_time__gte=since)

        rows = []
        previous_count = 0
        first_stage_count = 0

        for index, (stage, refs) in enumerate(FUNNEL_STAGES):
            qs = base_qs.filter(item_ref__in=refs)
            count = qs.count()
            unique_sessions = qs.exclude(session_id="").values("session_id").distinct().count()
            unique_users = qs.values("user_id").distinct().count()
            if index == 0:
                first_stage_count = count
            step_conversion = _safe_div(count, previous_count) * 100 if previous_count else 0.0
            overall_conversion = _safe_div(count, first_stage_count) * 100 if first_stage_count else 0.0
            dropoff_from_previous = max(previous_count - count, 0) if previous_count else 0
            rows.append(
                {
                    "stage": stage,
                    "event_refs": " | ".join(refs),
                    "count": count,
                    "unique_sessions": unique_sessions,
                    "unique_users": unique_users,
                    "step_conversion_pct": round(step_conversion, 4),
                    "overall_from_chat_open_pct": round(overall_conversion, 4),
                    "dropoff_from_previous": dropoff_from_previous,
                }
            )
            previous_count = count

        stage_df = pd.DataFrame(rows)

        summary = {
            "generated_at": timezone.now().isoformat(),
            "window_days": days,
            "stage_rows": rows,
            "headline_metrics": {
                "chatbot_opened_count": int(stage_df.loc[stage_df["stage"] == "chatbot_opened", "count"].sum()),
                "recommendation_shown_count": int(stage_df.loc[stage_df["stage"] == "recommendation_shown", "count"].sum()),
                "book_button_clicked_count": int(stage_df.loc[stage_df["stage"] == "book_button_clicked", "count"].sum()),
                "booking_completed_count": int(stage_df.loc[stage_df["stage"] == "booking_completed", "count"].sum()),
            },
        }

        chat_open = summary["headline_metrics"]["chatbot_opened_count"]
        rec_shown = summary["headline_metrics"]["recommendation_shown_count"]
        booked = summary["headline_metrics"]["booking_completed_count"]
        summary["headline_metrics"]["recommendation_show_rate_from_chat_open_pct"] = round(_safe_div(rec_shown, chat_open) * 100, 4)
        summary["headline_metrics"]["booking_completion_from_chat_open_pct"] = round(_safe_div(booked, chat_open) * 100, 4)
        summary["headline_metrics"]["booking_completion_from_recommendation_shown_pct"] = round(_safe_div(booked, rec_shown) * 100, 4)

        suffix = f"_{label}" if label else ""
        csv_path = out_dir / f"guest_funnel_report{suffix}.csv"
        json_path = out_dir / f"guest_funnel_report{suffix}.json"
        stage_df.to_csv(csv_path, index=False, encoding="utf-8")
        json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

        self.stdout.write(
            json.dumps(
                {
                    "csv": str(csv_path),
                    "json": str(json_path),
                    "window_days": days,
                    "headline_metrics": summary["headline_metrics"],
                },
                indent=2,
            )
        )
