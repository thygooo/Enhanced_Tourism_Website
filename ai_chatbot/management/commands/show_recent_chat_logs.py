import json

from django.core.management.base import BaseCommand

from ai_chatbot.models import RecommendationEvent, RecommendationResult, SystemMetricLog


def _short(value, max_len=180):
    text = str(value or "")
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


class Command(BaseCommand):
    help = "Print recent chatbot recommendation/event/system logs for quick backend inspection."

    def add_arguments(self, parser):
        parser.add_argument(
            "--limit",
            type=int,
            default=5,
            help="Number of recent rows per section (default: 5)",
        )
        parser.add_argument(
            "--section",
            choices=["all", "results", "events", "metrics"],
            default="all",
            help="Show only one section or all",
        )

    def handle(self, *args, **options):
        limit = max(int(options.get("limit") or 5), 1)
        section = options.get("section") or "all"

        if section in ("all", "results"):
            self._show_results(limit)
        if section in ("all", "events"):
            self._show_events(limit)
        if section in ("all", "metrics"):
            self._show_metrics(limit)

    def _show_results(self, limit):
        self.stdout.write("\n=== RecommendationResult (Recent) ===")
        rows = RecommendationResult.objects.select_related("user").order_by("-generated_at")[:limit]
        if not rows:
            self.stdout.write("No recommendation results found.")
            return

        for rr in rows:
            ctx = rr.context_json if isinstance(rr.context_json, dict) else {}
            items = rr.recommended_items_json if isinstance(rr.recommended_items_json, list) else []
            msg = ctx.get("message_text", "")
            intent = ctx.get("intent", "")
            session_id = ctx.get("session_id", "")
            params = ctx.get("params", {})
            booking_outcome = ctx.get("booking_outcome", {})

            self.stdout.write(
                f"[RR#{rr.result_id}] {rr.generated_at} | user={rr.user_id} | intent={intent} | top_k={rr.top_k}"
            )
            self.stdout.write(f"  session_id: {session_id}")
            self.stdout.write(f"  message: {_short(msg)}")
            self.stdout.write(f"  params: {_short(json.dumps(params, ensure_ascii=False))}")
            self.stdout.write(f"  clicked_item_ref: {_short(rr.clicked_item_ref)}")
            if booking_outcome:
                self.stdout.write(f"  booking_outcome: {_short(json.dumps(booking_outcome, ensure_ascii=False))}")
            if items:
                self.stdout.write(f"  recommended_items_json[{min(len(items), 3)} shown]:")
                for item in items[:3]:
                    self.stdout.write(f"    - {_short(json.dumps(item, ensure_ascii=False))}")
            self.stdout.write("")

    def _show_events(self, limit):
        self.stdout.write("\n=== RecommendationEvent (Recent) ===")
        rows = RecommendationEvent.objects.select_related("user").order_by("-event_time")[:limit]
        if not rows:
            self.stdout.write("No recommendation events found.")
            return

        for ev in rows:
            self.stdout.write(
                f"[EV#{ev.event_id}] {ev.event_time} | user={ev.user_id} | type={ev.event_type} | "
                f"item_ref={_short(ev.item_ref, 120)} | session_id={_short(ev.session_id, 40)}"
            )

    def _show_metrics(self, limit):
        self.stdout.write("\n=== SystemMetricLog (Recent) ===")
        rows = SystemMetricLog.objects.order_by("-logged_at")[:limit]
        if not rows:
            self.stdout.write("No system metrics found.")
            return

        for m in rows:
            self.stdout.write(
                f"[MT#{m.metric_id}] {m.logged_at} | {m.module} | {m.endpoint} | "
                f"{m.response_time_ms}ms | status={m.status_code} | success={m.success_flag}"
            )
            if m.error_message:
                self.stdout.write(f"  error: {_short(m.error_message)}")

