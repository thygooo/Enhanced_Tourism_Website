import json
from pathlib import Path

from django.core.management.base import BaseCommand

from ai_chatbot.sop_metrics import build_sop_metrics_snapshot


class Command(BaseCommand):
    help = (
        "Export a read-only SOP metrics snapshot from existing runtime tables. "
        "This command does not change transactional data."
    )

    def add_arguments(self, parser):
        parser.add_argument("--days", type=int, default=30, help="Window size in days (default: 30).")
        parser.add_argument(
            "--source",
            type=str,
            default="all",
            help="Data source filter: all|unlabeled|demo_seeded|pilot_test|real_world",
        )
        parser.add_argument(
            "--out",
            type=str,
            default="thesis_data_templates/sop_metrics_snapshot.json",
            help="Output JSON path.",
        )

    def handle(self, *args, **options):
        days = int(options.get("days") or 30)
        source = str(options.get("source") or "all").strip()
        out_path = Path(str(options.get("out") or "thesis_data_templates/sop_metrics_snapshot.json")).resolve()

        payload = build_sop_metrics_snapshot(days=days, source=source)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

        self.stdout.write(self.style.SUCCESS(f"SOP metrics snapshot exported: {out_path}"))

