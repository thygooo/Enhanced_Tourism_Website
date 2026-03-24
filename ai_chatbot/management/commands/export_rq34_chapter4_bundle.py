from django.core.management.base import BaseCommand

from ai_chatbot.rq34_metrics import export_rq34_bundle


class Command(BaseCommand):
    help = (
        "Export chapter-ready RQ3/RQ4 bundle (CSV + JSON) with locked formulas and "
        "clear true-accuracy vs proxy distinction. Read-only analytics export only."
    )

    def add_arguments(self, parser):
        parser.add_argument("--days", type=int, default=30)
        parser.add_argument(
            "--source",
            type=str,
            default="all",
            choices=["all", "unlabeled", "demo_seeded", "pilot_test", "real_world"],
        )
        parser.add_argument(
            "--out-rq3-csv",
            type=str,
            default="thesis_data_templates/chapter4_rq3_metrics.csv",
        )
        parser.add_argument(
            "--out-rq4-csv",
            type=str,
            default="thesis_data_templates/chapter4_rq4_acceptance.csv",
        )
        parser.add_argument(
            "--out-rq4-items-csv",
            type=str,
            default="thesis_data_templates/chapter4_rq4_item_stats.csv",
        )
        parser.add_argument(
            "--out-metric-defs-csv",
            type=str,
            default="thesis_data_templates/chapter4_metric_definitions.csv",
        )
        parser.add_argument(
            "--out-bundle-json",
            type=str,
            default="thesis_data_templates/chapter4_rq3_rq4_bundle.json",
        )

    def handle(self, *args, **options):
        result = export_rq34_bundle(
            days=max(int(options.get("days") or 30), 1),
            source=str(options.get("source") or "all"),
            out_rq3_csv=str(options.get("out_rq3_csv")),
            out_rq4_csv=str(options.get("out_rq4_csv")),
            out_rq4_items_csv=str(options.get("out_rq4_items_csv")),
            out_metric_defs_csv=str(options.get("out_metric_defs_csv")),
            out_bundle_json=str(options.get("out_bundle_json")),
        )
        self.stdout.write(self.style.SUCCESS("RQ3/RQ4 Chapter 4 bundle export complete."))
        for key, path in result.items():
            self.stdout.write(f"- {key}: {path}")
