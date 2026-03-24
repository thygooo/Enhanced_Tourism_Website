from django.core.management.base import BaseCommand

from chapter4_sop_evidence_export import main as export_main


class Command(BaseCommand):
    help = (
        "Export Chapter 4 SOP evidence bundle (matrix, formula lock, SOP2 traceability). "
        "Read-only analytics export; does not modify transactional workflows."
    )

    def add_arguments(self, parser):
        parser.add_argument("--days", type=int, default=30)
        parser.add_argument(
            "--data-source",
            type=str,
            default="all",
            choices=["all", "unlabeled", "demo_seeded", "pilot_test", "real_world"],
        )
        parser.add_argument(
            "--out-matrix-csv",
            type=str,
            default="thesis_data_templates/chapter4_sop_evidence_matrix.csv",
        )
        parser.add_argument(
            "--out-matrix-json",
            type=str,
            default="thesis_data_templates/chapter4_sop_evidence_matrix.json",
        )
        parser.add_argument(
            "--out-formula-json",
            type=str,
            default="thesis_data_templates/chapter4_formula_lock.json",
        )
        parser.add_argument(
            "--out-sop2-csv",
            type=str,
            default="thesis_data_templates/chapter4_sop2_traceability.csv",
        )
        parser.add_argument(
            "--out-sop2-json",
            type=str,
            default="thesis_data_templates/chapter4_sop2_traceability.json",
        )

    def handle(self, *args, **options):
        export_main(
            days=max(int(options.get("days") or 30), 1),
            data_source=str(options.get("data_source") or "all"),
            out_matrix_csv=str(options.get("out_matrix_csv")),
            out_matrix_json=str(options.get("out_matrix_json")),
            out_formula_json=str(options.get("out_formula_json")),
            out_sop2_csv=str(options.get("out_sop2_csv")),
            out_sop2_json=str(options.get("out_sop2_json")),
        )
        self.stdout.write(self.style.SUCCESS("Chapter 4 SOP evidence bundle export complete."))

