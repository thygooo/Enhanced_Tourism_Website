import csv
import json
from pathlib import Path

from django.core.management.base import BaseCommand
from django.utils import timezone

from ai_chatbot.models import UsabilitySurveyResponse


SOURCE_OPTIONS = {"all", "unlabeled", "demo_seeded", "pilot_test", "real_world"}
SUS_CODES = [f"SUS_Q{i}" for i in range(1, 11)]
PU_CODES = [f"PU_Q{i}" for i in range(1, 5)]
PEU_CODES = [f"PEU_Q{i}" for i in range(1, 5)]
DIFF_CODES = ["DIFF_DISCOVER", "DIFF_MATCH", "DIFF_PLAN", "DIFF_BOOKPAY"]
FULL_CODES = set(SUS_CODES + PU_CODES + PEU_CODES + DIFF_CODES)


def _normalize_source(source: str) -> str:
    value = str(source or "all").strip().lower()
    return value if value in SOURCE_OPTIONS else "all"


class Command(BaseCommand):
    help = (
        "Export survey readiness audit (complete/incomplete batches, missing codes, and gap-to-target). "
        "Read-only analytics command."
    )

    def add_arguments(self, parser):
        parser.add_argument(
            "--source",
            type=str,
            default="all",
            choices=sorted(SOURCE_OPTIONS),
        )
        parser.add_argument("--target-complete-batches", type=int, default=30)
        parser.add_argument(
            "--out-json",
            type=str,
            default="thesis_data_templates/survey_readiness_audit.json",
        )
        parser.add_argument(
            "--out-csv",
            type=str,
            default="thesis_data_templates/survey_readiness_incomplete_batches.csv",
        )

    def handle(self, *args, **options):
        source = _normalize_source(str(options.get("source") or "all"))
        target_batches = max(int(options.get("target_complete_batches") or 30), 1)
        out_json = Path(str(options.get("out_json")))
        out_csv = Path(str(options.get("out_csv")))

        qs = UsabilitySurveyResponse.objects.all().order_by("submitted_at")
        if source != "all":
            qs = qs.filter(data_source=source)

        batches = {}
        for row in qs.iterator():
            batch_id = str(row.survey_batch_id or "").strip()
            if not batch_id:
                batch_id = f"no_batch__row_{row.response_id}"
            payload = batches.get(
                batch_id,
                {
                    "batch_id": batch_id,
                    "codes": set(),
                    "rows": 0,
                    "first_submitted_at": row.submitted_at,
                    "last_submitted_at": row.submitted_at,
                    "data_sources": set(),
                    "user_ids": set(),
                },
            )
            code = str(row.statement_code or "").strip().upper()
            payload["codes"].add(code)
            payload["rows"] += 1
            payload["first_submitted_at"] = min(payload["first_submitted_at"], row.submitted_at)
            payload["last_submitted_at"] = max(payload["last_submitted_at"], row.submitted_at)
            payload["data_sources"].add(str(row.data_source or ""))
            if row.user_id is not None:
                payload["user_ids"].add(str(row.user_id))
            batches[batch_id] = payload

        complete_batches = 0
        sus_complete = 0
        pu_complete = 0
        peu_complete = 0
        diff_complete = 0
        incomplete_rows = []
        for batch_id, payload in batches.items():
            codes = payload["codes"]
            full_missing = sorted(list(FULL_CODES.difference(codes)))
            sus_missing = sorted(list(set(SUS_CODES).difference(codes)))
            pu_missing = sorted(list(set(PU_CODES).difference(codes)))
            peu_missing = sorted(list(set(PEU_CODES).difference(codes)))
            diff_missing = sorted(list(set(DIFF_CODES).difference(codes)))

            if not full_missing:
                complete_batches += 1
            if not sus_missing:
                sus_complete += 1
            if not pu_missing:
                pu_complete += 1
            if not peu_missing:
                peu_complete += 1
            if not diff_missing:
                diff_complete += 1

            if full_missing:
                incomplete_rows.append(
                    {
                        "batch_id": batch_id,
                        "rows": payload["rows"],
                        "missing_count": len(full_missing),
                        "missing_codes": ",".join(full_missing),
                        "first_submitted_at": payload["first_submitted_at"].isoformat()
                        if payload["first_submitted_at"]
                        else "",
                        "last_submitted_at": payload["last_submitted_at"].isoformat()
                        if payload["last_submitted_at"]
                        else "",
                        "data_sources": ",".join(sorted([v for v in payload["data_sources"] if v])),
                        "user_count": len(payload["user_ids"]),
                    }
                )

        audit = {
            "generated_at": timezone.now().isoformat(),
            "source": source,
            "target_complete_batches": target_batches,
            "totals": {
                "survey_rows": qs.count(),
                "distinct_batches": len(batches),
                "complete_full_batches": complete_batches,
                "complete_sus_batches": sus_complete,
                "complete_pu_batches": pu_complete,
                "complete_peu_batches": peu_complete,
                "complete_difficulty_batches": diff_complete,
            },
            "gaps_to_target": {
                "full_batches_needed": max(0, target_batches - complete_batches),
                "sus_batches_needed": max(0, target_batches - sus_complete),
                "pu_batches_needed": max(0, target_batches - pu_complete),
                "peu_batches_needed": max(0, target_batches - peu_complete),
                "difficulty_batches_needed": max(0, target_batches - diff_complete),
            },
            "notes": [
                "Use this audit to prioritize collection of complete survey batches.",
                "A complete full batch contains SUS + PU + PEU + difficulty codes.",
            ],
        }

        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")

        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with out_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "batch_id",
                    "rows",
                    "missing_count",
                    "missing_codes",
                    "first_submitted_at",
                    "last_submitted_at",
                    "data_sources",
                    "user_count",
                ],
            )
            writer.writeheader()
            writer.writerows(incomplete_rows)

        self.stdout.write(self.style.SUCCESS(f"Saved JSON: {out_json}"))
        self.stdout.write(self.style.SUCCESS(f"Saved CSV: {out_csv}"))
