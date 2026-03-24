import argparse
import csv
import json
from datetime import timedelta

from django.db.models import Avg
from django.utils import timezone

from admin_app.models import Accomodation, Room, TourismInformation
from ai_chatbot.models import (
    RecommendationEvent,
    RecommendationResult,
    SystemMetricLog,
    UsabilitySurveyResponse,
)
from guest_app.models import AccommodationBooking


SOURCE_OPTIONS = {"all", "unlabeled", "demo_seeded", "pilot_test", "real_world"}
DIFF_CODES = ["DIFF_DISCOVER", "DIFF_MATCH", "DIFF_PLAN", "DIFF_BOOKPAY"]
SUS_CODES = [f"SUS_Q{i}" for i in range(1, 11)]
PU_CODES = [f"PU_Q{i}" for i in range(1, 5)]
PEU_CODES = [f"PEU_Q{i}" for i in range(1, 5)]
ACCOM_INTENTS = {"get_accommodation_recommendation", "gethotelrecommendation"}


def _normalize_source(raw):
    value = str(raw or "all").strip().lower()
    if value not in SOURCE_OPTIONS:
        return "all"
    return value


def _mean(values):
    nums = [float(v) for v in values if isinstance(v, (int, float))]
    if not nums:
        return None
    return sum(nums) / len(nums)


def _sus_score(row_map):
    total = 0
    for idx in range(1, 11):
        code = f"SUS_Q{idx}"
        raw = row_map.get(code)
        if not isinstance(raw, int):
            return None
        if idx % 2 == 1:
            total += (raw - 1)
        else:
            total += (5 - raw)
    return total * 2.5


def _build_formula_lock():
    return {
        "booking_conversion_efficiency": {
            "metric_name": "book_rate_pct",
            "numerator": "count(RecommendationEvent where event_type='book')",
            "denominator": (
                "count(RecommendationEvent where event_type='view' "
                "and item_ref='chat:accommodation_recommendation_request')"
            ),
            "formula": "(numerator / denominator) * 100",
            "scope_note": "Accommodation recommendation funnel only.",
            "status": "locked",
        },
        "recommendation_accuracy_distinction": {
            "true_accuracy": {
                "definition": "Agreement against labeled relevance ground truth (e.g., relevant/not_relevant).",
                "required_inputs": [
                    "labeled evaluation dataset",
                    "prediction labels",
                    "ground-truth labels",
                ],
                "formula_examples": ["accuracy", "precision", "recall", "f1"],
                "status": "not_fully_measurable_online_yet",
            },
            "online_effectiveness_proxies": [
                {
                    "name": "queries_with_results_rate",
                    "formula": "queries_with_results / accommodation_queries * 100",
                },
                {
                    "name": "ctr_pct",
                    "formula": "clicks / views * 100",
                },
                {
                    "name": "book_rate_pct",
                    "formula": "books / views * 100",
                },
            ],
            "reporting_rule": (
                "Do not label online proxies as true recommendation accuracy in Chapter 4."
            ),
            "status": "locked",
        },
    }


def _filter_by_source(qs, source):
    if source == "all":
        return qs
    return qs.filter(data_source=source)


def _survey_batch_maps(survey_rows):
    batches = {}
    for row in survey_rows:
        code = str(row.statement_code or "").strip().upper()
        score = int(row.likert_score or 0)
        if score < 1 or score > 5:
            continue
        batch_id = str(row.survey_batch_id or "").strip()
        if not batch_id:
            continue
        payload = batches.get(batch_id, {})
        payload[code] = score
        batches[batch_id] = payload
    return batches


def _compute_metrics(days=30, source="all"):
    since = timezone.now() - timedelta(days=max(int(days), 1))
    source = _normalize_source(source)

    survey_qs = _filter_by_source(
        UsabilitySurveyResponse.objects.filter(submitted_at__gte=since),
        source,
    )
    reco_event_qs = _filter_by_source(
        RecommendationEvent.objects.filter(event_time__gte=since),
        source,
    )
    reco_result_qs = _filter_by_source(
        RecommendationResult.objects.filter(generated_at__gte=since),
        source,
    )
    metric_log_qs = _filter_by_source(
        SystemMetricLog.objects.filter(logged_at__gte=since, module="chat"),
        source,
    )

    diff_means = {}
    for code in DIFF_CODES:
        avg = survey_qs.filter(statement_code=code).aggregate(avg_val=Avg("likert_score"))["avg_val"]
        diff_means[code] = round(float(avg), 3) if avg is not None else ""

    batch_maps = _survey_batch_maps(list(survey_qs))
    sus_scores = []
    pu_means = []
    peu_means = []
    for row_map in batch_maps.values():
        sus = _sus_score(row_map)
        if isinstance(sus, (int, float)):
            sus_scores.append(float(sus))
        pu_vals = [row_map.get(code) for code in PU_CODES if isinstance(row_map.get(code), int)]
        peu_vals = [row_map.get(code) for code in PEU_CODES if isinstance(row_map.get(code), int)]
        if pu_vals:
            pu_means.append(_mean(pu_vals))
        if peu_vals:
            peu_means.append(_mean(peu_vals))

    views = reco_event_qs.filter(
        event_type="view", item_ref="chat:accommodation_recommendation_request"
    ).count()
    clicks = reco_event_qs.filter(event_type="click").count()
    books = reco_event_qs.filter(event_type="book").count()
    ctr_pct = (clicks / views * 100) if views else 0.0
    book_rate_pct = (books / views * 100) if views else 0.0

    accom_queries = 0
    queries_with_results = 0
    for row in reco_result_qs:
        ctx = row.context_json if isinstance(row.context_json, dict) else {}
        intent = str(ctx.get("intent") or "").strip().lower()
        if intent not in ACCOM_INTENTS:
            continue
        accom_queries += 1
        items = row.recommended_items_json if isinstance(row.recommended_items_json, list) else []
        if items:
            queries_with_results += 1
    queries_with_results_rate = (queries_with_results / accom_queries * 100) if accom_queries else 0.0

    chat_latencies = list(metric_log_qs.values_list("response_time_ms", flat=True))
    chat_avg_ms = round(_mean(chat_latencies), 2) if chat_latencies else ""
    chat_success = metric_log_qs.filter(success_flag=True).count()
    chat_total = metric_log_qs.count()
    chat_success_pct = round((chat_success / chat_total * 100), 2) if chat_total else 0.0

    approved_accommodations = Accomodation.objects.filter(approval_status="accepted").count()
    total_rooms = Room.objects.count()
    published_tourism_spots = TourismInformation.objects.filter(
        publication_status="published", is_active=True
    ).count()
    booking_total = AccommodationBooking.objects.count()

    return {
        "days": days,
        "source": source,
        "difficulty": diff_means,
        "sus_avg_score_0_100": round(_mean(sus_scores), 2) if sus_scores else "",
        "pu_avg_likert_1_5": round(_mean(pu_means), 3) if pu_means else "",
        "peu_avg_likert_1_5": round(_mean(peu_means), 3) if peu_means else "",
        "views": views,
        "clicks": clicks,
        "books": books,
        "ctr_pct": round(ctr_pct, 2),
        "book_rate_pct": round(book_rate_pct, 2),
        "accommodation_queries": accom_queries,
        "queries_with_results": queries_with_results,
        "queries_with_results_rate": round(queries_with_results_rate, 2),
        "chat_avg_response_ms": chat_avg_ms,
        "chat_success_rate_pct": chat_success_pct,
        "approved_accommodations": approved_accommodations,
        "total_rooms": total_rooms,
        "published_tourism_spots": published_tourism_spots,
        "accommodation_bookings_total": booking_total,
    }


def _build_sop2_traceability():
    return [
        {
            "sop_item": "2.1",
            "requirement": "Chatbot interaction and real-time assistance",
            "feature_evidence": "Chat endpoint with stateful handling and logging",
            "code_anchor": "ai_chatbot/views.py:openai_chat + ai_chatbot/models.py:ChatbotLog,SystemMetricLog",
            "testable_output": "chat logs, response-time logs",
            "status": "partially_ready",
            "gap_note": "Needs formal requirement acceptance checklist artifact.",
        },
        {
            "sop_item": "2.2",
            "requirement": "Personalized recommendations using CNN and Decision Trees",
            "feature_evidence": "Accommodation recommendation path with trace metadata and CNN intent support",
            "code_anchor": "ai_chatbot/recommenders.py + ai_chatbot/views.py + ai_chatbot/models.py:RecommendationResult",
            "testable_output": "recommendation result rows, score traces",
            "status": "partially_ready",
            "gap_note": "True labeled accuracy evidence still needs locked evaluation protocol.",
        },
        {
            "sop_item": "2.3",
            "requirement": "Automated booking and billing for hotels/inns",
            "feature_evidence": "Billing and booking endpoints with room-linked records",
            "code_anchor": "guest_app/views.py:accommodation_billing,accommodation_book + guest_app/models.py:AccommodationBooking",
            "testable_output": "booking records, bill computations, booking events",
            "status": "partially_ready",
            "gap_note": "Need Chapter 4 conversion denominator lock and reporting narrative.",
        },
        {
            "sop_item": "2.4",
            "requirement": "Alignment with records management, promotion, feedback, reports",
            "feature_evidence": "Owner/admin approvals, room management, tourism publication status, survey reporting",
            "code_anchor": "admin_app/views.py + admin_app/models.py:TourismInformation + admin_app/views.py:survey_results_api",
            "testable_output": "approval records, published spots, survey API/export outputs",
            "status": "partially_ready",
            "gap_note": "Needs explicit requirement-to-workflow matrix in Chapter 4 appendices.",
        },
    ]


def _build_sop_evidence_rows(metrics):
    return [
        {
            "sop_item": "1.1",
            "indicator": "Difficulty in discovering suitable hotels/inns",
            "variable": "DIFF_DISCOVER mean",
            "metric_value": metrics["difficulty"].get("DIFF_DISCOVER", ""),
            "output_type": "Likert mean (1-5)",
            "readiness": "partially_ready",
            "notes": "Needs larger real-world sample for defense confidence.",
        },
        {
            "sop_item": "1.2",
            "indicator": "Difficulty in matching options to preferences",
            "variable": "DIFF_MATCH mean",
            "metric_value": metrics["difficulty"].get("DIFF_MATCH", ""),
            "output_type": "Likert mean (1-5)",
            "readiness": "partially_ready",
            "notes": "",
        },
        {
            "sop_item": "1.3",
            "indicator": "Difficulty in planning stay-related itinerary",
            "variable": "DIFF_PLAN mean",
            "metric_value": metrics["difficulty"].get("DIFF_PLAN", ""),
            "output_type": "Likert mean (1-5)",
            "readiness": "partially_ready",
            "notes": "",
        },
        {
            "sop_item": "1.4",
            "indicator": "Difficulty in completing bookings/payments",
            "variable": "DIFF_BOOKPAY mean",
            "metric_value": metrics["difficulty"].get("DIFF_BOOKPAY", ""),
            "output_type": "Likert mean (1-5)",
            "readiness": "partially_ready",
            "notes": "Payment workflow is process-level; no external gateway completion metric yet.",
        },
        {
            "sop_item": "3.1",
            "indicator": "Recommendation accuracy distinction lock",
            "variable": "true_accuracy_vs_proxy",
            "metric_value": "see formula lock JSON",
            "output_type": "definition lock",
            "readiness": "partially_ready",
            "notes": "True accuracy requires labeled relevance; current runtime evidence is effectiveness proxy.",
        },
        {
            "sop_item": "3.2",
            "indicator": "System Usability Scale",
            "variable": "sus_avg_score_0_100",
            "metric_value": metrics["sus_avg_score_0_100"],
            "output_type": "SUS score (0-100)",
            "readiness": "partially_ready",
            "notes": "Thesis-ready after adequate respondent count.",
        },
        {
            "sop_item": "3.3",
            "indicator": "User engagement",
            "variable": "ctr_pct",
            "metric_value": metrics["ctr_pct"],
            "output_type": "percentage",
            "readiness": "thesis_ready",
            "notes": "Uses RecommendationEvent click/view logs.",
        },
        {
            "sop_item": "3.4",
            "indicator": "Booking conversion efficiency",
            "variable": "book_rate_pct",
            "metric_value": metrics["book_rate_pct"],
            "output_type": "percentage",
            "readiness": "partially_ready",
            "notes": "Denominator locked to accommodation recommendation views only.",
        },
        {
            "sop_item": "4.1",
            "indicator": "Perceived Usefulness",
            "variable": "pu_avg_likert_1_5",
            "metric_value": metrics["pu_avg_likert_1_5"],
            "output_type": "Likert mean (1-5)",
            "readiness": "partially_ready",
            "notes": "Needs enough complete survey batches for stronger inference.",
        },
        {
            "sop_item": "4.2",
            "indicator": "Perceived Ease of Use",
            "variable": "peu_avg_likert_1_5",
            "metric_value": metrics["peu_avg_likert_1_5"],
            "output_type": "Likert mean (1-5)",
            "readiness": "partially_ready",
            "notes": "Needs enough complete survey batches for stronger inference.",
        },
    ]


def _write_csv(path, rows, fieldnames):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main(
    *,
    days=30,
    data_source="all",
    out_matrix_csv="thesis_data_templates/chapter4_sop_evidence_matrix.csv",
    out_matrix_json="thesis_data_templates/chapter4_sop_evidence_matrix.json",
    out_formula_json="thesis_data_templates/chapter4_formula_lock.json",
    out_sop2_csv="thesis_data_templates/chapter4_sop2_traceability.csv",
    out_sop2_json="thesis_data_templates/chapter4_sop2_traceability.json",
):
    metrics = _compute_metrics(days=days, source=data_source)
    formula_lock = _build_formula_lock()
    sop2_rows = _build_sop2_traceability()
    matrix_rows = _build_sop_evidence_rows(metrics)

    _write_csv(
        out_matrix_csv,
        matrix_rows,
        [
            "sop_item",
            "indicator",
            "variable",
            "metric_value",
            "output_type",
            "readiness",
            "notes",
        ],
    )
    _write_csv(
        out_sop2_csv,
        sop2_rows,
        [
            "sop_item",
            "requirement",
            "feature_evidence",
            "code_anchor",
            "testable_output",
            "status",
            "gap_note",
        ],
    )

    with open(out_matrix_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "generated_at": timezone.now().isoformat(),
                "window": {"days": days, "source": _normalize_source(data_source)},
                "metrics_snapshot": metrics,
                "rows": matrix_rows,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    with open(out_formula_json, "w", encoding="utf-8") as f:
        json.dump(formula_lock, f, indent=2, ensure_ascii=False)
    with open(out_sop2_json, "w", encoding="utf-8") as f:
        json.dump(sop2_rows, f, indent=2, ensure_ascii=False)

    print(f"Saved: {out_matrix_csv}")
    print(f"Saved: {out_matrix_json}")
    print(f"Saved: {out_formula_json}")
    print(f"Saved: {out_sop2_csv}")
    print(f"Saved: {out_sop2_json}")
    print(f"Source filter: {_normalize_source(data_source)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Export chapter-ready SOP evidence tables and formula locks (read-only)."
    )
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--data-source", default="all", choices=sorted(SOURCE_OPTIONS))
    parser.add_argument("--out-matrix-csv", default="thesis_data_templates/chapter4_sop_evidence_matrix.csv")
    parser.add_argument("--out-matrix-json", default="thesis_data_templates/chapter4_sop_evidence_matrix.json")
    parser.add_argument("--out-formula-json", default="thesis_data_templates/chapter4_formula_lock.json")
    parser.add_argument("--out-sop2-csv", default="thesis_data_templates/chapter4_sop2_traceability.csv")
    parser.add_argument("--out-sop2-json", default="thesis_data_templates/chapter4_sop2_traceability.json")
    args = parser.parse_args()

    main(
        days=max(int(args.days), 1),
        data_source=args.data_source,
        out_matrix_csv=args.out_matrix_csv,
        out_matrix_json=args.out_matrix_json,
        out_formula_json=args.out_formula_json,
        out_sop2_csv=args.out_sop2_csv,
        out_sop2_json=args.out_sop2_json,
    )
