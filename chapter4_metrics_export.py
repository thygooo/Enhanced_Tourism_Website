import argparse
import csv
from datetime import timedelta
from statistics import mean

from django.db.models import Count
from django.utils import timezone

from ai_chatbot.models import RecommendationEvent, RecommendationResult, SystemMetricLog


DAYS = 30
OUT_FILE = "thesis_data_templates/chapter4_metrics_summary.csv"
ACCOM_INTENTS = {"get_accommodation_recommendation", "gethotelrecommendation"}
SOURCE_OPTIONS = {"all", "unlabeled", "demo_seeded", "pilot_test", "real_world"}


def _normalize_data_source(raw_value):
    source = str(raw_value or "all").strip().lower()
    if source not in SOURCE_OPTIONS:
        return "all"
    return source


def _resolve_source_filter(data_source, real_only=False):
    if real_only:
        return "real_world"
    return _normalize_data_source(data_source)


def main(*, days=DAYS, out_file=OUT_FILE, data_source="all", real_only=False):
    selected_source = _resolve_source_filter(data_source, real_only=real_only)
    since = timezone.now() - timedelta(days=days)
    window_label = f"last_{days}_days"
    if selected_source != "all":
        window_label = f"{window_label}_{selected_source}"

    rr_qs = RecommendationResult.objects.filter(generated_at__gte=since).order_by("generated_at")
    if selected_source != "all":
        rr_qs = rr_qs.filter(data_source=selected_source)

    accom_rr = []
    for row in rr_qs:
        ctx = row.context_json if isinstance(row.context_json, dict) else {}
        intent = str(ctx.get("intent") or "").strip().lower()
        if intent in ACCOM_INTENTS:
            accom_rr.append(row)

    total_queries = len(accom_rr)
    with_results = 0
    top1_scores = []
    dt_scores = []
    hybrid_count = 0
    surrogate_hybrid_count = 0
    fallback_count = 0
    no_match = 0

    for row in accom_rr:
        items = row.recommended_items_json if isinstance(row.recommended_items_json, list) else []
        if not items:
            no_match += 1
            continue

        with_results += 1
        first = items[0] if isinstance(items[0], dict) else {}

        score = first.get("score")
        if isinstance(score, (int, float)):
            top1_scores.append(float(score))

        meta = first.get("meta", {}) if isinstance(first.get("meta"), dict) else {}
        trace = meta.get("trace", {}) if isinstance(meta.get("trace"), dict) else {}

        dt_score = trace.get("decision_tree_score")
        if isinstance(dt_score, (int, float)):
            dt_scores.append(float(dt_score))

        mode = str(trace.get("scoring_mode") or "").strip().lower()
        if mode in ("hybrid_textcnn_decisiontree", "hybrid_textcnn_surrogate_tree"):
            hybrid_count += 1
            if mode == "hybrid_textcnn_surrogate_tree":
                surrogate_hybrid_count += 1
        elif mode == "hybrid_fallback_heuristic":
            fallback_count += 1

    hit_rate_pct = (with_results / total_queries * 100) if total_queries else 0
    avg_top1_score = mean(top1_scores) if top1_scores else None
    avg_dt_score = mean(dt_scores) if dt_scores else None
    no_match_pct = (no_match / total_queries * 100) if total_queries else 0

    events = RecommendationEvent.objects.filter(event_time__gte=since)
    if selected_source != "all":
        events = events.filter(data_source=selected_source)
    views = events.filter(event_type="view").count()
    clicks = events.filter(event_type="click").count()
    books = events.filter(event_type="book").count()
    ctr_pct = (clicks / views * 100) if views else 0
    book_rate_pct = (books / views * 100) if views else 0

    perf_qs = SystemMetricLog.objects.filter(logged_at__gte=since, module="chat")
    if selected_source != "all":
        perf_qs = perf_qs.filter(data_source=selected_source)
    latencies = list(perf_qs.values_list("response_time_ms", flat=True))
    total_req = len(latencies)
    ok_req = perf_qs.filter(success_flag=True).count()
    success_rate_pct = (ok_req / total_req * 100) if total_req else 0
    avg_response_ms = (sum(latencies) / total_req) if total_req else None

    p95_response_ms = None
    if latencies:
        srt = sorted(latencies)
        idx = int(0.95 * (len(srt) - 1))
        p95_response_ms = srt[idx]

    top_errors_qs = perf_qs.exclude(error_message="").values("error_message").annotate(n=Count("metric_id"))
    top_errors = top_errors_qs.order_by("-n")[:5]
    top_errors_text = " | ".join(
        [f"{entry['n']}x:{str(entry['error_message'])[:80]}" for entry in top_errors]
    )
    source_note = (
        "all data sources"
        if selected_source == "all"
        else f"filtered by data_source={selected_source}"
    )

    rows = [
        {
            "metric_group": "recommendation",
            "metric_name": "accommodation_queries",
            "metric_value": total_queries,
            "window": window_label,
            "notes": source_note,
        },
        {
            "metric_group": "recommendation",
            "metric_name": "queries_with_results",
            "metric_value": with_results,
            "window": window_label,
            "notes": "",
        },
        {
            "metric_group": "recommendation",
            "metric_name": "hit_rate_pct",
            "metric_value": round(hit_rate_pct, 2),
            "window": window_label,
            "notes": "with_results / accommodation_queries",
        },
        {
            "metric_group": "recommendation",
            "metric_name": "avg_top1_score",
            "metric_value": round(avg_top1_score, 4) if avg_top1_score is not None else "",
            "window": window_label,
            "notes": "top-ranked item score",
        },
        {
            "metric_group": "recommendation",
            "metric_name": "avg_decision_tree_score",
            "metric_value": round(avg_dt_score, 4) if avg_dt_score is not None else "",
            "window": window_label,
            "notes": "from recommendation trace",
        },
        {
            "metric_group": "recommendation",
            "metric_name": "hybrid_dt_mode_count",
            "metric_value": hybrid_count,
            "window": window_label,
            "notes": "scoring_mode in (hybrid_textcnn_decisiontree, hybrid_textcnn_surrogate_tree)",
        },
        {
            "metric_group": "recommendation",
            "metric_name": "hybrid_surrogate_tree_mode_count",
            "metric_value": surrogate_hybrid_count,
            "window": window_label,
            "notes": "scoring_mode=hybrid_textcnn_surrogate_tree",
        },
        {
            "metric_group": "recommendation",
            "metric_name": "fallback_mode_count",
            "metric_value": fallback_count,
            "window": window_label,
            "notes": "scoring_mode=hybrid_fallback_heuristic",
        },
        {
            "metric_group": "recommendation",
            "metric_name": "no_match_count",
            "metric_value": no_match,
            "window": window_label,
            "notes": "recommendation list empty",
        },
        {
            "metric_group": "recommendation",
            "metric_name": "no_match_pct",
            "metric_value": round(no_match_pct, 2),
            "window": window_label,
            "notes": "no_match_count / accommodation_queries",
        },
        {
            "metric_group": "funnel",
            "metric_name": "views",
            "metric_value": views,
            "window": window_label,
            "notes": "RecommendationEvent view",
        },
        {
            "metric_group": "funnel",
            "metric_name": "clicks",
            "metric_value": clicks,
            "window": window_label,
            "notes": "RecommendationEvent click",
        },
        {
            "metric_group": "funnel",
            "metric_name": "books",
            "metric_value": books,
            "window": window_label,
            "notes": "RecommendationEvent book",
        },
        {
            "metric_group": "funnel",
            "metric_name": "ctr_pct",
            "metric_value": round(ctr_pct, 2),
            "window": window_label,
            "notes": "clicks / views",
        },
        {
            "metric_group": "funnel",
            "metric_name": "book_rate_pct",
            "metric_value": round(book_rate_pct, 2),
            "window": window_label,
            "notes": "books / views",
        },
        {
            "metric_group": "performance",
            "metric_name": "chat_requests",
            "metric_value": total_req,
            "window": window_label,
            "notes": "SystemMetricLog module=chat",
        },
        {
            "metric_group": "performance",
            "metric_name": "chat_success_rate_pct",
            "metric_value": round(success_rate_pct, 2),
            "window": window_label,
            "notes": "success_flag=true / total",
        },
        {
            "metric_group": "performance",
            "metric_name": "chat_avg_response_ms",
            "metric_value": round(avg_response_ms, 2) if avg_response_ms is not None else "",
            "window": window_label,
            "notes": "",
        },
        {
            "metric_group": "performance",
            "metric_name": "chat_p95_response_ms",
            "metric_value": p95_response_ms if p95_response_ms is not None else "",
            "window": window_label,
            "notes": "",
        },
        {
            "metric_group": "errors",
            "metric_name": "top_error_messages",
            "metric_value": top_errors_text,
            "window": window_label,
            "notes": "up to top 5",
        },
    ]

    with open(out_file, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(
            csvfile,
            fieldnames=["metric_group", "metric_name", "metric_value", "window", "notes"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved: {out_file}")
    print(f"Rows: {len(rows)}")
    print(f"Data source filter: {selected_source}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Export Chapter 4 metrics summary with optional data_source filtering.",
    )
    parser.add_argument("--days", type=int, default=DAYS)
    parser.add_argument("--out-file", default=OUT_FILE)
    parser.add_argument(
        "--data-source",
        default="all",
        choices=sorted(SOURCE_OPTIONS),
        help="Filter logs by data_source. Use 'all' to include everything.",
    )
    parser.add_argument(
        "--real-only",
        action="store_true",
        help="Shortcut for --data-source real_world.",
    )
    args = parser.parse_args()

    main(
        days=max(int(args.days), 1),
        out_file=args.out_file,
        data_source=args.data_source,
        real_only=bool(args.real_only),
    )
