import os
from datetime import timedelta
from statistics import mean

from django.db.models import Count
from django.utils import timezone

from ai_chatbot.models import RecommendationEvent, RecommendationResult, SystemMetricLog


DAYS = 30
since = timezone.now() - timedelta(days=DAYS)
accom_intents = {"get_accommodation_recommendation", "gethotelrecommendation"}
source_filter = str(os.getenv("CHATBOT_METRICS_DATA_SOURCE", "all") or "all").strip().lower()
if source_filter not in {"all", "unlabeled", "demo_seeded", "pilot_test", "real_world"}:
    source_filter = "all"

print(f"\n=== WINDOW: last {DAYS} days (since {since}) ===")
print(f"=== DATA SOURCE FILTER: {source_filter} ===")

# 1) Recommendation quality proxy
rr = RecommendationResult.objects.filter(generated_at__gte=since).order_by("generated_at")
if source_filter != "all":
    rr = rr.filter(data_source=source_filter)
accom_rr = []
for row in rr:
    ctx = row.context_json if isinstance(row.context_json, dict) else {}
    intent = str(ctx.get("intent") or "").strip().lower()
    if intent in accom_intents:
        accom_rr.append(row)

total_queries = len(accom_rr)
with_results = 0
top1_scores = []
dt_scores = []
hybrid_count = 0
surrogate_hybrid_count = 0
fallback_count = 0

for row in accom_rr:
    items = row.recommended_items_json if isinstance(row.recommended_items_json, list) else []
    if items:
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

print("\n[1] Recommendation Quality Proxy")
print("accommodation_queries:", total_queries)
print("queries_with_results:", with_results)
print("hit_rate_pct:", round((with_results / total_queries * 100), 2) if total_queries else 0)
print("avg_top1_score:", round(mean(top1_scores), 4) if top1_scores else None)
print("avg_decision_tree_score:", round(mean(dt_scores), 4) if dt_scores else None)
print("hybrid_dt_mode_count:", hybrid_count)
print("hybrid_surrogate_tree_mode_count:", surrogate_hybrid_count)
print("fallback_mode_count:", fallback_count)

# 2) Funnel / conversion
events = RecommendationEvent.objects.filter(event_time__gte=since)
if source_filter != "all":
    events = events.filter(data_source=source_filter)
views = events.filter(event_type="view").count()
clicks = events.filter(event_type="click").count()
books = events.filter(event_type="book").count()

print("\n[2] Funnel / Conversion")
print("views:", views)
print("clicks:", clicks)
print("books:", books)
print("ctr_pct(click/view):", round((clicks / views * 100), 2) if views else 0)
print("book_rate_pct(book/view):", round((books / views * 100), 2) if views else 0)

# 3) System performance
metrics = list(
    SystemMetricLog.objects.filter(
        logged_at__gte=since,
        module="chat",
        **({"data_source": source_filter} if source_filter != "all" else {}),
    )
    .values_list("response_time_ms", flat=True)
)
total_m = len(metrics)
ok_m = SystemMetricLog.objects.filter(
    logged_at__gte=since,
    module="chat",
    success_flag=True,
    **({"data_source": source_filter} if source_filter != "all" else {}),
).count()

p95 = None
if metrics:
    sorted_metrics = sorted(metrics)
    idx = int(0.95 * (len(sorted_metrics) - 1))
    p95 = sorted_metrics[idx]

print("\n[3] Chat Performance")
print("requests:", total_m)
print("success_rate_pct:", round((ok_m / total_m * 100), 2) if total_m else 0)
print("avg_response_ms:", round(sum(metrics) / total_m, 2) if total_m else None)
print("p95_response_ms:", p95)

# 4) Failures / no-match
no_match = 0
for row in accom_rr:
    items = row.recommended_items_json if isinstance(row.recommended_items_json, list) else []
    if not items:
        no_match += 1

top_errors = (
    SystemMetricLog.objects.filter(
        logged_at__gte=since,
        **({"data_source": source_filter} if source_filter != "all" else {}),
    )
    .exclude(error_message="")
    .values("error_message")
    .annotate(n=Count("metric_id"))
    .order_by("-n")[:10]
)

print("\n[4] Failures / No-Match")
print("accommodation_no_match_count:", no_match)
print("accommodation_no_match_pct:", round((no_match / total_queries * 100), 2) if total_queries else 0)
print("top_error_messages:")
for err in top_errors:
    print(f"  - {err['n']}x | {str(err['error_message'])[:120]}")
