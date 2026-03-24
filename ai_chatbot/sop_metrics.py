import datetime as dt
from typing import Dict, Iterable, List

from django.db.models import Avg, Count
from django.utils import timezone

from .models import (
    RecommendationEvent,
    RecommendationResult,
    SystemMetricLog,
    UsabilitySurveyResponse,
)


SOURCE_OPTIONS = {"all", "unlabeled", "demo_seeded", "pilot_test", "real_world"}
ACCOM_INTENTS = {"get_accommodation_recommendation", "gethotelrecommendation"}
DIFF_CODES = ["DIFF_DISCOVER", "DIFF_MATCH", "DIFF_PLAN", "DIFF_BOOKPAY"]
SUS_CODES = [f"SUS_Q{i}" for i in range(1, 11)]
PU_CODES = [f"PU_Q{i}" for i in range(1, 5)]
PEU_CODES = [f"PEU_Q{i}" for i in range(1, 5)]


def _normalize_source(source: str) -> str:
    value = str(source or "all").strip().lower()
    return value if value in SOURCE_OPTIONS else "all"


def _to_positive_int(value, default: int = 30, max_value: int = 3650) -> int:
    try:
        parsed = int(str(value).strip())
    except Exception:
        parsed = default
    if parsed < 1:
        parsed = default
    if parsed > max_value:
        parsed = max_value
    return parsed


def _mean(values: Iterable[float]):
    rows = [float(v) for v in values if isinstance(v, (int, float))]
    if not rows:
        return None
    return sum(rows) / len(rows)


def _sample_variance(values: List[float]):
    n = len(values)
    if n < 2:
        return None
    avg = sum(values) / n
    return sum((x - avg) ** 2 for x in values) / (n - 1)


def _cronbach_alpha_from_maps(row_maps: List[Dict[str, int]], codes: List[str]):
    if len(codes) < 2:
        return None
    vectors = []
    for row in row_maps:
        if all(isinstance(row.get(code), int) for code in codes):
            vectors.append([row.get(code) for code in codes])
    if len(vectors) < 2:
        return None

    item_variances = []
    for idx in range(len(codes)):
        vals = [float(vec[idx]) for vec in vectors]
        var = _sample_variance(vals)
        if var is None:
            return None
        item_variances.append(var)

    total_scores = [float(sum(vec)) for vec in vectors]
    total_var = _sample_variance(total_scores)
    if total_var is None or total_var <= 0:
        return None

    k = len(codes)
    return (k / (k - 1)) * (1 - (sum(item_variances) / total_var))


def _sus_score(row_map: Dict[str, int]):
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


def build_sop_metrics_snapshot(days=30, source="all"):
    days = _to_positive_int(days, default=30, max_value=3650)
    source = _normalize_source(source)
    since = timezone.now() - dt.timedelta(days=days)

    rr_qs = RecommendationResult.objects.filter(generated_at__gte=since).order_by("generated_at")
    re_qs = RecommendationEvent.objects.filter(event_time__gte=since)
    perf_qs = SystemMetricLog.objects.filter(logged_at__gte=since, module="chat")
    survey_qs = UsabilitySurveyResponse.objects.filter(submitted_at__gte=since).order_by("submitted_at")

    if source != "all":
        rr_qs = rr_qs.filter(data_source=source)
        re_qs = re_qs.filter(data_source=source)
        perf_qs = perf_qs.filter(data_source=source)
        survey_qs = survey_qs.filter(data_source=source)

    total_accom_queries = 0
    queries_with_results = 0
    top1_scores = []
    dt_scores = []
    no_match_count = 0

    for row in rr_qs:
        context = row.context_json if isinstance(row.context_json, dict) else {}
        intent = str(context.get("intent") or "").strip().lower()
        if intent not in ACCOM_INTENTS:
            continue
        total_accom_queries += 1
        items = row.recommended_items_json if isinstance(row.recommended_items_json, list) else []
        if not items:
            no_match_count += 1
            continue
        queries_with_results += 1
        first = items[0] if isinstance(items[0], dict) else {}
        if isinstance(first.get("score"), (int, float)):
            top1_scores.append(float(first["score"]))
        meta = first.get("meta") if isinstance(first.get("meta"), dict) else {}
        trace = meta.get("trace") if isinstance(meta.get("trace"), dict) else {}
        if isinstance(trace.get("decision_tree_score"), (int, float)):
            dt_scores.append(float(trace["decision_tree_score"]))

    hit_rate_pct = (queries_with_results / total_accom_queries * 100) if total_accom_queries else 0.0
    no_match_pct = (no_match_count / total_accom_queries * 100) if total_accom_queries else 0.0

    views = re_qs.filter(event_type="view").count()
    clicks = re_qs.filter(event_type="click").count()
    books = re_qs.filter(event_type="book").count()
    ctr_pct = (clicks / views * 100) if views else 0.0
    book_rate_pct = (books / views * 100) if views else 0.0

    latencies = list(perf_qs.values_list("response_time_ms", flat=True))
    total_req = len(latencies)
    success_req = perf_qs.filter(success_flag=True).count()
    success_rate_pct = (success_req / total_req * 100) if total_req else 0.0
    avg_response_ms = _mean(latencies)
    p95_response_ms = None
    if latencies:
        srt = sorted(latencies)
        idx = int(0.95 * (len(srt) - 1))
        p95_response_ms = srt[idx]

    batch_maps = {}
    difficulty_scores = {code: [] for code in DIFF_CODES}
    for row in survey_qs:
        code = str(row.statement_code or "").strip().upper()
        score = int(row.likert_score or 0)
        if score < 1 or score > 5:
            continue
        if code in difficulty_scores:
            difficulty_scores[code].append(score)
        batch_id = str(row.survey_batch_id or "").strip()
        if batch_id:
            existing = batch_maps.get(batch_id, {})
            existing[code] = score
            batch_maps[batch_id] = existing

    sus_scores = []
    pu_means = []
    peu_means = []
    for row_map in batch_maps.values():
        sus = _sus_score(row_map)
        if isinstance(sus, (int, float)):
            sus_scores.append(float(sus))
        pu_values = [row_map.get(code) for code in PU_CODES if isinstance(row_map.get(code), int)]
        peu_values = [row_map.get(code) for code in PEU_CODES if isinstance(row_map.get(code), int)]
        if pu_values:
            pu_means.append(_mean(pu_values))
        if peu_values:
            peu_means.append(_mean(peu_values))

    difficulty_by_code = {
        code: round(_mean(values), 3) if _mean(values) is not None else None
        for code, values in difficulty_scores.items()
    }

    return {
        "generated_at": timezone.now().isoformat(),
        "window": {"days": days, "source": source},
        "sop_1_difficulty": {
            "difficulty_avg_by_code": difficulty_by_code,
            "difficulty_overall_avg": round(
                _mean([score for values in difficulty_scores.values() for score in values]), 3
            )
            if any(difficulty_scores.values())
            else None,
            "response_count": sum(len(v) for v in difficulty_scores.values()),
        },
        "sop_2_functionality_readiness": {
            "chatbot_logging_enabled": True,
            "recommendation_logging_enabled": True,
            "booking_funnel_logging_enabled": True,
            "records_management_workflow_enabled": True,
            "note": "Read-only snapshot from runtime tables; does not modify transactions.",
        },
        "sop_3_performance": {
            "accommodation_queries": total_accom_queries,
            "queries_with_results": queries_with_results,
            "hit_rate_pct": round(hit_rate_pct, 2),
            "avg_top1_score": round(_mean(top1_scores), 4) if _mean(top1_scores) is not None else None,
            "avg_decision_tree_score": round(_mean(dt_scores), 4) if _mean(dt_scores) is not None else None,
            "no_match_count": no_match_count,
            "no_match_pct": round(no_match_pct, 2),
            "views": views,
            "clicks": clicks,
            "books": books,
            "ctr_pct": round(ctr_pct, 2),
            "book_rate_pct": round(book_rate_pct, 2),
            "chat_success_rate_pct": round(success_rate_pct, 2),
            "chat_avg_response_ms": round(avg_response_ms, 2) if avg_response_ms is not None else None,
            "chat_p95_response_ms": p95_response_ms,
        },
        "sop_4_acceptance": {
            "survey_batches": len(batch_maps),
            "sus_avg_score_0_100": round(_mean(sus_scores), 2) if _mean(sus_scores) is not None else None,
            "pu_avg_likert_1_5": round(_mean(pu_means), 3) if _mean(pu_means) is not None else None,
            "peu_avg_likert_1_5": round(_mean(peu_means), 3) if _mean(peu_means) is not None else None,
            "pu_cronbach_alpha": round(_cronbach_alpha_from_maps(list(batch_maps.values()), PU_CODES), 4)
            if _cronbach_alpha_from_maps(list(batch_maps.values()), PU_CODES) is not None
            else None,
            "peu_cronbach_alpha": round(_cronbach_alpha_from_maps(list(batch_maps.values()), PEU_CODES), 4)
            if _cronbach_alpha_from_maps(list(batch_maps.values()), PEU_CODES) is not None
            else None,
        },
        "row_counts": {
            "recommendation_results": rr_qs.count(),
            "recommendation_events": re_qs.count(),
            "system_metric_logs": perf_qs.count(),
            "survey_rows": survey_qs.count(),
        },
    }

