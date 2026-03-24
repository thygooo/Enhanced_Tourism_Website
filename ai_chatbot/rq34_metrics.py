import csv
import json
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from django.db.models import Avg, Count
from django.utils import timezone

from ai_chatbot.models import (
    ChatbotLog,
    RecommendationEvent,
    RecommendationResult,
    SystemMetricLog,
    UsabilitySurveyResponse,
)


SOURCE_OPTIONS = {"all", "unlabeled", "demo_seeded", "pilot_test", "real_world"}
DIFF_CODES = ["DIFF_DISCOVER", "DIFF_MATCH", "DIFF_PLAN", "DIFF_BOOKPAY"]
SUS_CODES = [f"SUS_Q{i}" for i in range(1, 11)]
PU_CODES = [f"PU_Q{i}" for i in range(1, 5)]
PEU_CODES = [f"PEU_Q{i}" for i in range(1, 5)]
ACCOM_INTENTS = {"get_accommodation_recommendation", "gethotelrecommendation"}
ACCOM_VIEW_ITEM_REF = "chat:accommodation_recommendation_request"


def _normalize_source(source: str) -> str:
    value = str(source or "all").strip().lower()
    return value if value in SOURCE_OPTIONS else "all"


def _mean(values: Iterable[float]):
    nums = [float(v) for v in values if isinstance(v, (int, float))]
    if not nums:
        return None
    return sum(nums) / len(nums)


def _sample_std(values: Iterable[float]):
    nums = [float(v) for v in values if isinstance(v, (int, float))]
    n = len(nums)
    if n < 2:
        return None
    mu = sum(nums) / n
    var = sum((x - mu) ** 2 for x in nums) / (n - 1)
    return var ** 0.5


def _sus_score(batch_map: Dict[str, int]):
    total = 0
    for idx in range(1, 11):
        code = f"SUS_Q{idx}"
        raw = batch_map.get(code)
        if not isinstance(raw, int):
            return None
        if idx % 2 == 1:
            total += raw - 1
        else:
            total += 5 - raw
    return total * 2.5


def _cronbach_alpha(batch_maps: List[Dict[str, int]], codes: List[str]):
    if len(codes) < 2:
        return None

    vectors = []
    for row in batch_maps:
        if all(isinstance(row.get(code), int) for code in codes):
            vectors.append([row[code] for code in codes])
    if len(vectors) < 2:
        return None

    item_vars = []
    for i in range(len(codes)):
        col = [float(v[i]) for v in vectors]
        s = _sample_std(col)
        if s is None:
            return None
        item_vars.append(s * s)

    totals = [float(sum(v)) for v in vectors]
    total_std = _sample_std(totals)
    if total_std is None:
        return None
    total_var = total_std * total_std
    if total_var <= 0:
        return None

    k = len(codes)
    return (k / (k - 1)) * (1 - (sum(item_vars) / total_var))


def _write_csv(path: Path, rows: List[Dict], fieldnames: List[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _filter_source(qs, source: str):
    if source == "all":
        return qs
    return qs.filter(data_source=source)


def _survey_batches(survey_rows: Iterable[UsabilitySurveyResponse]):
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


def _extract_item_scores(batch_maps: List[Dict[str, int]], codes: List[str]):
    item_stats = []
    for code in codes:
        vals = [row.get(code) for row in batch_maps if isinstance(row.get(code), int)]
        mean_v = _mean(vals)
        std_v = _sample_std(vals)
        item_stats.append(
            {
                "statement_code": code,
                "n": len(vals),
                "mean_1_5": round(mean_v, 4) if mean_v is not None else "",
                "sd_1_5": round(std_v, 4) if std_v is not None else "",
            }
        )
    return item_stats


def build_rq34_bundle(*, days: int, source: str):
    days = max(int(days or 30), 1)
    source = _normalize_source(source)
    since = timezone.now() - timedelta(days=days)

    rr_qs = _filter_source(RecommendationResult.objects.filter(generated_at__gte=since), source)
    re_qs = _filter_source(RecommendationEvent.objects.filter(event_time__gte=since), source)
    sl_qs = _filter_source(SystemMetricLog.objects.filter(logged_at__gte=since, module="chat"), source)
    cl_qs = _filter_source(ChatbotLog.objects.filter(created_at__gte=since), source)
    us_qs = _filter_source(UsabilitySurveyResponse.objects.filter(submitted_at__gte=since), source)

    # RQ3.1: true accuracy vs online proxies
    accom_queries = 0
    queries_with_results = 0
    top1_scores = []
    for row in rr_qs.iterator():
        ctx = row.context_json if isinstance(row.context_json, dict) else {}
        intent = str(ctx.get("intent") or "").strip().lower()
        if intent not in ACCOM_INTENTS:
            continue
        accom_queries += 1
        items = row.recommended_items_json if isinstance(row.recommended_items_json, list) else []
        if not items:
            continue
        queries_with_results += 1
        top = items[0] if isinstance(items[0], dict) else {}
        score = top.get("score")
        if isinstance(score, (int, float)):
            top1_scores.append(float(score))

    queries_with_results_rate = (queries_with_results / accom_queries * 100) if accom_queries else 0.0
    views = re_qs.filter(event_type="view", item_ref=ACCOM_VIEW_ITEM_REF).count()
    clicks = re_qs.filter(event_type="click").count()
    books = re_qs.filter(event_type="book").count()
    ctr_pct = (clicks / views * 100) if views else 0.0
    book_rate_pct = (books / views * 100) if views else 0.0

    # RQ3.2: SUS
    batches = _survey_batches(us_qs.iterator())
    batch_rows = list(batches.values())
    sus_scores = []
    for row in batch_rows:
        sus = _sus_score(row)
        if isinstance(sus, (int, float)):
            sus_scores.append(float(sus))

    sus_mean = _mean(sus_scores)
    sus_sd = _sample_std(sus_scores)

    # RQ3.3: engagement
    unique_event_sessions = (
        re_qs.exclude(session_id="")
        .values("session_id")
        .distinct()
        .count()
    )
    unique_chat_users = cl_qs.values("user_id").distinct().count()

    # RQ3.4: conversion efficiency lock
    conversion_formula = "(book_events / recommendation_view_events) * 100"
    conversion_denominator_definition = (
        "RecommendationEvent where event_type='view' and item_ref='chat:accommodation_recommendation_request'"
    )

    # RQ4: acceptance (PU/PEU)
    pu_means = []
    peu_means = []
    for row in batch_rows:
        pu_vals = [row.get(code) for code in PU_CODES if isinstance(row.get(code), int)]
        peu_vals = [row.get(code) for code in PEU_CODES if isinstance(row.get(code), int)]
        if pu_vals:
            pu_means.append(float(_mean(pu_vals)))
        if peu_vals:
            peu_means.append(float(_mean(peu_vals)))

    pu_mean = _mean(pu_means)
    pu_sd = _sample_std(pu_means)
    peu_mean = _mean(peu_means)
    peu_sd = _sample_std(peu_means)
    pu_alpha = _cronbach_alpha(batch_rows, PU_CODES)
    peu_alpha = _cronbach_alpha(batch_rows, PEU_CODES)

    item_stats = []
    item_stats.extend(_extract_item_scores(batch_rows, PU_CODES))
    item_stats.extend(_extract_item_scores(batch_rows, PEU_CODES))
    item_stats.extend(_extract_item_scores(batch_rows, SUS_CODES))
    item_stats.extend(_extract_item_scores(batch_rows, DIFF_CODES))

    # Optional true-accuracy source from locked Decision Tree offline evaluation artifact.
    true_accuracy = None
    true_accuracy_n = 0
    true_accuracy_source = ""
    dt_eval_path = Path("artifacts/decision_tree_final/evaluation_metrics_v1.json")
    if dt_eval_path.exists():
        try:
            dt_eval = json.loads(dt_eval_path.read_text(encoding="utf-8"))
            acc_raw = dt_eval.get("accuracy")
            test_rows_raw = dt_eval.get("test_rows")
            if isinstance(acc_raw, (int, float)):
                true_accuracy = float(acc_raw) * 100.0
                true_accuracy_source = str(dt_eval_path)
            if isinstance(test_rows_raw, int):
                true_accuracy_n = int(test_rows_raw)
        except Exception:
            true_accuracy = None
            true_accuracy_n = 0
            true_accuracy_source = ""

    rq3_rows = [
        {
            "rq_item": "3.1",
            "indicator": "Recommendation accuracy (true metric)",
            "metric_name": "true_accuracy_labeled",
            "formula": "correct_predictions / total_labeled_samples * 100",
            "value": round(true_accuracy, 4) if true_accuracy is not None else "",
            "n": true_accuracy_n,
            "status": "available_offline_eval" if true_accuracy is not None else "missing",
            "notes": (
                f"Offline labeled evaluation artifact: {true_accuracy_source}"
                if true_accuracy_source
                else "Not measurable from online runtime logs alone; requires labeled relevance evaluation dataset."
            ),
        },
        {
            "rq_item": "3.1",
            "indicator": "Recommendation effectiveness proxy",
            "metric_name": "queries_with_results_rate_pct",
            "formula": "queries_with_results / accommodation_queries * 100",
            "value": round(queries_with_results_rate, 4),
            "n": accom_queries,
            "status": "available_proxy",
            "notes": "Use as online effectiveness proxy only, not true accuracy.",
        },
        {
            "rq_item": "3.1",
            "indicator": "Recommendation effectiveness proxy",
            "metric_name": "avg_top1_score",
            "formula": "mean(top_1_recommendation_score)",
            "value": round(_mean(top1_scores), 6) if _mean(top1_scores) is not None else "",
            "n": len(top1_scores),
            "status": "available_proxy",
            "notes": "Model score proxy from recommendation trace.",
        },
        {
            "rq_item": "3.2",
            "indicator": "System Usability Scale",
            "metric_name": "sus_mean_0_100",
            "formula": "mean(SUS batch scores)",
            "value": round(sus_mean, 4) if sus_mean is not None else "",
            "n": len(sus_scores),
            "status": "available",
            "notes": "Pair with SD and respondent count in Chapter 4.",
        },
        {
            "rq_item": "3.2",
            "indicator": "System Usability Scale",
            "metric_name": "sus_sd_0_100",
            "formula": "sample_sd(SUS batch scores)",
            "value": round(sus_sd, 4) if sus_sd is not None else "",
            "n": len(sus_scores),
            "status": "available",
            "notes": "",
        },
        {
            "rq_item": "3.3",
            "indicator": "User engagement",
            "metric_name": "ctr_pct",
            "formula": "click_events / recommendation_view_events * 100",
            "value": round(ctr_pct, 4),
            "n": views,
            "status": "available",
            "notes": "Recommendation event funnel.",
        },
        {
            "rq_item": "3.3",
            "indicator": "User engagement",
            "metric_name": "unique_recommendation_sessions",
            "formula": "count(distinct RecommendationEvent.session_id)",
            "value": unique_event_sessions,
            "n": unique_event_sessions,
            "status": "available",
            "notes": "Session-level engagement volume.",
        },
        {
            "rq_item": "3.3",
            "indicator": "User engagement",
            "metric_name": "unique_chat_users",
            "formula": "count(distinct ChatbotLog.user_id)",
            "value": unique_chat_users,
            "n": unique_chat_users,
            "status": "available",
            "notes": "User-level chat participation.",
        },
        {
            "rq_item": "3.4",
            "indicator": "Booking conversion efficiency",
            "metric_name": "book_rate_pct",
            "formula": conversion_formula,
            "value": round(book_rate_pct, 4),
            "n": views,
            "status": "available",
            "notes": "Denominator locked to accommodation recommendation view events only.",
        },
    ]

    rq4_rows = [
        {
            "rq_item": "4.1",
            "construct": "Perceived Usefulness (PU)",
            "metric_name": "pu_mean_1_5",
            "formula": "mean(mean(PU_Q1..PU_Q4) per complete batch)",
            "value": round(pu_mean, 4) if pu_mean is not None else "",
            "n": len(pu_means),
            "status": "available",
            "notes": "",
        },
        {
            "rq_item": "4.1",
            "construct": "Perceived Usefulness (PU)",
            "metric_name": "pu_sd_1_5",
            "formula": "sample_sd(mean(PU_Q1..PU_Q4) per complete batch)",
            "value": round(pu_sd, 4) if pu_sd is not None else "",
            "n": len(pu_means),
            "status": "available",
            "notes": "",
        },
        {
            "rq_item": "4.1",
            "construct": "Perceived Usefulness (PU)",
            "metric_name": "pu_cronbach_alpha",
            "formula": "Cronbach alpha across PU_Q1..PU_Q4",
            "value": round(pu_alpha, 6) if pu_alpha is not None else "",
            "n": len(batch_rows),
            "status": "available",
            "notes": "Internal consistency reliability.",
        },
        {
            "rq_item": "4.2",
            "construct": "Perceived Ease of Use (PEU)",
            "metric_name": "peu_mean_1_5",
            "formula": "mean(mean(PEU_Q1..PEU_Q4) per complete batch)",
            "value": round(peu_mean, 4) if peu_mean is not None else "",
            "n": len(peu_means),
            "status": "available",
            "notes": "",
        },
        {
            "rq_item": "4.2",
            "construct": "Perceived Ease of Use (PEU)",
            "metric_name": "peu_sd_1_5",
            "formula": "sample_sd(mean(PEU_Q1..PEU_Q4) per complete batch)",
            "value": round(peu_sd, 4) if peu_sd is not None else "",
            "n": len(peu_means),
            "status": "available",
            "notes": "",
        },
        {
            "rq_item": "4.2",
            "construct": "Perceived Ease of Use (PEU)",
            "metric_name": "peu_cronbach_alpha",
            "formula": "Cronbach alpha across PEU_Q1..PEU_Q4",
            "value": round(peu_alpha, 6) if peu_alpha is not None else "",
            "n": len(batch_rows),
            "status": "available",
            "notes": "Internal consistency reliability.",
        },
    ]

    definition_rows = [
        {
            "metric_name": "book_rate_pct",
            "locked_formula": conversion_formula,
            "denominator_lock": conversion_denominator_definition,
            "metric_type": "RQ3 proxy metric",
            "interpretation_rule": "Report as conversion efficiency only.",
        },
        {
            "metric_name": "true_accuracy_labeled",
            "locked_formula": "correct_predictions / total_labeled_samples * 100",
            "denominator_lock": "labeled relevance evaluation set size",
            "metric_type": "RQ3 true accuracy",
            "interpretation_rule": "Do not substitute online proxies for this metric.",
        },
        {
            "metric_name": "queries_with_results_rate_pct",
            "locked_formula": "queries_with_results / accommodation_queries * 100",
            "denominator_lock": "accommodation recommendation queries",
            "metric_type": "RQ3 online proxy",
            "interpretation_rule": "Label explicitly as effectiveness proxy.",
        },
        {
            "metric_name": "ctr_pct",
            "locked_formula": "click_events / recommendation_view_events * 100",
            "denominator_lock": "recommendation view events",
            "metric_type": "RQ3 online proxy",
            "interpretation_rule": "Engagement indicator, not true accuracy.",
        },
    ]

    metadata = {
        "generated_at": timezone.now().isoformat(),
        "window_days": days,
        "source": source,
        "counts": {
            "recommendation_results": rr_qs.count(),
            "recommendation_events": re_qs.count(),
            "system_metric_logs": sl_qs.count(),
            "chatbot_logs": cl_qs.count(),
            "survey_rows": us_qs.count(),
            "survey_batches_with_batch_id": len(batch_rows),
        },
        "accuracy_disclaimer": (
            "True recommendation accuracy is not computed from runtime logs only. "
            "Use labeled relevance evaluation dataset for true accuracy metrics."
        ),
        "offline_true_accuracy_source": true_accuracy_source,
        "offline_true_accuracy_pct": round(true_accuracy, 4) if true_accuracy is not None else "",
        "offline_true_accuracy_test_rows": true_accuracy_n,
    }

    return {
        "metadata": metadata,
        "rq3_rows": rq3_rows,
        "rq4_rows": rq4_rows,
        "rq4_item_rows": item_stats,
        "metric_definition_rows": definition_rows,
    }


def export_rq34_bundle(
    *,
    days: int = 30,
    source: str = "all",
    out_rq3_csv: str = "thesis_data_templates/chapter4_rq3_metrics.csv",
    out_rq4_csv: str = "thesis_data_templates/chapter4_rq4_acceptance.csv",
    out_rq4_items_csv: str = "thesis_data_templates/chapter4_rq4_item_stats.csv",
    out_metric_defs_csv: str = "thesis_data_templates/chapter4_metric_definitions.csv",
    out_bundle_json: str = "thesis_data_templates/chapter4_rq3_rq4_bundle.json",
):
    bundle = build_rq34_bundle(days=days, source=source)
    rq3_path = Path(out_rq3_csv)
    rq4_path = Path(out_rq4_csv)
    rq4_items_path = Path(out_rq4_items_csv)
    defs_path = Path(out_metric_defs_csv)
    bundle_path = Path(out_bundle_json)

    _write_csv(
        rq3_path,
        bundle["rq3_rows"],
        ["rq_item", "indicator", "metric_name", "formula", "value", "n", "status", "notes"],
    )
    _write_csv(
        rq4_path,
        bundle["rq4_rows"],
        ["rq_item", "construct", "metric_name", "formula", "value", "n", "status", "notes"],
    )
    _write_csv(
        rq4_items_path,
        bundle["rq4_item_rows"],
        ["statement_code", "n", "mean_1_5", "sd_1_5"],
    )
    _write_csv(
        defs_path,
        bundle["metric_definition_rows"],
        ["metric_name", "locked_formula", "denominator_lock", "metric_type", "interpretation_rule"],
    )

    bundle_path.parent.mkdir(parents=True, exist_ok=True)
    bundle_path.write_text(json.dumps(bundle, indent=2, ensure_ascii=False), encoding="utf-8")

    return {
        "rq3_csv": str(rq3_path),
        "rq4_csv": str(rq4_path),
        "rq4_items_csv": str(rq4_items_path),
        "metric_defs_csv": str(defs_path),
        "bundle_json": str(bundle_path),
    }
