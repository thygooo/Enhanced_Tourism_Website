import argparse
import csv
from collections import defaultdict
from datetime import timedelta
from statistics import mean, pstdev

from django.utils import timezone

from ai_chatbot.models import UsabilitySurveyResponse


OUT_FILE = "thesis_data_templates/chapter4_survey_summary.csv"
SOURCE_OPTIONS = {"all", "unlabeled", "demo_seeded", "pilot_test", "real_world"}
SUS_CODES = [f"SUS_Q{i}" for i in range(1, 11)]
PU_CODES = [f"PU_Q{i}" for i in range(1, 5)]
PEU_CODES = [f"PEU_Q{i}" for i in range(1, 5)]
DIFF_CODES = ["DIFF_DISCOVER", "DIFF_MATCH", "DIFF_PLAN", "DIFF_BOOKPAY"]


def _normalize_data_source(raw):
    source = str(raw or "all").strip().lower()
    if source not in SOURCE_OPTIONS:
        return "all"
    return source


def _sus_score_from_rows(rows_by_code):
    # SUS scoring: odd (x-1), even (5-x), then *2.5
    total = 0
    for idx in range(1, 11):
        code = f"SUS_Q{idx}"
        raw = rows_by_code.get(code)
        if raw is None:
            return None
        score = int(raw)
        if idx % 2 == 1:
            total += (score - 1)
        else:
            total += (5 - score)
    return round(total * 2.5, 2)


def _sample_variance(values):
    if values is None:
        return None
    nums = [float(v) for v in values if isinstance(v, (int, float))]
    n = len(nums)
    if n < 2:
        return None
    avg = sum(nums) / n
    return sum((val - avg) ** 2 for val in nums) / (n - 1)


def _cronbach_alpha_from_rows(rows, codes):
    """
    Compute Cronbach's alpha from complete rows for the given code list.
    rows is a list of dicts (each dict: statement_code -> likert_score).
    """
    if not rows:
        return None
    if len(codes) < 2:
        return None

    complete_vectors = []
    for row in rows:
        if all(isinstance(row.get(code), int) for code in codes):
            complete_vectors.append([row.get(code) for code in codes])

    if len(complete_vectors) < 2:
        return None

    k = len(codes)
    item_variances = []
    for idx in range(k):
        item_values = [vec[idx] for vec in complete_vectors]
        var = _sample_variance(item_values)
        if var is None:
            return None
        item_variances.append(var)

    total_scores = [sum(vec) for vec in complete_vectors]
    total_variance = _sample_variance(total_scores)
    if total_variance is None or total_variance <= 0:
        return None

    alpha = (k / (k - 1)) * (1 - (sum(item_variances) / total_variance))
    return float(alpha)


def _fmt_mean(values, decimals=3):
    return round(mean(values), decimals) if values else ""


def _fmt_sd(values, decimals=3):
    if not values:
        return ""
    if len(values) == 1:
        return 0.0
    return round(pstdev(values), decimals)


def main(*, days=30, out_file=OUT_FILE, data_source="all", real_only=False):
    selected_source = "real_world" if real_only else _normalize_data_source(data_source)
    since = timezone.now() - timedelta(days=max(int(days), 1))
    qs = UsabilitySurveyResponse.objects.filter(submitted_at__gte=since).order_by("submitted_at")
    if selected_source != "all":
        qs = qs.filter(data_source=selected_source)

    batches = defaultdict(dict)
    quick_scores = []
    difficulty_scores = {code: [] for code in DIFF_CODES}
    for row in qs:
        code = str(row.statement_code or "").strip().upper()
        score = int(row.likert_score or 0)
        if score < 1 or score > 5:
            continue

        if code in difficulty_scores:
            difficulty_scores[code].append(score)

        batch_id = str(row.survey_batch_id or "").strip()
        if batch_id:
            batches[batch_id][code] = score
        elif code == "CHAT_UX_HELPFULNESS":
            quick_scores.append(score)

    sus_scores = []
    pu_means = []
    peu_means = []
    for _batch, code_map in batches.items():
        sus = _sus_score_from_rows(code_map)
        if sus is not None:
            sus_scores.append(sus)

        pu_values = [code_map.get(code) for code in PU_CODES if isinstance(code_map.get(code), int)]
        if pu_values:
            pu_means.append(mean(pu_values))

        peu_values = [code_map.get(code) for code in PEU_CODES if isinstance(code_map.get(code), int)]
        if peu_values:
            peu_means.append(mean(peu_values))

    batch_rows = list(batches.values())
    pu_alpha = _cronbach_alpha_from_rows(batch_rows, PU_CODES)
    peu_alpha = _cronbach_alpha_from_rows(batch_rows, PEU_CODES)

    difficulty_all = []
    for code in DIFF_CODES:
        difficulty_all.extend(difficulty_scores[code])

    rows = [
        {
            "metric_group": "survey",
            "metric_name": "survey_batches",
            "metric_value": len(batches),
            "notes": f"data_source={selected_source}",
        },
        {
            "metric_group": "survey",
            "metric_name": "sus_avg_score_0_100",
            "metric_value": _fmt_mean(sus_scores, decimals=2),
            "notes": "average SUS score across complete batches",
        },
        {
            "metric_group": "survey",
            "metric_name": "pu_avg_likert_1_5",
            "metric_value": _fmt_mean(pu_means, decimals=3),
            "notes": "average of PU items across batches",
        },
        {
            "metric_group": "survey",
            "metric_name": "peu_avg_likert_1_5",
            "metric_value": _fmt_mean(peu_means, decimals=3),
            "notes": "average of PEU items across batches",
        },
        {
            "metric_group": "survey",
            "metric_name": "pu_cronbach_alpha",
            "metric_value": round(pu_alpha, 4) if isinstance(pu_alpha, (int, float)) else "",
            "notes": "Cronbach alpha for PU_Q1..PU_Q4 across complete batches",
        },
        {
            "metric_group": "survey",
            "metric_name": "peu_cronbach_alpha",
            "metric_value": round(peu_alpha, 4) if isinstance(peu_alpha, (int, float)) else "",
            "notes": "Cronbach alpha for PEU_Q1..PEU_Q4 across complete batches",
        },
        {
            "metric_group": "survey",
            "metric_name": "chat_ux_helpfulness_avg_likert_1_5",
            "metric_value": _fmt_mean(quick_scores, decimals=3),
            "notes": "quick feedback rows where statement_code=CHAT_UX_HELPFULNESS",
        },
        {
            "metric_group": "difficulty",
            "metric_name": "difficulty_overall_avg_likert_1_5",
            "metric_value": _fmt_mean(difficulty_all, decimals=3),
            "notes": "all DIFF_* responses combined",
        },
        {
            "metric_group": "difficulty",
            "metric_name": "difficulty_overall_sd",
            "metric_value": _fmt_sd(difficulty_all, decimals=3),
            "notes": "population SD of all DIFF_* responses",
        },
    ]

    for code in DIFF_CODES:
        values = difficulty_scores[code]
        rows.append(
            {
                "metric_group": "difficulty",
                "metric_name": f"{code.lower()}_avg_likert_1_5",
                "metric_value": _fmt_mean(values, decimals=3),
                "notes": f"mean for {code}",
            }
        )
        rows.append(
            {
                "metric_group": "difficulty",
                "metric_name": f"{code.lower()}_sd",
                "metric_value": _fmt_sd(values, decimals=3),
                "notes": f"population SD for {code}",
            }
        )

    with open(out_file, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["metric_group", "metric_name", "metric_value", "notes"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved: {out_file}")
    print(f"Rows: {len(rows)}")
    print(f"Data source filter: {selected_source}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export SUS/TAM survey summary metrics.")
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--out-file", default=OUT_FILE)
    parser.add_argument("--data-source", default="all", choices=sorted(SOURCE_OPTIONS))
    parser.add_argument("--real-only", action="store_true")
    args = parser.parse_args()

    main(
        days=args.days,
        out_file=args.out_file,
        data_source=args.data_source,
        real_only=bool(args.real_only),
    )
