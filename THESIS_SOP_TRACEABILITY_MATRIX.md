# Thesis SOP-to-Code Traceability Matrix

This maps your Chapter 1/3 Statement of the Problem (SOP) and metrics to the current implementation in this repository.

## 1) Completion Checklist (Defense-Ready)

Mark each as `DONE` only when the output artifact exists for the final real-world window.

| Item | Required Output | Current Code Anchor | Status |
|---|---|---|---|
| Scope freeze (hotel/inn AI module only) | Written scope lock note in Chapter 3/4 | `ai_chatbot/views.py`, `guest_app/views.py` | IN PROGRESS |
| SOP traceability finalized | This file + Chapter 4 table references | `THESIS_SOP_TRACEABILITY_MATRIX.md` | DONE |
| Real-world-only metric export | `chapter4_metrics_summary_real_only.csv` | `chapter4_metrics_export.py:31` | IN PROGRESS |
| Real-world-only survey export | `chapter4_survey_summary.csv` (real_only) | `chapter4_survey_export.py:92` | DONE |
| Recommendation training export | `text_cnn_messages_export*.csv`, `accommodation_reco_training_export*.csv` | `ai_chatbot/management/commands/export_training_data.py:159` | DONE (pipeline ready) |
| Model artifact lock | Final artifact names + date/version in Chapter 4 | `artifacts/text_cnn_demo`, `artifacts/decision_tree_demo` | IN PROGRESS |
| Reliability stats for adapted scales | Cronbach alpha (PU/PEU, optional SUS reporting note) | `chapter4_survey_export.py:54` | DONE |
| SOP question-by-question evidence table | Chapter 4 mapping table with figure/table IDs | Chapter 4 draft | IN PROGRESS |

## 2) SOP Traceability Matrix

## SOP 1: Difficulty experienced by tourists in existing system

| SOP Indicator | Measurement | Collection/Storage | Processing Script | Output Artifact | Notes |
|---|---|---|---|---|---|
| 1.1 Discovering suitable hotels/inns | Likert/weighted mean from difficulty items | `ai_chatbot.models.UsabilitySurveyResponse` (`statement_code`) at `ai_chatbot/models.py:101`; API: `submit_usability_feedback` at `ai_chatbot/views.py:2612` | `chapter4_survey_export.py:42` (extend with DIFF_* item groups) | `thesis_data_templates/chapter4_survey_summary.csv` | Needs explicit difficulty codes (e.g., `DIFF_DISCOVER`) if not yet collected |
| 1.2 Matching options to preferences | Same as above | Same as above | Same as above | Same as above | Same gap as 1.1 |
| 1.3 Planning stay-related itineraries | Same as above | Same as above | Same as above | Same as above | Same gap as 1.1 |
| 1.4 Completing bookings/payments | Same as above + booking completion counts | `guest_app.views.accommodation_book` at `guest_app/views.py:3104`; `guest_app.models.AccommodationBooking` at `guest_app/models.py:263` | `chapter4_metrics_export.py:31` + survey export | `chapter4_metrics_summary.csv` + survey summary | Payment is guided billing/status flow, not external payment gateway completion |

## SOP 2: Additional functionalities and end-user requirements

| SOP Indicator | Implementation Evidence | Code Anchor | Measurable Evidence |
|---|---|---|---|
| 2.1 Chatbot interaction + real-time assistance | Conversational endpoint with intent parsing, clarifications, and state | `openai_chat` at `ai_chatbot/views.py:1857`; parser at `ai_chatbot/views.py:1464`, `ai_chatbot/views.py:1688`; OpenAI/fallback extractor at `ai_chatbot/views.py:1790` | Response-time/success logs (`SystemMetricLog`) and chat usage events |
| 2.2 Personalized recommendation (CNN + Decision Tree) | Hybrid accommodation scorer and diagnostics | `recommend_accommodations_with_diagnostics` at `ai_chatbot/recommenders.py:567`; trace builder at `ai_chatbot/recommenders.py:282`; prediction endpoint at `ai_chatbot/views.py:2690` | `avg_decision_tree_score`, hit rate, no-match, fallback/hybrid mode counts from `chapter4_metrics_export.py` |
| 2.3 Automated booking + billing process | Billing estimate and booking flow for hotel/inn | Billing calc at `ai_chatbot/views.py:1017` and `ai_chatbot/recommenders.py:671`; booking flow at `ai_chatbot/views.py:1207`; guest endpoints at `guest_app/views.py:3006`, `guest_app/views.py:3104` | Booking conversion metrics + booking status logs |
| 2.4 Alignment with current office workflows | Existing admin/accommodation dashboards and approval pipeline still active | `admin_dashboard` at `admin_app/views.py:676`; `employee_dashboard` at `admin_app/views.py:531`; `accommodation_bookings` at `admin_app/views.py:416`; tour pending/approval at `tour_app/views.py:254`, `tour_app/views.py:338` | Operational continuity evidence in chapter narrative + screenshots + counts |

## SOP 3: Performance level of AI-enhanced system

| SOP Metric | Formula / Definition | Source | Script Output |
|---|---|---|---|
| 3.1 Recommendation accuracy proxy | `hit_rate_pct = queries_with_results / accommodation_queries * 100` | `RecommendationResult` (`ai_chatbot/models.py:48`) | `chapter4_metrics_export.py:86`, row `hit_rate_pct` |
| 3.1 Recommendation quality detail | `avg_top1_score`, `avg_decision_tree_score` | Recommendation trace metadata | `chapter4_metrics_export.py:157` |
| 3.2 SUS | Standard SUS scoring (odd/even transform x2.5) | `UsabilitySurveyResponse` (`ai_chatbot/models.py:101`) | `chapter4_survey_export.py:26`, `sus_avg_score_0_100` |
| 3.3 User engagement | views, clicks, CTR | `RecommendationEvent` (`ai_chatbot/models.py:12`) | `chapter4_metrics_export.py:97`, rows `views`, `clicks`, `ctr_pct` |
| 3.4 Booking conversion efficiency | books/view rate | `RecommendationEvent` + booking flow logs | `chapter4_metrics_export.py:98`, row `book_rate_pct` |
| System performance support metric | response time avg/p95, success rate | `SystemMetricLog` (`ai_chatbot/models.py:77`) | `chapter4_metrics_export.py`, row `chat_p95_response_ms` |

## SOP 4: Acceptance level (PU and PEU)

| SOP Indicator | Measurement | Source | Script Output | Gap |
|---|---|---|---|---|
| 4.1 Perceived Usefulness (PU) | Mean Likert score across PU items | `UsabilitySurveyResponse` (`statement_code` PU_Q1..PU_Q4) | `chapter4_survey_export.py`, `pu_avg_likert_1_5` + `pu_cronbach_alpha` | Needs sufficient completed real-world batches |
| 4.2 Perceived Ease of Use (PEU) | Mean Likert score across PEU items | `UsabilitySurveyResponse` (`statement_code` PEU_Q1..PEU_Q4) | `chapter4_survey_export.py`, `peu_avg_likert_1_5` + `peu_cronbach_alpha` | Needs sufficient completed real-world batches |

## 3) Thesis Data Pipeline (Operational)

| Stage | Command/Script | Main Artifact |
|---|---|---|
| Collect chat/reco/system/survey logs | Runtime via `/api/chat/*` endpoints in `ai_chatbot/urls.py:10` | DB tables |
| Export Chapter 4 system metrics | `chapter4_metrics_export.py` | `thesis_data_templates/chapter4_metrics_summary.csv` |
| Export Chapter 4 survey metrics | `chapter4_survey_export.py` | `thesis_data_templates/chapter4_survey_summary.csv` |
| Export model-training datasets | `python manage.py export_training_data` | `text_cnn_messages_export*.csv`, `accommodation_reco_training_export*.csv` |
| Train text CNN demo artifact | `ai_chatbot/train_text_cnn_demo.py:117` | `artifacts/text_cnn_demo/text_cnn_demo.keras` |
| Train decision tree demo artifact | `ai_chatbot/train_decision_tree_demo.py:129` | `artifacts/decision_tree_demo/decision_tree_demo.pkl` |

## 4) Remaining Gaps to Close Before Final Defense

1. Collect real-world responses for `DIFF_*`, `SUS_*`, `PU_*`, `PEU_*` so exported rows are populated.
2. Produce final **real-world-only** exports near submission date and lock them as Chapter 4 source files.
3. Create a final Chapter 4 table index that references each metric row from the generated CSVs.
