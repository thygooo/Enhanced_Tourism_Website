# RQ3 Live Demo Script (Data Provenance)

Use this script during thesis presentation to demonstrate where RQ3 values come from.

## 0) Panel-friendly flow (UI + data extraction)

1. Navigate the system (show where RQ3 data is generated):
- Guest user flow:
  - Open `http://127.0.0.1:8000/guest_app/login/`
  - Go to accommodations: `http://127.0.0.1:8000/guest_app/accommodations/`
  - Trigger recommendations and click recommended items (creates `view`/`click` events)
  - Create a booking (creates `book` event)
- Survey flow:
  - Submit usability feedback through chatbot survey endpoint/UI integration (stores SUS rows)
- Admin verification:
  - Open `http://127.0.0.1:8000/admin_app/survey-results/` to show survey records dashboard

2. Extract RQ3 from logs and survey tables:
- Run export command in Section 1.
- Open CSV in Section 2.
- Run provenance checks in Sections 3 to 5.

3. State the final RQ3 answers:
- Read from exported file `thesis_data_templates/chapter4_rq3_metrics_pilot_test.csv`.

## 1) Run the RQ3/RQ4 export (read-only)

```powershell
python manage.py export_rq34_chapter4_bundle --source pilot_test --days 365 --out-rq3-csv thesis_data_templates/chapter4_rq3_metrics_pilot_test.csv --out-rq4-csv thesis_data_templates/chapter4_rq4_acceptance_pilot_test.csv --out-rq4-items-csv thesis_data_templates/chapter4_rq4_item_stats_pilot_test.csv --out-metric-defs-csv thesis_data_templates/chapter4_metric_definitions_pilot_test.csv --out-bundle-json thesis_data_templates/chapter4_rq3_rq4_bundle_pilot_test.json
```

Primary RQ3 output:

- `thesis_data_templates/chapter4_rq3_metrics_pilot_test.csv`

## 2) Open the exported RQ3 table

```powershell
Get-Content thesis_data_templates\chapter4_rq3_metrics_pilot_test.csv
```

Expected current rows (pilot dataset):

- `true_accuracy_labeled = 75.0` (n=480, from offline labeled evaluation)
- `queries_with_results_rate_pct = 100.0` (n=123)
- `avg_top1_score = 0.720871` (n=89)
- `sus_mean_0_100 = 50.0` (n=1)
- `ctr_pct = 27.6596` (n=94)
- `book_rate_pct = 45.7447` (n=94)

## 3) Show raw log counts used by RQ3 proxies

```powershell
python manage.py shell -c "from datetime import timedelta; from django.utils import timezone; from ai_chatbot.models import RecommendationResult, RecommendationEvent, UsabilitySurveyResponse; since=timezone.now()-timedelta(days=365); src='pilot_test'; ACCOM_INTENTS={'get_accommodation_recommendation','gethotelrecommendation'}; rr=RecommendationResult.objects.filter(generated_at__gte=since,data_source=src); re=RecommendationEvent.objects.filter(event_time__gte=since,data_source=src); us=UsabilitySurveyResponse.objects.filter(submitted_at__gte=since,data_source=src); accom=0; with_results=0; 
for row in rr.iterator():
 ctx=row.context_json if isinstance(row.context_json,dict) else {}; intent=str(ctx.get('intent') or '').strip().lower();
 if intent not in ACCOM_INTENTS: continue
 accom+=1
 items=row.recommended_items_json if isinstance(row.recommended_items_json,list) else []
 if not items: continue
 with_results+=1
views=re.filter(event_type='view',item_ref='chat:accommodation_recommendation_request').count(); clicks=re.filter(event_type='click').count(); books=re.filter(event_type='book').count(); print({'accom_queries':accom,'queries_with_results':with_results,'views':views,'clicks':clicks,'books':books,'survey_rows':us.count()})"
```

Current output:

- `accom_queries: 123`
- `queries_with_results: 123`
- `views: 94`
- `clicks: 26`
- `books: 43`
- `survey_rows: 49`

Derived checks:

- `queries_with_results_rate_pct = 123/123 * 100 = 100.0`
- `ctr_pct = 26/94 * 100 = 27.6596`
- `book_rate_pct = 43/94 * 100 = 45.7447`

## 4) Show true recommendation accuracy source (RQ3.1 true metric)

```powershell
Get-Content artifacts\decision_tree_final\evaluation_metrics_v1.json
```

Key values:

- `accuracy = 0.75` -> `75.0%`
- `test_rows = 480`

This is why `true_accuracy_labeled` in RQ3 is reported from offline labeled evaluation, not from runtime clicks/views alone.

## 5) Show SUS provenance (RQ3.2)

```powershell
python manage.py shell -c "from datetime import timedelta; from django.utils import timezone; from ai_chatbot.models import UsabilitySurveyResponse; since=timezone.now()-timedelta(days=365); src='pilot_test'; rows=UsabilitySurveyResponse.objects.filter(submitted_at__gte=since,data_source=src); batches={}; 
for r in rows.iterator():
 code=str(r.statement_code or '').strip().upper(); score=int(r.likert_score or 0); bid=str(r.survey_batch_id or '').strip();
 if not bid or score<1 or score>5: continue
 d=batches.get(bid,{})
 d[code]=score; batches[bid]=d
sus=[]
for b in batches.values():
 ok=True; total=0
 for i in range(1,11):
  raw=b.get(f'SUS_Q{i}')
  if not isinstance(raw,int): ok=False; break
  total += (raw-1) if i%2==1 else (5-raw)
 if ok: sus.append(total*2.5)
mean=(sum(sus)/len(sus)) if sus else None
print({'survey_batches':len(batches),'sus_complete_batches':len(sus),'sus_mean':mean})"
```

Current output:

- `survey_batches: 8`
- `sus_complete_batches: 1`
- `sus_mean: 50.0`

## 6) If panel asks for real-world filter

```powershell
python manage.py export_rq34_chapter4_bundle --source real_world --days 365
Get-Content thesis_data_templates\chapter4_rq3_metrics.csv
```

Current DB state shows `n=0` for real-world RQ3 runtime rows, so pilot-test outputs are the analyzable dataset at this time.

## 7) Phase C evidence: conversation-level analytics and traceability

Use this when panelists ask: "Can you show the actual conversation evidence, linked with chatbot behavior and events?"

```powershell
python manage.py export_chat_conversation_evidence --days 30 --source all --out-csv thesis_data_templates/chatbot_conversation_evidence.csv --out-json thesis_data_templates/chatbot_conversation_evidence.json --out-kpi-csv thesis_data_templates/chatbot_quality_kpis.csv
```

Generated files:

- `thesis_data_templates/chatbot_conversation_evidence.csv`
- `thesis_data_templates/chatbot_conversation_evidence.json`
- `thesis_data_templates/chatbot_quality_kpis.csv`

Quick view command:

```powershell
Get-Content thesis_data_templates\chatbot_conversation_evidence.csv
```

What to explain while showing this file:

- Each row is one chatbot exchange with:
  - `resolved_intent`
  - `response_nlg_source`
  - `fallback_used`
  - `user_message` and `bot_response`
- Session linkage is shown via `session_id`.
- Evidence columns connect behavior to engagement/conversion:
  - `step_event_count_in_session`
  - `step_event_click_count_in_session`
  - `step_event_book_count_in_session`
  - `recommendation_result_count_in_session`
  - `latest_recommendation_trace_count`

Panel talking line:

- "Aside from aggregated RQ3 metrics, we can trace interaction-level evidence per conversation and session, including recommendation rendering, comparison/why interactions, and booking progression signals."
