# Final Defense Runbook

Last updated: April 10, 2026

## 1) Generate deployment readiness snapshot

Local safe check:

```powershell
python manage.py export_deployment_readiness_report
```

Defense-profile check (recommended before defense rehearsal):

```powershell
$env:DJANGO_DEBUG='False'
$env:SESSION_COOKIE_SECURE='True'
$env:CSRF_COOKIE_SECURE='True'
$env:CSRF_TRUSTED_ORIGINS='https://defense-demo.local'
$env:DJANGO_ALLOWED_HOSTS='127.0.0.1,testserver,defense-demo.local'
$env:DB_PASSWORD='your-db-password'
python manage.py export_deployment_readiness_report --out-json thesis_data_templates/deployment_readiness_report_defense_profile.json --out-md thesis_data_templates/deployment_readiness_report_defense_profile.md
```

Target:
- Overall should be `CONDITIONALLY_READY` or better.
- `DEBUG flag` must be `PASS`.
- No `FAIL` rows.

## 2) Generate Chapter 4 RQ3/RQ4 evidence with a defense window

Use a wider window to include all recent study data:

```powershell
python manage.py export_rq34_chapter4_bundle --source all --days 365 --out-rq3-csv thesis_data_templates/chapter4_rq3_metrics_final_window.csv --out-rq4-csv thesis_data_templates/chapter4_rq4_acceptance_final_window.csv --out-rq4-items-csv thesis_data_templates/chapter4_rq4_item_stats_final_window.csv --out-metric-defs-csv thesis_data_templates/chapter4_metric_definitions_final_window.csv --out-bundle-json thesis_data_templates/chapter4_rq3_rq4_bundle_final_window.json
```

Target:
- RQ3.1 true accuracy available from offline evaluation.
- RQ3.2 SUS has valid `n` and mean.
- RQ4 PU/PEU means have valid `n` and means.

## 3) Generate survey readiness audit

```powershell
python manage.py export_survey_readiness_audit --source all --target-complete-batches 30 --out-json thesis_data_templates/survey_readiness_audit_final_window.json --out-csv thesis_data_templates/survey_readiness_incomplete_batches_final_window.csv
```

Target:
- `complete_full_batches >= target_complete_batches`.
- If not yet met, continue data collection and rerun this command.

## 4) Current status snapshot (April 10, 2026)

Based on generated artifacts:
- `deployment_readiness_report_defense_profile.md`: `CONDITIONALLY_READY` (8 PASS, 0 WARN, 0 FAIL).
- `chapter4_rq3_rq4_bundle_final_window.json`:
  - RQ3.1 true accuracy = `75.0%` (`n=480`, offline labeled eval).
  - SUS/PU/PEU are populated but very low sample (`n=1` for complete batches).
- `survey_readiness_audit_final_window.json`:
  - complete full batches = `1`
  - target = `30`
  - gap = `29`

## 5) Final go/no-go rule for defense data quality

Go for final defense only if:
- Deployment report has no `FAIL`.
- UAT execution logs are complete for all roles.
- Survey full-batch sample size reaches your approved target (recommended 30+ complete batches).
- Chapter 4 exports are regenerated after final data collection and attached to appendix.

