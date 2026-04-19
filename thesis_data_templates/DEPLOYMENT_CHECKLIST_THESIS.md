# Deployment Checklist (Thesis)

Reference: `thesis_data_templates/FINAL_DEFENSE_RUNBOOK.md`

## A. Environment & Secrets
- [ ] `SECRET_KEY` is non-default and stored in environment.
- [ ] Gemini key is stored in environment, not hardcoded in source.
- [ ] Any exposed/old keys are rotated.
- [ ] `.env` is excluded from git and not committed.

## B. Django Security Baseline
- [ ] `DEBUG=False` for deployment profile.
- [ ] `ALLOWED_HOSTS` includes deployed domain/IP.
- [ ] `CSRF_TRUSTED_ORIGINS` includes deployed `https://` domain.
- [ ] `SESSION_COOKIE_SECURE=True` (HTTPS).
- [ ] `CSRF_COOKIE_SECURE=True` (HTTPS).

## C. Static/Media
- [ ] `STATIC_ROOT` configured for deployment.
- [ ] `python manage.py collectstatic` runs successfully.
- [ ] Media upload directory is writable and backed up.

## D. Database
- [ ] Database credentials come from environment variables.
- [ ] Backup and restore procedure documented and tested.
- [ ] Migration status checked and applied in deployment environment.

## E. Functional Smoke Tests
- [ ] Guest can sign in, request recommendation, and book.
- [ ] Owner can sign in, view owner hub, manage rooms.
- [ ] Admin can approve owner/accommodation records.
- [ ] Employee can access employee dashboard workflows.
- [ ] Chatbot role-based responses work for guest/owner/admin/employee.

## F. Chapter 4 Evidence Exports
- [ ] `export_rq34_chapter4_bundle` generated latest RQ3/RQ4 tables.
- [ ] `export_chapter4_sop_evidence` generated SOP matrix and formula lock.
- [ ] Exported data source label for defense is consistent (`pilot_test` or `real_world`).

## G. Demo Defense Pack
- [ ] UAT execution log is filled and signed by testers.
- [ ] Screenshots/recordings mapped to UAT test IDs.
- [ ] RQ3/RQ4 output tables are ready for slide deck and Chapter 4 appendix.

## H. Final Window Exports (Recommended)
- [ ] `export_rq34_chapter4_bundle --days 365` generated final-window outputs.
- [ ] `export_survey_readiness_audit --target-complete-batches <approved target>` generated gap report.
- [ ] Defense-profile deployment report exported (`deployment_readiness_report_defense_profile.md`).
