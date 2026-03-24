# SOP Alignment Status (Safe, Non-Disruptive)

Date: March 18, 2026  
Scope: AI-Powered Tourism Reservation and Monitoring System for Bayawan City

This status file documents what is already aligned to the thesis SOP and what remains.  
All listed next steps are designed to avoid changes to core transactional behavior.

## Safety Note

- No core workflow changes are required for this phase.
- Core functionality remains: account roles, owner approval, accommodation approval, room management, booking, billing, chatbot routing.
- This alignment phase focuses on measurement, traceability, and evidence outputs.

## SOP Alignment Summary

## SOP 1: Difficulty of the existing tourism process

- Already supported:
- Difficulty item codes exist in survey pipeline: `DIFF_DISCOVER`, `DIFF_MATCH`, `DIFF_PLAN`, `DIFF_BOOKPAY`.
- Survey storage exists in `ai_chatbot.UsabilitySurveyResponse`.

- Remaining:
- Ensure enough real-world responses are collected per difficulty item.
- Export and include weighted means/SD in Chapter 4 evidence tables.

## SOP 2: Additional AI-enhanced functionalities and requirements

- Already supported:
- Chatbot interaction endpoint and real-time request handling.
- Recommendation and billing/booking endpoints.
- Owner/admin records workflows (approval, room management, booking views).
- Room-level attributes available for recommendation data preparation.

- Remaining:
- Finalize Decision Tree integration and validation for recommendation path.
- Freeze functional scope note for defense documentation.

## SOP 3: Performance level of AI-enhanced system

- Already supported:
- Recommendation logs: `RecommendationEvent`, `RecommendationResult`.
- System performance logs: `SystemMetricLog`.
- Survey logs and dashboard endpoints.

- Remaining:
- Finalize and lock performance exports for defense window (real-world filtered).
- Produce stable KPI table set (accuracy proxy, engagement, conversion, latency, success rate).

## SOP 4: Acceptance level (PU, PEU)

- Already supported:
- PU/PEU/SUS item capture in survey flow.
- Reliability calculations available in export scripts.

- Remaining:
- Collect sufficient real-world PU/PEU responses for robust analysis.
- Lock final acceptance metrics in chapter tables.

## Recommended Safe Execution Order

1. Run read-only metric exports (no behavior changes).
2. Validate source tags (`data_source`) for real-world filtering.
3. Collect remaining survey responses for missing SOP indicators.
4. Freeze metric outputs for defense documentation.
5. Only then integrate Decision Tree runtime changes (feature-flagged).

## Change Control Policy for SOP Alignment

- If a proposed change touches booking/auth/approval/chat intent flow, require manual review before merge.
- Prefer additive, isolated files for analytics and reporting.
- Keep all SOP metrics generation read-only against production tables.

