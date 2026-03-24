# Phase 1 CNN Manual Validation Checklist (Intent Model)

## Scope
- Runtime code reviewed: `ai_chatbot/views.py`
- Runtime classifier path: `artifacts/text_cnn_intent/text_cnn_intent.h5`
- Runtime label map path: `artifacts/text_cnn_intent/label_map.json`
- Intents covered:
  - `book_accommodation`
  - `calculate_accommodation_billing`
  - `calculate_billing`
  - `get_accommodation_recommendation`
  - `get_recommendation`
  - `get_tourism_information`

## Execution Checklist
- [ ] Confirm model resolver in `ai_chatbot/views.py` points to intent artifact (`_resolve_intent_text_cnn_model_path`).
- [ ] Confirm label map is loaded from model sibling path (`_default_label_map_path_for_model`).
- [ ] Confirm intent normalization only allows the 6 thesis intents (`_normalize_intent_label` + `_ALLOWED_INTENTS`).
- [ ] Confirm fallback behavior when model is unavailable (`text_cnn_unavailable`) is logged.
- [ ] Run manual utility:
  - Command: `python ai_chatbot/run_intent_manual_validation.py`
  - Input: `thesis_data_templates/text_cnn_intent_manual_validation_queries_v1.csv`
  - Output CSV: `artifacts/text_cnn_intent/manual_validation/manual_validation_predictions_v1.csv`
  - Output JSON: `artifacts/text_cnn_intent/manual_validation/manual_validation_summary_v1.json`
- [ ] Review per-intent accuracy and confusion pairs.
- [ ] Manually inspect low-confidence samples (`confidence < 0.60`) and top-3 alternatives.

## Query Set Coverage
- `book_accommodation`: query IDs `BA_001` to `BA_015` (15)
- `calculate_accommodation_billing`: query IDs `CAB_001` to `CAB_015` (15)
- `calculate_billing`: query IDs `CB_001` to `CB_015` (15)
- `get_accommodation_recommendation`: query IDs `GAR_001` to `GAR_015` (15)
- `get_recommendation`: query IDs `GR_001` to `GR_015` (15)
- `get_tourism_information`: query IDs `GTI_001` to `GTI_015` (15)

Total manual validation queries: 90

## Most Likely Intent Confusions (from artifacts + semantics)
1. `get_accommodation_recommendation` vs `get_recommendation`
2. `get_tourism_information` vs `get_recommendation`
3. `book_accommodation` vs `get_accommodation_recommendation`
4. `calculate_accommodation_billing` vs `calculate_billing`
5. `calculate_billing` vs `get_tourism_information` for mixed "how much + place info" phrasing

## Notes
- This phase performs validation and inference checks only.
- No CNN retraining is performed in this phase.
