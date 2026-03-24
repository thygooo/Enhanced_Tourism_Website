# Phase 1 Report: CNN Validation and Inference Checking

## 1. Runtime Inference Flow Review (`ai_chatbot/views.py`)
- Intent classification entrypoint: `_classify_intent_with_text_cnn(message)`
- Runtime model resolver: `_resolve_intent_text_cnn_model_path()`
- Runtime default path target: `artifacts/text_cnn_intent/text_cnn_intent.h5`
- Runtime label map resolver: `_default_label_map_path_for_model(model_path)`
- Prediction function: `_predict_text_cnn_labels(...)` with confidence and top-3 outputs
- Intent normalization gate: `_normalize_intent_label(...)` restricted by `_ALLOWED_INTENTS`
- Fallback path: `_intent_from_message(...)` only used when CNN result has no valid intent

## 2. Runtime Artifact Verification (executed)
- Resolved model path: `C:\Users\Shenna Mae\SoftEngr\Tourism-2026\artifacts\text_cnn_intent\text_cnn_intent.h5`
- Resolved source: `final_default`
- Label map path: `C:\Users\Shenna Mae\SoftEngr\Tourism-2026\artifacts\text_cnn_intent\label_map.json`
- Existence check: both files exist (`True`)

## 3. Manual Query Validation Set
- Source file: `thesis_data_templates/text_cnn_intent_manual_validation_queries_v1.csv`
- Total queries: 90
- Coverage: 15 queries per intent for all 6 intents

## 4. Manual Utility Run Outputs
- Script: `ai_chatbot/run_intent_manual_validation.py`
- Predictions: `artifacts/text_cnn_intent/manual_validation/manual_validation_predictions_v1.csv`
- Summary: `artifacts/text_cnn_intent/manual_validation/manual_validation_summary_v1.json`

Observed summary from run:
- Overall manual-set accuracy: `0.1556`
- Per-intent manual-set accuracy:
  - `book_accommodation`: `0.0000`
  - `calculate_accommodation_billing`: `0.0000`
  - `calculate_billing`: `0.0667`
  - `get_accommodation_recommendation`: `0.0667`
  - `get_recommendation`: `0.0667`
  - `get_tourism_information`: `0.7333`

## 5. Likely Intent Confusions

### From confusion matrix artifact (`evaluation_metrics_v3_full.json`)
- `get_accommodation_recommendation -> book_accommodation` (1 test sample)
- `get_recommendation -> get_accommodation_recommendation` (1 test sample)
- `get_tourism_information -> get_recommendation` (1 test sample)

### From manual semantic probes (`manual_validation_summary_v1.json`)
- Heavy drift toward `get_tourism_information` in natural long-form prompts
- Frequent confusion pairs:
  - `get_accommodation_recommendation -> get_tourism_information` (12)
  - `get_recommendation -> get_tourism_information` (11)
  - `book_accommodation -> get_tourism_information` (9)
  - `calculate_billing -> book_accommodation` (7)
  - `calculate_accommodation_billing -> calculate_billing` (7)

## 6. Assumption Statement
- The manual query set uses realistic free-form user wording and was not sourced from training rows; this may increase distribution shift relative to benchmark splits.
- Minimal runtime alignment fix applied: CNN vectorization repair now prefers `text_cnn_messages_final_expanded_v3_clean.csv` when rebuilding lookup tables.
