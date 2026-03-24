# CNN Intent Error Analysis Report (Manual Validation v1)

## Inputs Analyzed
- `artifacts/text_cnn_intent/manual_validation/manual_validation_predictions_v1.csv`
- `artifacts/text_cnn_intent/manual_validation/manual_validation_summary_v1.json`
- `artifacts/text_cnn_intent/manual_validation/error_analysis_rows_v1.csv`
- `artifacts/text_cnn_intent/manual_validation/error_analysis_summary_v1.json`

## 1) Failing Intents (Manual Validation v1)
- Total samples: 90
- Total errors: 76
- Overall accuracy: 0.1556
- Errors by true intent:
  - `book_accommodation`: 15/15 errors
  - `calculate_accommodation_billing`: 15/15 errors
  - `calculate_billing`: 14/15 errors
  - `get_accommodation_recommendation`: 14/15 errors
  - `get_recommendation`: 14/15 errors
  - `get_tourism_information`: 4/15 errors

## 2) Main Confusion Patterns
- `get_accommodation_recommendation -> get_tourism_information` (12)
- `get_recommendation -> get_tourism_information` (11)
- `book_accommodation -> get_tourism_information` (9)
- `calculate_billing -> book_accommodation` (7)
- `calculate_accommodation_billing -> calculate_billing` (7)

## 3) Query Pattern Observations
- Most failing queries are question-style or polite long-form sentences.
- Most failing queries contain mixed intent words (e.g., booking + recommendation + budget).
- Error confidence is consistently low:
  - Mean error confidence: ~0.288
  - Max error confidence: ~0.488
  - Errors below 0.60 confidence: 76/76

## 4) Training-vs-Runtime Preprocessing Check
- Training path (`ai_chatbot/train_text_cnn_intent.py`):
  - `TextVectorization(standardize='lower_and_strip_punctuation')`
  - Vectorizer adapted during training and vocabulary saved to `text_cnn_intent_vocab.json`.
- Runtime path (`ai_chatbot/views.py`):
  - Model prediction uses the same embedded vectorizer.
  - If table-init error occurs, runtime repair rebuilds a vectorizer.
- Identified mismatch (important):
  - Previous repair logic re-adapted vocabulary from a corpus at runtime.
  - Re-adaptation can change token-index mapping against fixed embedding weights, causing degraded predictions.
- Applied fix:
  - Runtime repair now prefers loading saved vocabulary from artifact JSON before any corpus adaptation.

## 5) Confidence-Threshold Fallback Added
- New behavior in `ai_chatbot/views.py`:
  - If top intent confidence `< 0.60` (configurable via `CHATBOT_INTENT_CNN_CONFIDENCE_THRESHOLD`), source becomes `text_cnn_low_confidence`.
  - Intent is left empty so `_classify_intent_and_extract_params` routes to heuristic fallback instead of unsafe low-confidence CNN routing.
- Validation of routing safety:
  - `artifacts/text_cnn_intent/manual_validation/fallback_routing_summary_v1.json`
  - Final intent accuracy with fallback routing: 0.6889
  - Fallback used on 90/90 manual queries (all were low-confidence).

## 6) Row-Level Error Analysis Output
Row-level file includes required fields:
- true intent
- predicted intent
- confidence
- likely cause of failure
- suggested fix

See: `artifacts/text_cnn_intent/manual_validation/error_analysis_rows_v1.csv`

## 7) Suggested Fixes
- Keep confidence-threshold fallback enabled in runtime to prevent misrouting.
- Augment training data with realistic paraphrases from failing queries.
- Add contrastive examples for:
  - booking vs accommodation recommendation
  - accommodation billing vs tour billing
  - recommendation vs tourism information question style

## 8) Retraining Recommendation
Retraining is recommended before deployment-readiness claims.
- Reason: manual free-form performance is low and current runtime relies on fallback for all tested manual queries.
- Safe next step: retrain on merged dataset that includes strict-label paraphrase augmentation and re-run benchmark + manual validation.
