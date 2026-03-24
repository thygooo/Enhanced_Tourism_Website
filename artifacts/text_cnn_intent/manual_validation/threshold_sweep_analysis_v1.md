# Threshold/Fallback Tuning Analysis (Manual Realistic Set)

## Scope
- No retraining performed in this step.
- No default model replacement performed.
- Compared old model (`artifacts/text_cnn_intent/text_cnn_intent.h5`) vs robust-round model (`artifacts/text_cnn_intent_robust_round1/text_cnn_intent.h5`).
- Manual query set: `thesis_data_templates/text_cnn_intent_manual_validation_queries_v1.csv`.

## Why Final Routed Accuracy Dropped (0.6889 -> 0.5222 at threshold 0.60)
Main cause is **changed confidence distribution** in the new model:
- Old model at 0.60: fallback used on 90/90 queries.
- New model at 0.60: fallback used on 59/90 queries.
- 16 regressed rows were previously corrected by fallback but are now high-confidence wrong CNN predictions (`final_source_old=heuristic_intent_fallback`, `final_source_new=text_cnn_intent`).

This means fallback was bypassed more often, and those accepted CNN predictions were frequently incorrect.

## Row-by-Row Comparison Findings
- Compared files:
  - `fallback_routing_predictions_v1.csv`
  - `fallback_routing_predictions_round1.csv`
- Summary (`old_vs_new_row_compare_v1.csv`):
  - Regressed: 16
  - Improved: 1
  - Unchanged: 73
- Regressions by intent:
  - `calculate_accommodation_billing`: 9
  - `book_accommodation`: 5
  - `calculate_billing`: 1
  - `get_recommendation`: 1

## Threshold Sweep (0.30, 0.40, 0.50, 0.60, 0.70)

### Old model
| Threshold | Raw CNN Acc | Fallback Used | Final Routed Acc |
|---|---:|---:|---:|
| 0.30 | 0.1667 | 32 | 0.3333 |
| 0.40 | 0.1667 | 74 | 0.5333 |
| 0.50 | 0.1667 | 88 | 0.6667 |
| 0.60 | 0.1667 | 90 | 0.6889 |
| 0.70 | 0.1667 | 90 | 0.6889 |

### New robust-round model
| Threshold | Raw CNN Acc | Fallback Used | Final Routed Acc |
|---|---:|---:|---:|
| 0.30 | 0.3111 | 1 | 0.3222 |
| 0.40 | 0.3111 | 13 | 0.4000 |
| 0.50 | 0.3111 | 36 | 0.5111 |
| 0.60 | 0.3111 | 59 | 0.5222 |
| 0.70 | 0.3111 | 71 | 0.5778 |

## Interpretation Against Requested Causes
- Confidence threshold too high?
  - Not the main root issue. Raising threshold improves final routed accuracy for new model, but still does not recover old model final routed accuracy in tested range.
- Fallback heuristic misrouting?
  - Not the primary driver of the drop. Regressed rows were mostly old-correct via fallback, new-wrong via accepted CNN.
- Changed confidence distribution?
  - Yes, primary driver. New model is more confident on many wrong predictions.
- Specific intents less recoverable by fallback?
  - Yes. Most regressions are in billing/booking/recommendation intents, especially `calculate_accommodation_billing`.

## Recommended Threshold
For the **new robust-round model**, best among tested thresholds is **0.70** (final routed accuracy `0.5778`).

However, this is still below the old model at 0.60 (`0.6889`), so threshold tuning alone is insufficient to recover prior routed performance.

## Main Issue Now: CNN or Fallback?
Primary issue is now the **CNN confidence calibration/decision behavior** on realistic prompts (overconfident wrong predictions for several intents), not fallback logic alone.

Supporting outputs:
- `artifacts/text_cnn_intent/manual_validation/threshold_sweep_analysis_v1.json`
- `artifacts/text_cnn_intent/manual_validation/threshold_sweep_analysis_v1.csv`
- `artifacts/text_cnn_intent/manual_validation/old_vs_new_row_compare_v1.csv`
