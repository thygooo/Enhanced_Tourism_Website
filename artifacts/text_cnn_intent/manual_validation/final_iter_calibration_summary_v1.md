# Final Iteration CNN Calibration-Only Sweep Summary

## Scope
- Model: `artifacts/text_cnn_intent_final_iter_v1/text_cnn_intent.h5`
- Manual set: 90 realistic queries
- No retraining performed

## Threshold Sweep Results
| Threshold | Raw CNN Accuracy | Fallback Usage | Final Routed Accuracy |
|---|---:|---:|---:|
| 0.10 | 0.2778 | 0 | 0.2778 |
| 0.20 | 0.2778 | 2 | 0.2889 |
| 0.30 | 0.2778 | 40 | 0.4333 |
| 0.40 | 0.2778 | 81 | 0.6333 |
| 0.50 | 0.2778 | 90 | 0.6889 |
| 0.60 | 0.2778 | 90 | 0.6889 |

## Key Calibration Findings
- Lower thresholds allow more direct CNN routing, but this currently harms routed accuracy.
- Best routed result in tested range is tied at **0.50 and 0.60** (both 0.6889).
- No lower threshold produced better routed accuracy than 0.60.

## Intent-by-Intent Safety (lower-threshold direct routing)
- Safest direct-routing intent:
  - `calculate_billing` (accepted accuracy ~0.891 on accepted direct predictions)
- High-risk direct-routing intents:
  - `book_accommodation` (accepted accuracy 0.0)
  - `get_tourism_information` (accepted accuracy 0.0)
  - `get_recommendation` (accepted accuracy ~0.289)

## Recommended Calibration Strategy
1. Keep global threshold conservative at **0.60** for thesis demo (or 0.50; same routed accuracy in this sweep).
2. Apply selective intent-level gating:
   - Allow lower-confidence direct routing only for `calculate_billing` (e.g., >=0.35) with keyword confirmation.
   - Keep fallback-first for `book_accommodation`, `get_tourism_information`, `get_recommendation`, and accommodation-billing intents unless confidence is high.
3. Treat fallback as primary safety net for practical demo stability.

## Output Files
- JSON: `artifacts/text_cnn_intent/manual_validation/final_iter_calibration_sweep_v1.json`
- CSV summary: `artifacts/text_cnn_intent/manual_validation/final_iter_calibration_sweep_v1.csv`
- CSV detailed rows: `artifacts/text_cnn_intent/manual_validation/final_iter_calibration_rows_v1.csv`
