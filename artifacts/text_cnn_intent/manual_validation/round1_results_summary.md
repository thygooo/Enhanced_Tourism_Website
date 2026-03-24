# Final Focused CNN Round (Robustness Round 1) Summary

## What Was Done
1. Reviewed augmentation v1 and balanced it to 40 realistic paraphrases per intent.
2. Merged balanced augmentation into:
   - `thesis_data_templates/text_cnn_messages_final_expanded_v3_clean_robust_round_v1.csv`
3. Performed exact dedupe, cross-label conflict check, and near-duplicate check.
4. Retrained Text-CNN in separate directory:
   - `artifacts/text_cnn_intent_robust_round1/`
5. Re-ran benchmark and manual realistic validation.
6. Kept low-confidence fallback logic in runtime.

## Dataset Preparation Outputs
- Balanced augmentation:
  - `thesis_data_templates/text_cnn_intent_realistic_paraphrase_augmentation_v2_balanced.csv`
- Merged robust dataset:
  - `thesis_data_templates/text_cnn_messages_final_expanded_v3_clean_robust_round_v1.csv`
- Prep report:
  - `thesis_data_templates/text_cnn_messages_final_expanded_v3_clean_robust_round_v1_report.json`

## Retraining/Evaluation Outputs
- New model + benchmark metrics + confusion matrix:
  - `artifacts/text_cnn_intent_robust_round1/evaluation_metrics_v3_full.json`
  - `artifacts/text_cnn_intent_robust_round1/confusion_matrix_v3.csv`
  - `artifacts/text_cnn_intent_robust_round1/confusion_matrix_v3.json`
  - `artifacts/text_cnn_intent_robust_round1/confusion_matrix_v3.png`
- Manual validation (same manual query set):
  - `artifacts/text_cnn_intent/manual_validation/manual_validation_predictions_round1.csv`
  - `artifacts/text_cnn_intent/manual_validation/manual_validation_summary_round1.json`
- Comparison:
  - `artifacts/text_cnn_intent/manual_validation/round1_comparison_report.json`

## Old vs New (Key Comparison)
- Benchmark accuracy: `0.9697 -> 0.9798` (improved)
- Benchmark macro F1: `0.9680 -> 0.9753` (improved)
- Manual realistic raw accuracy: `0.1556 -> 0.3111` (improved but still low)
- Manual realistic with fallback final accuracy:
  - `0.6889 -> 0.5222` (decreased)

## Verdict
Manual realistic robustness improved in raw CNN scoring, but not enough to call deployment-ready.

Reason:
- The new model still fails completely on several intents in the manual realistic set (`book_accommodation`, `calculate_accommodation_billing`, `calculate_billing`, `get_recommendation`).
- Operational fallback-assisted accuracy decreased versus the prior model/fallback configuration.

Recommendation:
- Keep this round as an experimental artifact.
- Do not replace production/default intent artifact yet.
