# Chapter 4 Thesis-Ready Outputs (Artifacts-Based)

## 1. CNN Dataset Preparation Summary
The Text-CNN intent dataset was finalized using `thesis_data_templates/text_cnn_messages_final_expanded_v3_clean.csv`. The preparation process included expansion, duplicate/near-duplicate cleaning, and split-leakage reduction using recorded reports (`text_cnn_messages_final_expanded_v3_clean_report.json`). Leakage candidate pairs were reduced from 232 to 0 before final evaluation.

## 2. Decision Tree Dataset Preparation Summary
The Decision Tree dataset was prepared using the finalized schema in `thesis_data_templates/decision_tree_accommodation_schema_v1.md`, then generated reproducibly with `ai_chatbot/generate_decision_tree_dataset.py` (seed=42). The resulting file is `thesis_data_templates/accommodation_reco_training_final_v1.csv`, with generation assumptions logged in `artifacts/decision_tree_final/dataset_generation_summary_v1.json`.

## 3. Final Dataset Size Summaries
- CNN intent dataset: 1,920 rows total, 6 intents, 320 samples per intent (`text_cnn_messages_final_expanded_v3_clean.csv`).
- Decision Tree recommendation dataset: 2,400 rows total (`accommodation_reco_training_final_v1.csv`).
- Decision Tree target distribution: 1,735 `relevant`, 665 `not_relevant`.

## 4. Model Configuration Summaries
- Text-CNN intent model:
  - Model artifact: `artifacts/text_cnn_intent/text_cnn_intent.h5`
  - Label map: `artifacts/text_cnn_intent/label_map.json`
  - Input pipeline: `TextVectorization` (`max_tokens=4000`, `sequence_length=40`)
  - Core layers: Embedding(64), Conv1D(filters=64, kernel=3), GlobalMaxPooling1D, Dense(64), Dropout(0.3), Dense(softmax)
  - Training defaults recorded in `ai_chatbot/train_text_cnn_intent.py`: epochs=12, batch_size=8, early stopping enabled.
- Decision Tree model:
  - Pipeline artifact: `artifacts/decision_tree_final/decision_tree_final.pkl`
  - Preprocessing: median imputation (numeric), most-frequent + one-hot encoding (categorical)
  - Classifier: `DecisionTreeClassifier(max_depth=8, min_samples_leaf=5, class_weight='balanced', random_state=42)`
  - Evaluation split: test_size=0.20, stratified.

## 5. Evaluation Metric Summaries
- CNN intent metrics (`artifacts/text_cnn_intent/evaluation_metrics_v3_full.json`):
  - Accuracy: 0.9697
  - Precision (macro): 0.9698
  - Recall (macro): 0.9667
  - F1 (macro): 0.9680
  - Test size: 99
- Decision Tree metrics (`artifacts/decision_tree_final/evaluation_metrics_v1.json`):
  - Accuracy: 0.7500
  - Precision (macro): 0.7036
  - Recall (macro): 0.7344
  - F1 (macro): 0.7122
  - Test size: 480

## 6. Formal Academic Interpretation of CNN Results
The Text-CNN model achieved high multiclass performance on the cleaned evaluation split, indicating that the selected architecture and preprocessing were effective for intent discrimination within the curated thesis intent set. Macro-level metrics near 0.97 suggest strong and relatively balanced class-wise behavior under the final test partition. Residual confusions were limited and concentrated among semantically adjacent intents, which is expected in natural-language intent classification.

## 7. Formal Academic Interpretation of Decision Tree Results
The Decision Tree attained moderate-to-strong classification performance (accuracy=0.75, macro-F1=0.7122) on the synthetic recommendation dataset. The feature-importance profile shows that budget and room price are dominant decision signals, followed by guest-capacity fit, which aligns with practical accommodation selection behavior. Performance is adequate for backend ranking support, but additional calibration and real interaction logs are recommended to further improve minority-class precision and decision stability.

## 8. CNN Model Limitations
Although benchmark evaluation is strong, manual free-form probes (`artifacts/text_cnn_intent/manual_validation/manual_validation_summary_v1.json`) showed substantial performance degradation on longer and more varied conversational phrasing. This indicates sensitivity to wording distribution and possible mismatch between benchmark split style and live user language. The model may require broader paraphrase coverage, stricter domain-balanced validation sets, or confidence-threshold fallback strategies for robust deployment behavior.

## 9. Decision Tree Model Limitations
The current Decision Tree was trained on synthetic data, which is useful for controlled experimentation but may not fully capture real user click/booking behavior. As a result, decision boundaries can overfit generator assumptions and may yield overconfident predictions in some edge cases. Final thesis deployment readiness therefore depends on retraining or recalibration using real interaction logs and post-deployment monitoring.

## 10. Concluding Integration Paragraph
Together, the Text-CNN and Decision Tree components operationalize a two-stage intelligent tourism assistant: the Text-CNN maps user messages to service intents, while the Decision Tree supports accommodation relevance scoring using preference- and candidate-level features. This hybrid structure enables the chatbot to interpret user requests and provide data-driven recommendations in a unified backend workflow, directly supporting the system objectives of tourism assistance, reservation support, and explainable recommendation behavior.
