# Likely Thesis Defense Questions and Concise Answers (20)

1. **Q:** What is the source of your CNN intent dataset?  
   **A:** The final dataset is `thesis_data_templates/text_cnn_messages_final_expanded_v3_clean.csv` with 1,920 labeled messages across 6 intents.

2. **Q:** Why did you perform duplicate and near-duplicate removal?  
   **A:** To prevent inflated performance from memorization and reduce split leakage, improving validity of reported metrics.

3. **Q:** How did you verify leakage reduction?  
   **A:** Using the leakage-cleaning report `text_cnn_messages_final_expanded_v3_clean_report.json`, where leakage pairs decreased from 232 to 0.

4. **Q:** Why did you choose Text-CNN for intent classification?  
   **A:** Text-CNN is efficient for sentence-level classification, captures local n-gram patterns, and is lightweight for deployment in a Django backend.

5. **Q:** What CNN metrics support model quality?  
   **A:** Accuracy=0.9697, Macro-F1=0.9680 from `artifacts/text_cnn_intent/evaluation_metrics_v3_full.json`.

6. **Q:** What does the CNN confusion matrix tell you?  
   **A:** Most intents are classified correctly; errors are concentrated in semantically related intent pairs such as recommendation vs information requests.

7. **Q:** Did you only rely on benchmark metrics?  
   **A:** No. We added manual inference validation (`manual_validation_summary_v1.json`) to test more realistic free-form phrasing.

8. **Q:** Why is manual intent performance lower than benchmark results?  
   **A:** The manual set includes broader conversational phrasing and distribution shift compared with the cleaned split used during benchmark evaluation.

9. **Q:** What is your fallback if CNN intent confidence is weak?  
   **A:** The system has heuristic intent fallback logic in `ai_chatbot/views.py` when CNN output is unavailable or incompatible.

10. **Q:** Why did you choose Decision Tree for recommendation scoring?  
    **A:** Decision Trees are interpretable, fast, and suitable for mixed categorical/numeric tourism preference features.

11. **Q:** What is your Decision Tree target variable?  
    **A:** `relevance_label` with classes `relevant` and `not_relevant`.

12. **Q:** What are your most influential Decision Tree features?  
    **A:** Budget, room price, requested guests, and room capacity based on `feature_importance_v1.csv`.

13. **Q:** How did you preprocess mixed feature types?  
    **A:** Numeric features use median imputation; categorical features use most-frequent imputation plus one-hot encoding in a scikit-learn pipeline.

14. **Q:** What are the Decision Tree evaluation results?  
    **A:** Accuracy=0.75 and Macro-F1=0.7122 on a stratified 20% test split (`evaluation_metrics_v1.json`).

15. **Q:** How do you interpret the Decision Tree confusion matrix?  
    **A:** The model detects many relevant cases well, but still misclassifies some not-relevant samples, indicating room for calibration.

16. **Q:** Is the Decision Tree already integrated in Django runtime?  
    **A:** Yes. Runtime loads `artifacts/decision_tree_final/decision_tree_final.pkl` via `ai_chatbot/recommenders.py`.

17. **Q:** How do you handle missing recommendation inputs?  
    **A:** Inputs are normalized with safe defaults (e.g., guests, budget, nights, rank) before inference.

18. **Q:** How did you verify end-to-end behavior of both models?  
    **A:** Through `ai_chatbot/run_end_to_end_validation.py` with outputs in `artifacts/validation/end_to_end_validation_v1.json`.

19. **Q:** What are the main limitations of your current setup?  
    **A:** CNN sensitivity to free-form phrasing and Decision Tree dependence on synthetic training data.

20. **Q:** What is your next improvement for deployment readiness?  
    **A:** Collect real chatbot interaction logs, retrain/calibrate both models on real distributions, and apply confidence-threshold routing for safer fallback behavior.
