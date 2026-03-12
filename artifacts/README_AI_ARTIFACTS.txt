AI MODEL ARTIFACTS (SAFE WORKFLOW)

Purpose:
- Keep pilot/sample-data training outputs separate from final real-data training outputs.
- Avoid accidentally using pilot models for final thesis evaluation.

Recommended folders:
- artifacts/text_cnn_template_demo/      -> pilot Text-CNN model (sample dataset)
- artifacts/decision_tree_demo/          -> pilot Decision Tree model (sample dataset)
- artifacts/text_cnn_final/              -> final Text-CNN model (real labeled data)
- artifacts/decision_tree_final/         -> final Decision Tree model (real labeled data)
- artifacts/backups/                     -> optional backups before replacing models

Rules:
1) Do not overwrite pilot artifacts when training final models.
2) Put "demo" or "final" in folder names.
3) Keep a short notes file per artifact folder:
   - dataset used
   - date trained
   - metrics
   - target label
4) Final thesis metrics should come from final (real-data) artifacts only.

Safe workflow:
1. Pilot training (sample CSVs)
   - save to *_demo folders
2. Real data collection and labeling
3. Final training (real CSVs)
   - save to *_final folders
4. Update system loading path to final folder after validation

Re-training is safe:
- Training creates/replaces model files only.
- It does not permanently change your database schema.
- You can retrain multiple times as data improves.

