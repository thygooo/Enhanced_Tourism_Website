# Decision Tree Pruning Search (Defense Profile)

## Baseline vs Recommended Pruned Model
| Metric | Baseline (mean+/-std) | Recommended Pruned (mean+/-std) |
|---|---:|---:|
| accuracy | 0.8819 +/- 0.0142 | 0.8902 +/- 0.0100 |
| precision_macro | 0.8483 +/- 0.0175 | 0.8559 +/- 0.0115 |
| recall_macro | 0.8698 +/- 0.0149 | 0.8921 +/- 0.0112 |
| f1_macro | 0.8574 +/- 0.0160 | 0.8701 +/- 0.0112 |
| train-validation gap (f1_macro) | 0.1060 | 0.0295 |

## Recommended Final Parameters
```json
{
  "criterion": "entropy",
  "max_depth": 8,
  "max_features": null,
  "min_samples_leaf": 5,
  "min_samples_split": 30
}
```

## Recommendation
- Keep baseline model: **False**
- Switch to pruned model: **True**
- Reason: Pruned model reduces overfitting gap with only small validation-F1 tradeoff, making it safer for defense generalization claims.

## Thesis Alignment
- CNN remains intent classification.
- Decision Tree remains recommendation/refinement.
- Backend/database logic remains unchanged.
- Gemini remains phrasing-only.
