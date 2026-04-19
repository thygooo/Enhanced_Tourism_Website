# Decision Tree Stability Evaluation (Defense Profile)

## Configuration
- Algorithm: `DecisionTreeClassifier`
- Criterion: `entropy`
- Max depth: `None`
- Min samples leaf: `3`
- Min samples split: `2`
- Max features: `None`
- CV strategy: `RepeatedStratifiedKFold` (5 folds x 3 repeats)

## Validation Metrics (Validation Fold)
| Metric | Mean | Std | Min | Max |
|---|---:|---:|---:|---:|
| accuracy | 0.8826 | 0.0119 | 0.8604 | 0.9104 |
| precision_macro | 0.8495 | 0.0146 | 0.8220 | 0.8833 |
| recall_macro | 0.8686 | 0.0130 | 0.8486 | 0.8986 |
| f1_macro | 0.8577 | 0.0134 | 0.8361 | 0.8904 |

## Overfitting Check (Train vs Validation)
| Metric | Train Mean | Validation Mean | Gap |
|---|---:|---:|---:|
| accuracy | 0.9703 | 0.8826 | 0.0877 |
| precision_macro | 0.9535 | 0.8495 | 0.1040 |
| recall_macro | 0.9763 | 0.8686 | 0.1077 |
| f1_macro | 0.9639 | 0.8577 | 0.1062 |

## Stability / Risk Checks
- Split instability flag: **False**
- Overfitting concern flag: **True**
- Leakage risk flag: **False**
- Class imbalance severity: **moderate** (ratio=2.6090)

## Data Quality Checks
- Rows: 2400
- Duplicate rows: 0
- Missing cells (features + target): 0
- Feature-label conflict rows: 0

## Thesis Alignment
- CNN remains for intent classification.
- Decision Tree remains the recommendation/refinement algorithm.
- Backend/database logic remains unchanged.
- Gemini remains phrasing-only.
