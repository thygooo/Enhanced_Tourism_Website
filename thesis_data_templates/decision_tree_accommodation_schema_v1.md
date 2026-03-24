# Decision Tree Accommodation Recommendation Dataset Schema (Phase 2)

## Use Case Context
This dataset is designed for a tourism chatbot that recommends hotel/inn rooms based on user preferences and candidate room attributes. Each row represents one displayed room candidate for one recommendation request.

## Modeling Objective
- Task type: Supervised classification
- Algorithm: Decision Tree Classifier
- Prediction target: `relevance_label`
- Target meaning:
  - `relevant`: candidate is suitable for the user request and likely to be selected/booked
  - `not_relevant`: candidate does not fit preference/constraints or is unlikely to be selected

## Final Feature Set
| Feature | Description | Data Type Class | Recommended Preprocessing |
|---|---|---|---|
| `requested_guests` | User-requested number of guests | Numeric (integer) | Coerce to numeric; median imputation if missing |
| `requested_budget` | User budget ceiling (PHP per night) | Numeric (continuous) | Coerce to numeric; median imputation; optional clipping for outliers |
| `requested_location` | Preferred location text from user | Categorical | Lowercase/trim; mode imputation; one-hot encoding |
| `requested_accommodation_type` | Preferred type (`hotel`, `inn`, `either`) | Categorical | Normalize labels; mode imputation; one-hot encoding |
| `room_price_per_night` | Candidate room price (PHP/night) | Numeric (continuous) | Coerce to numeric; median imputation |
| `room_capacity` | Maximum room guest capacity | Numeric (integer) | Coerce to numeric; median imputation |
| `room_available` | Availability indicator/count (>=1 available) | Numeric (integer, bounded) | Coerce to numeric; median imputation |
| `accom_location` | Candidate accommodation location | Categorical | Lowercase/trim; mode imputation; one-hot encoding |
| `company_type` | Candidate business type (`hotel`/`inn`) | Binary categorical | Normalize labels; mode imputation; one-hot encoding |
| `nights_requested` | Number of nights inferred/requested | Ordinal numeric (integer) | Coerce to numeric; median imputation |
| `cnn_confidence` | Intent/CNN confidence used in hybrid scoring | Numeric (0-1) | Coerce to numeric; clip to [0,1]; median imputation |
| `shown_rank` | Position shown to user in recommendation list | Ordinal numeric (integer) | Coerce to numeric; median imputation |

## Target Variable
| Field | Type | Classes | Notes |
|---|---|---|---|
| `relevance_label` | Categorical (binary class) | `relevant`, `not_relevant` | Derived from booking/selection behavior and fit rules |

## Data Integrity Constraints
- `requested_guests >= 1`
- `requested_budget >= 0`
- `room_price_per_night > 0`
- `room_capacity >= 1`
- `room_available >= 0`
- `nights_requested >= 1`
- `0 <= cnn_confidence <= 1`
- `shown_rank >= 1`
- If `requested_accommodation_type` is not `either`, it should align with `company_type` for highly relevant cases.

## Thesis Notes
- This schema is aligned with the current recommendation pipeline fields already referenced in `ai_chatbot/recommenders.py` and `ai_chatbot/train_decision_tree_demo.py`.
- Categorical encoding plus median/mode imputation is appropriate for Decision Tree training in scikit-learn pipelines.
