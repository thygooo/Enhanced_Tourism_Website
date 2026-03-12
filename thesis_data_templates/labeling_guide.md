# Labeling Guide for Text-CNN and Decision Tree Datasets

This guide standardizes how the team labels data for:

- `text_cnn_messages.csv` (Text-CNN training/evaluation)
- `accommodation_reco_training.csv` (Decision Tree recommendation training/evaluation)

Use this guide before labeling to avoid inconsistent labels.

## 1. General Rules

- Use lowercase label values exactly as defined in this guide.
- Do not invent new labels without team agreement.
- If a message is unclear, choose the closest label and note it for review.
- Remove or anonymize personal information before exporting/labeling (name, phone, email).
- Keep timestamps in ISO format if possible: `YYYY-MM-DDTHH:MM:SS`.

## 2. Text-CNN Dataset (`text_cnn_messages.csv`)

## 2.1 Purpose

Label user chatbot messages so the Text-CNN can classify:

- user intent
- accommodation type preference (optional)
- budget signal (optional)
- location signal (optional)

## 2.2 Required Columns (Core)

- `message_id`
- `message_text`
- `label_intent`
- `source`
- `timestamp`
- `split`

Optional but recommended:

- `label_accommodation_type`
- `label_budget_level`
- `label_location_signal`

## 2.3 Intent Labels (Use Exactly)

- `get_accommodation_recommendation`
- `calculate_accommodation_billing`
- `book_accommodation`
- `get_tour_recommendation` (only if tour messages are included)
- `calculate_tour_billing` (only if tour billing messages are included)
- `greeting`
- `follow_up_clarification`
- `other`

## 2.4 Intent Label Definitions

- `get_accommodation_recommendation`
  - User asks for hotel/inn suggestions, options, or recommendations.
  - Examples: "recommend a hotel", "cheap inn in bayawan"

- `calculate_accommodation_billing`
  - User asks for room cost, estimate, total bill, or price computation.
  - Examples: "calculate hotel bill", "how much for 2 nights"

- `book_accommodation`
  - User wants to reserve/book a room or proceed with booking.
  - Examples: "book room 12", "reserve an inn for tomorrow"

- `get_tour_recommendation`
  - User asks for tour suggestions (not accommodation).
  - Examples: "recommend a tour", "what tour is good for family"

- `calculate_tour_billing`
  - User asks for tour cost/billing estimate.
  - Examples: "tour bill for 3 guests"

- `greeting`
  - Greeting/small talk with no actionable travel request.
  - Examples: "hello", "good morning"

- `follow_up_clarification`
  - User gives incomplete details or answers a previous chatbot question.
  - Examples: "for 2 guests", "near poblacion", "budget is 1500"

- `other`
  - Unsupported, irrelevant, or unclear messages.
  - Examples: unrelated topics, spam, non-tourism requests

## 2.5 Auxiliary Labels (Optional but Recommended)

### `label_accommodation_type`

- `hotel`
- `inn`
- `either`
- `unknown`

Rules:

- Use `hotel` if the message explicitly asks for hotel.
- Use `inn` if the message explicitly asks for inn.
- Use `either` if user asks generally for a place to stay/hotel or inn.
- Use `unknown` if not inferable or message is not accommodation-related.

### `label_budget_level`

- `low`
- `mid`
- `high`
- `unspecified`

Suggested thresholds (team may revise, but keep consistent):

- `low`: <= 1500
- `mid`: 1501 to 3000
- `high`: > 3000
- `unspecified`: no budget mentioned

If you change thresholds, update this guide and relabel consistently.

### `label_location_signal`

- `with_location`
- `without_location`

Use `with_location` if the message includes a place/area signal (e.g., Bayawan, Poblacion, terminal area).

## 2.6 Split Rules (`split`)

Allowed values:

- `train`
- `val`
- `test`

Recommendations:

- Early manual labeling can be `train` first.
- Assign `val` and `test` after enough data is collected.
- Avoid duplicate or near-duplicate messages across splits.

## 2.7 Text-CNN Labeling Examples

- "recommend a cheap inn in bayawan for 2 guests"
  - `label_intent = get_accommodation_recommendation`
  - `label_accommodation_type = inn`
  - `label_budget_level = low`
  - `label_location_signal = with_location`

- "calculate hotel bill for room 12 for 2 nights"
  - `label_intent = calculate_accommodation_billing`
  - `label_accommodation_type = hotel`
  - `label_budget_level = unspecified`
  - `label_location_signal = without_location`

- "for 3 guests only"
  - `label_intent = follow_up_clarification`
  - `label_accommodation_type = unknown`
  - `label_budget_level = unspecified`
  - `label_location_signal = without_location`

## 2.8 Edge Cases (Text-CNN)

- Mixed request in one message:
  - Example: "recommend a hotel and calculate the bill for 2 nights"
  - Rule: label by primary intent for now (usually first request) and flag for review.
  - Optional future improvement: multi-label classification.

- Ambiguous booking language:
  - "I want a room tomorrow"
  - If it clearly implies reservation -> `book_accommodation`
  - If unclear and missing booking action -> `follow_up_clarification`

- Message contains typo or grammar issues:
  - Keep original text; do not rewrite.
  - Label based on intended meaning.

## 3. Decision Tree Dataset (`accommodation_reco_training.csv`)

## 3.1 Purpose

Label structured recommendation outcomes so the Decision Tree can learn which recommendations are relevant or likely to be selected/booked.

One row should represent one recommendation candidate shown to a user/session.

## 3.2 Key Inputs (Features)

Examples of important fields:

- user request fields: `requested_guests`, `requested_budget`, `requested_location`, `requested_accommodation_type`
- candidate room fields: `room_price_per_night`, `room_capacity`, `room_available`, `company_type`, `accom_location`
- AI context fields: `cnn_intent_label`, `cnn_confidence`
- serving context: `shown_rank`, `timestamp`

## 3.3 Target Columns (Labels)

Recommended target fields:

- `was_clicked` (`0` or `1`)
- `was_booked` (`0` or `1`)
- `was_selected` (`0` or `1`)
- `relevance_label` (`relevant` or `not_relevant`)

Use at least one target for model training. Recommended primary target:

- `was_booked`

If booking volume is low, use:

- `was_clicked` or `was_selected`

## 3.4 Label Definitions (Decision Tree)

- `was_clicked = 1`
  - User clicked/opened the recommended item (room/hotel card/link/button)

- `was_booked = 1`
  - The recommended item resulted in a completed booking submission for the same room/accommodation within the same query/session window

- `was_selected = 1`
  - User selected the item as preferred choice (even if no final booking yet)

- `relevance_label = relevant`
  - Recommendation matches user constraints/preferences well enough to be considered suitable

- `relevance_label = not_relevant`
  - Recommendation violates or poorly matches key constraints/preferences

## 3.5 Decision Tree Labeling Rules

- If room exceeds requested budget significantly and budget is strict:
  - usually `not_relevant`

- If room capacity is below guest count:
  - `not_relevant`

- If room unavailable (`room_available = 0`):
  - `not_relevant`

- If room matches budget, capacity, and location preference:
  - likely `relevant`

- A row can be `relevant` even if `was_booked = 0`
  - User may choose not to book for non-system reasons.

## 3.6 Decision Tree Examples

- Candidate within budget, enough capacity, available, correct location:
  - `relevance_label = relevant`

- Candidate unavailable but shown:
  - `room_available = 0`
  - `relevance_label = not_relevant`

- Candidate slightly over budget but all else matches:
  - Team decision needed:
  - Option A (strict): `not_relevant`
  - Option B (soft budget): `relevant`
  - Choose one policy and apply consistently.

## 3.7 Linking Recommendation Rows to Outcomes

To make training useful, preserve linkage:

- `query_id` ties multiple shown candidates to one user request
- `session_id` groups related events
- `room_id` identifies the exact candidate
- booking logs should include `room_id` and timestamp so `was_booked` can be derived

## 4. Quality Control Process (Team Workflow)

Recommended workflow:

1. One member labels a batch.
2. Another member reviews 10 to 20 percent of rows.
3. Resolve disagreements and update this guide if needed.
4. Recheck label counts for imbalance or accidental new labels.

## 4.1 Quick QC Checklist

- No blank `label_intent` in Text-CNN dataset
- No invalid label names (typos)
- `split` values only `train`, `val`, `test`
- Binary fields only `0` or `1`
- `relevance_label` only `relevant` or `not_relevant`
- No personally identifiable information in exported text fields

## 5. Versioning and Changes

- If the team changes labels, thresholds, or rules:
  - update this file first
  - note the date of change
  - avoid mixing old/new labeling standards in the same training batch

Suggested version note format:

- `Labeling guide v1.0 - 2026-02-25`

