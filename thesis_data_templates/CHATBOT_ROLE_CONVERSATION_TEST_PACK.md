# Chatbot Role Conversation Test Pack

Use this script pack for regression checks and panel rehearsal.

## Guest Flow (Recommendation + Explainability)
1. `show default hotel suggestions`
2. `why option 1`
3. `compare top 3`
4. `recommend a quiet and affordable hotel`
5. `remember my preferences: hotel in bayawan budget 2000 for 2 guests`
6. `recommend options`
7. `forget my preferences`

Expected:
- Recommendation list is shown.
- `why option 1` returns explanation/reasons.
- `compare top 3` returns comparative summary.
- Preference-first prompt returns recommendations.
- Remember/forget commands return confirmation text.

## Owner Flow (Operations)
1. `help`
2. `show my rooms`
3. `show my bookings`
4. `open owner hub`
5. `open reports and analytics`

Expected:
- Owner-scoped guidance and snapshot.
- Rooms/bookings summary shown.
- Owner links open correctly.

## Admin Flow (Moderation + Oversight)
1. `help`
2. `show pending accommodations`
3. `show pending owner accounts`
4. `open accommodation bookings`
5. `open dashboard`

Expected:
- Admin-scoped guidance and snapshot.
- Pending summaries appear.
- Admin navigation links resolve.

## Employee Flow (Task Navigation)
1. `help`
2. `open assigned tours`
3. `open tour calendar`
4. `open map`
5. `open profile`

Expected:
- Employee-scoped guidance and snapshot.
- Dashboard navigation works.

## Out-of-Scope Safety
Run for each role:
- `write me a poem about stars`

Expected:
- Scope-limited response appears.
- Quick replies redirect user to supported tasks.
