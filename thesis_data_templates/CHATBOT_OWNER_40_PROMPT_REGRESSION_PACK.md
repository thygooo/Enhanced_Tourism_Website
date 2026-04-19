# Accommodation Owner 40-Prompt Regression Pack

Source pack: `thesis_data_templates/CHATBOT_OWNER_40_PROMPT_REGRESSION_PACK.csv`

## One-command test run

```powershell
python manage.py test ai_chatbot.tests.OwnerChatbotRegressionPackTests --keepdb
```

This verifies:
- Owner-specific support topic routing (registration, room ops, bookings, availability, billing, dashboard support)
- Role-safe owner response behavior and authorized owner-only summaries
- Short/messy owner prompts trigger follow-up guidance where needed
- Endpoint remains stable for mixed Filipino-English owner prompts
