# Guest/Tourist 40-Prompt Regression Pack

Source pack: `thesis_data_templates/CHATBOT_GUEST_40_PROMPT_REGRESSION_PACK.csv`

## One-command test run

```powershell
python manage.py test ai_chatbot.tests.ChatbotGuestPromptRegressionPackTests --keepdb
```

This verifies:
- CNN + heuristic intent routing for all 40 required prompts
- Guest chatbot endpoint returns safe non-empty responses for all prompts
- Clarification behavior is triggered for incomplete prompts
- No hard crash on mixed Filipino-English, short, messy, or underspecified inputs
