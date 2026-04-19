# Admin 40-Prompt Regression Pack

Source pack: `thesis_data_templates/CHATBOT_ADMIN_40_PROMPT_REGRESSION_PACK.csv`

## One-command test run

```powershell
python manage.py test ai_chatbot.tests.AdminChatbotRegressionPackTests --keepdb
```

This verifies:
- Admin support topic routing (approval, destination/content, records, monitoring, account management, reports)
- Backend-driven admin summaries and safe navigation links
- Clarification behavior for short/messy admin inputs
- Endpoint stability for mixed Filipino-English admin prompts
