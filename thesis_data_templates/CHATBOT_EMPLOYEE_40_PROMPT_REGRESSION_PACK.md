# Employee 40-Prompt Regression Pack

Source pack: `thesis_data_templates/CHATBOT_EMPLOYEE_40_PROMPT_REGRESSION_PACK.csv`

## One-command test run

```powershell
python manage.py test ai_chatbot.tests.EmployeeChatbotRegressionPackTests --keepdb
```

This verifies:
- Employee support topic routing (tourist monitoring, bookings, records, reports, workflow, account support)
- Role-safe employee behavior with backend-driven summaries and links
- Clarification behavior for short/messy employee inputs
- Endpoint stability for mixed Filipino-English employee prompts
