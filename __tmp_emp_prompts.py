from django.test import Client
from admin_app.models import Employee
import json

emp = Employee.objects.filter(status='accepted').first()
print(f"USING_EMPLOYEE: {getattr(emp, 'emp_id', None)} {getattr(emp, 'username', None)} {getattr(emp, 'role', None)}")
if not emp:
    raise SystemExit(0)

c = Client(HTTP_HOST='127.0.0.1:8000')
s = c.session
s['user_type'] = 'employee'
s['employee_id'] = emp.emp_id
s['is_admin'] = str(getattr(emp, 'role', '')).strip().lower() == 'admin'
s.save()

prompts = [
    "How can I view all bookings in the system?",
    "Can I check pending reservations?",
    "How do I know which bookings are confirmed?",
    "Where can I monitor accommodation bookings?",
    "how many tour bookings do we have?",
    "how much total revenue do we have all time when it comes to tour bookings?",
    "Where can I see booking summaries?",
    "Can I view tourist statistics?",
    "How do I access monitoring reports?",
    "How do I view registered accommodations?",
    "Can I check which hotels or inns are active?",
    "Can I view tourism destination records?",
    "How do I approve accommodation registrations?",
    "Paano makita ang tourist records sa system?",
    "pending bookings",
]

for i, p in enumerate(prompts, 1):
    r = c.post('/api/chat/', data=json.dumps({'message': p}), content_type='application/json')
    d = r.json() if r.status_code == 200 else {}
    t = str(d.get('fulfillmentText') or '').replace('\n', ' | ')
    if len(t) > 300:
        t = t[:300] + '...'
    print(f"{i:02d}. {p}")
    print(f"    status={r.status_code} link={d.get('billing_link_label', '')} needs={d.get('needs_clarification')}")
    print(f"    {t}")
