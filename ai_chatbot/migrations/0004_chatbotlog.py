from django.conf import settings
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("ai_chatbot", "0003_usabilitysurveyresponse_survey_batch_id"),
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
    ]

    operations = [
        migrations.CreateModel(
            name="ChatbotLog",
            fields=[
                ("log_id", models.BigAutoField(primary_key=True, serialize=False)),
                ("user_message", models.TextField()),
                ("resolved_intent", models.CharField(blank=True, default="", max_length=80)),
                ("resolved_params_json", models.JSONField(blank=True, default=dict)),
                ("bot_response", models.TextField()),
                ("intent_classifier_source", models.CharField(blank=True, default="", max_length=80)),
                ("response_nlg_source", models.CharField(blank=True, default="", max_length=80)),
                ("fallback_used", models.BooleanField(default=False)),
                ("provenance_json", models.JSONField(blank=True, default=dict)),
                (
                    "data_source",
                    models.CharField(
                        choices=[
                            ("unlabeled", "Unlabeled"),
                            ("demo_seeded", "Demo Seeded"),
                            ("pilot_test", "Pilot Test"),
                            ("real_world", "Real World"),
                        ],
                        db_index=True,
                        default="unlabeled",
                        max_length=20,
                    ),
                ),
                ("created_at", models.DateTimeField(auto_now_add=True, db_index=True)),
                (
                    "user",
                    models.ForeignKey(
                        blank=True,
                        null=True,
                        on_delete=models.SET_NULL,
                        related_name="chatbot_logs",
                        to=settings.AUTH_USER_MODEL,
                    ),
                ),
            ],
            options={
                "db_table": "chatbot_logs",
                "ordering": ["-created_at"],
            },
        ),
    ]
