from django.conf import settings
from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    initial = True

    dependencies = [
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
    ]

    operations = [
        migrations.CreateModel(
            name="RecommendationEvent",
            fields=[
                ("event_id", models.BigAutoField(primary_key=True, serialize=False)),
                ("event_type", models.CharField(choices=[("view", "View"), ("click", "Click"), ("save", "Save"), ("rate", "Rate"), ("book", "Book")], max_length=10)),
                ("item_ref", models.CharField(blank=True, default="", max_length=100)),
                ("rating_score", models.PositiveSmallIntegerField(blank=True, null=True)),
                ("dwell_time_sec", models.PositiveIntegerField(blank=True, null=True)),
                ("session_id", models.CharField(blank=True, default="", max_length=64)),
                ("event_time", models.DateTimeField(auto_now_add=True)),
                (
                    "user",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        related_name="recommendation_events",
                        to=settings.AUTH_USER_MODEL,
                    ),
                ),
            ],
            options={
                "db_table": "recommendation_events",
                "ordering": ["-event_time"],
            },
        ),
        migrations.CreateModel(
            name="RecommendationResult",
            fields=[
                ("result_id", models.BigAutoField(primary_key=True, serialize=False)),
                ("algorithm_version", models.CharField(default="v1", max_length=50)),
                ("context_json", models.JSONField(blank=True, default=dict)),
                ("recommended_items_json", models.JSONField(blank=True, default=list)),
                ("top_k", models.PositiveIntegerField(default=3)),
                ("clicked_item_ref", models.CharField(blank=True, default="", max_length=100)),
                ("feedback_rating", models.PositiveSmallIntegerField(blank=True, null=True)),
                ("generated_at", models.DateTimeField(auto_now_add=True)),
                (
                    "user",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        related_name="recommendation_results",
                        to=settings.AUTH_USER_MODEL,
                    ),
                ),
            ],
            options={
                "db_table": "recommendation_results",
                "ordering": ["-generated_at"],
            },
        ),
        migrations.CreateModel(
            name="SystemMetricLog",
            fields=[
                ("metric_id", models.BigAutoField(primary_key=True, serialize=False)),
                ("module", models.CharField(max_length=50)),
                ("endpoint", models.CharField(blank=True, default="", max_length=150)),
                ("response_time_ms", models.PositiveIntegerField()),
                ("success_flag", models.BooleanField(default=True)),
                ("status_code", models.PositiveIntegerField(blank=True, null=True)),
                ("error_message", models.TextField(blank=True, default="")),
                ("logged_at", models.DateTimeField(auto_now_add=True)),
            ],
            options={
                "db_table": "system_metric_logs",
                "ordering": ["-logged_at"],
            },
        ),
        migrations.CreateModel(
            name="UsabilitySurveyResponse",
            fields=[
                ("response_id", models.BigAutoField(primary_key=True, serialize=False)),
                ("statement_code", models.CharField(max_length=30)),
                ("likert_score", models.PositiveSmallIntegerField()),
                ("comment", models.TextField(blank=True, default="")),
                ("submitted_at", models.DateTimeField(auto_now_add=True)),
                (
                    "user",
                    models.ForeignKey(
                        blank=True,
                        null=True,
                        on_delete=django.db.models.deletion.SET_NULL,
                        related_name="usability_survey_responses",
                        to=settings.AUTH_USER_MODEL,
                    ),
                ),
            ],
            options={
                "db_table": "usability_survey_responses",
                "ordering": ["-submitted_at"],
            },
        ),
    ]
