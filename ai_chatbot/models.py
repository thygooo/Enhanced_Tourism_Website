from django.conf import settings
from django.db import models

DATA_SOURCE_CHOICES = [
    ("unlabeled", "Unlabeled"),
    ("demo_seeded", "Demo Seeded"),
    ("pilot_test", "Pilot Test"),
    ("real_world", "Real World"),
]


class RecommendationEvent(models.Model):
    EVENT_CHOICES = [
        ("view", "View"),
        ("click", "Click"),
        ("save", "Save"),
        ("rate", "Rate"),
        ("book", "Book"),
    ]

    event_id = models.BigAutoField(primary_key=True)
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="recommendation_events",
    )
    event_type = models.CharField(max_length=10, choices=EVENT_CHOICES)
    item_ref = models.CharField(max_length=100, blank=True, default="")
    rating_score = models.PositiveSmallIntegerField(null=True, blank=True)
    dwell_time_sec = models.PositiveIntegerField(null=True, blank=True)
    session_id = models.CharField(max_length=64, blank=True, default="")
    data_source = models.CharField(
        max_length=20,
        choices=DATA_SOURCE_CHOICES,
        default="unlabeled",
        db_index=True,
    )
    event_time = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "recommendation_events"
        ordering = ["-event_time"]

    def __str__(self):
        return f"{self.user_id} {self.event_type} {self.item_ref}".strip()


class RecommendationResult(models.Model):
    result_id = models.BigAutoField(primary_key=True)
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="recommendation_results",
    )
    algorithm_version = models.CharField(max_length=50, default="v1")
    context_json = models.JSONField(default=dict, blank=True)
    recommended_items_json = models.JSONField(default=list, blank=True)
    top_k = models.PositiveIntegerField(default=3)
    clicked_item_ref = models.CharField(max_length=100, blank=True, default="")
    feedback_rating = models.PositiveSmallIntegerField(null=True, blank=True)
    data_source = models.CharField(
        max_length=20,
        choices=DATA_SOURCE_CHOICES,
        default="unlabeled",
        db_index=True,
    )
    generated_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "recommendation_results"
        ordering = ["-generated_at"]

    def __str__(self):
        return f"RecommendationResult #{self.result_id} for {self.user_id}"


class SystemMetricLog(models.Model):
    metric_id = models.BigAutoField(primary_key=True)
    module = models.CharField(max_length=50)
    endpoint = models.CharField(max_length=150, blank=True, default="")
    response_time_ms = models.PositiveIntegerField()
    success_flag = models.BooleanField(default=True)
    status_code = models.PositiveIntegerField(null=True, blank=True)
    error_message = models.TextField(blank=True, default="")
    data_source = models.CharField(
        max_length=20,
        choices=DATA_SOURCE_CHOICES,
        default="unlabeled",
        db_index=True,
    )
    logged_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "system_metric_logs"
        ordering = ["-logged_at"]

    def __str__(self):
        return f"{self.module} {self.endpoint} ({self.response_time_ms}ms)"


class UsabilitySurveyResponse(models.Model):
    response_id = models.BigAutoField(primary_key=True)
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="usability_survey_responses",
    )
    statement_code = models.CharField(max_length=30)
    likert_score = models.PositiveSmallIntegerField()
    comment = models.TextField(blank=True, default="")
    survey_batch_id = models.CharField(max_length=64, blank=True, default="", db_index=True)
    data_source = models.CharField(
        max_length=20,
        choices=DATA_SOURCE_CHOICES,
        default="unlabeled",
        db_index=True,
    )
    submitted_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "usability_survey_responses"
        ordering = ["-submitted_at"]

    def __str__(self):
        return f"{self.statement_code}={self.likert_score}"


class ChatbotLog(models.Model):
    log_id = models.BigAutoField(primary_key=True)
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="chatbot_logs",
    )
    user_message = models.TextField()
    resolved_intent = models.CharField(max_length=80, blank=True, default="")
    resolved_params_json = models.JSONField(default=dict, blank=True)
    bot_response = models.TextField()
    intent_classifier_source = models.CharField(max_length=80, blank=True, default="")
    response_nlg_source = models.CharField(max_length=80, blank=True, default="")
    fallback_used = models.BooleanField(default=False)
    provenance_json = models.JSONField(default=dict, blank=True)
    data_source = models.CharField(
        max_length=20,
        choices=DATA_SOURCE_CHOICES,
        default="unlabeled",
        db_index=True,
    )
    created_at = models.DateTimeField(auto_now_add=True, db_index=True)

    class Meta:
        db_table = "chatbot_logs"
        ordering = ["-created_at"]

    def __str__(self):
        return f"ChatbotLog #{self.log_id} intent={self.resolved_intent or 'n/a'}"
