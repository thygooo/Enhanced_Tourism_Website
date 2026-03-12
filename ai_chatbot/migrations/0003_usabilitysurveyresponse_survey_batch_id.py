from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("ai_chatbot", "0002_add_data_source_fields"),
    ]

    operations = [
        migrations.AddField(
            model_name="usabilitysurveyresponse",
            name="survey_batch_id",
            field=models.CharField(blank=True, db_index=True, default="", max_length=64),
        ),
    ]
