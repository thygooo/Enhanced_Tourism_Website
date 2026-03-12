from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("ai_chatbot", "0001_initial"),
    ]

    operations = [
        migrations.AddField(
            model_name="recommendationevent",
            name="data_source",
            field=models.CharField(
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
        migrations.AddField(
            model_name="recommendationresult",
            name="data_source",
            field=models.CharField(
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
        migrations.AddField(
            model_name="systemmetriclog",
            name="data_source",
            field=models.CharField(
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
        migrations.AddField(
            model_name="usabilitysurveyresponse",
            name="data_source",
            field=models.CharField(
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
    ]
