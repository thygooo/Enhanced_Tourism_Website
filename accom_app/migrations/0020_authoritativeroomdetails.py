from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('admin_app', '0025_accomodation_approval_status_and_more'),
        ('accom_app', '0019_alter_other_estab_intended_month_alter_room_accom_id_and_more'),
    ]

    operations = [
        migrations.CreateModel(
            name='AuthoritativeRoomDetails',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('room_type', models.CharField(blank=True, default='', max_length=100)),
                ('amenities', models.TextField(blank=True, default='')),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('updated_at', models.DateTimeField(auto_now=True)),
                ('room', models.OneToOneField(on_delete=django.db.models.deletion.CASCADE, related_name='owner_details', to='admin_app.room')),
            ],
        ),
    ]
