from django.contrib.auth.models import AbstractBaseUser, BaseUserManager
from django.utils.translation import gettext_lazy as _
from django.db import models
from django.contrib.auth.hashers import make_password, check_password
from django.db import models
from django.contrib.auth.hashers import make_password, check_password
from django.utils import timezone
from django.conf import settings

class EmployeeManager(BaseUserManager):
    def create_user(self, email, password=None, **extra_fields):
        if not email:
            raise ValueError(_('The Email field is required'))
        email = self.normalize_email(email)
        user = self.model(email=email, **extra_fields)
        user.set_password(password)  # hashes the password
        user.save(using=self._db)
        return user

    def create_superuser(self, email, password=None, **extra_fields):
        extra_fields.setdefault('is_staff', True)
        extra_fields.setdefault('is_superuser', True)
        return self.create_user(email, password, **extra_fields)

class Employee(AbstractBaseUser):
    emp_id = models.AutoField(primary_key=True)
    first_name = models.CharField(max_length=100)
    last_name = models.CharField(max_length=100)
    middle_name = models.CharField(max_length=100, blank=True, null=True)
    username = models.CharField(max_length=100, unique=True, default='default_username')
    age = models.IntegerField()
    phone_number = models.CharField(max_length=15, unique=True)
    email = models.EmailField(unique=True)
    sex_choices = [('M', 'Male'), ('F', 'Female')]
    sex = models.CharField(max_length=1, choices=sex_choices)
    profile_picture = models.ImageField(upload_to='employee_pictures/', blank=True, null=True)
    role = models.CharField(max_length=50, default='Employee', editable=False)
    status = models.CharField(max_length=50, default='pending', editable=False)
    last_login = models.DateTimeField(null=True, blank=True)  # Track last login time

    # Required by Django permissions
    is_active = models.BooleanField(default=True)
    is_staff = models.BooleanField(default=False)
    is_superuser = models.BooleanField(default=False)

    objects = EmployeeManager()

    USERNAME_FIELD = 'email'  # we log in by email
    REQUIRED_FIELDS = ['first_name', 'last_name', 'age', 'phone_number', 'sex']

    def __str__(self):
        return f'{self.first_name} {self.last_name}'


class UserActivity(models.Model):
    """Model to track user activity for employees and admins"""
    ACTIVITY_TYPES = [
        ('login', 'Login'),
        ('logout', 'Logout'),
        ('view_page', 'View Page'),
        ('update', 'Update Data'),
        ('create', 'Create Data'),
        ('delete', 'Delete Data'),
        ('approve', 'Approve Request'),
        ('reject', 'Reject Request'),
        ('other', 'Other Action')
    ]
    
    employee = models.ForeignKey(Employee, on_delete=models.CASCADE, related_name='activities')
    activity_type = models.CharField(max_length=20, choices=ACTIVITY_TYPES)
    timestamp = models.DateTimeField(auto_now_add=True)
    page = models.CharField(max_length=255, blank=True, null=True)
    description = models.TextField(blank=True, null=True)
    ip_address = models.GenericIPAddressField(blank=True, null=True)
    user_agent = models.TextField(blank=True, null=True)
    
    class Meta:
        verbose_name_plural = "User Activities"
        ordering = ['-timestamp']
    
    def __str__(self):
        return f"{self.employee} - {self.get_activity_type_display()} - {self.timestamp}"


class Accomodation(models.Model):
    APPROVAL_STATUS_CHOICES = [
        ("pending", "Pending"),
        ("accepted", "Accepted"),
        ("declined", "Declined"),
    ]

    accom_id = models.AutoField(primary_key=True)
    owner = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="owned_accommodations",
        null=True,
        blank=True,
    )
    company_name = models.CharField(max_length=200)
    email_address = models.EmailField(unique=True)
    location = models.CharField(max_length=300)
    company_type = models.CharField(max_length=100)
    description = models.TextField(blank=True, default="")
    password = models.CharField(max_length=128)
    phone_number = models.CharField(max_length=20)
    status = models.CharField(max_length=50, null=True, blank=True, default="Pending")
    approval_status = models.CharField(
        max_length=20,
        choices=APPROVAL_STATUS_CHOICES,
        default="pending",
    )
    profile_picture = models.ImageField(upload_to='accommodation_profiles/', blank=True, null=True)

    def save(self, *args, **kwargs):
        # If the password is not already hashed, hash it.
        # Django password hashes usually start with a prefix like 'pbkdf2_'
        if self.password and not self.password.startswith('pbkdf2_'):
            self.password = make_password(self.password)
        # Keep legacy status field in sync with normalized approval_status.
        if self.approval_status:
            self.status = self.approval_status
        elif self.status:
            self.approval_status = self.status.strip().lower()
        super().save(*args, **kwargs)

    def __str__(self):
        return self.company_name


class AccommodationCertification(models.Model):
    """Model to store multiple certification images for accommodations"""
    accommodation = models.ForeignKey(Accomodation, on_delete=models.CASCADE, related_name='certifications')
    image = models.ImageField(upload_to='accommodation_certifications/')
    uploaded_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"Certification for {self.accommodation.company_name} ({self.id})"


class AdminInfo(models.Model):
    username = models.CharField(max_length=255, unique=True)
    password = models.CharField(max_length=255)
    first_name = models.CharField(max_length=100)
    last_name = models.CharField(max_length=100)
    is_staff = models.BooleanField(default=False)  # Only set True for admin users
    last_login = models.DateTimeField(null=True, blank=True)  # Add last_login field

    def set_password(self, raw_password):
        self.password = make_password(raw_password)

    def check_password(self, raw_password):
        return check_password(raw_password, self.password)

    def __str__(self):
        return self.username


from django.db import models

class Region(models.Model):
    name = models.CharField(max_length=255)

    def __str__(self):
        return self.name

class Country(models.Model):
    name = models.CharField(max_length=255)
    region = models.ForeignKey(Region, related_name="countries", on_delete=models.CASCADE)

    def __str__(self):
        return self.name

class Entry(models.Model):
    title = models.CharField(max_length=200)
    description = models.TextField(null=True, blank=True)
    is_hotel = models.BooleanField(default=False)  # Existing field for reference

    def __str__(self):
        return self.title


class TourismInformationQuerySet(models.QuerySet):
    def published(self):
        return self.filter(publication_status="published", is_active=True)


class TourismInformation(models.Model):
    PUBLICATION_STATUS_CHOICES = [
        ("draft", "Draft"),
        ("published", "Published"),
        ("archived", "Archived"),
    ]

    tourism_info_id = models.BigAutoField(primary_key=True)
    spot_name = models.CharField(max_length=200, db_index=True)
    description = models.TextField(blank=True, default="")
    location = models.CharField(max_length=300, blank=True, default="")
    contact_information = models.CharField(max_length=255, blank=True, default="")
    operating_hours = models.CharField(max_length=255, blank=True, default="")
    publication_status = models.CharField(
        max_length=20,
        choices=PUBLICATION_STATUS_CHOICES,
        default="draft",
        db_index=True,
    )
    is_active = models.BooleanField(default=True, db_index=True)
    image = models.ImageField(upload_to="tourism_information/", null=True, blank=True)
    created_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="created_tourism_information",
    )
    updated_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="updated_tourism_information",
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    objects = TourismInformationQuerySet.as_manager()

    class Meta:
        ordering = ["spot_name", "-updated_at"]
        verbose_name = "Tourism Information"
        verbose_name_plural = "Tourism Information"

    def __str__(self):
        return self.spot_name

    @property
    def is_published(self):
        return self.publication_status == "published" and self.is_active

class HotelConfirmation(models.Model):
    entry = models.OneToOneField(Entry, on_delete=models.CASCADE)
    confirmed = models.CharField(max_length=3, default="no")  # Will store "yes" if confirmed
    confirmed_on = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Hotel Confirmation for {self.entry.title}: {self.confirmed}"

class EstablishmentForm(models.Model):
    regions = models.ManyToManyField(Region, related_name="establishment_forms")
    countries = models.ManyToManyField(Country, related_name="establishment_forms")
    entries = models.ManyToManyField(Entry, related_name="establishment_forms")

    def __str__(self):
        return f"Establishment Form"

class Summary(models.Model):
    # Ensure the reference matches the model name exactly
    accom_id = models.ForeignKey('accom_app.Accommodation', on_delete=models.CASCADE)
    month_submitted = models.CharField(max_length=20)
    entry_ans = models.TextField(blank=True, null=True)
    hotel = models.CharField(max_length=1, default="0")  # use "1" for hotel, "0" when not marked as hotel

    def __str__(self):
        return f"Summary for {self.accom_id} on {self.month_submitted}"


class Room(models.Model):
    """Model to store room information for accommodations"""
    ROOM_STATUS_CHOICES = [
        ('AVAILABLE', 'Available'),
        ('OCCUPIED', 'Occupied'),
        ('UNAVAILABLE', 'Unavailable')
    ]
    
    room_id = models.AutoField(primary_key=True)
    accommodation = models.ForeignKey(Accomodation, on_delete=models.CASCADE, related_name='rooms')
    room_name = models.CharField(max_length=100)
    person_limit = models.IntegerField(default=0)
    current_availability = models.IntegerField(null=True, blank=True)
    price_per_night = models.DecimalField(max_digits=10, decimal_places=2, default=0)
    status = models.CharField(max_length=15, choices=ROOM_STATUS_CHOICES, default='AVAILABLE')
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        # Add unique constraint for room_name per accommodation
        unique_together = ['accommodation', 'room_name']
    
    def save(self, *args, **kwargs):
        # Set current_availability to person_limit if not specified
        if self.current_availability is None:
            self.current_availability = self.person_limit
        super().save(*args, **kwargs)
    
    def __str__(self):
        return f"{self.room_name} (Capacity: {self.person_limit}, Status: {self.get_status_display()})"


class RoomAssignment(models.Model):
    """Model to track guest assignments to rooms"""
    assignment_id = models.AutoField(primary_key=True)
    room = models.ForeignKey(Room, on_delete=models.CASCADE, related_name='assignments')
    guest = models.ForeignKey('guest_app.Guest', on_delete=models.CASCADE, related_name='room_assignments')
    is_owner = models.BooleanField(default=False)
    checked_in = models.DateTimeField(blank=True, null=True)
    checked_out = models.DateTimeField(blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"{self.guest} in {self.room}"


class TourAssignment(models.Model):
    """Model to assign tours to employees"""
    employee = models.ForeignKey(Employee, on_delete=models.CASCADE, related_name='tour_assignments')
    schedule = models.ForeignKey('tour_app.Tour_Schedule', on_delete=models.CASCADE, related_name='employee_assignments')
    assigned_date = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        unique_together = ['employee', 'schedule']
    
    def __str__(self):
        return f"{self.employee} assigned to {self.schedule}"
