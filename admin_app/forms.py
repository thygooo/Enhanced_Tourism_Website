# admin_app/forms.py

from django import forms
from django.contrib.auth import get_user_model
from django.contrib.auth.password_validation import validate_password
from django.core.exceptions import ValidationError
from django.utils import timezone

from .models import Employee
from .models import Accomodation, AccommodationCertification, TourismInformation


class MultipleFileInput(forms.ClearableFileInput):
    allow_multiple_selected = True
class EmployeeRegistrationForm(forms.ModelForm):
    password1 = forms.CharField(widget=forms.PasswordInput(), label="Password")
    password2 = forms.CharField(widget=forms.PasswordInput(), label="Confirm Password")

    class Meta:
        model = Employee
        fields = [
            'first_name',
            'last_name',
            'middle_name',
            'username',
            'age',
            'phone_number',
            'email',
            'sex',
            'profile_picture',
        ]

    def clean(self):
        cleaned_data = super().clean()
        password1 = cleaned_data.get('password1')
        password2 = cleaned_data.get('password2')
        if password1 and password2 and password1 != password2:
            self.add_error('password2', 'Passwords do not match.')
        return cleaned_data

    def save(self, commit=True):
        user = super().save(commit=False)
        # if password1 is valid, hash it before saving
        password = self.cleaned_data.get('password1')
        if password:
            user.set_password(password)
        if commit:
            user.save()
        return user



class AccommodationRegistrationForm(forms.ModelForm):
    password_confirm = forms.CharField(
        widget=forms.PasswordInput(),
        label="Confirm Accommodation Account Password",
    )
    certifications = forms.FileField(
        required=False,
        widget=MultipleFileInput(),
        help_text="Upload one or more business permits/certifications.",
    )

    def __init__(self, *args, **kwargs):
        self.owner = kwargs.pop("owner", None)
        super().__init__(*args, **kwargs)

    class Meta:
        model = Accomodation
        fields = [
            "company_name",
            "company_type",
            "location",
            "phone_number",
            "email_address",
            "description",
            "accommodation_amenities",
            "password",
            "profile_picture",
        ]
        labels = {
            "company_name": "Business Name",
            "company_type": "Business Type",
            "location": "Address",
            "phone_number": "Contact Number",
            "email_address": "Contact Email",
            "description": "Business Description",
            "accommodation_amenities": "Accommodation Amenities",
            "password": "Accommodation Account Password",
        }
        widgets = {
            "company_type": forms.Select(
                choices=[
                    ("Hotel", "Hotel"),
                    ("Inn", "Inn"),
                    ("Resort", "Resort"),
                    ("Homestay", "Homestay"),
                    ("Establishment", "Establishment"),
                ]
            ),
            "description": forms.Textarea(attrs={"rows": 4}),
            "accommodation_amenities": forms.Textarea(
                attrs={
                    "rows": 3,
                    "placeholder": "e.g., WiFi, Parking, 24/7 Front Desk, Restaurant",
                }
            ),
            "password": forms.PasswordInput(),
        }

    def clean_company_name(self):
        company_name = str(self.cleaned_data.get("company_name") or "").strip()
        if not company_name:
            raise forms.ValidationError("Business name is required.")
        qs = Accomodation.objects.filter(company_name__iexact=company_name)
        if self.owner is not None:
            qs = qs.filter(owner=self.owner).exclude(approval_status="declined")
        if qs.exists():
            raise forms.ValidationError(
                "You already submitted this business name. Please use a different name."
            )
        return company_name

    def clean_email_address(self):
        email_address = str(self.cleaned_data.get("email_address") or "").strip().lower()
        qs = Accomodation.objects.filter(email_address__iexact=email_address).exclude(
            approval_status="declined"
        )
        if self.owner is not None:
            qs = qs.exclude(owner=self.owner)
        if qs.exists():
            raise forms.ValidationError(
                "This accommodation email is already used in an active or pending registration."
            )
        return email_address

    def clean_phone_number(self):
        phone = str(self.cleaned_data.get("phone_number") or "").strip()
        digits = "".join(ch for ch in phone if ch.isdigit())
        if len(digits) < 7:
            raise forms.ValidationError("Enter a valid contact number.")
        return phone

    def clean(self):
        cleaned_data = super().clean()
        password = cleaned_data.get("password")
        password_confirm = cleaned_data.get("password_confirm")
        if password and password_confirm and password != password_confirm:
            self.add_error("password_confirm", "Passwords do not match.")
        if password:
            try:
                validate_password(password)
            except ValidationError as exc:
                self.add_error("password", exc)
        return cleaned_data

    def save(self, commit=True):
        instance = super().save(commit=False)
        if self.owner is not None:
            instance.owner = self.owner
        instance.approval_status = "pending"
        instance.status = "pending"
        instance.reviewed_at = None
        instance.reviewed_by = None
        instance.rejection_reason = ""
        if commit:
            instance.save()

            certification_files = self.files.getlist("certifications")
            for cert_file in certification_files:
                AccommodationCertification.objects.create(
                    accommodation=instance,
                    image=cert_file,
                )

        return instance


class AdminAccommodationEncodeForm(AccommodationRegistrationForm):
    owner_user = forms.ModelChoiceField(
        queryset=get_user_model().objects.none(),
        required=False,
        label="Link to Owner Account (Optional)",
        help_text="Leave blank for admin-encoded demo records not tied to an owner login.",
    )
    approval_status = forms.ChoiceField(
        choices=Accomodation.APPROVAL_STATUS_CHOICES,
        initial="accepted",
        label="Initial Approval Status",
    )
    rejection_reason = forms.CharField(
        required=False,
        widget=forms.Textarea(attrs={"rows": 3}),
        label="Decline Reason (Optional)",
    )

    def __init__(self, *args, **kwargs):
        self.reviewer = kwargs.pop("reviewer", None)
        super().__init__(*args, **kwargs)
        self.fields["owner_user"].queryset = get_user_model().objects.all().order_by("username")
        self.fields["approval_status"].help_text = "Use Accepted for quick demo-ready accommodations."

    def clean(self):
        cleaned_data = super().clean()
        approval_status = str(cleaned_data.get("approval_status") or "pending").strip().lower()
        rejection_reason = str(cleaned_data.get("rejection_reason") or "").strip()
        if approval_status == "declined" and not rejection_reason:
            self.add_error("rejection_reason", "Decline reason is required when status is Declined.")
        return cleaned_data

    def save(self, commit=True):
        instance = super().save(commit=False)

        selected_owner = self.cleaned_data.get("owner_user")
        instance.owner = selected_owner

        approval_status = str(self.cleaned_data.get("approval_status") or "pending").strip().lower()
        rejection_reason = str(self.cleaned_data.get("rejection_reason") or "").strip()

        instance.approval_status = approval_status
        instance.status = approval_status

        if approval_status in {"accepted", "declined"}:
            instance.reviewed_at = timezone.now()
            instance.reviewed_by = self.reviewer
            instance.rejection_reason = rejection_reason if approval_status == "declined" else ""
        else:
            instance.reviewed_at = None
            instance.reviewed_by = None
            instance.rejection_reason = ""

        if commit:
            instance.save()
            certification_files = self.files.getlist("certifications")
            for cert_file in certification_files:
                AccommodationCertification.objects.create(
                    accommodation=instance,
                    image=cert_file,
                )

        return instance


# Backward-compatible alias for existing imports/usages.
AccomodationForm = AccommodationRegistrationForm


from django import forms
from .models import EstablishmentForm, Region, Country, Entry

class EstablishmentFormAdmin(forms.ModelForm):
    class Meta:
        model = EstablishmentForm
        fields = ['regions', 'countries', 'entries']

    # Fields to allow text input for new regions, countries, and entries
    new_region = forms.CharField(max_length=255, required=False, widget=forms.TextInput(attrs={
        'placeholder': 'Enter new region',
        'class': 'form-control',
        'style': 'width: 100%; padding: 10px; border-radius: 5px;'
    }))
    new_country = forms.CharField(max_length=255, required=False, widget=forms.TextInput(attrs={
        'placeholder': 'Enter new country',
        'class': 'form-control',
        'style': 'width: 100%; padding: 10px; border-radius: 5px;'
    }))
    new_entry = forms.CharField(max_length=255, required=False, widget=forms.TextInput(attrs={
        'placeholder': 'Enter new entry',
        'class': 'form-control',
        'style': 'width: 100%; padding: 10px; border-radius: 5px;'
    }))

    # Use ModelMultipleChoiceField for selecting existing regions, countries, and entries
    regions = forms.ModelMultipleChoiceField(queryset=Region.objects.all(), widget=forms.CheckboxSelectMultiple, required=False)
    countries = forms.ModelMultipleChoiceField(queryset=Country.objects.all(), widget=forms.CheckboxSelectMultiple, required=False)
    entries = forms.ModelMultipleChoiceField(queryset=Entry.objects.all(), widget=forms.CheckboxSelectMultiple, required=False)

    def clean(self):
        cleaned_data = super().clean()

        # Handle adding new regions, countries, or entries
        new_region = cleaned_data.get('new_region')
        new_country = cleaned_data.get('new_country')
        new_entry = cleaned_data.get('new_entry')

        # If new values are provided, create new records in the database
        if new_region:
            region, created = Region.objects.get_or_create(name=new_region)
            cleaned_data['regions'] = Region.objects.filter(name=new_region)

        if new_country:
            country, created = Country.objects.get_or_create(name=new_country)
            cleaned_data['countries'] = Country.objects.filter(name=new_country)

        if new_entry:
            entry, created = Entry.objects.get_or_create(title=new_entry)
            cleaned_data['entries'] = Entry.objects.filter(title=new_entry)

        return cleaned_data


class TourismInformationForm(forms.ModelForm):
    class Meta:
        model = TourismInformation
        fields = [
            "spot_name",
            "description",
            "location",
            "contact_information",
            "operating_hours",
            "publication_status",
            "is_active",
            "image",
        ]
        widgets = {
            "description": forms.Textarea(attrs={"rows": 4}),
        }
