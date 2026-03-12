from django.contrib import admin

# Register your models here.
from django.contrib import admin
from .models import (
    Region,
    Country,
    Entry,
    EstablishmentForm,
    Accomodation,
    AccommodationCertification,
    TourismInformation,
)

admin.site.register(Region)
admin.site.register(Country)
admin.site.register(Entry)
admin.site.register(EstablishmentForm)


@admin.action(description="Mark selected accommodations as accepted")
def mark_accepted(modeladmin, request, queryset):
    queryset.update(approval_status="accepted", status="accepted")


@admin.action(description="Mark selected accommodations as declined")
def mark_declined(modeladmin, request, queryset):
    queryset.update(approval_status="declined", status="declined")


@admin.register(Accomodation)
class AccomodationAdmin(admin.ModelAdmin):
    list_display = (
        "company_name",
        "company_type",
        "email_address",
        "phone_number",
        "owner",
        "approval_status",
    )
    list_filter = ("approval_status", "company_type")
    search_fields = ("company_name", "email_address", "phone_number")
    readonly_fields = ("owner",)
    actions = [mark_accepted, mark_declined]


@admin.register(AccommodationCertification)
class AccommodationCertificationAdmin(admin.ModelAdmin):
    list_display = ("id", "accommodation", "uploaded_at")
    search_fields = ("accommodation__company_name",)


@admin.action(description="Publish selected tourism information")
def publish_tourism_information(modeladmin, request, queryset):
    queryset.update(publication_status="published", is_active=True)


@admin.action(description="Archive selected tourism information")
def archive_tourism_information(modeladmin, request, queryset):
    queryset.update(publication_status="archived", is_active=False)


@admin.register(TourismInformation)
class TourismInformationAdmin(admin.ModelAdmin):
    list_display = (
        "spot_name",
        "location",
        "publication_status",
        "is_active",
        "updated_at",
    )
    list_filter = ("publication_status", "is_active")
    search_fields = ("spot_name", "location", "description", "contact_information")
    readonly_fields = ("created_at", "updated_at")
    actions = [publish_tourism_information, archive_tourism_information]


