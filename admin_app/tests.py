from django.contrib.auth import get_user_model
from django.contrib.auth.models import Group
from django.test import TestCase
from django.urls import reverse

from admin_app.models import Accomodation, TourismInformation


class AccommodationRegistrationRBACTests(TestCase):
    def setUp(self):
        user_model = get_user_model()
        self.user = user_model.objects.create_user(
            username="normal_guest",
            email="normal_guest@example.com",
            password="secure-pass-123",
            first_name="Normal",
            last_name="Guest",
        )
        self.owner_user = user_model.objects.create_user(
            username="accom_owner_user",
            email="accom_owner@example.com",
            password="secure-pass-456",
            first_name="Accom",
            last_name="Owner",
        )
        owner_group, _ = Group.objects.get_or_create(name="accommodation_owner")
        self.owner_user.groups.add(owner_group)
        self.url = reverse("admin_app:accommodation_register")

    def _payload(self, *, suffix):
        return {
            "company_name": f"Bayawan Test Stay {suffix}",
            "company_type": "Hotel",
            "location": "Bayawan City",
            "phone_number": "09990000000",
            "email_address": f"accom-{suffix}@example.com",
            "description": "A database-driven registration test record.",
            "password": "accom-pass-123",
        }

    def test_non_owner_cannot_access_registration_endpoint(self):
        self.client.force_login(self.user)
        response = self.client.get(self.url)
        self.assertEqual(response.status_code, 403)

    def test_non_owner_cannot_submit_registration(self):
        self.client.force_login(self.user)
        response = self.client.post(self.url, data=self._payload(suffix="non-owner"))
        self.assertEqual(response.status_code, 403)
        self.assertFalse(Accomodation.objects.filter(email_address="accom-non-owner@example.com").exists())

    def test_accommodation_owner_can_submit_registration_and_is_pending(self):
        self.client.force_login(self.owner_user)
        response = self.client.post(self.url, data=self._payload(suffix="owner"))
        self.assertEqual(response.status_code, 302)

        accom = Accomodation.objects.get(email_address="accom-owner@example.com")
        self.assertEqual(accom.owner_id, self.owner_user.pk)
        self.assertEqual(accom.approval_status, "pending")


class AccommodationDashboardTemplateRouteTests(TestCase):
    def test_dashboard_renders_without_stale_url_reverse_errors(self):
        accom = Accomodation.objects.create(
            company_name="Dashboard Stay",
            email_address="dashboard-stay@example.com",
            location="Bayawan",
            company_type="hotel",
            description="Template route regression test",
            password="accom-pass-123",
            phone_number="09995550123",
            approval_status="accepted",
            status="accepted",
        )
        session = self.client.session
        session["user_type"] = "accomodation"
        session["accom_id"] = accom.accom_id
        session["company_name"] = accom.company_name
        session["company_type"] = accom.company_type
        session.save()

        response = self.client.get(reverse("admin_app:accommodation_dashboard"))
        self.assertEqual(response.status_code, 200)


class AccommodationRegisterOwnerApprovalRequiredTests(TestCase):
    def setUp(self):
        user_model = get_user_model()
        self.user = user_model.objects.create_user(
            username="upgrade_guest_user",
            email="upgrade_guest_user@example.com",
            password="secure-pass-123",
            first_name="Upgrade",
            last_name="Guest",
        )
        self.url = reverse("admin_app:accommodation_register")

    def test_authenticated_guest_is_redirected_to_owner_signup_flow(self):
        self.client.force_login(self.user)
        response = self.client.get(self.url)
        self.assertEqual(response.status_code, 302)
        self.assertIn("owner_signup=1", response.url)
        self.user.refresh_from_db()
        self.assertFalse(self.user.groups.filter(name__iexact="accommodation_owner").exists())


class AccommodationOwnerApprovalDashboardTests(TestCase):
    def setUp(self):
        user_model = get_user_model()
        self.owner_candidate = user_model.objects.create_user(
            username="pending_owner_candidate",
            email="pending_owner_candidate@example.com",
            password="secure-pass-123",
            first_name="Pending",
            last_name="Owner",
        )
        pending_group, _ = Group.objects.get_or_create(name="accommodation_owner_pending")
        self.owner_candidate.groups.add(pending_group)

        session = self.client.session
        session["user_type"] = "employee"
        session["is_admin"] = True
        session["employee_id"] = 1
        session.save()

    def test_pending_owner_page_lists_owner_candidates(self):
        response = self.client.get(reverse("admin_app:pending_accommodation_owners"))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "pending_owner_candidate@example.com")

    def test_admin_can_accept_pending_owner(self):
        response = self.client.post(
            reverse("admin_app:accommodation_owner_update", kwargs={"user_id": self.owner_candidate.pk}),
            data={"action": "accept"},
        )
        self.assertEqual(response.status_code, 302)
        self.owner_candidate.refresh_from_db()
        self.assertTrue(self.owner_candidate.groups.filter(name__iexact="accommodation_owner").exists())
        self.assertFalse(self.owner_candidate.groups.filter(name__iexact="accommodation_owner_pending").exists())


class TourismInformationModelTests(TestCase):
    def setUp(self):
        user_model = get_user_model()
        self.user = user_model.objects.create_user(
            username="tourism_admin_seed",
            email="tourism_admin_seed@example.com",
            password="secure-tourism-pass-123",
            first_name="Tourism",
            last_name="Admin",
        )

    def test_defaults_to_draft_and_active(self):
        row = TourismInformation.objects.create(
            spot_name="Danjugan Falls",
            description="Scenic falls in Bayawan area.",
            location="Bayawan City",
            created_by=self.user,
            updated_by=self.user,
        )

        self.assertEqual(row.publication_status, "draft")
        self.assertTrue(row.is_active)
        self.assertFalse(row.is_published)

    def test_published_queryset_only_returns_active_published(self):
        TourismInformation.objects.create(
            spot_name="Published Spot",
            publication_status="published",
            is_active=True,
            created_by=self.user,
            updated_by=self.user,
        )
        TourismInformation.objects.create(
            spot_name="Archived Spot",
            publication_status="archived",
            is_active=False,
            created_by=self.user,
            updated_by=self.user,
        )
        TourismInformation.objects.create(
            spot_name="Unpublished Spot",
            publication_status="draft",
            is_active=True,
            created_by=self.user,
            updated_by=self.user,
        )

        rows = TourismInformation.objects.published()
        self.assertEqual(rows.count(), 1)
        self.assertEqual(rows.first().spot_name, "Published Spot")


class TourismInformationAdminAccessTests(TestCase):
    def _set_admin_session(self):
        session = self.client.session
        session["user_type"] = "employee"
        session["is_admin"] = True
        session["employee_id"] = 1
        session.save()

    def _set_non_admin_session(self):
        session = self.client.session
        session["user_type"] = "employee"
        session["is_admin"] = False
        session["employee_id"] = 2
        session.save()

    def test_non_admin_cannot_access_tourism_information_manage(self):
        self._set_non_admin_session()
        response = self.client.get(reverse("admin_app:tourism_information_manage"))
        self.assertEqual(response.status_code, 302)
        self.assertIn(reverse("admin_app:login"), response.url)

    def test_admin_can_create_publish_archive_tourism_information(self):
        self._set_admin_session()

        create_response = self.client.post(
            reverse("admin_app:tourism_information_create"),
            data={
                "spot_name": "Bayawan Heritage Park",
                "description": "A local cultural and historical attraction.",
                "location": "Bayawan City Proper",
                "contact_information": "09171234567",
                "operating_hours": "08:00 AM - 05:00 PM",
                "publication_status": "draft",
                "is_active": "on",
            },
        )
        self.assertEqual(create_response.status_code, 302)

        row = TourismInformation.objects.get(spot_name="Bayawan Heritage Park")
        self.assertEqual(row.publication_status, "draft")
        self.assertTrue(row.is_active)

        publish_response = self.client.post(
            reverse("admin_app:tourism_information_publish", kwargs={"tourism_info_id": row.tourism_info_id})
        )
        self.assertEqual(publish_response.status_code, 302)
        row.refresh_from_db()
        self.assertEqual(row.publication_status, "published")
        self.assertTrue(row.is_active)

        archive_response = self.client.post(
            reverse("admin_app:tourism_information_archive", kwargs={"tourism_info_id": row.tourism_info_id})
        )
        self.assertEqual(archive_response.status_code, 302)
        row.refresh_from_db()
        self.assertEqual(row.publication_status, "archived")
        self.assertFalse(row.is_active)
