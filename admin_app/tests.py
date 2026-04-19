from datetime import date, timedelta

from django.contrib.auth import get_user_model
from django.contrib.auth.models import Group
from django.test import TestCase
from django.urls import reverse

from admin_app.models import Accomodation, TourismInformation, Room
from guest_app.models import AccommodationBooking


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
            "password_confirm": "accom-pass-123",
        }

    def test_non_owner_cannot_access_registration_endpoint(self):
        self.client.force_login(self.user)
        response = self.client.get(self.url)
        self.assertEqual(response.status_code, 302)
        self.assertIn(reverse("admin_app:login"), response.url)

    def test_non_owner_cannot_submit_registration(self):
        self.client.force_login(self.user)
        response = self.client.post(self.url, data=self._payload(suffix="non-owner"))
        self.assertEqual(response.status_code, 302)
        self.assertIn(reverse("admin_app:login"), response.url)
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
        user_model = get_user_model()
        owner_user = user_model.objects.create_user(
            username="owner_dashboard_user",
            email="owner_dashboard@example.com",
            password="secure-pass-789",
            first_name="Owner",
            last_name="Dashboard",
        )
        owner_group, _ = Group.objects.get_or_create(name="accommodation_owner")
        owner_user.groups.add(owner_group)
        accom = Accomodation.objects.create(
            owner=owner_user,
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
        self.client.force_login(owner_user)

        response = self.client.get(reverse("admin_app:accommodation_dashboard"))
        self.assertEqual(response.status_code, 200)


class OwnerRoomBookingsJsonTests(TestCase):
    def setUp(self):
        user_model = get_user_model()
        owner_group, _ = Group.objects.get_or_create(name="accommodation_owner")

        self.owner_user = user_model.objects.create_user(
            username="owner_room_json_user",
            email="owner_room_json@example.com",
            password="secure-pass-789",
            first_name="Owner",
            last_name="Json",
        )
        self.owner_user.groups.add(owner_group)

        self.accom = Accomodation.objects.create(
            owner=self.owner_user,
            company_name="JSON Stay",
            email_address="json-stay@example.com",
            location="Bayawan",
            company_type="hotel",
            description="Owner room booking JSON test",
            password="accom-pass-123",
            phone_number="09995550199",
            approval_status="accepted",
            status="accepted",
        )
        self.room = Room.objects.create(
            accommodation=self.accom,
            room_name="Executive Twin",
            person_limit=5,
            current_availability=5,
            price_per_night="3400.00",
            status="AVAILABLE",
        )
        self.guest_user = user_model.objects.create_user(
            username="owner_room_guest",
            email="owner_room_guest@example.com",
            password="secure-pass-111",
            first_name="Jade",
            last_name="Guest",
        )

    def test_owner_room_bookings_json_returns_room_scoped_non_cancelled_rows(self):
        AccommodationBooking.objects.create(
            guest=self.guest_user,
            accommodation=self.accom,
            room=self.room,
            check_in=date(2026, 5, 10),
            check_out=date(2026, 5, 13),
            num_guests=2,
            status="confirmed",
            total_amount="6800.00",
        )
        AccommodationBooking.objects.create(
            guest=self.guest_user,
            accommodation=self.accom,
            room=self.room,
            check_in=date(2026, 5, 20),
            check_out=date(2026, 5, 22),
            num_guests=1,
            status="cancelled",
            total_amount="3400.00",
        )

        self.client.force_login(self.owner_user)
        response = self.client.get(
            reverse("admin_app:owner_room_bookings_json", kwargs={"room_id": self.room.room_id})
        )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload.get("status"), "success")
        self.assertEqual(payload.get("count"), 1)
        self.assertEqual(payload["guests"][0]["first_name"], "Jade")
        self.assertEqual(payload["guests"][0]["status_raw"], "confirmed")

    def test_owner_room_bookings_json_rejects_room_from_other_owner(self):
        user_model = get_user_model()
        owner_group = Group.objects.get(name="accommodation_owner")
        other_owner = user_model.objects.create_user(
            username="other_owner_room_json_user",
            email="other_owner_room_json@example.com",
            password="secure-pass-222",
            first_name="Other",
            last_name="Owner",
        )
        other_owner.groups.add(owner_group)

        other_accom = Accomodation.objects.create(
            owner=other_owner,
            company_name="Other Stay",
            email_address="other-stay@example.com",
            location="Bayawan",
            company_type="hotel",
            description="Other room scope",
            password="accom-pass-222",
            phone_number="09995550200",
            approval_status="accepted",
            status="accepted",
        )
        other_room = Room.objects.create(
            accommodation=other_accom,
            room_name="Other Room",
            person_limit=2,
            current_availability=2,
            price_per_night="1200.00",
            status="AVAILABLE",
        )

        self.client.force_login(self.owner_user)
        response = self.client.get(
            reverse("admin_app:owner_room_bookings_json", kwargs={"room_id": other_room.room_id})
        )
        self.assertEqual(response.status_code, 404)


class OwnerRoomBookingsCheckInTests(TestCase):
    def setUp(self):
        user_model = get_user_model()
        owner_group, _ = Group.objects.get_or_create(name="accommodation_owner")

        self.owner_user = user_model.objects.create_user(
            username="owner_room_checkin_user",
            email="owner_room_checkin@example.com",
            password="secure-pass-789",
            first_name="Owner",
            last_name="CheckIn",
        )
        self.owner_user.groups.add(owner_group)

        self.accom = Accomodation.objects.create(
            owner=self.owner_user,
            company_name="Checkin Stay",
            email_address="checkin-stay@example.com",
            location="Bayawan",
            company_type="hotel",
            description="Owner room booking check-in test",
            password="accom-pass-123",
            phone_number="09995550333",
            approval_status="accepted",
            status="accepted",
        )
        self.room = Room.objects.create(
            accommodation=self.accom,
            room_name="Business Single",
            person_limit=2,
            current_availability=2,
            price_per_night="1200.00",
            status="AVAILABLE",
        )
        self.guest_user = user_model.objects.create_user(
            username="owner_room_checkin_guest",
            email="owner_room_checkin_guest@example.com",
            password="secure-pass-111",
            first_name="Cyril",
            last_name="Guest",
        )

    def test_owner_room_check_in_accepts_confirmed_booking_for_today(self):
        today = date.today()
        booking = AccommodationBooking.objects.create(
            guest=self.guest_user,
            accommodation=self.accom,
            room=self.room,
            check_in=today,
            check_out=today + timedelta(days=1),
            num_guests=1,
            status="confirmed",
            total_amount="1200.00",
        )

        self.client.force_login(self.owner_user)
        response = self.client.post(
            reverse("admin_app:owner_room_bookings_check_in"),
            data={"room_id": self.room.room_id, "booking_ids": [booking.booking_id]},
        )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload.get("status"), "success")
        self.assertEqual(payload.get("checked_in_count"), 1)
        self.assertIn(booking.booking_id, payload.get("checked_in_ids", []))

    def test_owner_room_check_in_rejects_future_checkin_date(self):
        today = date.today()
        booking = AccommodationBooking.objects.create(
            guest=self.guest_user,
            accommodation=self.accom,
            room=self.room,
            check_in=today + timedelta(days=10),
            check_out=today + timedelta(days=12),
            num_guests=1,
            status="confirmed",
            total_amount="2400.00",
        )

        self.client.force_login(self.owner_user)
        response = self.client.post(
            reverse("admin_app:owner_room_bookings_check_in"),
            data={"room_id": self.room.room_id, "booking_ids": [booking.booking_id]},
        )
        self.assertEqual(response.status_code, 400)
        payload = response.json()
        self.assertEqual(payload.get("status"), "error")
        self.assertEqual(payload.get("checked_in_count"), 0)


class OwnerAccommodationBookingEditTests(TestCase):
    def setUp(self):
        user_model = get_user_model()
        owner_group, _ = Group.objects.get_or_create(name="accommodation_owner")

        self.owner_user = user_model.objects.create_user(
            username="owner_booking_edit_user",
            email="owner_booking_edit@example.com",
            password="secure-pass-123",
            first_name="Owner",
            last_name="Edit",
        )
        self.owner_user.groups.add(owner_group)

        self.accom = Accomodation.objects.create(
            owner=self.owner_user,
            company_name="Edit Stay",
            email_address="edit-stay@example.com",
            location="Bayawan",
            company_type="hotel",
            description="Owner booking edit test",
            password="accom-pass-123",
            phone_number="09995550777",
            approval_status="accepted",
            status="accepted",
        )
        self.room = Room.objects.create(
            accommodation=self.accom,
            room_name="Executive Twin",
            person_limit=5,
            current_availability=5,
            price_per_night="3400.00",
            status="AVAILABLE",
        )
        self.guest_user = user_model.objects.create_user(
            username="owner_booking_edit_guest",
            email="owner_booking_edit_guest@example.com",
            password="secure-pass-111",
            first_name="Jade",
            last_name="Guest",
        )
        self.booking = AccommodationBooking.objects.create(
            guest=self.guest_user,
            accommodation=self.accom,
            room=self.room,
            check_in=date(2026, 5, 10),
            check_out=date(2026, 5, 13),
            num_guests=1,
            status="pending",
            total_amount="3400.00",
        )

    def test_edit_action_updates_details_without_auto_confirm(self):
        self.client.force_login(self.owner_user)
        response = self.client.post(
            reverse("admin_app:owner_accommodation_booking_update", kwargs={"booking_id": self.booking.booking_id}),
            data={
                "action": "edit",
                "check_in": "2026-05-11",
                "check_out": "2026-05-14",
                "num_guests": "2",
            },
        )
        self.assertEqual(response.status_code, 302)
        self.booking.refresh_from_db()
        self.assertEqual(str(self.booking.check_in), "2026-05-11")
        self.assertEqual(str(self.booking.check_out), "2026-05-14")
        self.assertEqual(self.booking.num_guests, 2)
        self.assertEqual(self.booking.status, "pending")


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

    def test_authenticated_guest_is_redirected_to_admin_login_for_owner_flow(self):
        self.client.force_login(self.user)
        response = self.client.get(self.url)
        self.assertEqual(response.status_code, 302)
        self.assertIn(reverse("admin_app:login"), response.url)
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
