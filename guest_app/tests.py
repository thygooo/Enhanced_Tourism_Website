import json
from datetime import timedelta
from decimal import Decimal

from django.contrib.auth.models import Group
from django.contrib.auth import get_user_model
from django.test import TestCase
from django.urls import reverse
from django.utils import timezone

from admin_app.models import Accomodation, Room
from guest_app.models import AccommodationBooking, Billing, AccommodationBookingCompanion


class GuestAccommodationApprovalVisibilityTests(TestCase):
    def setUp(self):
        user_model = get_user_model()
        self.user = user_model.objects.create_user(
            username="guest_filter_user",
            email="guest_filter_user@example.com",
            password="secure-pass-123",
            first_name="Guest",
            last_name="Filter",
        )
        self.client.force_login(self.user)
        self.other_user = user_model.objects.create_user(
            username="guest_filter_other_user",
            email="guest_filter_other_user@example.com",
            password="secure-pass-123",
            first_name="Guest",
            last_name="Other",
        )

        self.accepted_accom = Accomodation.objects.create(
            company_name="Accepted Hotel",
            email_address="accepted@example.com",
            location="Bayawan",
            company_type="hotel",
            password="accom-pass-1",
            phone_number="09990000001",
            approval_status="accepted",
            status="accepted",
        )
        self.pending_accom = Accomodation.objects.create(
            company_name="Pending Hotel",
            email_address="pending@example.com",
            location="Bayawan",
            company_type="hotel",
            password="accom-pass-2",
            phone_number="09990000002",
            approval_status="pending",
            status="pending",
        )
        self.declined_accom = Accomodation.objects.create(
            company_name="Declined Inn",
            email_address="declined@example.com",
            location="Bayawan",
            company_type="inn",
            password="accom-pass-3",
            phone_number="09990000003",
            approval_status="declined",
            status="declined",
        )

        self.accepted_room = Room.objects.create(
            accommodation=self.accepted_accom,
            room_name="Accepted Room",
            person_limit=2,
            current_availability=2,
            price_per_night=Decimal("1200.00"),
            status="AVAILABLE",
        )
        self.pending_room = Room.objects.create(
            accommodation=self.pending_accom,
            room_name="Pending Room",
            person_limit=2,
            current_availability=2,
            price_per_night=Decimal("1200.00"),
            status="AVAILABLE",
        )
        self.declined_room = Room.objects.create(
            accommodation=self.declined_accom,
            room_name="Declined Room",
            person_limit=2,
            current_availability=2,
            price_per_night=Decimal("1200.00"),
            status="AVAILABLE",
        )

    def test_pending_and_declined_do_not_appear_on_guest_accommodation_page(self):
        response = self.client.get(reverse("accommodation_page"))
        self.assertEqual(response.status_code, 200)
        content = response.content.decode("utf-8")
        self.assertIn("Accepted Hotel", content)
        self.assertNotIn("Pending Hotel", content)
        self.assertNotIn("Declined Inn", content)

    def test_only_accepted_accommodation_is_bookable(self):
        check_in = timezone.now().date() + timedelta(days=2)
        check_out = check_in + timedelta(days=2)

        pending_response = self.client.post(
            reverse("accommodation_book"),
            data={
                "room_id": self.pending_room.room_id,
                "check_in": check_in.isoformat(),
                "check_out": check_out.isoformat(),
                "num_guests": 1,
            },
        )
        self.assertEqual(pending_response.status_code, 404)

        accepted_response = self.client.post(
            reverse("accommodation_book"),
            data={
                "room_id": self.accepted_room.room_id,
                "check_in": check_in.isoformat(),
                "check_out": check_out.isoformat(),
                "num_guests": 1,
            },
        )
        self.assertEqual(accepted_response.status_code, 200)
        body = accepted_response.json()
        self.assertTrue(body.get("success"))
        self.assertTrue(
            AccommodationBooking.objects.filter(
                guest=self.user,
                room=self.accepted_room,
                accommodation=self.accepted_accom,
            ).exists()
        )
        booking = AccommodationBooking.objects.get(
            guest=self.user,
            room=self.accepted_room,
            accommodation=self.accepted_accom,
        )
        billing = Billing.objects.filter(booking=booking).first()
        self.assertIsNotNone(billing)
        self.assertEqual(billing.booking_reference, f"AB-{booking.booking_id}")
        self.assertEqual(billing.payment_status, "unpaid")
        self.assertEqual(billing.total_amount, booking.total_amount)

    def test_pending_and_declined_rooms_are_not_billable(self):
        pending_response = self.client.post(
            reverse("accommodation_billing"),
            data={
                "room_id": self.pending_room.room_id,
                "nights": 2,
            },
        )
        self.assertEqual(pending_response.status_code, 404)

        accepted_response = self.client.post(
            reverse("accommodation_billing"),
            data={
                "room_id": self.accepted_room.room_id,
                "nights": 2,
            },
        )
        self.assertEqual(accepted_response.status_code, 200)
        self.assertTrue(accepted_response.json().get("success"))

    def test_overlapping_room_booking_is_blocked(self):
        check_in = timezone.now().date() + timedelta(days=5)
        check_out = check_in + timedelta(days=2)
        AccommodationBooking.objects.create(
            guest=self.other_user,
            accommodation=self.accepted_accom,
            room=self.accepted_room,
            check_in=check_in,
            check_out=check_out,
            num_guests=1,
            status="confirmed",
            total_amount=Decimal("2400.00"),
        )

        response = self.client.post(
            reverse("accommodation_book"),
            data={
                "room_id": self.accepted_room.room_id,
                "check_in": (check_in + timedelta(days=1)).isoformat(),
                "check_out": (check_out + timedelta(days=1)).isoformat(),
                "num_guests": 1,
            },
        )
        self.assertEqual(response.status_code, 409)
        body = response.json()
        self.assertFalse(body.get("success"))
        self.assertIn("already booked", str(body.get("message", "")).lower())
        self.assertEqual(
            AccommodationBooking.objects.filter(room=self.accepted_room).count(),
            1,
        )

    def test_booking_with_companions_creates_linked_companion_records(self):
        check_in = timezone.now().date() + timedelta(days=10)
        check_out = check_in + timedelta(days=2)
        payload = [
            {"name": "Juan Dela Cruz", "contact_info": "09171234567"},
            {"name": "Maria Cruz", "contact_info": "maria@example.com"},
        ]

        response = self.client.post(
            reverse("accommodation_book"),
            data={
                "room_id": self.accepted_room.room_id,
                "check_in": check_in.isoformat(),
                "check_out": check_out.isoformat(),
                "num_guests": 2,
                "companions_json": json.dumps(payload),
            },
        )
        self.assertEqual(response.status_code, 200)
        booking = AccommodationBooking.objects.get(
            guest=self.user,
            room=self.accepted_room,
            check_in=check_in,
            check_out=check_out,
        )
        companions = AccommodationBookingCompanion.objects.filter(booking=booking)
        self.assertEqual(companions.count(), 2)
        self.assertTrue(
            companions.filter(
                companion_name="Juan Dela Cruz",
                companion_contact="09171234567",
            ).exists()
        )
        self.assertTrue(
            companions.filter(
                companion_name="Maria Cruz",
                companion_contact="maria@example.com",
            ).exists()
        )

    def test_booking_with_invalid_companion_payload_returns_validation_error(self):
        check_in = timezone.now().date() + timedelta(days=12)
        check_out = check_in + timedelta(days=2)

        response = self.client.post(
            reverse("accommodation_book"),
            data={
                "room_id": self.accepted_room.room_id,
                "check_in": check_in.isoformat(),
                "check_out": check_out.isoformat(),
                "num_guests": 1,
                "companions_json": '{"invalid": "object"}',
            },
        )
        self.assertEqual(response.status_code, 400)
        body = response.json()
        self.assertFalse(body.get("success"))
        self.assertIn("companions", body.get("errors", {}))


class GuestAccommodationRoleEnforcementTests(TestCase):
    def setUp(self):
        user_model = get_user_model()
        self.guest_user = user_model.objects.create_user(
            username="rbac_guest_user",
            email="rbac_guest_user@example.com",
            password="secure-pass-123",
            first_name="Rbac",
            last_name="Guest",
        )
        self.owner_user = user_model.objects.create_user(
            username="rbac_owner_user",
            email="rbac_owner_user@example.com",
            password="secure-pass-123",
            first_name="Rbac",
            last_name="Owner",
        )
        owner_group, _ = Group.objects.get_or_create(name="accommodation_owner")
        self.owner_user.groups.add(owner_group)

        self.accepted_accom = Accomodation.objects.create(
            company_name="RBAC Accepted Hotel",
            email_address="rbac-accepted@example.com",
            location="Bayawan",
            company_type="hotel",
            password="accom-pass-rbac",
            phone_number="09990000111",
            approval_status="accepted",
            status="accepted",
        )
        self.accepted_room = Room.objects.create(
            accommodation=self.accepted_accom,
            room_name="RBAC Room",
            person_limit=2,
            current_availability=2,
            price_per_night=Decimal("1500.00"),
            status="AVAILABLE",
        )

    def test_accommodation_owner_cannot_access_guest_accommodation_page(self):
        self.client.force_login(self.owner_user)
        response = self.client.get(reverse("accommodation_page"))
        self.assertEqual(response.status_code, 403)

    def test_accommodation_owner_cannot_access_guest_booking_history(self):
        self.client.force_login(self.owner_user)
        response = self.client.get(reverse("my_accommodation_bookings"))
        self.assertEqual(response.status_code, 403)

    def test_accommodation_owner_cannot_preview_billing_or_create_booking(self):
        self.client.force_login(self.owner_user)
        check_in = timezone.now().date() + timedelta(days=3)
        check_out = check_in + timedelta(days=2)

        billing_response = self.client.post(
            reverse("accommodation_billing"),
            data={
                "room_id": self.accepted_room.room_id,
                "check_in": check_in.isoformat(),
                "check_out": check_out.isoformat(),
            },
        )
        self.assertEqual(billing_response.status_code, 403)

        before_count = AccommodationBooking.objects.count()
        booking_response = self.client.post(
            reverse("accommodation_book"),
            data={
                "room_id": self.accepted_room.room_id,
                "check_in": check_in.isoformat(),
                "check_out": check_out.isoformat(),
                "num_guests": 1,
            },
        )
        self.assertEqual(booking_response.status_code, 403)
        self.assertEqual(AccommodationBooking.objects.count(), before_count)

    def test_accommodation_owner_cannot_request_guest_recommendations_endpoint(self):
        self.client.force_login(self.owner_user)
        response = self.client.post(
            reverse("accommodation_recommend"),
            data={
                "location": "Bayawan",
                "budget": 2000,
                "guests": 1,
            },
        )
        self.assertEqual(response.status_code, 403)

    def test_accommodation_owner_cannot_cancel_guest_booking(self):
        check_in = timezone.now().date() + timedelta(days=6)
        check_out = check_in + timedelta(days=2)
        booking = AccommodationBooking.objects.create(
            guest=self.guest_user,
            accommodation=self.accepted_accom,
            room=self.accepted_room,
            check_in=check_in,
            check_out=check_out,
            num_guests=1,
            status="pending",
            total_amount=Decimal("3000.00"),
        )

        self.client.force_login(self.owner_user)
        response = self.client.post(
            reverse("cancel_my_accommodation_booking", args=[booking.booking_id]),
            data={},
        )
        self.assertEqual(response.status_code, 403)
        booking.refresh_from_db()
        self.assertEqual(booking.status, "pending")


class AccommodationRoomAvailabilityLifecycleTests(TestCase):
    def setUp(self):
        user_model = get_user_model()
        self.guest_user = user_model.objects.create_user(
            username="lifecycle_guest_user",
            email="lifecycle_guest_user@example.com",
            password="secure-pass-123",
            first_name="Life",
            last_name="Cycle",
        )
        self.owner_user = user_model.objects.create_user(
            username="lifecycle_owner_user",
            email="lifecycle_owner_user@example.com",
            password="secure-pass-123",
            first_name="Life",
            last_name="Owner",
        )
        owner_group, _ = Group.objects.get_or_create(name="accommodation_owner")
        self.owner_user.groups.add(owner_group)

        self.accepted_accom = Accomodation.objects.create(
            company_name="Lifecycle Hotel",
            email_address="lifecycle-hotel@example.com",
            location="Bayawan",
            company_type="hotel",
            password="accom-pass-lifecycle",
            phone_number="09990000222",
            approval_status="accepted",
            status="accepted",
        )
        self.room = Room.objects.create(
            accommodation=self.accepted_accom,
            room_name="Lifecycle Room",
            person_limit=3,
            current_availability=3,
            price_per_night=Decimal("1800.00"),
            status="AVAILABLE",
        )

    def _set_admin_session(self):
        session = self.client.session
        session["user_type"] = "employee"
        session["is_admin"] = True
        session["employee_id"] = 1
        session.save()

    def test_pending_booking_creation_keeps_room_operationally_available(self):
        self.client.force_login(self.guest_user)
        today = timezone.localdate()
        response = self.client.post(
            reverse("accommodation_book"),
            data={
                "room_id": self.room.room_id,
                "check_in": today.isoformat(),
                "check_out": (today + timedelta(days=1)).isoformat(),
                "num_guests": 1,
            },
        )
        self.assertEqual(response.status_code, 200)
        self.room.refresh_from_db()
        self.assertEqual(self.room.status, "AVAILABLE")
        self.assertEqual(self.room.current_availability, self.room.person_limit)

    def test_confirmed_booking_for_today_marks_current_availability_zero(self):
        today = timezone.localdate()
        booking = AccommodationBooking.objects.create(
            guest=self.guest_user,
            accommodation=self.accepted_accom,
            room=self.room,
            check_in=today,
            check_out=today + timedelta(days=1),
            num_guests=1,
            status="pending",
            total_amount=Decimal("1800.00"),
        )

        self.client.force_login(self.guest_user)
        self._set_admin_session()
        response = self.client.post(
            reverse("admin_app:accommodation_booking_update", args=[booking.booking_id]),
            data={"action": "confirm"},
        )
        self.assertEqual(response.status_code, 302)
        booking.refresh_from_db()
        self.room.refresh_from_db()
        self.assertEqual(booking.status, "confirmed")
        self.assertEqual(self.room.status, "AVAILABLE")
        self.assertEqual(self.room.current_availability, 0)

    def test_declined_booking_restores_room_current_availability(self):
        today = timezone.localdate()
        booking = AccommodationBooking.objects.create(
            guest=self.guest_user,
            accommodation=self.accepted_accom,
            room=self.room,
            check_in=today,
            check_out=today + timedelta(days=1),
            num_guests=1,
            status="confirmed",
            total_amount=Decimal("1800.00"),
        )
        self.room.current_availability = 0
        self.room.save(update_fields=["current_availability", "updated_at"])

        self.client.force_login(self.guest_user)
        self._set_admin_session()
        response = self.client.post(
            reverse("admin_app:accommodation_booking_update", args=[booking.booking_id]),
            data={"action": "decline"},
        )
        self.assertEqual(response.status_code, 302)
        booking.refresh_from_db()
        self.room.refresh_from_db()
        self.assertEqual(booking.status, "declined")
        self.assertEqual(self.room.current_availability, self.room.person_limit)

    def test_guest_cancellation_restores_room_current_availability(self):
        today = timezone.localdate()
        booking = AccommodationBooking.objects.create(
            guest=self.guest_user,
            accommodation=self.accepted_accom,
            room=self.room,
            check_in=today,
            check_out=today + timedelta(days=1),
            num_guests=1,
            status="confirmed",
            total_amount=Decimal("1800.00"),
        )
        self.room.current_availability = 0
        self.room.save(update_fields=["current_availability", "updated_at"])

        self.client.force_login(self.guest_user)
        response = self.client.post(
            reverse("cancel_my_accommodation_booking", args=[booking.booking_id]),
            data={"reason": "Change of plans"},
        )
        self.assertEqual(response.status_code, 302)
        booking.refresh_from_db()
        self.room.refresh_from_db()
        self.assertEqual(booking.status, "cancelled")
        self.assertEqual(self.room.current_availability, self.room.person_limit)
