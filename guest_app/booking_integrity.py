from django.db import transaction
from django.utils import timezone

from admin_app.models import Room as AdminRoom

from .models import AccommodationBooking, Billing, AccommodationBookingCompanion


ACTIVE_BOOKING_STATUSES = ("pending", "confirmed")


def has_room_booking_overlap(*, room, check_in, check_out):
    """
    Detect whether a room already has an active booking that overlaps
    the requested [check_in, check_out) range.
    """
    return AccommodationBooking.objects.filter(
        room=room,
        status__in=ACTIVE_BOOKING_STATUSES,
        check_in__lt=check_out,
        check_out__gt=check_in,
    ).exists()


def sync_room_current_availability(room):
    """
    Compatibility-safe room availability sync.

    Meaning:
    - Room.status remains an operational/admin flag (AVAILABLE/OCCUPIED/UNAVAILABLE).
    - Date-based reservability is enforced by overlap checks on bookings.
    - current_availability is treated as today's occupancy snapshot for AVAILABLE rooms.
    """
    if room is None:
        return

    locked_room = AdminRoom.objects.select_for_update().filter(room_id=room.room_id).first()
    if locked_room is None:
        return

    if str(locked_room.status or "").upper() != "AVAILABLE":
        return

    today = timezone.localdate()
    has_confirmed_stay_today = AccommodationBooking.objects.filter(
        room=locked_room,
        status="confirmed",
        check_in__lte=today,
        check_out__gt=today,
    ).exists()

    target_availability = 0 if has_confirmed_stay_today else max(int(locked_room.person_limit or 0), 0)
    if locked_room.current_availability != target_availability:
        locked_room.current_availability = target_availability
        locked_room.save(update_fields=["current_availability", "updated_at"])


def create_accommodation_booking_with_integrity(
    *,
    guest,
    room,
    check_in,
    check_out,
    num_guests,
    total_amount,
    status="pending",
    companions=None,
):
    """
    Create an accommodation booking with overlap protection.

    Returns:
      (booking, None) on success
      (None, "room_unavailable") if room is not accepted/available at commit time
      (None, "date_overlap") if date range overlaps an existing active booking
    """
    with transaction.atomic():
        # Lock room row first to serialize booking attempts for the same room.
        locked_room = (
            AdminRoom.objects.select_related("accommodation")
            .select_for_update()
            .filter(
                room_id=room.room_id,
                status="AVAILABLE",
                accommodation__approval_status="accepted",
            )
            .first()
        )
        if locked_room is None:
            return None, "room_unavailable"

        overlap_exists = (
            AccommodationBooking.objects.select_for_update()
            .filter(
                room=locked_room,
                status__in=ACTIVE_BOOKING_STATUSES,
                check_in__lt=check_out,
                check_out__gt=check_in,
            )
            .exists()
        )
        if overlap_exists:
            return None, "date_overlap"

        booking = AccommodationBooking.objects.create(
            guest=guest,
            accommodation=locked_room.accommodation,
            room=locked_room,
            check_in=check_in,
            check_out=check_out,
            num_guests=num_guests,
            status=status,
            total_amount=total_amount,
        )
        Billing.objects.get_or_create(
            booking=booking,
            defaults={
                "booking_reference": f"AB-{booking.booking_id}",
                "total_amount": booking.total_amount,
                "payment_status": booking.payment_status,
                "amount_paid": booking.amount_paid,
            },
        )
        for companion in companions or []:
            AccommodationBookingCompanion.objects.create(
                booking=booking,
                companion_name=str(companion.get("name") or "").strip()[:120],
                companion_contact=str(companion.get("contact_info") or "").strip()[:150],
            )
        sync_room_current_availability(locked_room)
        return booking, None
