from django.core.management.base import BaseCommand
from django.db import transaction

from admin_app.models import Accomodation, Room


DEMO_ACCOMMODATIONS = [
    {
        "company_name": "Demo Bayawan Inn",
        "email_address": "demo_bayawan_inn@example.com",
        "location": "Bayawan City Proper",
        "company_type": "inn",
        "password": "demo12345",
        "phone_number": "09170000001",
        "status": "Approved",
        "rooms": [
            {"room_name": "Standard Twin", "person_limit": 2, "price_per_night": "1800.00", "current_availability": 2, "status": "AVAILABLE"},
            {"room_name": "Family Room", "person_limit": 4, "price_per_night": "3200.00", "current_availability": 1, "status": "AVAILABLE"},
        ],
    },
    {
        "company_name": "Demo Poblacion Hotel",
        "email_address": "demo_poblacion_hotel@example.com",
        "location": "Poblacion, Bayawan City",
        "company_type": "hotel",
        "password": "demo12345",
        "phone_number": "09170000002",
        "status": "Approved",
        "rooms": [
            {"room_name": "Deluxe Queen", "person_limit": 2, "price_per_night": "2400.00", "current_availability": 3, "status": "AVAILABLE"},
            {"room_name": "Executive Suite", "person_limit": 4, "price_per_night": "4100.00", "current_availability": 1, "status": "AVAILABLE"},
        ],
    },
    {
        "company_name": "Demo Terminal Travelers Inn",
        "email_address": "demo_terminal_inn@example.com",
        "location": "Terminal Area, Bayawan City",
        "company_type": "inn",
        "password": "demo12345",
        "phone_number": "09170000003",
        "status": "Approved",
        "rooms": [
            {"room_name": "Solo Room", "person_limit": 1, "price_per_night": "900.00", "current_availability": 1, "status": "AVAILABLE"},
            {"room_name": "Budget Double", "person_limit": 2, "price_per_night": "1200.00", "current_availability": 2, "status": "AVAILABLE"},
        ],
    },
    {
        "company_name": "Demo Suba Lodge Hotel",
        "email_address": "demo_suba_lodge@example.com",
        "location": "Suba, Bayawan City",
        "company_type": "hotel",
        "password": "demo12345",
        "phone_number": "09170000004",
        "status": "Approved",
        "rooms": [
            {"room_name": "Garden Room", "person_limit": 3, "price_per_night": "1600.00", "current_availability": 2, "status": "AVAILABLE"},
            {"room_name": "Group Room", "person_limit": 6, "price_per_night": "4500.00", "current_availability": 1, "status": "AVAILABLE"},
        ],
    },
    {
        "company_name": "Demo Banga Seaside Inn",
        "email_address": "demo_banga_seaside_inn@example.com",
        "location": "Banga, Bayawan City",
        "company_type": "inn",
        "password": "demo12345",
        "phone_number": "09170000005",
        "status": "Approved",
        "rooms": [
            {"room_name": "Solo Room (WiFi)", "person_limit": 1, "price_per_night": "850.00", "current_availability": 2, "status": "AVAILABLE"},
            {"room_name": "Standard Double (Aircon)", "person_limit": 2, "price_per_night": "1300.00", "current_availability": 3, "status": "AVAILABLE"},
            {"room_name": "Family Room (Beach View)", "person_limit": 4, "price_per_night": "2500.00", "current_availability": 2, "status": "AVAILABLE"},
        ],
    },
    {
        "company_name": "Demo Villareal Garden Hotel",
        "email_address": "demo_villareal_garden_hotel@example.com",
        "location": "Villareal, Bayawan City",
        "company_type": "hotel",
        "password": "demo12345",
        "phone_number": "09170000006",
        "status": "Approved",
        "rooms": [
            {"room_name": "Economy Twin (WiFi)", "person_limit": 2, "price_per_night": "1200.00", "current_availability": 3, "status": "AVAILABLE"},
            {"room_name": "Deluxe Queen (Aircon)", "person_limit": 2, "price_per_night": "1950.00", "current_availability": 2, "status": "AVAILABLE"},
            {"room_name": "Family Suite (Kitchenette)", "person_limit": 5, "price_per_night": "2900.00", "current_availability": 1, "status": "AVAILABLE"},
        ],
    },
    {
        "company_name": "Demo Poblacion Plaza Stay",
        "email_address": "demo_poblacion_plaza_stay@example.com",
        "location": "Poblacion, Bayawan City",
        "company_type": "inn",
        "password": "demo12345",
        "phone_number": "09170000007",
        "status": "Approved",
        "rooms": [
            {"room_name": "Budget Single (Fan Room)", "person_limit": 1, "price_per_night": "800.00", "current_availability": 2, "status": "AVAILABLE"},
            {"room_name": "Standard Triple (WiFi)", "person_limit": 3, "price_per_night": "1650.00", "current_availability": 2, "status": "AVAILABLE"},
            {"room_name": "Family Quad (Aircon)", "person_limit": 4, "price_per_night": "2300.00", "current_availability": 2, "status": "AVAILABLE"},
        ],
    },
    {
        "company_name": "Demo Baywalk Breeze Resort",
        "email_address": "demo_baywalk_breeze_resort@example.com",
        "location": "Baywalk Area, Bayawan City",
        "company_type": "resort",
        "password": "demo12345",
        "phone_number": "09170000008",
        "status": "Approved",
        "rooms": [
            {"room_name": "Garden Cabin (WiFi)", "person_limit": 2, "price_per_night": "1700.00", "current_availability": 2, "status": "AVAILABLE"},
            {"room_name": "Seaview Double (Aircon)", "person_limit": 2, "price_per_night": "2200.00", "current_availability": 2, "status": "AVAILABLE"},
            {"room_name": "Family Loft (Beach View)", "person_limit": 6, "price_per_night": "3000.00", "current_availability": 1, "status": "AVAILABLE"},
        ],
    },
    {
        "company_name": "Demo Nangka Travelers Lodge",
        "email_address": "demo_nangka_travelers_lodge@example.com",
        "location": "Nangka, Bayawan City",
        "company_type": "lodge",
        "password": "demo12345",
        "phone_number": "09170000009",
        "status": "Approved",
        "rooms": [
            {"room_name": "Backpacker Single (WiFi)", "person_limit": 1, "price_per_night": "900.00", "current_availability": 3, "status": "AVAILABLE"},
            {"room_name": "Twin Shared (Aircon)", "person_limit": 2, "price_per_night": "1250.00", "current_availability": 3, "status": "AVAILABLE"},
            {"room_name": "Barkada Room (Family Type)", "person_limit": 6, "price_per_night": "2750.00", "current_availability": 1, "status": "AVAILABLE"},
        ],
    },
    {
        "company_name": "Demo Malabugas Highland Inn",
        "email_address": "demo_malabugas_highland_inn@example.com",
        "location": "Malabugas, Bayawan City",
        "company_type": "inn",
        "password": "demo12345",
        "phone_number": "09170000010",
        "status": "Approved",
        "rooms": [
            {"room_name": "Mountain Single (Fan)", "person_limit": 1, "price_per_night": "850.00", "current_availability": 2, "status": "AVAILABLE"},
            {"room_name": "Deluxe Twin (Aircon)", "person_limit": 2, "price_per_night": "1550.00", "current_availability": 2, "status": "AVAILABLE"},
            {"room_name": "Panorama Family Room", "person_limit": 5, "price_per_night": "2600.00", "current_availability": 1, "status": "AVAILABLE"},
        ],
    },
    {
        "company_name": "Demo San Roque Family Suites",
        "email_address": "demo_san_roque_family_suites@example.com",
        "location": "San Roque, Bayawan City",
        "company_type": "hotel",
        "password": "demo12345",
        "phone_number": "09170000011",
        "status": "Approved",
        "rooms": [
            {"room_name": "Couple Suite (WiFi)", "person_limit": 2, "price_per_night": "1850.00", "current_availability": 2, "status": "AVAILABLE"},
            {"room_name": "Family Room (Aircon)", "person_limit": 4, "price_per_night": "2450.00", "current_availability": 2, "status": "AVAILABLE"},
            {"room_name": "Connecting Rooms (Large Family)", "person_limit": 6, "price_per_night": "2950.00", "current_availability": 1, "status": "AVAILABLE"},
        ],
    },
    {
        "company_name": "Demo Ubos Marina Hotel",
        "email_address": "demo_ubos_marina_hotel@example.com",
        "location": "Ubos, Bayawan City",
        "company_type": "hotel",
        "password": "demo12345",
        "phone_number": "09170000012",
        "status": "Approved",
        "rooms": [
            {"room_name": "Marina Twin (WiFi)", "person_limit": 2, "price_per_night": "1600.00", "current_availability": 2, "status": "AVAILABLE"},
            {"room_name": "Premium Queen (Aircon)", "person_limit": 2, "price_per_night": "2100.00", "current_availability": 2, "status": "AVAILABLE"},
            {"room_name": "Harbor Family Room (View)", "person_limit": 4, "price_per_night": "2800.00", "current_availability": 1, "status": "AVAILABLE"},
        ],
    },
    {
        "company_name": "Demo Kalumboyan Riverside Inn",
        "email_address": "demo_kalumboyan_riverside_inn@example.com",
        "location": "Kalumboyan, Bayawan City",
        "company_type": "inn",
        "password": "demo12345",
        "phone_number": "09170000013",
        "status": "Approved",
        "rooms": [
            {"room_name": "Riverside Single (Fan)", "person_limit": 1, "price_per_night": "900.00", "current_availability": 2, "status": "AVAILABLE"},
            {"room_name": "Riverside Double (WiFi)", "person_limit": 2, "price_per_night": "1400.00", "current_availability": 3, "status": "AVAILABLE"},
            {"room_name": "Family Room (Aircon)", "person_limit": 4, "price_per_night": "2250.00", "current_availability": 2, "status": "AVAILABLE"},
        ],
    },
    {
        "company_name": "Demo Bawis Eco Stay",
        "email_address": "demo_bawis_eco_stay@example.com",
        "location": "Bawis, Bayawan City",
        "company_type": "lodge",
        "password": "demo12345",
        "phone_number": "09170000014",
        "status": "Approved",
        "rooms": [
            {"room_name": "Eco Pod Single", "person_limit": 1, "price_per_night": "820.00", "current_availability": 2, "status": "AVAILABLE"},
            {"room_name": "Eco Twin (WiFi)", "person_limit": 2, "price_per_night": "1350.00", "current_availability": 2, "status": "AVAILABLE"},
            {"room_name": "Eco Family Nook", "person_limit": 4, "price_per_night": "2150.00", "current_availability": 2, "status": "AVAILABLE"},
        ],
    },
    {
        "company_name": "Demo Bayawan Boulevard Inn",
        "email_address": "demo_bayawan_boulevard_inn@example.com",
        "location": "Bayawan Boulevard, Bayawan City",
        "company_type": "inn",
        "password": "demo12345",
        "phone_number": "09170000015",
        "status": "Approved",
        "rooms": [
            {"room_name": "Transit Single", "person_limit": 1, "price_per_night": "880.00", "current_availability": 3, "status": "AVAILABLE"},
            {"room_name": "Transit Double (WiFi)", "person_limit": 2, "price_per_night": "1280.00", "current_availability": 3, "status": "AVAILABLE"},
            {"room_name": "Family Stopover Room", "person_limit": 5, "price_per_night": "2400.00", "current_availability": 1, "status": "AVAILABLE"},
        ],
    },
    {
        "company_name": "Demo Bayawan Central Suites",
        "email_address": "demo_bayawan_central_suites@example.com",
        "location": "City Proper, Bayawan City",
        "company_type": "hotel",
        "password": "demo12345",
        "phone_number": "09170000016",
        "status": "Approved",
        "rooms": [
            {"room_name": "Business Single (WiFi)", "person_limit": 1, "price_per_night": "1100.00", "current_availability": 2, "status": "AVAILABLE"},
            {"room_name": "Executive Twin (Aircon)", "person_limit": 2, "price_per_night": "1900.00", "current_availability": 2, "status": "AVAILABLE"},
            {"room_name": "Premium Family Suite", "person_limit": 6, "price_per_night": "3000.00", "current_availability": 1, "status": "AVAILABLE"},
            {"room_name": "Balcony Double (City View)", "person_limit": 2, "price_per_night": "2050.00", "current_availability": 2, "status": "AVAILABLE"},
        ],
    },
]


class Command(BaseCommand):
    help = "Seed demo accommodations and rooms for local chatbot recommendation testing."

    def add_arguments(self, parser):
        parser.add_argument(
            "--purge",
            action="store_true",
            help="Delete previously seeded demo accommodations/rooms (matched by demo email prefix).",
        )
        parser.add_argument(
            "--show-only",
            action="store_true",
            help="Print what would be created/updated without writing changes.",
        )

    @transaction.atomic
    def handle(self, *args, **options):
        purge = bool(options.get("purge"))
        show_only = bool(options.get("show_only"))

        if purge:
            return self._purge(show_only=show_only)
        return self._seed(show_only=show_only)

    def _demo_queryset(self):
        return Accomodation.objects.filter(email_address__startswith="demo_")

    def _purge(self, *, show_only=False):
        qs = self._demo_queryset()
        accom_count = qs.count()
        room_count = Room.objects.filter(accommodation__in=qs).count()

        if show_only:
            self.stdout.write(self.style.WARNING(f"[SHOW ONLY] Would delete {room_count} demo room(s) and {accom_count} demo accommodation(s)."))
            return

        qs.delete()  # cascades to Room via FK
        self.stdout.write(self.style.SUCCESS(f"Deleted {room_count} demo room(s) and {accom_count} demo accommodation(s)."))

    def _seed(self, *, show_only=False):
        created_accom = 0
        updated_accom = 0
        created_rooms = 0
        updated_rooms = 0

        # Compatibility cleanup for legacy "Pamplona" demo records so only the
        # Bayawan-specific replacement remains in the dataset.
        legacy_email_map = {
            "demo_pamplona_road_inn@example.com": "demo_bayawan_boulevard_inn@example.com",
        }
        if not show_only:
            for old_email, new_email in legacy_email_map.items():
                old_qs = Accomodation.objects.filter(email_address=old_email)
                if not old_qs.exists():
                    continue

                if Accomodation.objects.filter(email_address=new_email).exists():
                    # If replacement already exists, drop legacy rows.
                    old_qs.delete()
                else:
                    # Otherwise migrate one legacy row to the replacement email
                    # then remove any accidental duplicates.
                    old_obj = old_qs.first()
                    old_obj.email_address = new_email
                    old_obj.save(update_fields=["email_address"])
                    Accomodation.objects.filter(email_address=old_email).delete()

        for accom_def in DEMO_ACCOMMODATIONS:
            room_defs = accom_def["rooms"]
            accom_fields = {k: v for k, v in accom_def.items() if k != "rooms"}

            if show_only:
                exists = Accomodation.objects.filter(email_address=accom_fields["email_address"]).exists()
                action = "UPDATE" if exists else "CREATE"
                self.stdout.write(f"[SHOW ONLY] {action} accommodation: {accom_fields['company_name']} ({accom_fields['email_address']})")
                for room_def in room_defs:
                    self.stdout.write(f"  - room: {room_def['room_name']} | PHP {room_def['price_per_night']} | pax {room_def['person_limit']}")
                continue

            accom_obj, accom_created = Accomodation.objects.update_or_create(
                email_address=accom_fields["email_address"],
                defaults=accom_fields,
            )
            if accom_created:
                created_accom += 1
            else:
                updated_accom += 1

            for room_def in room_defs:
                room_obj, room_created = Room.objects.update_or_create(
                    accommodation=accom_obj,
                    room_name=room_def["room_name"],
                    defaults=room_def,
                )
                if room_created:
                    created_rooms += 1
                else:
                    updated_rooms += 1

                self.stdout.write(
                    f"{'Created' if room_created else 'Updated'} room #{room_obj.room_id}: "
                    f"{accom_obj.company_name} - {room_obj.room_name} "
                    f"(PHP {room_obj.price_per_night}, pax {room_obj.person_limit}, {room_obj.status})"
                )

        self.stdout.write(self.style.SUCCESS(
            f"Demo seed complete | accommodations: +{created_accom} created / {updated_accom} updated | "
            f"rooms: +{created_rooms} created / {updated_rooms} updated"
        ))
        self.stdout.write("These demo records are local-only and safe to purge later with `--purge`.")
