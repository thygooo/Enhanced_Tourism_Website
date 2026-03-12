import json
import os
import json
import time
from datetime import timedelta
from decimal import Decimal
from unittest.mock import patch

from django.contrib.auth import get_user_model
from django.test import Client, TestCase, SimpleTestCase, override_settings
from django.utils import timezone

from admin_app.models import Accomodation, Room, TourismInformation
from guest_app.models import AccommodationBooking, Billing
from ai_chatbot.views import (
    _classify_intent_and_extract_params,
    _extract_params_from_message,
    _format_cnn_prediction_for_chat,
    _is_my_accommodation_booking_status_command,
    _resolve_intent_text_cnn_model_path,
    _resolve_accommodation_text_cnn_model_path,
)
from ai_chatbot.recommenders import recommend_accommodations, _resolve_decision_tree_model_path
from ai_chatbot.models import ChatbotLog, RecommendationEvent, UsabilitySurveyResponse
from tour_app.models import Admission_Rates, Tour_Add, Tour_Schedule


class OpenAIChatEndpointTests(TestCase):
    def setUp(self):
        self.client = Client()
        self.url = "/api/chat/"
        user_model = get_user_model()
        self.user = user_model.objects.create_user(
            username="chat_test_user",
            email="chat_test_user@example.com",
            password="secure-pass-123",
            first_name="Chat",
            last_name="Tester",
        )
        self.client.force_login(self.user)

        self.tour = Tour_Add.objects.create(
            tour_id="00001",
            tour_name="River Adventure",
            description="A relaxing river and nature tour.",
        )

        now = timezone.now()
        self.schedule = Tour_Schedule.objects.create(
            tour=self.tour,
            start_time=now + timedelta(days=1),
            end_time=now + timedelta(days=2),
            price=Decimal("500.00"),
            slots_available=20,
            slots_booked=2,
            duration_days=1,
            status="active",
        )

        Admission_Rates.objects.create(
            tour_id=self.tour,
            payables="Environmental Fee",
            price=Decimal("50.00"),
        )

        self.accommodation = Accomodation.objects.create(
            company_name="Bayawan Stay Hotel",
            email_address="bayawan-stay@example.com",
            location="Bayawan",
            company_type="hotel",
            password="demo-password-123",
            phone_number="09999999999",
            approval_status="accepted",
            status="accepted",
        )
        self.room = Room.objects.create(
            accommodation=self.accommodation,
            room_name="Pool View Standard Room",
            person_limit=4,
            current_availability=2,
            price_per_night=Decimal("1200.00"),
            status="AVAILABLE",
        )
        self.other_accommodation = Accomodation.objects.create(
            company_name="Suba Budget Inn",
            email_address="suba-inn@example.com",
            location="Suba",
            company_type="hotel",
            password="demo-password-456",
            phone_number="09999999998",
            approval_status="accepted",
            status="accepted",
        )
        self.other_room = Room.objects.create(
            accommodation=self.other_accommodation,
            room_name="Economy Room",
            person_limit=2,
            current_availability=2,
            price_per_night=Decimal("1100.00"),
            status="AVAILABLE",
        )
        self.inn_accommodation = Accomodation.objects.create(
            company_name="Bayawan Comfort Inn",
            email_address="comfort-inn@example.com",
            location="Bayawan",
            company_type="inn",
            password="demo-password-789",
            phone_number="09999999997",
            approval_status="accepted",
            status="accepted",
        )
        self.inn_room = Room.objects.create(
            accommodation=self.inn_accommodation,
            room_name="Inn Deluxe",
            person_limit=2,
            current_availability=2,
            price_per_night=Decimal("1200.00"),
            status="AVAILABLE",
        )
        TourismInformation.objects.create(
            spot_name="Bayawan Heritage Park",
            description="A cultural landmark in Bayawan City.",
            location="Bayawan City Proper",
            contact_information="09171234567",
            operating_hours="08:00 AM - 05:00 PM",
            publication_status="published",
            is_active=True,
            created_by=self.user,
            updated_by=self.user,
        )
        TourismInformation.objects.create(
            spot_name="Draft Hidden Falls",
            description="Should not be visible to chatbot users.",
            location="Bayawan Hinterlands",
            publication_status="draft",
            is_active=True,
            created_by=self.user,
            updated_by=self.user,
        )

    def test_get_recommendation_intent_returns_tour(self):
        payload = {"message": "recommend a river tour for 2 guests under 700"}

        response = self.client.post(
            self.url,
            data=json.dumps(payload),
            content_type="application/json",
        )

        self.assertEqual(response.status_code, 200)
        text = response.json().get("fulfillmentText", "")
        self.assertIn("Top recommendations for you", text)
        self.assertIn("River Adventure", text)

    def test_calculate_billing_intent_returns_total(self):
        payload = {"message": f"calculate bill for {self.schedule.sched_id} for 2 guests"}

        response = self.client.post(
            self.url,
            data=json.dumps(payload),
            content_type="application/json",
        )

        self.assertEqual(response.status_code, 200)
        text = response.json().get("fulfillmentText", "")
        self.assertIn("Billing Summary", text)
        self.assertIn("Total amount due: PHP 1100.00", text)

    @patch("ai_chatbot.views._classify_intent_and_extract_params")
    @patch.dict(os.environ, {"OPENAI_API_KEY": ""}, clear=False)
    def test_tourism_information_intent_returns_published_records_only(self, mock_classify):
        mock_classify.return_value = {
            "intent": "get_tourism_information",
            "params": {"tourism_query": "heritage"},
            "source": "text_cnn_intent",
            "confidence": 0.91,
            "needs_clarification": False,
            "clarification_question": "",
            "clarification_field": "",
            "intent_classifier": {
                "intent": "get_tourism_information",
                "source": "text_cnn_intent",
                "confidence": 0.91,
                "top_3": [],
                "error": "",
            },
        }

        response = self.client.post(
            self.url,
            data=json.dumps({"message": "tell me about heritage park"}),
            content_type="application/json",
        )

        self.assertEqual(response.status_code, 200)
        text = response.json().get("fulfillmentText", "")
        self.assertIn("Bayawan Heritage Park", text)
        self.assertNotIn("Draft Hidden Falls", text)

    @patch("ai_chatbot.views._openai_extract_intent_and_params", side_effect=AssertionError("deprecated_parser_called"))
    @patch("ai_chatbot.views._classify_intent_and_extract_params")
    @patch.dict(os.environ, {"OPENAI_API_KEY": ""}, clear=False)
    def test_intent_routing_uses_text_cnn_classifier_path_first(self, mock_classify, _mock_legacy):
        mock_classify.return_value = {
            "intent": "calculate_billing",
            "params": {"sched_id": self.schedule.sched_id, "guests": 2},
            "source": "text_cnn_intent",
            "confidence": 0.93,
            "needs_clarification": False,
            "clarification_question": "",
            "clarification_field": "",
            "intent_classifier": {
                "intent": "calculate_billing",
                "source": "text_cnn_intent",
                "confidence": 0.93,
                "top_3": [],
                "error": "",
            },
        }
        response = self.client.post(
            self.url,
            data=json.dumps({"message": "anything"}),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertIn("Billing Summary", body.get("fulfillmentText", ""))
        self.assertEqual(body.get("response_nlg_source"), "openai_nlg_unavailable")
        self.assertEqual(body.get("intent_classifier", {}).get("source"), "text_cnn_intent")

    @patch("ai_chatbot.views._classify_intent_and_extract_params")
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=False)
    def test_openai_is_used_for_final_nlg_only(self, mock_classify):
        mock_classify.return_value = {
            "intent": "calculate_billing",
            "params": {"sched_id": self.schedule.sched_id, "guests": 2},
            "source": "text_cnn_intent",
            "confidence": 0.93,
            "needs_clarification": False,
            "clarification_question": "",
            "clarification_field": "",
            "intent_classifier": {
                "intent": "calculate_billing",
                "source": "text_cnn_intent",
                "confidence": 0.93,
                "top_3": [],
                "error": "",
            },
        }

        with patch("ai_chatbot.views.OpenAI") as mock_openai_cls:
            mock_client = mock_openai_cls.return_value
            mock_client.chat.completions.create.return_value.choices = [
                type("Choice", (), {"message": type("Msg", (), {"content": "Refined billing response."})()})()
            ]
            response = self.client.post(
                self.url,
                data=json.dumps({"message": "calculate bill"}),
                content_type="application/json",
            )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body.get("fulfillmentText"), "Refined billing response.")
        self.assertEqual(body.get("response_nlg_source"), "openai_nlg")
        self.assertTrue(mock_openai_cls.called)

    @patch.dict(os.environ, {"OPENAI_API_KEY": ""}, clear=False)
    def test_message_mode_works_without_openai_key(self):
        payload = {"message": "recommend a river tour for 2 guests under 700"}

        response = self.client.post(
            self.url,
            data=json.dumps(payload),
            content_type="application/json",
        )

        self.assertEqual(response.status_code, 200)
        text = response.json().get("fulfillmentText", "")
        self.assertIn("Top recommendations for you", text)

    def test_chat_requires_authenticated_user(self):
        self.client.logout()
        payload = {"message": "recommend a hotel in bayawan"}

        response = self.client.post(
            self.url,
            data=json.dumps(payload),
            content_type="application/json",
        )

        self.assertEqual(response.status_code, 401)
        body = response.json()
        self.assertEqual(body.get("error_code"), "chat_requires_login")
        self.assertIn("Please log in first", body.get("fulfillmentText", ""))

    @patch.dict(os.environ, {"OPENAI_API_KEY": ""}, clear=False)
    def test_accommodation_slot_filling_flow(self):
        response_1 = self.client.post(
            self.url,
            data=json.dumps({"message": "hotel in bayawan"}),
            content_type="application/json",
        )
        self.assertEqual(response_1.status_code, 200)
        first_body = response_1.json()
        first_text = first_body.get("fulfillmentText", "").lower()
        self.assertIn("top hotel/inn recommendations", first_text)
        self.assertIn("to continue booking", first_text)
        self.assertIn("budget", first_text)
        self.assertIn("guests", first_text)
        self.assertIn("check-in/check-out dates", first_text)
        self.assertIn("recommendation_trace", first_body)
        self.assertIn("recommendation_fallback", first_body)

        response_2 = self.client.post(
            self.url,
            data=json.dumps(
                {"message": "budget 1500, 2 guests, 2026-03-10 to 2026-03-12"}
            ),
            content_type="application/json",
        )
        self.assertEqual(response_2.status_code, 200)
        second_body = response_2.json()
        self.assertIn("top hotel/inn recommendations", second_body.get("fulfillmentText", "").lower())
        self.assertIn("recommendation_trace", second_body)

    @patch.dict(os.environ, {"OPENAI_API_KEY": ""}, clear=False)
    def test_single_digit_dates_are_normalized_in_accommodation_flow(self):
        response = self.client.post(
            self.url,
            data=json.dumps(
                {
                    "message": (
                        "recommend a hotel in bayawan, budget 1500, 2 guests, "
                        "2026-3-29 to 2026-03-31"
                    )
                }
            ),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 200)
        text = response.json().get("fulfillmentText", "").lower()
        self.assertIn("top hotel/inn recommendations", text)
        self.assertNotIn("please provide both check-in and check-out dates", text)

    @patch.dict(os.environ, {"OPENAI_API_KEY": ""}, clear=False)
    def test_location_first_flow_shows_preview_then_asks_budget(self):
        first = self.client.post(
            self.url,
            data=json.dumps({"message": "recommend a hotel for me"}),
            content_type="application/json",
        )
        self.assertEqual(first.status_code, 200)
        first_text = first.json().get("fulfillmentText", "").lower()
        self.assertIn("send all missing details in one reply", first_text)
        self.assertIn("location", first_text)
        self.assertIn("budget", first_text)
        self.assertIn("guests", first_text)

        second = self.client.post(
            self.url,
            data=json.dumps({"message": "bayawan"}),
            content_type="application/json",
        )
        self.assertEqual(second.status_code, 200)
        text = second.json().get("fulfillmentText", "").lower()
        self.assertIn("top hotel/inn recommendations", text)
        self.assertIn("to continue booking", text)
        self.assertIn("budget", text)
        self.assertIn("guests", text)

    @patch.dict(os.environ, {"OPENAI_API_KEY": ""}, clear=False)
    def test_reset_clears_slot_filling_state(self):
        first = self.client.post(
            self.url,
            data=json.dumps({"message": "hotel in bayawan"}),
            content_type="application/json",
        )
        self.assertEqual(first.status_code, 200)
        self.assertIn("budget", first.json().get("fulfillmentText", "").lower())

        session_before_reset = self.client.session
        self.assertIn("ai_chatbot_state", session_before_reset)

        reset = self.client.post(
            self.url,
            data=json.dumps({"message": "reset"}),
            content_type="application/json",
        )
        self.assertEqual(reset.status_code, 200)
        self.assertIn("context cleared", reset.json().get("fulfillmentText", "").lower())

        session_after_reset = self.client.session
        self.assertNotIn("ai_chatbot_state", session_after_reset)

    def test_accommodation_recommendations_rank_by_reason_score(self):
        params = {
            "guests": 2,
            "budget": 1500,
            "company_type": "hotel",
            "amenities": "pool",
        }
        results = recommend_accommodations(params, limit=5)

        self.assertGreaterEqual(len(results), 2)
        top = results[0]
        second = results[1]
        self.assertEqual(top.meta.get("room_id"), self.room.room_id)
        self.assertGreater(top.score, second.score)

    def test_pending_and_declined_accommodations_are_excluded_from_recommendations(self):
        pending_accom = Accomodation.objects.create(
            company_name="Pending Visibility Hotel",
            email_address="pending-visibility@example.com",
            location="Bayawan",
            company_type="hotel",
            password="demo-password-pending",
            phone_number="09999999991",
            approval_status="pending",
            status="pending",
        )
        declined_accom = Accomodation.objects.create(
            company_name="Declined Visibility Hotel",
            email_address="declined-visibility@example.com",
            location="Bayawan",
            company_type="hotel",
            password="demo-password-declined",
            phone_number="09999999992",
            approval_status="declined",
            status="declined",
        )
        pending_room = Room.objects.create(
            accommodation=pending_accom,
            room_name="Pending Room",
            person_limit=3,
            current_availability=2,
            price_per_night=Decimal("800.00"),
            status="AVAILABLE",
        )
        declined_room = Room.objects.create(
            accommodation=declined_accom,
            room_name="Declined Room",
            person_limit=3,
            current_availability=2,
            price_per_night=Decimal("850.00"),
            status="AVAILABLE",
        )

        results = recommend_accommodations(
            {
                "guests": 2,
                "budget": 2000,
                "company_type": "hotel",
                "location": "Bayawan",
            },
            limit=10,
        )
        room_ids = {item.meta.get("room_id") for item in results}
        self.assertNotIn(pending_room.room_id, room_ids)
        self.assertNotIn(declined_room.room_id, room_ids)
        self.assertIn(self.room.room_id, room_ids)

    @patch.dict(os.environ, {"OPENAI_API_KEY": ""}, clear=False)
    def test_chat_payload_includes_ranking_score_and_strength(self):
        payload = {
            "message": (
                "recommend a hotel in bayawan for 2 guests under 1500 "
                "from 2026-03-10 to 2026-03-12 with pool"
            )
        }
        response = self.client.post(
            self.url,
            data=json.dumps(payload),
            content_type="application/json",
        )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertIn("recommendation_trace", body)
        trace = body.get("recommendation_trace") or []
        self.assertGreaterEqual(len(trace), 1)
        first_item = trace[0]
        self.assertIn("ranking_score", first_item)
        self.assertIn("match_strength", first_item)
        self.assertIn("decision_tree_score", first_item)
        self.assertIn("cnn_alignment", first_item)
        self.assertIn("scoring_mode", first_item)
        self.assertIn(first_item.get("match_strength"), ("High", "Medium", "Low"))

    def test_cnn_predicted_type_influences_accommodation_ranking(self):
        base_params = {
            "guests": 2,
            "budget": 2000,
            "location": "bayawan",
        }

        inn_pref = dict(base_params)
        inn_pref["predicted_accommodation_type"] = "inn"
        inn_pref["predicted_accommodation_confidence"] = 0.95
        inn_results = recommend_accommodations(inn_pref, limit=5)
        self.assertGreaterEqual(len(inn_results), 2)
        self.assertEqual(inn_results[0].meta.get("room_id"), self.inn_room.room_id)

        hotel_pref = dict(base_params)
        hotel_pref["predicted_accommodation_type"] = "hotel"
        hotel_pref["predicted_accommodation_confidence"] = 0.95
        hotel_results = recommend_accommodations(hotel_pref, limit=5)
        self.assertGreaterEqual(len(hotel_results), 2)
        self.assertEqual(hotel_results[0].meta.get("room_id"), self.room.room_id)

    @patch.dict(os.environ, {"OPENAI_API_KEY": ""}, clear=False)
    def test_user_with_past_accommodation_bookings_gets_personalized_default_offer(self):
        today = timezone.now().date()
        AccommodationBooking.objects.create(
            guest=self.user,
            accommodation=self.accommodation,
            room=self.room,
            check_in=today - timedelta(days=40),
            check_out=today - timedelta(days=38),
            num_guests=2,
            status="confirmed",
            total_amount=Decimal("2400.00"),
        )

        response = self.client.post(
            self.url,
            data=json.dumps({"message": "recommend a hotel in bayawan"}),
            content_type="application/json",
        )

        self.assertEqual(response.status_code, 200)
        text = response.json().get("fulfillmentText", "").lower()
        self.assertIn("based on your", text)
        self.assertIn("want similar options", text)

    @patch.dict(os.environ, {"OPENAI_API_KEY": ""}, clear=False)
    def test_new_user_without_history_gets_standard_missing_slot_question(self):
        user_model = get_user_model()
        fresh_user = user_model.objects.create_user(
            username="fresh_chat_user",
            email="fresh_chat_user@example.com",
            password="secure-pass-456",
            first_name="Fresh",
            last_name="User",
        )
        self.client.force_login(fresh_user)

        response = self.client.post(
            self.url,
            data=json.dumps({"message": "recommend a hotel in bayawan"}),
            content_type="application/json",
        )

        self.assertEqual(response.status_code, 200)
        text = response.json().get("fulfillmentText", "").lower()
        self.assertIn("budget", text)
        self.assertNotIn("based on your", text)

    @patch.dict(os.environ, {"OPENAI_API_KEY": ""}, clear=False)
    def test_no_response_declines_personalized_defaults(self):
        today = timezone.now().date()
        AccommodationBooking.objects.create(
            guest=self.user,
            accommodation=self.accommodation,
            room=self.room,
            check_in=today - timedelta(days=25),
            check_out=today - timedelta(days=23),
            num_guests=2,
            status="confirmed",
            total_amount=Decimal("2400.00"),
        )

        first = self.client.post(
            self.url,
            data=json.dumps({"message": "recommend a hotel in bayawan"}),
            content_type="application/json",
        )
        self.assertEqual(first.status_code, 200)
        self.assertIn("want similar options", first.json().get("fulfillmentText", "").lower())

        decline = self.client.post(
            self.url,
            data=json.dumps({"message": "no"}),
            content_type="application/json",
        )
        self.assertEqual(decline.status_code, 200)
        decline_text = decline.json().get("fulfillmentText", "").lower()
        self.assertIn("budget", decline_text)
        self.assertNotIn("similar options", decline_text)

    @patch.dict(os.environ, {"OPENAI_API_KEY": ""}, clear=False)
    def test_plain_integer_guest_reply_works_when_waiting_for_guests(self):
        response_1 = self.client.post(
            self.url,
            data=json.dumps({"message": "recommend a hotel in bayawan"}),
            content_type="application/json",
        )
        self.assertEqual(response_1.status_code, 200)
        first_text = response_1.json().get("fulfillmentText", "").lower()
        self.assertIn("budget", first_text)
        self.assertIn("guests", first_text)
        self.assertIn("check-in/check-out", first_text)

        response_2 = self.client.post(
            self.url,
            data=json.dumps({"message": "1500"}),
            content_type="application/json",
        )
        self.assertEqual(response_2.status_code, 200)
        second_text = response_2.json().get("fulfillmentText", "").lower()
        self.assertIn("guests", second_text)
        self.assertIn("check-in/check-out", second_text)

        response_3 = self.client.post(
            self.url,
            data=json.dumps({"message": "2"}),
            content_type="application/json",
        )
        self.assertEqual(response_3.status_code, 200)
        self.assertIn("check-in/check-out", response_3.json().get("fulfillmentText", "").lower())

    @patch.dict(os.environ, {"OPENAI_API_KEY": ""}, clear=False)
    def test_date_only_follow_up_stays_in_accommodation_context(self):
        self.client.post(
            self.url,
            data=json.dumps(
                {"message": "recommend a hotel in bayawan, budget 1500, 2 guests"}
            ),
            content_type="application/json",
        )

        first_dates = self.client.post(
            self.url,
            data=json.dumps({"message": "2026-06-17 to 2026-06-20"}),
            content_type="application/json",
        )
        self.assertEqual(first_dates.status_code, 200)
        first_text = first_dates.json().get("fulfillmentText", "").lower()
        self.assertTrue(
            ("hotel or inn" in first_text) or ("top hotel/inn recommendations" in first_text)
        )

        second_dates = self.client.post(
            self.url,
            data=json.dumps({"message": "2026-03-15 to 2026-03-17"}),
            content_type="application/json",
        )
        self.assertEqual(second_dates.status_code, 200)
        second_text = second_dates.json().get("fulfillmentText", "").lower()
        self.assertTrue(
            ("hotel or inn" in second_text) or ("top hotel/inn recommendations" in second_text)
        )
        self.assertNotIn("matching tour", second_text)

    @patch.dict(os.environ, {"OPENAI_API_KEY": ""}, clear=False)
    def test_no_match_returns_budget_guidance_metadata(self):
        response = self.client.post(
            self.url,
            data=json.dumps(
                {
                    "message": (
                        "recommend a hotel in bayawan for 2 guests under 500 "
                        "from 2026-03-25 to 2026-03-27"
                    )
                }
            ),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 200)
        body = response.json()
        text = body.get("fulfillmentText", "").lower()
        self.assertIn("couldn't find a matching hotel or inn", text)
        self.assertIn("do you want to use", text)
        self.assertIn("no_match_reasons", body)
        self.assertIn("budget_too_low", body.get("no_match_reasons", []))
        self.assertEqual(body.get("suggested_budget_min"), 1200.0)

    @patch.dict(os.environ, {"OPENAI_API_KEY": ""}, clear=False)
    def test_relaxed_location_fallback_returns_recommendations(self):
        response = self.client.post(
            self.url,
            data=json.dumps(
                {
                    "message": (
                        "recommend a hotel in poblacion for 2 guests under 1500 "
                        "from 2026-03-25 to 2026-03-27"
                    )
                }
            ),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 200)
        body = response.json()
        text = body.get("fulfillmentText", "").lower()
        self.assertIn("couldn't find a matching hotel or inn", text)
        self.assertIn("broaden location", text)
        self.assertNotIn("recommendation_trace", body)

    @patch.dict(os.environ, {"OPENAI_API_KEY": ""}, clear=False)
    def test_compact_guest_reply_does_not_override_budget(self):
        self.client.post(
            self.url,
            data=json.dumps(
                {"message": "recommend a hotel in bayawan, budget 1500, 2 guests"}
            ),
            content_type="application/json",
        )

        state = self.client.session.get("ai_chatbot_state", {})
        params = state.get("params", {})
        self.assertEqual(params.get("budget"), 1500)
        self.assertEqual(params.get("guests"), 2)

    @patch.dict(os.environ, {"OPENAI_API_KEY": ""}, clear=False)
    def test_with_pool_refinement_stays_in_accommodation_flow(self):
        self.client.post(
            self.url,
            data=json.dumps(
                {
                    "message": (
                        "recommend a hotel in bayawan, budget 1500, 2 guests, "
                        "2026-03-25 to 2026-03-27"
                    )
                }
            ),
            content_type="application/json",
        )

        refine = self.client.post(
            self.url,
            data=json.dumps({"message": "with pool"}),
            content_type="application/json",
        )
        self.assertEqual(refine.status_code, 200)
        text = refine.json().get("fulfillmentText", "").lower()
        self.assertIn("hotel/inn", text)
        self.assertNotIn("matching tour", text)

    @patch.dict(os.environ, {"OPENAI_API_KEY": ""}, clear=False)
    def test_yes_after_budget_offer_stays_in_accommodation_flow(self):
        initial = self.client.post(
            self.url,
            data=json.dumps(
                {
                    "message": (
                        "recommend a hotel in bayawan for 2 guests under 1000 "
                        "from 2026-03-25 to 2026-03-27"
                    )
                }
            ),
            content_type="application/json",
        )
        self.assertEqual(initial.status_code, 200)
        self.assertIn("do you want to use php", initial.json().get("fulfillmentText", "").lower())

        follow_up = self.client.post(
            self.url,
            data=json.dumps({"message": "yes"}),
            content_type="application/json",
        )
        self.assertEqual(follow_up.status_code, 200)
        text = follow_up.json().get("fulfillmentText", "").lower()
        self.assertIn("top hotel/inn recommendations", text)
        self.assertNotIn("matching tour", text)

    @patch.dict(os.environ, {"OPENAI_API_KEY": ""}, clear=False)
    def test_custom_budget_after_offer_stays_in_accommodation_flow(self):
        initial = self.client.post(
            self.url,
            data=json.dumps(
                {
                    "message": (
                        "recommend a hotel in bayawan for 2 guests under 1000 "
                        "from 2026-03-25 to 2026-03-27"
                    )
                }
            ),
            content_type="application/json",
        )
        self.assertEqual(initial.status_code, 200)
        self.assertIn("do you want to use php", initial.json().get("fulfillmentText", "").lower())

        follow_up = self.client.post(
            self.url,
            data=json.dumps({"message": "1600"}),
            content_type="application/json",
        )
        self.assertEqual(follow_up.status_code, 200)
        text = follow_up.json().get("fulfillmentText", "").lower()
        self.assertIn("top hotel/inn recommendations", text)
        self.assertNotIn("matching tour", text)

    @patch.dict(os.environ, {"OPENAI_API_KEY": ""}, clear=False)
    def test_non_budget_updates_are_preserved_while_waiting_budget_confirmation(self):
        initial = self.client.post(
            self.url,
            data=json.dumps(
                {
                    "message": (
                        "recommend a hotel in bayawan for 2 guests under 1000 "
                        "from 2026-03-25 to 2026-03-27"
                    )
                }
            ),
            content_type="application/json",
        )
        self.assertEqual(initial.status_code, 200)
        self.assertIn("do you want to use php", initial.json().get("fulfillmentText", "").lower())

        update_dates = self.client.post(
            self.url,
            data=json.dumps({"message": "2026-04-01 to 2026-04-03"}),
            content_type="application/json",
        )
        self.assertEqual(update_dates.status_code, 200)
        self.assertIn(
            "i saved your dates/filters",
            update_dates.json().get("fulfillmentText", "").lower(),
        )
        state_before_confirmation = self.client.session.get("ai_chatbot_state", {})
        params_before_confirmation = state_before_confirmation.get("params", {})
        self.assertEqual(params_before_confirmation.get("check_in"), "2026-04-01")
        self.assertEqual(params_before_confirmation.get("check_out"), "2026-04-03")

        accepted = self.client.post(
            self.url,
            data=json.dumps({"message": "yes"}),
            content_type="application/json",
        )
        self.assertEqual(accepted.status_code, 200)
        payload = accepted.json()
        self.assertIn("recommendation_trace", payload)
        trace = payload.get("recommendation_trace") or []
        self.assertGreaterEqual(len(trace), 1)
        state_after_confirmation = self.client.session.get("ai_chatbot_state", {})
        params_after_confirmation = state_after_confirmation.get("params", {})
        self.assertEqual(params_after_confirmation.get("check_in"), "2026-04-01")
        self.assertEqual(params_after_confirmation.get("check_out"), "2026-04-03")

    @patch.dict(os.environ, {"OPENAI_API_KEY": ""}, clear=False)
    def test_booking_requires_confirmation_before_creating_record(self):
        before_count = AccommodationBooking.objects.filter(guest=self.user).count()

        draft = self.client.post(
            self.url,
            data=json.dumps(
                {"message": f"book room {self.room.room_id} for 2 guests from 2026-03-25 to 2026-03-27"}
            ),
            content_type="application/json",
        )
        self.assertEqual(draft.status_code, 200)
        draft_text = draft.json().get("fulfillmentText", "").lower()
        self.assertIn("confirm booking to generate lgu billing reference/link", draft_text)
        self.assertEqual(
            AccommodationBooking.objects.filter(guest=self.user).count(),
            before_count,
        )

        confirmed = self.client.post(
            self.url,
            data=json.dumps({"message": "yes"}),
            content_type="application/json",
        )
        self.assertEqual(confirmed.status_code, 200)
        confirmed_text = confirmed.json().get("fulfillmentText", "").lower()
        self.assertIn("booking receipt / summary", confirmed_text)
        self.assertEqual(
            AccommodationBooking.objects.filter(guest=self.user).count(),
            before_count + 1,
        )
        created_booking = AccommodationBooking.objects.filter(guest=self.user).latest("booking_id")
        self.assertTrue(Billing.objects.filter(booking=created_booking).exists())

    @patch.dict(os.environ, {"OPENAI_API_KEY": ""}, clear=False)
    def test_booking_confirmation_no_cancels_without_creating_record(self):
        before_count = AccommodationBooking.objects.filter(guest=self.user).count()

        draft = self.client.post(
            self.url,
            data=json.dumps(
                {"message": f"book room {self.room.room_id} for 2 guests from 2026-03-25 to 2026-03-27"}
            ),
            content_type="application/json",
        )
        self.assertEqual(draft.status_code, 200)
        self.assertIn(
            "confirm booking to generate lgu billing reference/link",
            draft.json().get("fulfillmentText", "").lower(),
        )

        cancelled = self.client.post(
            self.url,
            data=json.dumps({"message": "no"}),
            content_type="application/json",
        )
        self.assertEqual(cancelled.status_code, 200)
        self.assertIn("cancelled that draft booking", cancelled.json().get("fulfillmentText", "").lower())
        self.assertEqual(
            AccommodationBooking.objects.filter(guest=self.user).count(),
            before_count,
        )

    @patch.dict(os.environ, {"OPENAI_API_KEY": ""}, clear=False)
    def test_booking_confirmation_rejects_overlapping_existing_room_booking(self):
        other_user = get_user_model().objects.create_user(
            username="chat_overlap_user",
            email="chat_overlap_user@example.com",
            password="secure-pass-123",
            first_name="Overlap",
            last_name="Tester",
        )
        AccommodationBooking.objects.create(
            guest=other_user,
            accommodation=self.accommodation,
            room=self.room,
            check_in=timezone.now().date() + timedelta(days=5),
            check_out=timezone.now().date() + timedelta(days=7),
            num_guests=2,
            status="confirmed",
            total_amount=Decimal("2400.00"),
        )

        before_count = AccommodationBooking.objects.filter(guest=self.user).count()
        self.client.post(
            self.url,
            data=json.dumps(
                {
                    "message": (
                        f"book room {self.room.room_id} for 2 guests from "
                        f"{(timezone.now().date() + timedelta(days=5)).isoformat()} to "
                        f"{(timezone.now().date() + timedelta(days=7)).isoformat()}"
                    )
                }
            ),
            content_type="application/json",
        )

        confirm = self.client.post(
            self.url,
            data=json.dumps({"message": "yes"}),
            content_type="application/json",
        )
        self.assertEqual(confirm.status_code, 200)
        text = confirm.json().get("fulfillmentText", "").lower()
        self.assertIn("already booked for the selected dates", text)
        self.assertEqual(
            AccommodationBooking.objects.filter(guest=self.user).count(),
            before_count,
        )

    @patch.dict(os.environ, {"OPENAI_API_KEY": ""}, clear=False)
    def test_booking_confirmation_accept_is_one_time_only(self):
        before_count = AccommodationBooking.objects.filter(guest=self.user).count()

        draft = self.client.post(
            self.url,
            data=json.dumps(
                {"message": f"book room {self.room.room_id} for 2 guests from 2026-03-25 to 2026-03-27"}
            ),
            content_type="application/json",
        )
        self.assertEqual(draft.status_code, 200)
        self.assertIn(
            "confirm booking to generate lgu billing reference/link",
            draft.json().get("fulfillmentText", "").lower(),
        )

        first_yes = self.client.post(
            self.url,
            data=json.dumps({"message": "yes"}),
            content_type="application/json",
        )
        self.assertEqual(first_yes.status_code, 200)
        self.assertIn("booking receipt / summary", first_yes.json().get("fulfillmentText", "").lower())
        self.assertEqual(
            AccommodationBooking.objects.filter(guest=self.user).count(),
            before_count + 1,
        )

        second_yes = self.client.post(
            self.url,
            data=json.dumps({"message": "yes"}),
            content_type="application/json",
        )
        self.assertEqual(second_yes.status_code, 200)
        self.assertEqual(
            AccommodationBooking.objects.filter(guest=self.user).count(),
            before_count + 1,
        )

    @patch.dict(os.environ, {"OPENAI_API_KEY": ""}, clear=False)
    def test_yes_confirmation_returns_lgu_payment_link_and_view_bookings_option(self):
        self.client.post(
            self.url,
            data=json.dumps(
                {"message": f"book room {self.room.room_id} for 2 guests from 2026-03-25 to 2026-03-27"}
            ),
            content_type="application/json",
        )

        confirmed = self.client.post(
            self.url,
            data=json.dumps({"message": "yes"}),
            content_type="application/json",
        )
        self.assertEqual(confirmed.status_code, 200)
        body = confirmed.json()
        self.assertIn("billing_link", body)
        self.assertEqual(body.get("billing_link_label"), "Proceed to LGU Payment")
        self.assertIn("quick_replies", body)
        self.assertIn("view my accommodation bookings", body.get("quick_replies", []))
        self.assertTrue(body.get("show_feedback_prompt"))

    def test_accommodation_booking_notifications_returns_confirmed_or_declined(self):
        today = timezone.now().date()
        AccommodationBooking.objects.create(
            guest=self.user,
            accommodation=self.accommodation,
            room=self.room,
            check_in=today + timedelta(days=2),
            check_out=today + timedelta(days=4),
            num_guests=2,
            status="confirmed",
            total_amount=Decimal("2400.00"),
        )

        response = self.client.get("/api/chat/accommodation-booking-notifications/")
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body.get("status"), "ok")
        bookings = body.get("bookings") or []
        self.assertTrue(any(str(item.get("status")) == "confirmed" for item in bookings))

    @patch.dict(os.environ, {"OPENAI_API_KEY": ""}, clear=False)
    def test_pending_booking_confirmation_expires_after_10_minutes(self):
        before_count = AccommodationBooking.objects.filter(guest=self.user).count()

        draft = self.client.post(
            self.url,
            data=json.dumps(
                {"message": f"book room {self.room.room_id} for 2 guests from 2026-03-25 to 2026-03-27"}
            ),
            content_type="application/json",
        )
        self.assertEqual(draft.status_code, 200)

        session = self.client.session
        state = session.get("ai_chatbot_state", {})
        if "pending_booking" in state and isinstance(state["pending_booking"], dict):
            state["pending_booking"]["created_at"] = int(time.time()) - 601
        session["ai_chatbot_state"] = state
        session.save()

        expired = self.client.post(
            self.url,
            data=json.dumps({"message": "yes"}),
            content_type="application/json",
        )
        self.assertEqual(expired.status_code, 200)
        self.assertIn("expired after 10 minutes", expired.json().get("fulfillmentText", "").lower())
        self.assertEqual(
            AccommodationBooking.objects.filter(guest=self.user).count(),
            before_count,
        )

    @patch.dict(os.environ, {"OPENAI_API_KEY": ""}, clear=False)
    def test_invalid_tampered_room_id_is_handled_safely(self):
        before_count = AccommodationBooking.objects.filter(guest=self.user).count()

        response = self.client.post(
            self.url,
            data=json.dumps({"message": "book room xyz for 2 guests from 2026-03-25 to 2026-03-27"}),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 200)
        text = response.json().get("fulfillmentText", "").lower()
        self.assertIn("valid room id", text)
        self.assertEqual(
            AccommodationBooking.objects.filter(guest=self.user).count(),
            before_count,
        )

    def test_recommendation_click_endpoint_logs_click_event(self):
        response = self.client.post(
            "/api/chat/recommendation-click/",
            data=json.dumps(
                {
                    "room_id": self.room.room_id,
                    "accom_id": self.accommodation.accom_id,
                    "rank": 1,
                    "scoring_mode": "hybrid_textcnn_decisiontree",
                }
            ),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 200)
        self.assertTrue(
            RecommendationEvent.objects.filter(
                user=self.user,
                event_type="click",
                item_ref__icontains=f"room:{self.room.room_id}",
            ).exists()
        )

    def test_usability_feedback_endpoint_persists_sus_response(self):
        response = self.client.post(
            "/api/chat/usability-feedback/",
            data=json.dumps(
                {
                    "statement_code": "CHAT_UX_HELPFULNESS",
                    "likert_score": 5,
                    "comment": "Helpful recommendation card.",
                }
            ),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 200)
        self.assertTrue(
            UsabilitySurveyResponse.objects.filter(
                user=self.user,
                statement_code="CHAT_UX_HELPFULNESS",
                likert_score=5,
            ).exists()
        )

    def test_usability_feedback_endpoint_accepts_full_sus_tam_batch(self):
        responses = []
        for idx in range(1, 11):
            responses.append({"statement_code": f"SUS_Q{idx}", "likert_score": 4})
        for idx in range(1, 5):
            responses.append({"statement_code": f"PU_Q{idx}", "likert_score": 5})
            responses.append({"statement_code": f"PEU_Q{idx}", "likert_score": 4})

        response = self.client.post(
            "/api/chat/usability-feedback/",
            data=json.dumps(
                {
                    "instrument": "sus_tam_full",
                    "responses": responses,
                    "comment": "Batch submission test",
                }
            ),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body.get("saved_count"), 18)
        batch_id = body.get("survey_batch_id")
        self.assertTrue(batch_id)
        self.assertEqual(
            UsabilitySurveyResponse.objects.filter(user=self.user, survey_batch_id=batch_id).count(),
            18,
        )

    def test_usability_feedback_endpoint_rejects_incomplete_full_batch(self):
        responses = [{"statement_code": "SUS_Q1", "likert_score": 4}]
        response = self.client.post(
            "/api/chat/usability-feedback/",
            data=json.dumps(
                {
                    "instrument": "sus_tam_full",
                    "responses": responses,
                }
            ),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("missing_codes", response.json())

    @patch.dict(os.environ, {"OPENAI_API_KEY": ""}, clear=False)
    def test_chatbot_log_created_for_chat_interaction(self):
        message = "recommend a river tour for 2 guests under 700"
        response = self.client.post(
            self.url,
            data=json.dumps({"message": message}),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 200)
        row = ChatbotLog.objects.filter(user=self.user).latest("created_at")
        self.assertEqual(row.user_message, message)
        self.assertTrue(bool(row.bot_response))
        self.assertTrue(bool(row.resolved_intent))
        self.assertEqual(row.response_nlg_source, "openai_nlg_unavailable")
        self.assertIn("intent_source", row.provenance_json)

    @patch.dict(os.environ, {"OPENAI_API_KEY": ""}, clear=False)
    def test_chatbot_log_created_for_reset_command_early_return(self):
        message = "reset"
        response = self.client.post(
            self.url,
            data=json.dumps({"message": message}),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 200)
        row = ChatbotLog.objects.filter(user=self.user).latest("created_at")
        self.assertEqual(row.user_message, message)
        self.assertEqual(row.resolved_intent, "reset_command")
        self.assertIn("context cleared", row.bot_response.lower())

    @patch.dict(os.environ, {"OPENAI_API_KEY": ""}, clear=False)
    def test_chatbot_log_created_for_default_suggestions_shortcut(self):
        message = "show default hotel suggestions"
        response = self.client.post(
            self.url,
            data=json.dumps({"message": message}),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 200)
        row = ChatbotLog.objects.filter(user=self.user).latest("created_at")
        self.assertEqual(row.user_message, message)
        self.assertEqual(row.resolved_intent, "get_accommodation_recommendation_init")
        self.assertIn("suggested stays", row.bot_response.lower())

    @patch.dict(os.environ, {"OPENAI_API_KEY": ""}, clear=False)
    def test_chatbot_log_created_for_view_my_bookings_shortcut(self):
        message = "view my accommodation bookings"
        response = self.client.post(
            self.url,
            data=json.dumps({"message": message}),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 200)
        row = ChatbotLog.objects.filter(user=self.user).latest("created_at")
        self.assertEqual(row.user_message, message)
        self.assertEqual(row.resolved_intent, "view_my_accommodation_bookings")
        self.assertIn("booking status page", row.bot_response.lower())


class ChatbotFallbackParserTests(SimpleTestCase):
    @patch("ai_chatbot.views._classify_intent_with_text_cnn")
    def test_intent_classifier_has_controlled_heuristic_fallback_for_incompatible_labels(self, mock_cnn):
        mock_cnn.return_value = {
            "intent": "",
            "source": "text_cnn_incompatible_label_space",
            "confidence": 0.0,
            "top_3": [],
            "error": "incompatible_label_space",
        }
        parsed = _classify_intent_and_extract_params("recommend a hotel in bayawan")
        self.assertEqual(parsed.get("intent"), "get_accommodation_recommendation")
        self.assertEqual(parsed.get("source"), "heuristic_intent_fallback")
        self.assertEqual(
            parsed.get("intent_classifier", {}).get("source"),
            "text_cnn_incompatible_label_space",
        )

    @patch("ai_chatbot.views._classify_intent_with_text_cnn")
    def test_tourism_information_keywords_route_to_tourism_information_intent(self, mock_cnn):
        mock_cnn.return_value = {
            "intent": "",
            "source": "text_cnn_incompatible_label_space",
            "confidence": 0.0,
            "top_3": [],
            "error": "incompatible_label_space",
        }
        parsed = _classify_intent_and_extract_params(
            "Show tourism information about attractions in Bayawan."
        )
        self.assertEqual(parsed.get("intent"), "get_tourism_information")
        self.assertEqual(parsed.get("source"), "heuristic_intent_fallback")

    def test_budget_extraction_keyword_plain_number(self):
        params = _extract_params_from_message("budget 1500")
        self.assertEqual(params.get("budget"), 1500)

    def test_budget_extraction_with_comma(self):
        params = _extract_params_from_message("budget 1,500 for hotel")
        self.assertEqual(params.get("budget"), 1500)

    def test_budget_extraction_with_k_suffix(self):
        params = _extract_params_from_message("budget 1.5k")
        self.assertEqual(params.get("budget"), 1500)

    def test_budget_extraction_below_keyword(self):
        params = _extract_params_from_message("recommend an inn below 2000")
        self.assertEqual(params.get("budget"), 2000)

    def test_budget_extraction_numeric_only_message(self):
        params = _extract_params_from_message("1500")
        self.assertEqual(params.get("budget"), 1500)

    def test_type_extraction_hotel(self):
        params = _extract_params_from_message("recommend a hotel in bayawan")
        self.assertEqual(params.get("company_type"), "hotel")

    def test_type_extraction_inn(self):
        params = _extract_params_from_message("recommend an inn near terminal")
        self.assertEqual(params.get("company_type"), "inn")

    def test_type_extraction_either_when_both_words_present(self):
        params = _extract_params_from_message("show hotel and inn options")
        self.assertEqual(params.get("company_type"), "either")

    def test_location_extraction_near_terminal(self):
        params = _extract_params_from_message("recommend an inn near terminal for 1 guest")
        self.assertEqual(params.get("location"), "terminal area")

    def test_location_extraction_in_bayawan(self):
        params = _extract_params_from_message("recommend hotel in bayawan for 2 guests")
        self.assertEqual(params.get("location"), "bayawan")

    def test_location_extraction_around_poblacion(self):
        params = _extract_params_from_message("show inns around poblacion under 1200")
        self.assertEqual(params.get("location"), "poblacion")

    def test_date_extraction_with_month_words(self):
        params = _extract_params_from_message(
            "book room 12 for 2 guests from March 27, 2026 to March 29, 2026"
        )
        self.assertEqual(params.get("check_in"), "2026-03-27")
        self.assertEqual(params.get("check_out"), "2026-03-29")

    def test_date_extraction_with_abbrev_month_words(self):
        params = _extract_params_from_message(
            "book room 12 from Mar 7, 2026 to Mar 9, 2026"
        )
        self.assertEqual(params.get("check_in"), "2026-03-07")
        self.assertEqual(params.get("check_out"), "2026-03-09")

    def test_booking_status_command_recognized_show_my_hotel_bookings(self):
        self.assertTrue(
            _is_my_accommodation_booking_status_command("show my hotel bookings")
        )

    def test_booking_status_command_recognized_hotel_booking_status(self):
        self.assertTrue(
            _is_my_accommodation_booking_status_command("hotel booking status")
        )

    def test_booking_status_command_not_recognized_unrelated(self):
        self.assertFalse(
            _is_my_accommodation_booking_status_command("recommend an inn near terminal")
        )

    def test_cnn_output_remaps_hostel_like_labels_to_hotel(self):
        text = _format_cnn_prediction_for_chat(
            {
                "predicted_class": "hostel",
                "confidence": 0.82,
                "top_3": [
                    {"label": "hostel", "confidence": 0.82},
                    {"label": "transient_house", "confidence": 0.12},
                    {"label": "inn", "confidence": 0.06},
                ],
            }
        ).lower()
        self.assertIn("cnn predicted type: hotel", text)
        self.assertNotIn("cnn predicted type: hostel", text)
        self.assertNotIn("- hostel:", text)
        self.assertIn("- hotel:", text)


class ChatbotArtifactResolutionTests(SimpleTestCase):
    @override_settings(DEBUG=False)
    @patch.dict(
        os.environ,
        {
            "CHATBOT_ALLOW_DEMO_ARTIFACT_FALLBACK": "0",
            "CHATBOT_INTENT_CNN_MODEL_PATH": "",
        },
        clear=False,
    )
    def test_intent_text_cnn_prefers_final_path_when_demo_fallback_disabled(self):
        path, source = _resolve_intent_text_cnn_model_path()
        path_str = str(path).replace("\\", "/").lower()
        self.assertIn("/artifacts/text_cnn_intent/text_cnn_intent.keras", path_str)
        self.assertTrue(source.startswith("final"))

    @override_settings(DEBUG=True)
    @patch.dict(
        os.environ,
        {
            "CHATBOT_ALLOW_DEMO_ARTIFACT_FALLBACK": "1",
            "CHATBOT_INTENT_CNN_MODEL_PATH": "",
        },
        clear=False,
    )
    def test_intent_text_cnn_uses_demo_fallback_only_when_enabled(self):
        path, source = _resolve_intent_text_cnn_model_path()
        path_str = str(path).replace("\\", "/").lower()
        self.assertIn("/artifacts/text_cnn_demo/text_cnn_demo.keras", path_str)
        self.assertEqual(source, "demo_fallback")

    @override_settings(DEBUG=False)
    @patch.dict(
        os.environ,
        {
            "CHATBOT_ALLOW_DEMO_ARTIFACT_FALLBACK": "0",
            "CHATBOT_ACCOM_CNN_MODEL_PATH": "",
        },
        clear=False,
    )
    def test_accommodation_text_cnn_prefers_final_path_when_demo_fallback_disabled(self):
        path, source = _resolve_accommodation_text_cnn_model_path()
        path_str = str(path).replace("\\", "/").lower()
        self.assertIn("/artifacts/text_cnn_accommodation/text_cnn_accommodation.keras", path_str)
        self.assertTrue(source.startswith("final"))

    @override_settings(DEBUG=True)
    @patch.dict(
        os.environ,
        {
            "CHATBOT_ALLOW_DEMO_ARTIFACT_FALLBACK": "1",
            "CHATBOT_DECISION_TREE_MODEL_PATH": "",
        },
        clear=False,
    )
    def test_decision_tree_uses_demo_fallback_only_when_enabled(self):
        path, source = _resolve_decision_tree_model_path()
        path_str = str(path).replace("\\", "/").lower()
        self.assertIn("/artifacts/decision_tree_demo/decision_tree_demo.pkl", path_str)
        self.assertEqual(source, "demo_fallback")

    @override_settings(DEBUG=False)
    @patch.dict(
        os.environ,
        {
            "CHATBOT_ALLOW_DEMO_ARTIFACT_FALLBACK": "0",
            "CHATBOT_DECISION_TREE_MODEL_PATH": "",
        },
        clear=False,
    )
    def test_decision_tree_prefers_final_path_when_demo_fallback_disabled(self):
        path, source = _resolve_decision_tree_model_path()
        path_str = str(path).replace("\\", "/").lower()
        self.assertIn("/artifacts/decision_tree_final/decision_tree_final.pkl", path_str)
        self.assertTrue(source.startswith("final"))
