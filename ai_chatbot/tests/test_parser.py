from django.test import SimpleTestCase

from ai_chatbot.views import (
    _extract_params_from_message,
    _extract_params_with_confidence,
    _is_my_accommodation_booking_status_command,
)


class ChatbotFallbackParserTests(SimpleTestCase):
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

    def test_budget_extraction_below_compact_k_confident(self):
        parsed = _extract_params_with_confidence("below 1.2k")
        self.assertEqual(parsed.get("params", {}).get("budget"), 1200)
        self.assertFalse(parsed.get("needs_clarification"))
        self.assertGreaterEqual(float(parsed.get("confidence", 0)), 0.8)

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

    def test_month_range_without_year_needs_clarification(self):
        parsed = _extract_params_with_confidence("March 10-12")
        self.assertTrue(parsed.get("needs_clarification"))
        self.assertNotIn("check_in", parsed.get("params", {}))
        self.assertNotIn("check_out", parsed.get("params", {}))
        self.assertIn("year", str(parsed.get("clarification_question", "")).lower())

    def test_unknown_near_location_needs_area_clarification(self):
        parsed = _extract_params_with_confidence("near city hall")
        self.assertTrue(parsed.get("needs_clarification"))
        self.assertNotIn("location", parsed.get("params", {}))
        self.assertIn("barangay", str(parsed.get("clarification_question", "")).lower())

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
