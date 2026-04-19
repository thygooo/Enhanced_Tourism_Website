import json

from django.core.management.base import BaseCommand

from ai_chatbot.recommenders import get_decision_tree_runtime_status


class Command(BaseCommand):
    help = "Show runtime Decision Tree artifact status for defense/demo verification."

    def add_arguments(self, parser):
        parser.add_argument(
            "--pretty",
            action="store_true",
            help="Pretty-print JSON output.",
        )

    def handle(self, *args, **options):
        payload = {
            "status": "ok",
            "decision_tree_runtime": get_decision_tree_runtime_status(force_reload=True),
        }
        if options.get("pretty"):
            self.stdout.write(json.dumps(payload, indent=2))
        else:
            self.stdout.write(json.dumps(payload))
