import json
import os
from pathlib import Path

from django.conf import settings
from django.core.management.base import BaseCommand


def _status_row(check, status, current, target, notes):
    return {
        "check": check,
        "status": status,
        "current": current,
        "target": target,
        "notes": notes,
    }


class Command(BaseCommand):
    help = (
        "Export a low-risk deployment readiness report (JSON + Markdown). "
        "Read-only checks only; no runtime flow changes."
    )

    def add_arguments(self, parser):
        parser.add_argument(
            "--out-json",
            type=str,
            default="thesis_data_templates/deployment_readiness_report.json",
        )
        parser.add_argument(
            "--out-md",
            type=str,
            default="thesis_data_templates/deployment_readiness_report.md",
        )

    def handle(self, *args, **options):
        rows = []

        debug_enabled = bool(getattr(settings, "DEBUG", False))
        rows.append(
            _status_row(
                "DEBUG flag",
                "FAIL" if debug_enabled else "PASS",
                f"DEBUG={debug_enabled}",
                "DEBUG=False",
                "Production must not run with DEBUG=True.",
            )
        )

        secret_key = str(getattr(settings, "SECRET_KEY", "") or "")
        default_secret = "django-insecure-your-secret-key-here"
        weak_secret = (not secret_key) or (secret_key == default_secret)
        rows.append(
            _status_row(
                "SECRET_KEY strength",
                "FAIL" if weak_secret else "PASS",
                "default/empty" if weak_secret else "custom value set",
                "Non-default secret from environment",
                "Rotate immediately if key was exposed.",
            )
        )

        allowed_hosts = list(getattr(settings, "ALLOWED_HOSTS", []) or [])
        non_local_hosts = [h for h in allowed_hosts if h not in {"127.0.0.1", "localhost"}]
        rows.append(
            _status_row(
                "ALLOWED_HOSTS production host",
                "PASS" if non_local_hosts else "WARN",
                ", ".join(allowed_hosts) if allowed_hosts else "(empty)",
                "Include deployed domain/IP",
                "Local-only hosts are okay for development only.",
            )
        )

        session_cookie_secure = bool(getattr(settings, "SESSION_COOKIE_SECURE", False))
        csrf_cookie_secure = bool(getattr(settings, "CSRF_COOKIE_SECURE", False))
        rows.append(
            _status_row(
                "Secure cookies",
                "PASS" if (session_cookie_secure and csrf_cookie_secure) else "WARN",
                f"SESSION_COOKIE_SECURE={session_cookie_secure}, CSRF_COOKIE_SECURE={csrf_cookie_secure}",
                "Both True under HTTPS",
                "Enable before internet deployment.",
            )
        )

        trusted_origins = list(getattr(settings, "CSRF_TRUSTED_ORIGINS", []) or [])
        has_https_origin = any(str(origin).startswith("https://") for origin in trusted_origins)
        rows.append(
            _status_row(
                "CSRF trusted origins",
                "PASS" if has_https_origin else "WARN",
                ", ".join(trusted_origins) if trusted_origins else "(empty)",
                "Include deployed https:// domain",
                "Required for secure cross-site POST forms.",
            )
        )

        static_root = getattr(settings, "STATIC_ROOT", None)
        rows.append(
            _status_row(
                "STATIC_ROOT configured",
                "PASS" if static_root else "WARN",
                str(static_root or "(not set)"),
                "Set STATIC_ROOT for collectstatic in deployment",
                "Current setup may still work in dev but not ideal for production serving.",
            )
        )

        db_cfg = settings.DATABASES.get("default", {})
        db_engine = str(db_cfg.get("ENGINE", ""))
        db_host = str(db_cfg.get("HOST", ""))
        db_name = str(db_cfg.get("NAME", ""))
        db_password = str(db_cfg.get("PASSWORD", ""))
        db_password_from_env = bool(str(os.environ.get("DB_PASSWORD", "") or "").strip())
        db_password_hardcoded = bool(db_password) and not db_password_from_env
        rows.append(
            _status_row(
                "Database credential handling",
                "WARN" if db_password_hardcoded else ("PASS" if db_password_from_env else "WARN"),
                f"ENGINE={db_engine}, HOST={db_host}, NAME={db_name}, PASSWORD={'set' if db_password else 'empty'}",
                "Credentials from environment variables",
                (
                    "DB password sourced from environment."
                    if db_password_from_env
                    else "Hardcoded DB password in settings should be moved to .env for production."
                ),
            )
        )

        recaptcha_key = str(getattr(settings, "RECAPTCHA_SECRET_KEY", "") or "")
        rows.append(
            _status_row(
                "reCAPTCHA secret configured",
                "PASS" if recaptcha_key else "WARN",
                "set" if recaptcha_key else "missing",
                "Secret key present in environment",
                "Needed if captcha validation is enabled in forms.",
            )
        )

        report = {
            "generated_at": __import__("datetime").datetime.now().isoformat(),
            "summary": {
                "pass_count": sum(1 for r in rows if r["status"] == "PASS"),
                "warn_count": sum(1 for r in rows if r["status"] == "WARN"),
                "fail_count": sum(1 for r in rows if r["status"] == "FAIL"),
                "overall": "NOT_READY" if any(r["status"] == "FAIL" for r in rows) else "CONDITIONALLY_READY",
            },
            "checks": rows,
            "next_actions": [
                "Set DEBUG=False in production settings profile.",
                "Set non-default SECRET_KEY from environment and rotate if exposed.",
                "Set SESSION_COOKIE_SECURE=True and CSRF_COOKIE_SECURE=True under HTTPS.",
                "Add deployed HTTPS domain to ALLOWED_HOSTS and CSRF_TRUSTED_ORIGINS.",
                "Set STATIC_ROOT and run collectstatic during deployment.",
                "Move DB password to environment variable and remove hardcoded literal.",
            ],
        }

        out_json = Path(str(options.get("out_json") or "thesis_data_templates/deployment_readiness_report.json"))
        out_md = Path(str(options.get("out_md") or "thesis_data_templates/deployment_readiness_report.md"))
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

        md_lines = [
            "# Deployment Readiness Report",
            "",
            f"- Generated at: {report['generated_at']}",
            f"- Overall: **{report['summary']['overall']}**",
            f"- PASS: {report['summary']['pass_count']}",
            f"- WARN: {report['summary']['warn_count']}",
            f"- FAIL: {report['summary']['fail_count']}",
            "",
            "## Checks",
            "",
            "| Check | Status | Current | Target | Notes |",
            "|---|---|---|---|---|",
        ]
        for row in rows:
            md_lines.append(
                f"| {row['check']} | {row['status']} | {row['current']} | {row['target']} | {row['notes']} |"
            )
        md_lines.extend(["", "## Next Actions", ""])
        for action in report["next_actions"]:
            md_lines.append(f"- {action}")
        out_md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

        self.stdout.write(self.style.SUCCESS(f"Saved JSON: {out_json}"))
        self.stdout.write(self.style.SUCCESS(f"Saved Markdown: {out_md}"))
