# Deployment Readiness Report

- Generated at: 2026-03-24T02:32:39.557616
- Overall: **NOT_READY**
- PASS: 2
- WARN: 5
- FAIL: 1

## Checks

| Check | Status | Current | Target | Notes |
|---|---|---|---|---|
| DEBUG flag | FAIL | DEBUG=True | DEBUG=False | Production must not run with DEBUG=True. |
| SECRET_KEY strength | PASS | custom value set | Non-default secret from environment | Rotate immediately if key was exposed. |
| ALLOWED_HOSTS production host | PASS | unetymological-earnestine-aneurysmally.ngrok-free.dev, 127.0.0.1 | Include deployed domain/IP | Local-only hosts are okay for development only. |
| Secure cookies | WARN | SESSION_COOKIE_SECURE=False, CSRF_COOKIE_SECURE=False | Both True under HTTPS | Enable before internet deployment. |
| CSRF trusted origins | WARN | http://localhost:8000, http://127.0.0.1:8000 | Include deployed https:// domain | Required for secure cross-site POST forms. |
| STATIC_ROOT configured | WARN | (not set) | Set STATIC_ROOT for collectstatic in deployment | Current setup may still work in dev but not ideal for production serving. |
| Database credential handling | WARN | ENGINE=django.db.backends.mysql, HOST=127.0.0.1, NAME=project_db, PASSWORD=set | Credentials from environment variables | Hardcoded DB password in settings should be moved to .env for production. |
| reCAPTCHA secret configured | WARN | missing | Secret key present in environment | Needed if captcha validation is enabled in forms. |

## Next Actions

- Set DEBUG=False in production settings profile.
- Set non-default SECRET_KEY from environment and rotate if exposed.
- Set SESSION_COOKIE_SECURE=True and CSRF_COOKIE_SECURE=True under HTTPS.
- Add deployed HTTPS domain to ALLOWED_HOSTS and CSRF_TRUSTED_ORIGINS.
- Set STATIC_ROOT and run collectstatic during deployment.
- Move DB password to environment variable and remove hardcoded literal.
