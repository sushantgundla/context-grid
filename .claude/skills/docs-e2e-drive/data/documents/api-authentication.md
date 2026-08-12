# API authentication

Every request to the Northwind Cloud API carries a bearer token in the `Authorization` header.
There is no session, no cookie, and no unauthenticated endpoint apart from `/health`.

```bash
curl https://api.northwind.cloud/v2/projects \
  -H "Authorization: Bearer nw_live_..." \
  -H "Northwind-Version: 2026-01-15"
```

## Token types

| Type | Prefix | Expires | Scope |
|---|---|---|---|
| Personal access token | `nw_pat_` | 90 days | Everything the user can do |
| Service token | `nw_live_` | Never, until revoked | Whatever scopes you grant it |
| Test token | `nw_test_` | Never | Sandbox workspace only, no real data |

Personal access tokens expire after 90 days and cannot be extended. This is deliberate: a token
tied to a human should die when the human stops using it.

## Creating a service token

Service tokens are created under **Settings → Developers → Tokens**. You choose the scopes at
creation time and they cannot be widened afterwards — create a new token instead. The token
value is shown exactly once, at creation. We do not store it in a form we can show you again.

## Rotating a token

Create the new token first, deploy it, then revoke the old one. Revocation takes effect within
five seconds across every region. There is no grace period and no soft-revoke.

## Version pinning

The `Northwind-Version` header pins the API to a dated contract. Omitting it pins you to the
oldest version your account has ever used, which is almost never what you want. Versions are
supported for 24 months after their replacement ships.

## Rate limits

Rate limits are per workspace, not per token, and are returned on every response:

```
Northwind-RateLimit-Limit: 600
Northwind-RateLimit-Remaining: 597
Northwind-RateLimit-Reset: 1736899200
```

Exceeding the limit returns `429` with a `Retry-After` header in seconds. Retrying before that
time does not reset the window, it just costs you another request.
