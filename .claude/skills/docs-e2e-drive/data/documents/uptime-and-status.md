# Uptime, status and incidents

Our status page is at `status.northwind.cloud`. It is hosted outside our own infrastructure on
purpose, so it stays up when we do not.

## The commitment

| Plan | Monthly uptime target | Credit if missed |
|---|---:|---|
| Starter | 99.5% | None |
| Growth | 99.9% | 10% of the monthly fee |
| Enterprise | 99.95% | 25% of the monthly fee, contractual |

Uptime is measured as successful responses to the `/health` endpoint from three independent
regions, sampled every 30 seconds. Scheduled maintenance announced at least 72 hours in
advance does not count against the target.

## Claiming a credit

Service credits are not automatic. Email support within 30 days of the incident with your
workspace ID and the dates affected. Credits are applied to the next invoice and cannot be
paid out in cash.

## Incident severity

- **Sev 1** — the product is unusable for most customers. Status page updated within 15
  minutes, then every 30 minutes until resolved.
- **Sev 2** — a major feature is broken or one region is degraded. Updated within an hour.
- **Sev 3** — a minor feature is degraded, with a workaround. Noted on the status page, no
  running commentary.

Post-incident reviews are published for every Sev 1 within five business days, on the status
page, in public.

## Maintenance windows

Routine maintenance runs Sundays 02:00–06:00 UTC. It is designed to be invisible; if a window
will cause downtime we announce it 72 hours ahead and say how long.
