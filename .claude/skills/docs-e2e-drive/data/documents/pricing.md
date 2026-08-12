# Plans and pricing

Northwind Cloud has three plans. Every plan includes unlimited read-only viewers at no extra
cost, because charging for people who only look at data is a bad way to run a company.

| Plan | Price per seat / month | Included storage | API rate limit | Support response |
|---|---:|---:|---:|---|
| Starter | $12 | 10 GB | 60 requests/minute | 2 business days |
| Growth | $29 | 250 GB | 600 requests/minute | 8 business hours |
| Enterprise | Custom | 5 TB | 6,000 requests/minute | 1 hour, 24/7 |

## What counts as a seat

A seat is any user who can write — create, edit or delete anything in the workspace. Viewers
are free and unlimited on every plan. Service accounts and API tokens do not consume a seat.

## Annual billing

Paying annually is a 20% discount against the monthly price, taken as two free months. The
discount applies to seats only, not to storage overage or support add-ons.

## Storage overage

Storage beyond the included allowance is billed at **$0.12 per GB per month**, measured as the
average of hourly samples across the billing period rather than the peak. A workspace that
spikes to 400 GB for one afternoon and sits at 200 GB the rest of the month is billed close to
200 GB.

## Changing plan

Upgrades take effect immediately and are prorated to the day. Downgrades take effect at the
end of the current billing period, so you keep what you paid for. There is no fee to change
plan in either direction.

## Free trial

Every new workspace gets a 14-day trial of the Growth plan, no card required. When the trial
ends, the workspace drops to Starter limits rather than locking you out. If your data exceeds
the Starter storage allowance, it stays readable and you are asked to upgrade or delete before
new writes are accepted.
