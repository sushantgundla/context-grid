# Closing your Northwind account

Closing an **account** ends your relationship with Northwind Cloud entirely: every workspace
you own, your billing relationship, and your login. This is different from deleting a single
workspace — see [deleting a workspace](deleting-a-workspace.md) for that.

## Who can do it

Only the account Owner. If there is more than one Owner, all of them must confirm within 72
hours of the first request, otherwise the request lapses and nothing happens.

## What happens, and when

| Day | What happens |
|---|---|
| 0 | Account marked for closure. All workspaces become read-only. Exports still work. |
| 0–30 | Any Owner can cancel the closure and everything returns to normal. |
| 30 | Data is deleted from primary storage. Cancellation is no longer possible. |
| 90 | Data ages out of encrypted backups. Nothing recoverable remains. |

Export your data before day 30. After that we cannot retrieve it for you, at any price — the
backups exist for disaster recovery, not for restoring individual deleted accounts.

## Your final invoice

Closing does not cancel money already owed. Any usage in the current period is invoiced on the
normal billing date. Annual plans are not refunded pro rata; you keep access, read-only, until
the term you paid for ends.

## What we keep

Invoices, for seven years, because tax law requires it. Nothing else — not your documents, not
your logs, not your tokens.
