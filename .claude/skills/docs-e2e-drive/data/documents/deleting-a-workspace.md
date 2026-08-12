# Deleting a workspace

Deleting a **workspace** removes one workspace and its contents. Your account, your other
workspaces, and your billing relationship are untouched. To end everything, see
[closing your account](closing-your-account.md) instead.

## Who can do it

Any Owner or Admin of that workspace. Unlike account closure, no second confirmation from
another Owner is needed — a workspace is a smaller thing to lose.

## What happens, and when

| Day | What happens |
|---|---|
| 0 | Workspace hidden from everyone. Seats freed immediately. |
| 0–30 | Any Owner or Admin can restore it from **Settings → Workspaces → Recently deleted**. |
| 30 | Workspace data is deleted from primary storage. Restore is no longer possible. |
| 90 | Data ages out of encrypted backups. |

Seats are freed on day 0, so deleting a workspace lowers your next invoice straight away. This
is the opposite of a plan downgrade, which only takes effect at the end of the period.

## Exports

Start an export before you delete. A deleted workspace cannot start a new export, even during
the 30-day restore window — restore it first, export, then delete again.

## Shared content

Links shared out of the workspace stop working immediately on day 0, not on day 30. Anyone
outside your company holding a share link sees a 404 the moment the workspace is deleted.
