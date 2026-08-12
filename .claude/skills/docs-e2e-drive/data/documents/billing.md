# Billing and invoices

Northwind Cloud bills monthly, in arrears, on the calendar day you first subscribed. If that
day does not exist in a given month — the 31st in February, for example — we bill on the last
day of that month instead.

## When the invoice arrives

Invoices are emailed to the account's billing contact within 24 hours of the charge being
attempted. You can also download every past invoice from **Settings → Billing → Invoice
history**, going back to the day the workspace was created. Invoices are kept for seven years
and are never deleted, even after a workspace is closed.

## Payment methods

We accept Visa, Mastercard and American Express. Bank transfer is available on Enterprise
plans only, and requires a signed order form before the first invoice is issued.

| Method | Plans | Settlement time | Notes |
|---|---|---|---|
| Credit card | All | Immediate | Retried three times over six days on failure |
| SEPA direct debit | Growth, Enterprise | 3 business days | EUR only |
| Bank transfer | Enterprise | Up to 10 business days | Requires a signed order form |
| PayPal | None | — | Not supported, and not on the roadmap |

## Failed payments

A failed charge is retried three times over six days. After the third failure the workspace
moves to a read-only state: existing data stays intact and readable, but no new writes are
accepted. Nothing is deleted at this stage.

A workspace left read-only for 30 days is scheduled for deletion, and the billing contact is
warned by email twice before that happens.

## Changing the billing contact

Only an account Owner can change the billing contact. Admins cannot, because the billing
contact is the address that receives dunning notices and a compromised admin account should
not be able to redirect them.

## Tax

Prices exclude VAT and sales tax. Adding a valid VAT number under **Settings → Billing → Tax**
removes VAT from subsequent invoices for EU customers under the reverse charge mechanism. We
cannot retroactively remove tax from an invoice that has already been issued.
