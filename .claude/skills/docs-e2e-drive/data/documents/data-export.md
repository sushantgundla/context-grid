# Exporting your data

You can export everything in a workspace at any time, on every plan, including during a trial
and including while a workspace is read-only after a failed payment. Data portability is not a
paid feature.

## Starting an export

Go to **Settings → Data → Export**, or call the API:

```bash
curl -X POST https://api.northwind.cloud/v2/exports \
  -H "Authorization: Bearer nw_live_..." \
  -d '{"format": "ndjson", "include_attachments": true}'
```

Exports run in the background. A workspace under 10 GB usually finishes in under an hour;
larger workspaces are chunked into 5 GB files and can take most of a day.

## Formats

| Format | Contains | Good for |
|---|---|---|
| `ndjson` | Every record, one JSON object per line | Loading into another system |
| `csv` | One file per record type, flattened | Spreadsheets and analysts |
| `parquet` | Columnar, typed, compressed | Warehouses and large workspaces |

Attachments are always exported as original files alongside a manifest, never inlined as
base64, whatever the record format.

## Download links

When an export finishes we email the billing contact and every Owner a link. The link is valid
for **seven days** and can be downloaded up to five times. After that the export is deleted
from our storage and you must start a new one.

## What is not exported

Audit logs older than the retention period of your plan, deleted records past their 30-day
recovery window, and other users' personal access tokens. Everything else in the workspace is
included.

## Scheduled exports

Enterprise plans can schedule a recurring export to an S3 bucket you own. You provide a role
ARN, we assume it, and we write only to the prefix you specify. We never hold long-lived
credentials to your bucket.
