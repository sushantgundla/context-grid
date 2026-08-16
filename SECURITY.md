# Security

## Reporting a vulnerability

Email **gundla.sushant@gmail.com** with `context-grid security` in the subject, or use GitHub's
[private vulnerability reporting](https://github.com/sushantgundla/context-grid/security/advisories/new).

Please do not open a public issue for a vulnerability. You should get a reply within a week. If
you do not, assume the mail went astray and send it again — a silence is much more likely to be
an accident than a decision.

Tell me what you can: what the flaw is, how to reproduce it, and what an attacker gets. A rough
report is far better than no report.

## What is in scope

context-grid is a local measurement tool, not a service, so the interesting cases are about what
it does with input it did not write:

- **Documents it parses.** A corpus is untrusted input. A crafted PDF, HTML or Markdown file that
  causes code execution, an unbounded read, or a write outside the output directory is a
  vulnerability.
- **Config files.** `contextgrid run` executes a YAML file. It should never be able to run
  arbitrary code, load arbitrary Python, or read files the config did not name.
- **Eval sets.** Same reasoning: JSONL, CSV and BEIR imports come from elsewhere.
- **Cached artefacts.** `DiskCache` reads what an earlier run wrote. A cache entry that can be
  crafted into code execution is in scope.
- **Leaked credentials.** Any path where an API key set in the environment or a config ends up in
  a report, a manifest, a log line or an exception message.

## What is not

- Optional dependencies' own vulnerabilities. Report those upstream; tell me too if context-grid
  makes them reachable in a way the library itself would not.
- Cost. A sweep that spends real money on hosted models is doing what it was told to do. Use
  `run.budget_usd` — but if the budget is *not honoured*, that is a bug worth reporting.
- Running an obviously dangerous config on purpose. Pointing the tool at `/` is not an exploit.

## Supported versions

Pre-1.0, so only the latest release gets fixes. Currently `0.9.x`.
