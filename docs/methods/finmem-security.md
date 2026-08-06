# Security Policy

## Public repository boundary

This repository must not contain API credentials, exchange account snapshots,
orders, strategy state, model traces, checkpoints, Qdrant storage, or local
machine paths. Use `.env.example` only as a variable template and keep real
values in an untracked file passed with `--env-file` and restrictive file
permissions.

Paper mode never reads an OKX account. Account reads require
`--query-account`; orders require `--execute-orders` and either
`DEMO_ORDERS` or `LIVE_ORDERS`. Do not weaken these independent gates in a
method adapter.

Before every public push, run:

```bash
python tools/check_finmem_public.py
python tools/check_release.py
git status --short
git diff --cached
```

If a credential was ever committed, deleting it in a later commit is not
sufficient. Revoke or rotate it immediately, then remove it from Git history
with an appropriate history-rewrite tool before publishing the repository.

## Reporting a vulnerability

Do not open a public issue containing a credential, account identifier, order,
or exploit detail. Contact the repository owner privately through the security
contact configured on the GitHub repository.
