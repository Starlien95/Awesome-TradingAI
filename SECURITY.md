# Security policy

## Supported code

Security fixes target the latest `main` development line until versioned releases exist.

## Reporting

Report vulnerabilities through a private repository security advisory or another private channel provided by the repository owner. Do not include credentials, account snapshots, orders, fills, private news data, or unrestricted production logs in a public issue.

## Credential handling

- Credentials must come from environment variables or an untracked local secret provider.
- Configuration, examples, tests, fixtures, documentation, and artifacts must not contain credential values.
- Diagnostic commands may report only whether a variable is present.
- Research and dashboard code must not query accounts or place orders.

## Model artifact handling

Pickle and joblib can execute code during deserialization. Public loaders reject arbitrary Qlib pickle artifacts. Legacy checkpoints may be read only through a future explicit trusted-local migration command after the operator verifies their origin.

## Runtime safety

The base package has no live-order path. Demo and live execution require separate dependencies, mode validation, risk limits, an audit log, and explicit operator confirmation. Liquidation and account-alignment tools are manual operations.
