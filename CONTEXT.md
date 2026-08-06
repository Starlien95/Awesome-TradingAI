# Domain Context

## Method

A Method is a complete quantitative approach that a user can select and run.
It may be a canonical prediction model, a Qlib workflow, an agent system, or a
runtime strategy family. A Method declares its supported Execution Modes before
it can be run.

## Execution Mode

An Execution Mode is one of:

- `backtest`: historical evaluation without account access or order placement.
- `paper`: local portfolio simulation without exchange account access.
- `simulated`: exchange-provided simulated trading, including optional account
  reads and explicitly confirmed order placement.
- `live`: production exchange trading, including optional account reads and
  explicitly confirmed order placement.

## Run Spec

A Run Spec is the complete user request for one Method execution. It contains
the Method, Execution Mode, workspace, inputs, common portfolio settings,
network permissions, account permissions, order permissions, and
method-specific parameters.

## Method Run

A Method Run is the durable control-plane record created for one Run Spec. It
tracks planning, execution status, adapter output, artifacts, and failure
context without replacing the Method's native artifacts.
