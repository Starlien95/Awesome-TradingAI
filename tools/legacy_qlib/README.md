# Legacy Qlib command implementations

These modules preserve the command behavior of the former script collection while the supported interface moves to `quant-bench qlib`, `quant-bench sweep`, `quant-bench report`, and `quant-bench artifacts promote`.

They remain source-tree tools for migration and regression work. New automation should use the installed CLI because it validates inputs, writes runs to an explicit workspace, and records manifests and checksums.
