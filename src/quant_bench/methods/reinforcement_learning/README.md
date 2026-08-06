# Reinforcement-learning methods

MacroHFT v1 is the first registered reinforcement-learning runtime method. Its official source and checkpoints remain external because the audited upstream repository has no software license. Quant-bench loads user-exported TorchScript bundles and does not redistribute the upstream implementation. RL methods use the same run manifest, prediction, signal, metrics, trade, volume, and execution audit contracts as traditional ML.

New policies should keep environment-specific state inside the method package, expose model capability and provenance metadata, and use the shared runtime adapter and safety gates for execution.
