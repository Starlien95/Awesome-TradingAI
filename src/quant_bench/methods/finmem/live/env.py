import os
from pathlib import Path

from dotenv import load_dotenv


DEFAULT_ENV_PATH = Path(os.getenv("FINMEM_ENV_FILE", "~/.config/quant-bench/finmem.env")).expanduser()


def load_project_env(verbose=False, path=None, override=False):
    """Load one explicit local env file without overriding the shell by default."""
    env_path = Path(path).expanduser().resolve() if path else DEFAULT_ENV_PATH
    loaded = load_dotenv(dotenv_path=env_path, override=override) if env_path.is_file() else False
    if not os.getenv("OPENAI_API_KEY") and os.getenv("DASHSCOPE_API_KEY"):
        os.environ["OPENAI_API_KEY"] = os.getenv("DASHSCOPE_API_KEY")

    if verbose:
        print(f"[ENV] env file: {env_path}")
        print(f"[ENV] loaded: {loaded}")

        for key in [
            "QDRANT_ENDPOINT",
            "STRATEGY_INITIAL_CAPITAL_USDT",
            "TRADE_AMOUNT_USDT",
            "DASHSCOPE_API_KEY",
            "OPENAI_API_KEY",
            "OKX_API_KEY_SIMU",
            "OKX_SECRET_KEY_SIMU",
            "OKX_PASSPHRASE",
        ]:
            val = os.getenv(key)
            if val:
                if "KEY" in key or "SECRET" in key or "PASSPHRASE" in key:
                    print(f"[ENV] {key}=***{val[-4:]}")
                else:
                    print(f"[ENV] {key}={val}")
            else:
                print(f"[ENV] {key}=None")

    return loaded


def require_env(keys):
    missing = [k for k in keys if not os.getenv(k)]

    if missing:
        raise EnvironmentError(
            "Missing required environment variables: "
            + ", ".join(missing)
            + f"\nPlease export them or check: {DEFAULT_ENV_PATH}"
        )


def get_env_bool(name, default=False):
    value = os.getenv(name)
    if value is None:
        return bool(default)
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}
