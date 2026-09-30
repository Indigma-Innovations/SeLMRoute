import os
from pathlib import Path


def benchmark_data_root(root: str | Path = ".") -> Path:
    """Benchmark material regenerated from LLMRouterBench.
    """
    base = Path(root)
    configured = os.getenv("SELMROUTE_BENCHMARK_DATA_ROOT")
    if configured:
        p = Path(configured).expanduser()
        return p if p.is_absolute() else base / p
    return base / "data"
