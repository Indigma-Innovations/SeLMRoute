"""Build the frozen SeLMRoute tables from a local LLMRouterBench release.

This module deliberately contains no downloader.  The benchmark's licence and
distribution terms remain the responsibility of the user who obtained it.
"""

import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


class IncompatibleBenchmarkError(ValueError):
    """Raised when a directory is not the benchmark snapshot used by the paper."""


@dataclass(frozen=True)
class FrozenSetting:
    name: str
    source_names: tuple[str, ...]
    sample_count: int
    outcome_count: int
    datasets: tuple[str, ...]
    models: tuple[str, ...]
    id_style: str


PERFORMANCE = FrozenSetting(
    "performance",
    ("performance",),
    11_481,
    229_620,
    ("aime", "bbh", "emorynlp", "finqa", "gpqa", "humaneval", "kandk", "korbench",
     "livecodebench", "math500", "mathbench", "mbpp", "medqa", "meld", "mmlupro"),
    ("DeepHermes-3-Llama-3-8B-Preview", "DeepSeek-R1-0528-Qwen3-8B",
     "DeepSeek-R1-Distill-Qwen-7B", "Fin-R1", "GLM-Z1-9B-0414", "Intern-S1-mini",
     "Llama-3.1-8B-Instruct", "Llama-3.1-8B-UltraMedical", "Llama-3.1-Nemotron-Nano-8B-v1",
     "MiMo-7B-RL-0530", "MiniCPM4.1-8B", "NVIDIA-Nemotron-Nano-9B-v2", "OpenThinker3-7B",
     "Qwen2.5-Coder-7B-Instruct", "Qwen3-8B", "cogito-v1-preview-llama-8B", "gemma-2-9b-it",
     "glm-4-9b-chat", "granite-3.3-8b-instruct", "internlm3-8b-instruct"),
    "indexed",
)

PERFORMANCE_COST = FrozenSetting(
    "performance_cost",
    ("performance_cost", "cost"),
    12_446,
    161_520,
    ("aime", "arenahard", "gpqa", "hle", "livecodebench", "livemathbench", "mmlupro",
     "simpleqa", "swe-bench", "tau2"),
    ("claude-sonnet-4", "deepseek-r1-0528", "deepseek-v3-0324", "deepseek-v3.1-terminus",
     "gemini-2.5-flash", "gemini-2.5-pro", "glm-4.6", "gpt-5", "gpt-5-chat", "intern-s1",
     "kimi-k2-0905", "qwen3-235b-a22b-2507", "qwen3-235b-a22b-thinking-2507"),
    "content",
)

SETTINGS = (PERFORMANCE, PERFORMANCE_COST)
SAMPLE_COLUMNS = ("sample_id", "dataset", "split", "record_index", "query", "query_hash", "domain")
OUTCOME_COLUMNS = ("sample_id", "dataset", "split", "model", "score", "cost", "prompt_tokens",
                   "completion_tokens")


def _fail(message: str) -> IncompatibleBenchmarkError:
    return IncompatibleBenchmarkError(
        f"Incompatible LLMRouterBench revision/content: {message}. "
        "SeLMRoute will not substitute a different dataset, model pool, or benchmark revision."
    )


def _locate_pair(root: Path, setting: FrozenSetting) -> tuple[Path, Path]:
    candidates = []
    for name in setting.source_names:
        candidates.extend((root / name, root / "data" / name, root / ".local_data" / name))
    pairs = [(p / "samples.csv", p / "outcomes.csv") for p in candidates]
    found = [(s, o) for s, o in pairs if s.is_file() and o.is_file()]
    if len(found) != 1:
        checked = ", ".join(str(s.parent.relative_to(root)) for s, _ in pairs)
        detail = "multiple matching table pairs" if found else "samples.csv/outcomes.csv not found"
        raise _fail(f"{setting.name}: {detail}; checked {checked}")
    return found[0]


def _require_columns(frame: pd.DataFrame, columns: tuple[str, ...], label: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise _fail(f"{label} is missing columns {missing}")


def _make_ids(samples: pd.DataFrame, setting: FrozenSetting) -> pd.Series:
    ids: list[str] = []
    for row in samples.itertuples(index=False):
        dataset, digest = str(row.dataset), str(row.query_hash).lower()
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise _fail(f"{setting.name}: invalid query_hash for dataset {dataset!r}")
        if setting.id_style == "indexed":
            ids.append(f"{dataset}:{row.split}:{row.record_index}:{digest[:12]}")
        elif dataset == "arenahard":
            ids.append(f"arenahard:idx:{row.record_index}")
        else:
            ids.append(f"{dataset}:{digest[:16]}")
    return pd.Series(ids, index=samples.index, dtype="string")


def _build_setting(root: Path, destination: Path, setting: FrozenSetting) -> dict[str, object]:
    sample_path, outcome_path = _locate_pair(root, setting)
    samples = pd.read_csv(sample_path, dtype={"sample_id": str, "record_index": str, "query_hash": str})
    outcomes = pd.read_csv(outcome_path, dtype={"sample_id": str})
    _require_columns(samples, SAMPLE_COLUMNS, f"{setting.name}/samples.csv")
    _require_columns(outcomes, OUTCOME_COLUMNS, f"{setting.name}/outcomes.csv")

    if len(samples) != setting.sample_count:
        raise _fail(f"{setting.name}: found {len(samples):,} samples; expected {setting.sample_count:,}")
    datasets = tuple(sorted(samples["dataset"].astype(str).unique()))
    if datasets != setting.datasets:
        raise _fail(f"{setting.name}: datasets are {list(datasets)!r}; expected {list(setting.datasets)!r}")
    models = tuple(sorted(outcomes["model"].astype(str).unique()))
    if models != tuple(sorted(setting.models)):
        raise _fail(f"{setting.name}: candidate models are {list(models)!r}; expected {list(setting.models)!r}")

    old_ids = samples["sample_id"].astype(str)
    if old_ids.duplicated().any():
        raise _fail(f"{setting.name}: duplicate source sample_id values")
    new_ids = _make_ids(samples, setting)
    if new_ids.duplicated().any():
        raise _fail(f"{setting.name}: frozen sample-id rule produced collisions")
    id_map = dict(zip(old_ids, new_ids, strict=True))
    unknown = sorted(set(outcomes["sample_id"].astype(str)) - set(id_map))
    if unknown:
        raise _fail(f"{setting.name}: outcomes reference {len(unknown)} unknown sample IDs")
    samples["sample_id"] = new_ids
    outcomes["sample_id"] = outcomes["sample_id"].astype(str).map(id_map)

    if len(outcomes) != setting.outcome_count:
        raise _fail(
            f"{setting.name}: found {len(outcomes):,} outcomes; expected {setting.outcome_count:,}"
        )
    if outcomes.duplicated(["sample_id", "model"]).any():
        raise _fail(f"{setting.name}: duplicate (sample_id, model) outcomes")
    support = outcomes.groupby(["dataset", "model"], sort=False)["sample_id"].nunique()
    expected_support = {
        (dataset, model): int((samples["dataset"] == dataset).sum())
        for dataset in setting.datasets
        for model in setting.models
        # GPT-5 Chat was not evaluated on tau2 in the frozen cost protocol.
        if not (setting is PERFORMANCE_COST and dataset == "tau2" and model == "gpt-5-chat")
    }
    actual_support = {tuple(key): int(value) for key, value in support.items()}
    if actual_support != expected_support:
        raise _fail(f"{setting.name}: outcome support does not match the frozen dataset/model matrix")

    # Lexical ordering is part of the frozen export and makes reruns byte-stable.
    samples = samples.loc[:, SAMPLE_COLUMNS].sort_values("sample_id", kind="stable")
    outcomes = outcomes.loc[:, OUTCOME_COLUMNS].sort_values(["sample_id", "model"], kind="stable")
    destination.mkdir(parents=True)
    samples.to_csv(destination / "samples.csv", index=False, lineterminator="\n")
    outcomes.to_csv(destination / "outcomes.csv", index=False, lineterminator="\n")
    return {"samples": len(samples), "datasets": len(datasets), "models": len(models),
            "outcomes": len(outcomes)}


def prepare_data(benchmark_root: Path, output_root: Path = Path(".local_data")) -> dict[str, object]:
    """Validate a local release and atomically create both frozen table pairs."""
    benchmark_root = benchmark_root.resolve()
    if not benchmark_root.is_dir():
        raise _fail(f"benchmark root is not a directory: {benchmark_root}")
    output_root = output_root.resolve()
    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_root.name}-", dir=output_root.parent))
    try:
        result = {s.name: _build_setting(benchmark_root, staging / s.name, s) for s in SETTINGS}
        if output_root.exists():
            shutil.rmtree(output_root)
        staging.replace(output_root)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return {"status": "ok", "source": str(benchmark_root), "output_dir": str(output_root), **result}
