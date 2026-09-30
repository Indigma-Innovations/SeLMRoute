import json
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from selmroute.util import query_hash


DEFAULT_DOMAINS = {
    "aime": "math", "math500": "math", "mathbench": "math", "livemathbench": "math",
    "humaneval": "code", "mbpp": "code", "livecodebench": "code", "swe_bench": "code", "swe-bench": "code",
    "bbh": "logic", "korbench": "logic", "kandk": "logic", "knights_and_knaves": "logic", "knights-knaves": "logic",
    "mmlu_pro": "knowledge", "mmlu-pro": "knowledge", "mmlupro": "knowledge", "gpqa": "knowledge", "finqa": "knowledge", "medqa": "knowledge", "hle": "knowledge", "simpleqa": "knowledge",
    "emorynlp": "affective", "meld": "affective",
    "arenahard": "instruction_following", "arena_hard": "instruction_following",
    "tau2": "tool_use", "tau2_bench": "tool_use", "tau2-bench": "tool_use",
}


OFFICIAL_COST_DATASETS = {
    "aime", "livemathbench", "gpqa", "hle", "livecodebench",
    "mmlupro", "swe-bench", "simpleqa", "tau2", "arenahard",
}
OFFICIAL_COST_MODELS = {
    "claude-sonnet-4", "deepseek-v3-0324", "deepseek-v3.1-terminus",
    "deepseek-r1-0528", "gemini-2.5-flash", "gemini-2.5-pro",
    "gpt-5-chat", "gpt-5", "qwen3-235b-a22b-2507",
    "qwen3-235b-a22b-thinking-2507", "glm-4.6", "kimi-k2-0905",
    "intern-s1",
}
OFFICIAL_COST_SPLITS = {"test", "hybrid", "v1", "test_3000", "verified"}
# Only matters when the same dataset/model/prompt exists in more than one
# whitelisted split.  In the released cost pool this primarily resolves the
# older MMLU-Pro ``test`` subset against the canonical ``test_3000`` result.
OFFICIAL_COST_SPLIT_PRIORITY = ("test_3000", "verified", "hybrid", "v1", "test")
OFFICIAL_COST_EXPECTED_COUNTS = {
    "aime": 60,
    "livemathbench": 121,
    "livecodebench": 1055,
    "swe_bench": 500,
    "gpqa": 198,
    "hle": 2158,
    "mmlupro": 3000,
    "simpleqa": 4326,
    "arenahard": 750,
    "tau2": 278,
}

OFFICIAL_MAIN_DATASETS = {
    "aime", "bbh", "emorynlp", "finqa", "gpqa", "humaneval", "kandk",
    "korbench", "livecodebench", "math500", "mathbench", "mbpp", "medqa",
    "meld", "mmlupro",
}
OFFICIAL_MAIN_MODELS = {
    "cogito-v1-preview-llama-8B", "DeepHermes-3-Llama-3-8B-Preview",
    "DeepSeek-R1-0528-Qwen3-8B", "DeepSeek-R1-Distill-Qwen-7B", "Fin-R1",
    "gemma-2-9b-it", "glm-4-9b-chat", "GLM-Z1-9B-0414",
    "granite-3.3-8b-instruct", "Intern-S1-mini", "internlm3-8b-instruct",
    "Llama-3.1-8B-Instruct", "Llama-3.1-8B-UltraMedical",
    "Llama-3.1-Nemotron-Nano-8B-v1", "MiMo-7B-RL-0530", "MiniCPM4.1-8B",
    "NVIDIA-Nemotron-Nano-9B-v2", "OpenThinker3-7B", "Qwen3-8B",
    "Qwen2.5-Coder-7B-Instruct",
}
OFFICIAL_MAIN_EXPECTED_COUNTS = {
    "aime": 60, "bbh": 1080, "emorynlp": 697, "finqa": 1147,
    "gpqa": 198, "humaneval": 164, "kandk": 700, "korbench": 1250,
    "livecodebench": 1055, "math500": 500, "mathbench": 150, "mbpp": 974,
    "medqa": 1273, "meld": 1232, "mmlupro": 1001,
}


def normalize_name(name: str) -> str:
    text = name.strip().lower().replace("²", "2")
    text = re.sub(r"[\s-]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    compact = text.replace("_", "")
    aliases = {
        "swebench": "swe_bench",
        "tau2bench": "tau2",
        "tau2": "tau2",
        "mmlupro": "mmlupro",
    }
    return aliases.get(compact, text)


@dataclass(frozen=True)
class IngestedData:
    samples: pd.DataFrame
    outcomes: pd.DataFrame


def _timestamp_key(path: Path) -> tuple[int, str]:
    """Match LLMRouterBench's latest-artifact rule deterministically."""
    m = re.search(r"(\d{8})_(\d{6})\.json$", path.name)
    if m:
        return (int(m.group(1) + m.group(2)), str(path))
    return (0, str(path))


def _read_header(path: Path, root: Path) -> tuple[str, str, str, bool] | None:
    """Read canonical identifiers from JSON payload metadata."""
    try:
        with path.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
    except Exception:
        return None
    dataset = payload.get("dataset_name") or payload.get("dataset_id")
    split = payload.get("split")
    model = payload.get("model_name")
    if all(isinstance(x, str) and x.strip() for x in (dataset, split, model)):
        return dataset.strip(), split.strip(), model.strip(), bool(payload.get("demo", False))
    # Backward-compatible fallback for synthetic/legacy fixtures.
    rel = path.relative_to(root)
    if len(rel.parts) >= 4:
        return rel.parts[0], rel.parts[1], rel.parts[2], False
    return None


def _select_result_files(
    root: Path,
    *,
    datasets: set[str] | None = None,
    models: set[str] | None = None,
    splits: set[str] | None = None,
    skip_demo: bool = True,
) -> list[tuple[Path, str, str, str]]:
    """Return latest JSON per dataset/split/model using payload metadata."""
    wanted_datasets = {normalize_name(x) for x in datasets} if datasets else None
    wanted_models = {x.strip() for x in models} if models else None
    wanted_splits = {x.strip() for x in splits} if splits else None
    groups: dict[tuple[str, str, str], list[Path]] = {}
    for path in root.rglob("*.json"):
        header = _read_header(path, root)
        if header is None:
            continue
        dataset, split, model, is_demo = header
        if skip_demo and is_demo:
            continue
        if wanted_datasets is not None and normalize_name(dataset) not in wanted_datasets:
            continue
        if wanted_models is not None and model not in wanted_models:
            continue
        if wanted_splits is not None and split not in wanted_splits:
            continue
        groups.setdefault((dataset, split, model), []).append(path)
    selected: list[tuple[Path, str, str, str]] = []
    for (dataset, split, model), paths in sorted(groups.items()):
        selected.append((max(paths, key=_timestamp_key), dataset, split, model))
    return selected


def _record_text(record: dict, identity_field: str) -> str | None:
    if identity_field == "prompt":
        value = record.get("prompt")
        if not isinstance(value, str) or not value.strip():
            value = record.get("origin_query")
    elif identity_field == "origin_query":
        value = record.get("origin_query")
        if not isinstance(value, str) or not value.strip():
            value = record.get("prompt")
    else:
        raise ValueError("identity_field must be 'origin_query' or 'prompt'")
    return value if isinstance(value, str) and value.strip() else None


def _split_rank(split: str, priority: tuple[str, ...] | list[str] | None) -> int:
    if not priority:
        return 0
    table = {name: len(priority) - idx for idx, name in enumerate(priority)}
    return table.get(split, 0)


def ingest_llmrouterbench(
    results_root: str | Path,
    complete_case_only: bool = True,
    datasets: set[str] | None = None,
    models: set[str] | None = None,
    splits: set[str] | None = None,
    *,
    identity_field: str = "origin_query",
    identity_field_by_dataset: dict[str, str] | None = None,
    collapse_splits_by_identity: bool = False,
    split_priority: tuple[str, ...] | list[str] | None = None,
    skip_demo: bool = True,
) -> IngestedData:
    """Ingest a filtered LLMRouterBench result pool.

    ``identity_field='origin_query'`` preserves SemRoute's historical 15-dataset
    ingestion behavior.  The official performance-cost protocol instead groups
    by benchmark ``prompt``; set ``identity_field='prompt'`` and
    ``collapse_splits_by_identity=True`` to collapse overlapping split variants
    (e.g. MMLU-Pro ``test`` inside ``test_3000``). Dataset-specific identity
    overrides may be supplied with ``identity_field_by_dataset``; this is used
    by the exact cost protocol for ArenaHard, whose released model files contain
    one prompt-text variant for the same canonical record index.
    """
    root = Path(results_root)
    if not root.exists():
        raise FileNotFoundError(root)
    selected = _select_result_files(
        root, datasets=datasets, models=models, splits=splits, skip_demo=skip_demo
    )
    if not selected:
        raise ValueError(f"No LLMRouterBench result JSON files found under {root}")

    sample_rows: dict[tuple[str, ...], dict] = {}
    outcome_map: dict[tuple[str, str], dict] = {}
    seen_models: set[str] = set()

    for path, dataset, split, model in selected:
        seen_models.add(model)
        split_rank = _split_rank(split, split_priority)
        with path.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
        for pos, record in enumerate(payload.get("records", [])):
            idx = str(record.get("index", pos))
            dataset_norm = normalize_name(dataset)
            effective_identity = identity_field
            if identity_field_by_dataset:
                effective_identity = identity_field_by_dataset.get(dataset_norm, identity_field)

            if effective_identity == "record_index":
                # Keep the actual benchmark prompt as the model input while using
                # the canonical record index only for cross-model identity.
                query = _record_text(record, "prompt") or _record_text(record, "origin_query")
                if query is None:
                    continue
                identity_value = f"idx:{idx}"
                qhash = query_hash(query)
                if collapse_splits_by_identity:
                    key = (dataset, identity_value)
                    sample_id = f"{dataset}:idx:{idx}"
                else:
                    key = (dataset, split, identity_value)
                    sample_id = f"{dataset}:{split}:idx:{idx}"
            else:
                query = _record_text(record, effective_identity)
                if query is None:
                    continue
                qhash = query_hash(query)
                if collapse_splits_by_identity:
                    key = (dataset, qhash)
                    sample_id = f"{dataset}:{qhash[:16]}"
                else:
                    key = (dataset, split, qhash)
                    sample_id = f"{dataset}:{split}:{qhash[:16]}"

            existing = sample_rows.get(key)
            if existing is None or split_rank > int(existing.get("_split_rank", -1)):
                sample_rows[key] = {
                    "sample_id": sample_id,
                    "dataset": dataset,
                    "split": split,
                    "record_index": idx,
                    "query": query,
                    "query_hash": qhash,
                    "domain": DEFAULT_DOMAINS.get(normalize_name(dataset), "unknown"),
                    "_split_rank": split_rank,
                }

            outcome = {
                "sample_id": sample_id,
                "dataset": dataset,
                "split": split,
                "model": model,
                "score": float(record.get("score", 0.0) or 0.0),
                "cost": float(record.get("cost", 0.0) or 0.0),
                "prompt_tokens": int(record.get("prompt_tokens", 0) or 0),
                "completion_tokens": int(record.get("completion_tokens", 0) or 0),
                "_split_rank": split_rank,
            }
            okey = (sample_id, model)
            prior = outcome_map.get(okey)
            if prior is None or split_rank >= int(prior.get("_split_rank", -1)):
                outcome_map[okey] = outcome

    samples = pd.DataFrame(sample_rows.values())
    if not samples.empty:
        samples = samples.drop(columns=["_split_rank"], errors="ignore")
        samples = samples.sort_values("sample_id").reset_index(drop=True)
    outcomes = pd.DataFrame(outcome_map.values())
    if not outcomes.empty:
        outcomes = outcomes.drop(columns=["_split_rank"], errors="ignore")
        outcomes = outcomes.sort_values(["sample_id", "model"]).reset_index(drop=True)

    if complete_case_only:
        expected = len(seen_models)
        counts = outcomes.groupby("sample_id")["model"].nunique()
        keep = counts[counts == expected].index
        samples = samples[samples.sample_id.isin(keep)].reset_index(drop=True)
        outcomes = outcomes[outcomes.sample_id.isin(keep)].reset_index(drop=True)
    if complete_case_only and samples.empty:
        raise ValueError(
            "No complete samples remain. The selected dataset/model pool is non-rectangular; "
            "use --no-complete-case-only or choose a common model pool."
        )
    return IngestedData(samples=samples, outcomes=outcomes)



def _select_historical_main_result_files(
    root: Path,
    *,
    datasets: set[str],
    models: set[str],
) -> list[Path]:
    wanted_datasets = {normalize_name(x) for x in datasets}
    wanted_models = {x.strip() for x in models}
    groups: dict[tuple[str, str, str], list[Path]] = {}
    for path in root.rglob("*.json"):
        rel = path.relative_to(root)
        if len(rel.parts) < 4:
            continue
        dataset, split, model = rel.parts[0], rel.parts[1], rel.parts[2]
        if normalize_name(dataset) not in wanted_datasets:
            continue
        if model not in wanted_models:
            continue
        groups.setdefault((dataset, split, model), []).append(path)
    return [sorted(paths)[-1] for _, paths in sorted(groups.items())]


def ingest_llmrouterbench_main_frozen(
    results_root: str | Path,
    *,
    complete_case_only: bool = True,
) -> IngestedData:
    root = Path(results_root)
    if not root.exists():
        raise FileNotFoundError(root)

    files = _select_historical_main_result_files(
        root,
        datasets=set(OFFICIAL_MAIN_DATASETS),
        models=set(OFFICIAL_MAIN_MODELS),
    )
    if not files:
        raise ValueError(f"No LLMRouterBench result JSON files found under {root}")

    sample_rows: dict[tuple[str, str, str], dict] = {}
    outcome_rows: list[dict] = []
    seen_models: set[str] = set()

    for path in files:
        rel = path.relative_to(root)
        dataset, split, model = rel.parts[0], rel.parts[1], rel.parts[2]
        seen_models.add(model)
        with path.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)

        for pos, record in enumerate(payload.get("records", [])):
            idx = str(record.get("index", pos))
            query = record.get("origin_query") or record.get("prompt")
            if not isinstance(query, str) or not query.strip():
                continue

            key = (dataset, split, idx)
            qhash = query_hash(query)
            existing = sample_rows.get(key)
            if existing is not None and existing["query_hash"] != qhash:
                raise ValueError(
                    "Historical main-pool query mismatch across models for "
                    f"{key}: {path}. This bundle does not match the bundle used "
                    "to create the frozen SeLMRoute semantic features."
                )

            sample_id = f"{dataset}:{split}:{idx}:{qhash[:12]}"
            sample_rows[key] = {
                "sample_id": sample_id,
                "dataset": dataset,
                "split": split,
                "record_index": idx,
                "query": query,
                "query_hash": qhash,
                "domain": DEFAULT_DOMAINS.get(normalize_name(dataset), "unknown"),
            }
            outcome_rows.append({
                "sample_id": sample_id,
                "dataset": dataset,
                "split": split,
                "model": model,
                "score": float(record.get("score", 0.0) or 0.0),
                "cost": float(record.get("cost", 0.0) or 0.0),
                "prompt_tokens": int(record.get("prompt_tokens", 0) or 0),
                "completion_tokens": int(record.get("completion_tokens", 0) or 0),
            })

    samples = pd.DataFrame(sample_rows.values()).sort_values("sample_id").reset_index(drop=True)
    outcomes = (
        pd.DataFrame(outcome_rows)
        .drop_duplicates(["sample_id", "model"], keep="last")
        .sort_values(["sample_id", "model"])
        .reset_index(drop=True)
    )

    if complete_case_only:
        expected = len(seen_models)
        counts = outcomes.groupby("sample_id")["model"].nunique()
        keep = counts[counts == expected].index
        samples = samples[samples["sample_id"].isin(keep)].reset_index(drop=True)
        outcomes = outcomes[outcomes["sample_id"].isin(keep)].reset_index(drop=True)

    if complete_case_only and samples.empty:
        raise ValueError(
            "No complete samples remain. The supplied benchmark does not match "
            "the historical SeLMRoute 20-model performance pool."
        )
    return IngestedData(samples=samples, outcomes=outcomes)

def _count_by_normalized_dataset(samples: pd.DataFrame) -> dict[str, int]:
    if samples.empty:
        return {}
    tmp = samples.copy()
    tmp["_dataset_norm"] = tmp["dataset"].astype(str).map(normalize_name)
    return {str(k): int(v) for k, v in tmp.groupby("_dataset_norm").size().items()}


def validate_expected_dataset_counts(
    samples: pd.DataFrame,
    expected: dict[str, int],
) -> pd.DataFrame:
    actual = _count_by_normalized_dataset(samples)
    rows = []
    for dataset in sorted(set(expected) | set(actual)):
        rows.append({
            "dataset": dataset,
            "expected": int(expected.get(dataset, 0)),
            "actual": int(actual.get(dataset, 0)),
            "delta": int(actual.get(dataset, 0) - expected.get(dataset, 0)),
        })
    return pd.DataFrame(rows)


def validate_record_index_alignment(
    results_root: str | Path,
    *,
    dataset: str,
    models: set[str],
    splits: set[str] | None = None,
    expected_count: int | None = None,
) -> dict[str, object]:
    """Validate that every selected model exposes the same unique record-index set.

    This is intentionally stricter than row-position alignment: records may appear
    in different file order, but their explicit benchmark ``index`` values must
    agree exactly across models.
    """
    root = Path(results_root)
    selected = _select_result_files(
        root, datasets={dataset}, models=models, splits=splits, skip_demo=True
    )
    by_model: dict[str, set[str]] = {}
    duplicate_indices: dict[str, list[str]] = {}
    for path, _dataset, _split, model in selected:
        with path.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
        values: list[str] = []
        for pos, record in enumerate(payload.get("records", [])):
            values.append(str(record.get("index", pos)))
        unique = set(values)
        if len(unique) != len(values):
            seen: set[str] = set()
            dupes: list[str] = []
            for value in values:
                if value in seen:
                    dupes.append(value)
                seen.add(value)
            duplicate_indices[model] = sorted(set(dupes))
        by_model.setdefault(model, set()).update(unique)

    missing_models = sorted(models - set(by_model))
    if missing_models:
        raise ValueError(
            f"{dataset} record-index alignment failed: missing model files {missing_models}"
        )
    if duplicate_indices:
        raise ValueError(
            f"{dataset} record-index alignment failed: duplicate indices {duplicate_indices}"
        )

    reference_model = sorted(by_model)[0]
    reference = by_model[reference_model]
    mismatches = {}
    for model, indices in sorted(by_model.items()):
        if indices != reference:
            mismatches[model] = {
                "missing_vs_reference": sorted(reference - indices)[:20],
                "extra_vs_reference": sorted(indices - reference)[:20],
                "count": len(indices),
            }
    if mismatches:
        raise ValueError(
            f"{dataset} record-index alignment failed across models: {mismatches}"
        )
    if expected_count is not None and len(reference) != int(expected_count):
        raise ValueError(
            f"{dataset} record-index alignment failed: expected {expected_count} unique "
            f"indices but found {len(reference)}"
        )
    return {
        "dataset": normalize_name(dataset),
        "n_models": len(by_model),
        "n_unique_indices": len(reference),
        "reference_model": reference_model,
    }


def ingest_llmrouterbench_cost_exact(
    results_root: str | Path,
    *,
    assert_counts: bool = True,
) -> tuple[IngestedData, pd.DataFrame]:
    """Ingest the official 13-model/10-dataset performance-cost pool exactly.

    Mirrors ``baseline_config_performance_cost.yaml`` filtering and the official
    prompt-grouping semantics.  The pool is intentionally non-rectangular.
    """
    # ArenaHard contains one released prompt-text variant across model files.
    # The benchmark provides an explicit record index, so validate that all 13
    # model files share the same 750-index set and use that canonical identity
    # only for ArenaHard. Other cost datasets retain prompt-based identity.
    if "arenahard" in {normalize_name(x) for x in OFFICIAL_COST_DATASETS}:
        validate_record_index_alignment(
            results_root,
            dataset="arenahard",
            models=set(OFFICIAL_COST_MODELS),
            splits=set(OFFICIAL_COST_SPLITS),
            expected_count=OFFICIAL_COST_EXPECTED_COUNTS.get("arenahard"),
        )
    data = ingest_llmrouterbench(
        results_root,
        complete_case_only=False,
        datasets=set(OFFICIAL_COST_DATASETS),
        models=set(OFFICIAL_COST_MODELS),
        splits=set(OFFICIAL_COST_SPLITS),
        identity_field="prompt",
        identity_field_by_dataset=(
            {"arenahard": "record_index"}
            if "arenahard" in {normalize_name(x) for x in OFFICIAL_COST_DATASETS}
            else None
        ),
        collapse_splits_by_identity=True,
        split_priority=OFFICIAL_COST_SPLIT_PRIORITY,
        skip_demo=True,
    )
    audit = validate_expected_dataset_counts(data.samples, OFFICIAL_COST_EXPECTED_COUNTS)
    if assert_counts and (audit["delta"] != 0).any():
        bad = audit[audit["delta"] != 0].to_dict(orient="records")
        raise ValueError(
            "Exact cost-pool count assertion failed; expected the official 12,446 instances. "
            f"Mismatches: {bad}"
        )
    return data, audit


def audit_identity_counts(
    results_root: str | Path,
    *,
    datasets: set[str],
    models: set[str],
    splits: set[str] | None = None,
) -> pd.DataFrame:
    """Compare origin_query and prompt identity counts without training anything."""
    root = Path(results_root)
    selected = _select_result_files(root, datasets=datasets, models=models, splits=splits)
    grouped: dict[str, dict[str, set[str]]] = {}
    model_counts: dict[str, dict[str, int]] = {}
    for path, dataset, split, model in selected:
        dnorm = normalize_name(dataset)
        grouped.setdefault(dnorm, {"origin_query": set(), "prompt": set()})
        model_counts.setdefault(dnorm, {})
        with path.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
        model_counts[dnorm][model] = model_counts[dnorm].get(model, 0) + len(payload.get("records", []))
        for record in payload.get("records", []):
            for field in ("origin_query", "prompt"):
                value = record.get(field)
                if isinstance(value, str) and value.strip():
                    grouped[dnorm][field].add(query_hash(value))
    rows = []
    for dataset in sorted(grouped):
        counts = list(model_counts.get(dataset, {}).values())
        rows.append({
            "dataset": dataset,
            "n_models": len(model_counts.get(dataset, {})),
            "records_min_model": min(counts) if counts else 0,
            "records_max_model": max(counts) if counts else 0,
            "origin_query_union": len(grouped[dataset]["origin_query"]),
            "prompt_union": len(grouped[dataset]["prompt"]),
            "prompt_minus_origin": len(grouped[dataset]["prompt"]) - len(grouped[dataset]["origin_query"]),
        })
    return pd.DataFrame(rows)


def write_ingested(data: IngestedData, output_dir: str | Path) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    data.samples.to_csv(out / "samples.csv", index=False)
    data.outcomes.to_csv(out / "outcomes.csv", index=False)
    coverage = data.outcomes.pivot_table(index="sample_id", columns="model", values="score", aggfunc="first").notna().sum()
    coverage.rename("samples").to_csv(out / "model_coverage.csv")
