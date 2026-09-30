import hashlib
import json
import shutil
import tarfile
import tempfile
from pathlib import Path

from selmroute.data.llmrouterbench import (
    OFFICIAL_COST_MODELS,
    OFFICIAL_MAIN_DATASETS,
    OFFICIAL_MAIN_EXPECTED_COUNTS,
    OFFICIAL_MAIN_MODELS,
    IngestedData,
    ingest_llmrouterbench_main_frozen,
    ingest_llmrouterbench_cost_exact,
    validate_expected_dataset_counts,
    write_ingested,
)

LLMROUTERBENCH_HF_REPO = "NPULH/LLMRouterBench"
LLMROUTERBENCH_HF_REVISION = "0e5af1b84bf73437a01a1849c0f1d2468baa93fc"
LLMROUTERBENCH_BUNDLE_NAME = "bench-release.tar.gz"
LLMROUTERBENCH_BUNDLE_SHA256 = "b79f8cde1a6f029c2efa663a3a3b6f7748defb22341fe59f328cebef6648c8f1"

PERF_SAMPLES = 11_481
PERF_OUTCOMES = 229_620
COST_SAMPLES = 12_446
COST_OUTCOMES = 161_520


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _safe_extract_bundle(bundle: Path, destination: Path) -> Path:
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(bundle, "r:gz") as tf:
        # Python 3.13's data filter rejects absolute paths, links escaping the
        # destination, device files and other unsafe archive members.
        tf.extractall(destination, filter="data")

    direct = destination / "bench-release"
    if direct.is_dir():
        return direct

    candidates = [p for p in destination.rglob("bench-release") if p.is_dir()]
    if len(candidates) == 1:
        return candidates[0]
    raise ValueError(
        "Could not locate the extracted 'bench-release' directory. "
        f"Found {len(candidates)} candidates under {destination}."
    )


def _validate_main(data: IngestedData) -> dict[str, object]:
    audit = validate_expected_dataset_counts(data.samples, OFFICIAL_MAIN_EXPECTED_COUNTS)
    bad = audit[audit["delta"] != 0]
    problems: list[str] = []
    if len(data.samples) != PERF_SAMPLES:
        problems.append(f"samples={len(data.samples)} expected={PERF_SAMPLES}")
    if len(data.outcomes) != PERF_OUTCOMES:
        problems.append(f"outcomes={len(data.outcomes)} expected={PERF_OUTCOMES}")
    if data.samples["dataset"].nunique() != len(OFFICIAL_MAIN_DATASETS):
        problems.append(
            f"datasets={data.samples['dataset'].nunique()} expected={len(OFFICIAL_MAIN_DATASETS)}"
        )
    if data.outcomes["model"].nunique() != len(OFFICIAL_MAIN_MODELS):
        problems.append(
            f"models={data.outcomes['model'].nunique()} expected={len(OFFICIAL_MAIN_MODELS)}"
        )
    if not bad.empty:
        problems.append(f"dataset-count mismatches={bad.to_dict(orient='records')}")
    if problems:
        raise ValueError(
            "The supplied LLMRouterBench bundle does not match the frozen "
            "SeLMRoute performance pool: " + "; ".join(problems)
        )
    return {
        "samples": int(len(data.samples)),
        "outcomes": int(len(data.outcomes)),
        "datasets": int(data.samples["dataset"].nunique()),
        "models": int(data.outcomes["model"].nunique()),
        "dataset_counts": audit.to_dict(orient="records"),
    }


def _validate_cost(data: IngestedData) -> dict[str, object]:
    problems: list[str] = []
    if len(data.samples) != COST_SAMPLES:
        problems.append(f"samples={len(data.samples)} expected={COST_SAMPLES}")
    if len(data.outcomes) != COST_OUTCOMES:
        problems.append(f"outcomes={len(data.outcomes)} expected={COST_OUTCOMES}")
    if data.outcomes["model"].nunique() != len(OFFICIAL_COST_MODELS):
        problems.append(
            f"models={data.outcomes['model'].nunique()} expected={len(OFFICIAL_COST_MODELS)}"
        )
    if problems:
        raise ValueError(
            "The supplied LLMRouterBench bundle does not match the frozen "
            "SeLMRoute performance-cost pool: " + "; ".join(problems)
        )
    return {
        "samples": int(len(data.samples)),
        "outcomes": int(len(data.outcomes)),
        "datasets": int(data.samples["dataset"].nunique()),
        "models": int(data.outcomes["model"].nunique()),
    }


def prepare_benchmark_data(
    *,
    output_root: str | Path,
    results_root: str | Path | None = None,
    bundle: str | Path | None = None,
    verify_bundle: bool = True,
    keep_extracted_source: bool = False,
) -> dict[str, object]:
    """Rebuild SeLMRoute's private ``samples.csv`` and ``outcomes.csv`` files.

    Exactly one of ``results_root`` or ``bundle`` must be supplied. The function
    never downloads benchmark content and never writes benchmark-derived rows
    outside ``output_root``.
    """
    if (results_root is None) == (bundle is None):
        raise ValueError("Provide exactly one of results_root or bundle")

    out = Path(output_root).resolve()
    out.mkdir(parents=True, exist_ok=True)

    source_info: dict[str, object] = {
        "hf_repo": LLMROUTERBENCH_HF_REPO,
        "hf_revision": LLMROUTERBENCH_HF_REVISION,
        "expected_bundle_sha256": LLMROUTERBENCH_BUNDLE_SHA256,
    }

    temporary: tempfile.TemporaryDirectory[str] | None = None
    try:
        if bundle is not None:
            bundle_path = Path(bundle).expanduser().resolve()
            if not bundle_path.is_file():
                raise FileNotFoundError(bundle_path)
            actual_sha = sha256_file(bundle_path)
            source_info.update({"bundle": str(bundle_path), "bundle_sha256": actual_sha})
            if verify_bundle and actual_sha != LLMROUTERBENCH_BUNDLE_SHA256:
                raise ValueError(
                    "LLMRouterBench bundle SHA-256 mismatch. "
                    f"Expected {LLMROUTERBENCH_BUNDLE_SHA256}, got {actual_sha}. "
                    "Use --no-verify-bundle only if you intentionally want to test a different bundle."
                )
            if keep_extracted_source:
                extract_base = out / "_source"
                if extract_base.exists():
                    shutil.rmtree(extract_base)
            else:
                temporary = tempfile.TemporaryDirectory(prefix="selmroute-llmrouterbench-")
                extract_base = Path(temporary.name)
            source = _safe_extract_bundle(bundle_path, extract_base)
        else:
            source = Path(results_root).expanduser().resolve()  # type: ignore[arg-type]
            if not source.is_dir():
                raise FileNotFoundError(source)
            source_info["results_root"] = str(source)

        # The frozen JEV/Laya artifacts were created on 2026-09-19 with the
        # original SeLMRoute v0.1.2 importer.  Reproduce that identity exactly:
        # dataset + split + benchmark record index, with the query hash included
        # only in sample_id.  Do not use the later query-hash grouping importer.
        main = ingest_llmrouterbench_main_frozen(
            source,
            complete_case_only=True,
        )
        main_info = _validate_main(main)
        perf_dir = out / "performance"
        write_ingested(main, perf_dir)
        # model_coverage.csv is a derived convenience table, not needed by the paper.
        (perf_dir / "model_coverage.csv").unlink(missing_ok=True)

        cost, cost_audit = ingest_llmrouterbench_cost_exact(source, assert_counts=True)
        cost_info = _validate_cost(cost)
        cost_dir = out / "cost"
        write_ingested(cost, cost_dir)
        (cost_dir / "model_coverage.csv").unlink(missing_ok=True)
        cost_audit.to_csv(cost_dir / "official_count_audit.csv", index=False)

        manifest = {
            "format": "selmroute-local-benchmark-v1",
            "source": source_info,
            "performance": {**main_info, "identity_protocol": "historical-v0.1.2: dataset+split+record_index"},
            "performance_cost": {
                **cost_info,
                "dataset_counts": cost_audit.to_dict(orient="records"),
            },
            "files": {
                "performance_samples": "performance/samples.csv",
                "performance_outcomes": "performance/outcomes.csv",
                "cost_samples": "cost/samples.csv",
                "cost_outcomes": "cost/outcomes.csv",
            },
            "notice": (
                "These files are reconstructed locally from third-party benchmark material. "
                "They are gitignored and are not licensed or redistributed by SeLMRoute."
            ),
        }
        (out / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
        )
        return manifest
    finally:
        if temporary is not None:
            temporary.cleanup()
