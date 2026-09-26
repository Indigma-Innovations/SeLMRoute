import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from selmroute.features.embeddings import load_embeddings
from selmroute.paper.ablation import build_semantic_ablation_features
from selmroute.routing.catboost_router import CatBoostQualityRouter

PERF_SAMPLES = 11_481
PERF_DATASETS = 15
PERF_MODELS = 20
PERF_OUTCOMES = 229_620
COST_SAMPLES = 12_446
COST_DATASETS = 10
COST_MODELS = 13
COST_OUTCOMES = 161_520
PM_FEATURES = 40


def sha256_file(path: str | Path) -> str:
    p = Path(path)
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def default_paths(root: str | Path = ".") -> dict[str, Path]:
    root = Path(root)
    return {
        "perf_samples": root / "data/performance/samples.csv",
        "perf_outcomes": root / "data/performance/outcomes.csv",
        "jev_features": root / "data/performance/features_jev.csv",
        "laya_features": root / "data/performance/features_laya.csv",
        "gte_embeddings": root / "data/performance/gte_qwen2.npz",
        "compact12_features": root / "data/performance/features_jev_compact12.csv",
        "cost_samples": root / "data/cost/samples.csv",
        "cost_outcomes": root / "data/cost/outcomes.csv",
        "cost_features": root / "data/cost/features_jev.csv",
        "jev_direct_predictions": root / "data/frozen/jev_direct/oof_predictions.csv",
        "jev_direct_metrics": root / "data/frozen/jev_direct/jev_direct_oof_metrics.json",
        "jev_model": root / "models/jev/model.cbm",
        "jev_model_meta": root / "models/jev/metadata.json",
        "laya_model": root / "models/laya/model.cbm",
        "laya_model_meta": root / "models/laya/metadata.json",
    }


def _require(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Required release artifact is missing: {path}")


def _normalise_ids(*frames: pd.DataFrame) -> None:
    for frame in frames:
        if "sample_id" in frame.columns:
            frame["sample_id"] = frame["sample_id"].astype(str)


def _validate_semantic_features(features: pd.DataFrame, *, backend: str, n_samples: int) -> dict[str, object]:
    if len(features) != n_samples:
        raise ValueError(f"{backend} feature rows={len(features)}; expected {n_samples}")
    if "sample_id" not in features.columns or features["sample_id"].duplicated().any():
        raise ValueError(f"{backend} features require unique sample_id")
    pm = build_semantic_ablation_features(features, "probability_mass", semantic_prefix=backend)
    if len(pm.columns) - 1 != PM_FEATURES:
        raise ValueError(f"{backend} ProbabilityMass feature count={len(pm.columns)-1}; expected {PM_FEATURES}")
    vals = pm.drop(columns="sample_id").to_numpy(float)
    if not np.isfinite(vals).all():
        raise ValueError(f"{backend} ProbabilityMass contains non-finite values")
    if vals.min() < -1e-9 or vals.max() > 1.0 + 1e-9:
        raise ValueError(f"{backend} ProbabilityMass contains values outside [0,1]")
    return {"rows": int(len(features)), "probability_mass_features": PM_FEATURES}


def validate_release(root: str | Path = ".", *, require_models: bool = True) -> dict[str, object]:
    paths = default_paths(root)
    required = [
        "perf_samples", "perf_outcomes", "jev_features", "laya_features", "gte_embeddings",
        "compact12_features", "cost_samples", "cost_outcomes", "cost_features",
        "jev_direct_predictions", "jev_direct_metrics",
    ]
    if require_models:
        required += ["jev_model", "jev_model_meta", "laya_model", "laya_model_meta"]
    for key in required:
        _require(paths[key])

    ps = pd.read_csv(paths["perf_samples"])
    po = pd.read_csv(paths["perf_outcomes"])
    jf = pd.read_csv(paths["jev_features"])
    lf = pd.read_csv(paths["laya_features"])
    cf12 = pd.read_csv(paths["compact12_features"])
    cs = pd.read_csv(paths["cost_samples"])
    co = pd.read_csv(paths["cost_outcomes"])
    cfeat = pd.read_csv(paths["cost_features"])
    jdp = pd.read_csv(paths["jev_direct_predictions"])
    _normalise_ids(ps, po, jf, lf, cf12, cs, co, cfeat, jdp)

    for name, frame, cols in [
        ("performance samples", ps, {"sample_id", "dataset", "domain", "query"}),
        ("performance outcomes", po, {"sample_id", "model", "score"}),
        ("cost samples", cs, {"sample_id", "dataset", "query"}),
        ("cost outcomes", co, {"sample_id", "model", "score", "cost"}),
    ]:
        missing = cols - set(frame.columns)
        if missing:
            raise ValueError(f"{name} missing columns: {sorted(missing)}")

    if len(ps) != PERF_SAMPLES or ps["dataset"].nunique() != PERF_DATASETS:
        raise ValueError("Performance sample pool does not match the frozen release")
    if len(po) != PERF_OUTCOMES or po["model"].nunique() != PERF_MODELS:
        raise ValueError("Performance outcome pool does not match the frozen release")
    if po.duplicated(["sample_id", "model"]).any():
        raise ValueError("Duplicate performance (sample_id, model) outcomes")
    if set(ps.sample_id) != set(po.sample_id):
        raise ValueError("Performance sample/outcome support mismatch")

    jev_info = _validate_semantic_features(jf, backend="jev", n_samples=PERF_SAMPLES)
    laya_info = _validate_semantic_features(lf, backend="laya", n_samples=PERF_SAMPLES)
    if set(ps.sample_id) != set(jf.sample_id) or set(ps.sample_id) != set(lf.sample_id):
        raise ValueError("Performance feature/sample support mismatch")

    if len(cf12) != PERF_SAMPLES or "sample_id" not in cf12.columns:
        raise ValueError("Compact-12 feature file does not match performance samples")

    emb_ids, emb, emb_meta = load_embeddings(paths["gte_embeddings"])
    if len(emb_ids) != PERF_SAMPLES or len(set(emb_ids)) != PERF_SAMPLES or set(emb_ids) != set(ps.sample_id):
        raise ValueError("GTE embedding support does not match performance samples")

    if len(cs) != COST_SAMPLES or cs["dataset"].nunique() != COST_DATASETS:
        raise ValueError("Cost sample pool does not match the frozen release")
    if len(co) != COST_OUTCOMES or co["model"].nunique() != COST_MODELS:
        raise ValueError("Cost outcome pool does not match the frozen release")
    if "gpt-5" not in set(co["model"].astype(str)):
        raise ValueError("Cost pool must contain the fixed GPT-5 reference")
    if len(cfeat) != COST_SAMPLES or "sample_id" not in cfeat.columns:
        raise ValueError("Cost feature file does not match cost samples")
    _validate_semantic_features(cfeat, backend="jev", n_samples=COST_SAMPLES)

    if len(jdp) != PERF_SAMPLES or jdp["sample_id"].duplicated().any():
        raise ValueError("Frozen JEV-Direct OOF predictions must contain one row per performance sample")
    if set(jdp.sample_id) != set(ps.sample_id):
        raise ValueError("JEV-Direct prediction support mismatch")

    model_info: dict[str, object] = {}
    if require_models:
        for backend in ("jev", "laya"):
            model_dir = Path(root) / "models" / backend
            router = CatBoostQualityRouter.load(model_dir)
            if len(router.feature_names) != PM_FEATURES or len(router.model_names) != PERF_MODELS:
                raise ValueError(f"{backend} pretrained router metadata does not match the release")
            if not all(c.startswith(f"{backend}_") for c in router.feature_names):
                raise ValueError(f"{backend} pretrained router feature prefix mismatch")
            model_info[backend] = {
                "feature_count": len(router.feature_names),
                "model_count": len(router.model_names),
            }

    return {
        "performance": {
            "samples": PERF_SAMPLES,
            "datasets": PERF_DATASETS,
            "models": PERF_MODELS,
            "outcomes": PERF_OUTCOMES,
            "jev": jev_info,
            "laya": laya_info,
            "gte_dimension": int(emb.shape[1]),
            "gte_model": emb_meta,
        },
        "cost": {
            "samples": COST_SAMPLES,
            "datasets": COST_DATASETS,
            "models": COST_MODELS,
            "outcomes": COST_OUTCOMES,
            "baseline": "gpt-5",
        },
        "pretrained_models": model_info,
        "hashes": {key: sha256_file(path) for key, path in paths.items() if path.exists() and path.is_file()},
    }


def write_manifest(root: str | Path = ".", output: str | Path = "artifacts/release_validation.json", *, require_models: bool = True) -> dict[str, object]:
    result = validate_release(root, require_models=require_models)
    target = Path(root) / output if not Path(output).is_absolute() else Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return result
