import asyncio
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv

from selmroute.config import load_probe_set
from selmroute.features.jev import encode_answer
from selmroute.inference import load_router, rank_from_features
from selmroute.jev.client import AsyncJevClient


def _answers_to_frame(
    answers: dict[str, dict[str, Any]],
    *,
    backend: str,
) -> pd.DataFrame:
    """Convert typed semantic answers into the flat feature representation."""

    row: dict[str, object] = {
        "sample_id": "live-query",
    }

    for qid, answer in answers.items():
        row.update(
            encode_answer(
                f"{backend}_{qid}",
                answer,
            )
        )

    return pd.DataFrame([row])


def _jsonable(value: Any) -> Any:
    """Conversion for trace/debug output.

    The helper is used only for presentation. It does not
    alter the object passed to either semantic backend.
    """

    if value is None:
        return None

    if isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, dict):
        return {
            str(k): _jsonable(v)
            for k, v in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [
            _jsonable(v)
            for v in value
        ]

    if hasattr(value, "model_dump"):
        try:
            return _jsonable(
                value.model_dump()
            )
        except Exception:
            pass

    if hasattr(value, "dict"):
        try:
            return _jsonable(
                value.dict()
            )
        except Exception:
            pass

    if hasattr(value, "__dict__"):
        try:
            return {
                str(k): _jsonable(v)
                for k, v in vars(value).items()
                if not str(k).startswith("_")
            }
        except Exception:
            pass

    return str(value)


def semantic_trace_from_features(
    feature_values: pd.Series,
    *,
    backend: str,
) -> dict[str, object]:
    """Convert the 40 flat PM features into a human-readable probe view."""

    prefix = f"{backend}_"
    result: dict[str, object] = {}

    for name, raw_value in feature_values.items():
        if not name.startswith(prefix):
            continue

        value = float(raw_value)
        semantic_name = name[len(prefix):]

        if "__noul" in semantic_name:
            probe = semantic_name.replace(
                "__noul",
                "",
            )

            result[probe] = {
                "type": "binary",
                "probability_yes": value,
                "probability_no": 1.0 - value,
            }

        elif "__p__" in semantic_name:
            probe, level = semantic_name.split(
                "__p__",
                1,
            )

            if probe not in result:
                result[probe] = {
                    "type": "score",
                    "probabilities": {},
                }

            result[probe]["probabilities"][
                int(level)
            ] = value

    return result


async def _jev_answers(
    query: str,
    *,
    probes_path: Path,
    env_file: Path | None,
    return_trace: bool = False,
):
    """Run the live TypeSafe JEV semantic extractor."""

    load_dotenv(
        dotenv_path=env_file,
        override=False,
    )

    probes = load_probe_set(
        probes_path
    )

    base_url = os.getenv(
        "TYPESAFE_BASE_URL",
        "https://api.typesafe.ai",
    )

    # These are the semantic arguments supplied to
    # AsyncJevClient.evaluate().
    #
    # API credentials are intentionally excluded.
    backend_trace = {
        "backend": "jev",
        "query": str(query),
        "base_url": base_url,
        "probes_path": str(probes_path),
        "request": {
            "query": str(query),
            "probes": _jsonable(probes),
        },
        "actual_call": (
            "AsyncJevClient.evaluate(query, probes)"
        ),
    }

    async with AsyncJevClient(
        base_url=base_url,
        concurrency=1,
        cache=None,
    ) as client:
        response, cached, latency = (
            await client.evaluate(
                str(query),
                probes,
            )
        )

    answers = response.answers

    if return_trace:
        backend_trace["response_metadata"] = {
            "cached": bool(cached),
            "latency_ms": (
                float(latency)
                if latency is not None
                else None
            ),
        }

        return answers, backend_trace

    return answers


def _laya_answers(
    query: str,
    *,
    probes_path: Path,
    model: str,
    revision: str | None,
    subfolder: str | None,
    device: str | None,
    return_trace: bool = False,
):
    """Run the live local/open-weight Laya semantic extractor."""

    try:
        import laya
    except ImportError as exc:
        raise RuntimeError(
            "Live Laya inference is optional. "
            "Install it with: uv sync --extra live-laya"
        ) from exc

    from selmroute.laya.evaluate import (
        _questions_from_probes,
        _resolve_laya_model,
    )

    probes = load_probe_set(
        probes_path
    )

    model_path, resolved_revision = (
        _resolve_laya_model(
            model,
            revision,
            subfolder,
        )
    )

    questions = _questions_from_probes(
        probes
    )

    # These are the exact two arguments subsequently supplied to
    # agent.predict().
    laya_input = {
        "query": str(query),
    }

    backend_trace = {
        "backend": "laya",
        "model": model,
        "resolved_model_path": str(model_path),
        "requested_revision": revision,
        "resolved_revision": resolved_revision,
        "subfolder": subfolder,
        "device": device,
        "query": str(query),
        "probes_path": str(probes_path),
        "request": {
            "input": _jsonable(laya_input),
            "questions": _jsonable(questions),
        },
        "actual_call": (
            "agent.predict(input, questions)"
        ),
    }

    agent = laya.load(
        model_path,
        device=device,
        subfolder=subfolder,
    )

    response = agent.predict(
        laya_input,
        questions,
    )

    answers = response.get(
        "answers",
        {},
    )

    if return_trace:
        return answers, backend_trace

    return answers


def _print_probe_view(
    probe_view: dict[str, object],
) -> None:
    """Pretty-print the 16 semantic judgments."""

    for probe, info in probe_view.items():
        probe_type = info["type"]

        if probe_type == "binary":
            yes = info["probability_yes"]
            no = info["probability_no"]

            print(
                f"{probe:<28} "
                f"P(yes)={yes:.4f}  "
                f"P(no)={no:.4f}"
            )

        elif probe_type == "score":
            probabilities = (
                info["probabilities"]
            )

            values = "  ".join(
                f"L{level}={prob:.4f}"
                for level, prob
                in sorted(
                    probabilities.items()
                )
            )

            print(
                f"{probe:<28} {values}"
            )


def _print_live_trace(
    trace: dict[str, object],
) -> None:
    """Print an inspectable end-to-end live routing trace."""

    backend_call = trace[
        "backend_call"
    ]

    print()
    print("=" * 88)
    print("1. SEMANTIC BACKEND CALL")
    print("=" * 88)

    print(
        "Backend:",
        backend_call["backend"],
    )

    if backend_call.get("model"):
        print(
            "Model:",
            backend_call["model"],
        )

    if backend_call.get(
        "resolved_revision"
    ):
        print(
            "Revision:",
            backend_call[
                "resolved_revision"
            ],
        )

    if backend_call.get(
        "device"
    ) is not None:
        print(
            "Device:",
            backend_call["device"],
        )

    print("\nQuery:")
    print(
        backend_call["query"]
    )

    print("\nExact semantic-backend input:")
    print(
        json.dumps(
            backend_call["request"],
            indent=2,
            ensure_ascii=False,
            default=str,
        )
    )

    print(
        "\nActual backend call:",
        backend_call.get(
            "actual_call",
            "",
        ),
    )

    response_metadata = (
        backend_call.get(
            "response_metadata"
        )
    )

    if response_metadata:
        print(
            "\nBackend response metadata:"
        )
        print(
            json.dumps(
                response_metadata,
                indent=2,
                ensure_ascii=False,
                default=str,
            )
        )

    print()
    print("=" * 88)
    print("2. RAW SEMANTIC BACKEND OUTPUT")
    print("=" * 88)

    print(
        json.dumps(
            _jsonable(
                trace[
                    "semantic_output"
                ]["answers"]
            ),
            indent=2,
            ensure_ascii=False,
            default=str,
        )
    )

    print()
    print("=" * 88)
    print("3. INTERPRETABLE SEMANTIC STATE")
    print("=" * 88)

    print(
        "Representation:",
        trace[
            "semantic_output"
        ]["representation"],
    )

    print(
        "CatBoost feature count:",
        trace[
            "semantic_output"
        ]["feature_count"],
    )

    print()

    _print_probe_view(
        trace[
            "semantic_output"
        ]["probe_view"]
    )

    print()
    print("=" * 88)
    print("4. EXACT CATBOOST INPUT")
    print("=" * 88)

    cb = trace[
        "catboost_input"
    ]

    print(
        "ProbabilityMass features:",
        cb["feature_count"],
    )

    print()

    for i, (name, value) in enumerate(
        zip(
            cb["feature_order"],
            cb["vector"],
            strict=True,
        )
    ):
        print(
            f"{i:02d}  "
            f"{name:<48} "
            f"{value:.6f}"
        )

    print()
    print("=" * 88)
    print("5. CATBOOST OUTPUT")
    print("=" * 88)

    for i, item in enumerate(
        trace[
            "catboost_output"
        ]["ranking"],
        start=1,
    ):
        predicted_score = item.get(
            "predicted_score"
        )

        if predicted_score is None:
            score_text = "n/a"
        else:
            score_text = (
                f"{float(predicted_score):.6f}"
            )

        print(
            f"{i:2d}. "
            f"{item['model']:<45} "
            f"predicted_score={score_text}"
        )

    print(
        "\nSelected model:",
        trace[
            "catboost_output"
        ]["selected_model"],
    )


def route_live(
    query: str,
    *,
    backend: str = "laya",
    root: str | Path = ".",
    top_k: int = 5,
    env_file: str | Path | None = ".env",
    laya_model: str = "convaiinnovations/laya",
    laya_revision: str | None = None,
    laya_subfolder: str | None = None,
    device: str | None = None,
    verbose: bool = False,
) -> dict[str, object]:
    """Route one unseen query through Laya or TypeSafe JEV.

    Laya is the default live semantic backend.

    Offline paper reproduction never calls this function.

    When verbose=True, the result exposes the complete inference
    pipeline:

        query
          -> actual semantic-backend request
          -> raw backend semantic answers
          -> 16 human-readable semantic judgments
          -> exact 40-dimensional ProbabilityMass vector
          -> released backend-specific CatBoost router
          -> candidate-model ranking

    TypeSafe credentials are loaded from ``env_file`` through
    python-dotenv. Credentials, environment variables, authorization
    headers, and API keys are never included in the returned trace.
    """

    query = str(query)

    if not query.strip():
        raise ValueError(
            "query must not be empty"
        )

    backend = (
        backend
        .strip()
        .lower()
    )

    if backend not in {
        "laya",
        "jev",
    }:
        raise ValueError(
            "backend must be 'laya' or 'jev'"
        )

    if top_k < 1:
        raise ValueError(
            "top_k must be >= 1"
        )

    root = Path(root)

    probes_path = (
        root
        / "configs"
        / "probes_manual_v1.yaml"
    )

    if not probes_path.exists():
        raise FileNotFoundError(
            f"Probe configuration not found: "
            f"{probes_path}"
        )

    # Load first so we fail immediately if the matching
    # deployment checkpoint is absent.
    router = load_router(
        backend,
        root=root,
    )

    if backend == "jev":
        answers, backend_trace = (
            asyncio.run(
                _jev_answers(
                    query,
                    probes_path=probes_path,
                    env_file=(
                        None
                        if env_file is None
                        else Path(env_file)
                    ),
                    return_trace=True,
                )
            )
        )

    else:
        # Laya itself does not require the TypeSafe key, but loading
        # .env here keeps optional local configuration consistent.
        load_dotenv(
            dotenv_path=(
                None
                if env_file is None
                else Path(env_file)
            ),
            override=False,
        )

        answers, backend_trace = (
            _laya_answers(
                query,
                probes_path=probes_path,
                model=laya_model,
                revision=laya_revision,
                subfolder=laya_subfolder,
                device=device,
                return_trace=True,
            )
        )

    if not answers:
        raise RuntimeError(
            f"{backend} returned no semantic answers"
        )

    semantic = _answers_to_frame(
        answers,
        backend=backend,
    )

    # The released CatBoost checkpoint fixes both the features
    # and their order. Live semantic extraction must exactly
    # satisfy that contract.
    missing = [
        feature
        for feature
        in router.feature_names
        if feature
        not in semantic.columns
    ]

    if missing:
        raise ValueError(
            "Semantic backend did not produce all features "
            f"required by the {backend} router. "
            f"Missing: {missing}"
        )

    catboost_input = (
        semantic[
            router.feature_names
        ]
        .iloc[0]
        .astype(float)
    )

    if catboost_input.isna().any():
        bad = (
            catboost_input[
                catboost_input.isna()
            ]
            .index
            .tolist()
        )

        raise ValueError(
            "Semantic backend produced NaN values "
            f"for CatBoost features: {bad}"
        )

    probe_view = (
        semantic_trace_from_features(
            catboost_input,
            backend=backend,
        )
    )

    ranking = rank_from_features(
        semantic,
        backend=backend,
        root=root,
        top_k=top_k,
    )

    if not ranking:
        raise RuntimeError(
            "Router returned an empty ranking"
        )

    result: dict[str, object] = {
        "query": query,
        "backend": backend,
        "selected_model": (
            ranking[0]["model"]
        ),
        "ranking": ranking,
        "semantic_features": {
            name: float(
                catboost_input[name]
            )
            for name
            in router.feature_names
        },
    }

    if verbose:
        trace = {
            "backend_call": (
                backend_trace
            ),

            "semantic_output": {
                "answers": answers,
                "representation": (
                    "ProbabilityMass"
                ),
                "probe_count": len(
                    probe_view
                ),
                "feature_count": len(
                    router.feature_names
                ),
                "probe_view": probe_view,
            },

            "catboost_input": {
                "feature_count": len(
                    router.feature_names
                ),
                "feature_order": list(
                    router.feature_names
                ),
                "vector": [
                    float(
                        catboost_input[
                            name
                        ]
                    )
                    for name
                    in router.feature_names
                ],
                "features": {
                    name: float(
                        catboost_input[
                            name
                        ]
                    )
                    for name
                    in router.feature_names
                },
            },

            "catboost_output": {
                "selected_model": (
                    ranking[0]["model"]
                ),
                "ranking": ranking,
            },
        }

        result["trace"] = trace

        _print_live_trace(
            trace
        )

    return result
