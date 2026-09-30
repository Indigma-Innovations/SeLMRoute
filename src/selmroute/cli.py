from pathlib import Path

import pandas as pd
import typer
from rich.console import Console

from selmroute.data.prepare import prepare_benchmark_data
from selmroute.inference import route_published_sample
from selmroute.live import route_live
from selmroute.paths import benchmark_data_root
from selmroute.reproduce import reproduce_all
from selmroute.validation import write_manifest

app = typer.Typer(add_completion=False, no_args_is_help=True, help="SeLMRoute public reproduction and inference CLI")
console = Console()


@app.command("prepare-data")
def prepare_data_cmd(
    root: Path = typer.Option(Path("."), exists=True, file_okay=False, help="SeLMRoute repository root."),
    results_root: Path | None = typer.Option(None, exists=True, file_okay=False, help="Extracted LLMRouterBench bench-release directory."),
    bundle: Path | None = typer.Option(None, exists=True, dir_okay=False, help="Pinned LLMRouterBench bench-release.tar.gz."),
    output_root: Path | None = typer.Option(None, help="Local output root. Defaults to ./data."),
    verify_bundle: bool = typer.Option(True, "--verify-bundle/--no-verify-bundle", help="Verify the frozen bundle SHA-256."),
    keep_extracted_source: bool = typer.Option(False, help="Keep bundle extraction under OUTPUT_ROOT/_source."),
):
    """Rebuild samples.csv/outcomes.csv locally from LLMRouterBench.

    Exactly one of --bundle or --results-root is required. Benchmark rows are
    never downloaded by SeLMRoute and are written only to the gitignored local
    data root.
    """
    if (results_root is None) == (bundle is None):
        raise typer.BadParameter("Provide exactly one of --bundle or --results-root")
    target = output_root or benchmark_data_root(root)
    result = prepare_benchmark_data(
        output_root=target,
        results_root=results_root,
        bundle=bundle,
        verify_bundle=verify_bundle,
        keep_extracted_source=keep_extracted_source,
    )
    console.print_json(data=result)


@app.command("validate")
def validate_cmd(
    root: Path = typer.Option(Path("."), exists=True, file_okay=False),
    output: Path = typer.Option(Path("artifacts/release_validation.json")),
    require_models: bool = typer.Option(True, help="Require both pretrained CatBoost router bundles."),
):
    """Validate the complete frozen public-release artifact layout."""
    result = write_manifest(root, output, require_models=require_models)
    console.print_json(data=result)


@app.command("reproduce")
def reproduce_cmd(
    root: Path = typer.Option(Path("."), exists=True, file_okay=False),
    output_dir: Path = typer.Option(Path("artifacts/reproduction")),
):
    """Re-run the paper experiments from frozen data. Makes no JEV/Laya calls."""
    result = reproduce_all(root, output_dir)
    console.print_json(data={
        "status": "ok",
        "output_dir": str(output_dir),
        "note": result["note"],
    })


@app.command("route-published")
def route_published_cmd(
    sample_id: str = typer.Option(...),
    backend: str = typer.Option("laya", help="laya or jev"),
    root: Path = typer.Option(Path("."), exists=True, file_okay=False),
    top_k: int = typer.Option(5, min=1, max=20),
):
    """Use a released CatBoost router with precomputed features for one published query."""
    console.print_json(data=route_published_sample(sample_id, backend=backend, root=root, top_k=top_k))


@app.command("live-route")
def live_route_cmd(
    query: str = typer.Option(...),
    backend: str = typer.Option("laya", help="laya (default) or jev"),
    root: Path = typer.Option(Path("."), exists=True, file_okay=False),
    top_k: int = typer.Option(5, min=1, max=20),
    env_file: Path = typer.Option(Path(".env"), help="Used for TYPESAFE_API_KEY and optional HF_TOKEN."),
    device: str | None = typer.Option(None, help="Optional Laya device, e.g. cuda or cpu."),
):
    """Route one unseen query through a live semantic backend and a released CatBoost router."""
    result = route_live(query, backend=backend, root=root, top_k=top_k, env_file=env_file, device=device)
    console.print_json(data=result)


@app.command("show-sample")
def show_sample_cmd(
    row: int = typer.Option(0, min=0),
    root: Path = typer.Option(Path("."), exists=True, file_okay=False),
):
    """Print a sample_id/query pair for the pretrained-router notebook or CLI demo."""
    frame = pd.read_csv(root / "data/performance/samples.csv", dtype={"sample_id": str})
    if row >= len(frame):
        raise typer.BadParameter(f"row must be < {len(frame)}")
    console.print_json(data=frame.iloc[row].to_dict())


if __name__ == "__main__":
    app()
