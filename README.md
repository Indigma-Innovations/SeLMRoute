# SeLMRoute

SeLMRoute studies multi-LLM routing through an explicit probabilistic semantic state. A typed decision model describes a query through 16 interpretable judgments, the resulting 40-dimensional ProbabilityMass representation is passed to a supervised performance predictor, and a routing policy selects a candidate model.

The default workflow is fully offline. **Paper reproduction makes no TypeSafe/JEV calls, downloads no Laya model, and requires no API key.** It uses the frozen semantic features and embeddings released with the paper.

## What is included

- Duplicate-query-safe grouped train/test and grouped OOF evaluation.
- ProbabilityMass, Full, Hard, Primary, Primary+Entropy, and No-Entropy semantic ablations.
- CatBoost, Random Forest, MLP, Ridge, and OLS performance-learner ablations.
- Frozen GTE-Qwen2 embedding baseline.
    - You must download the `.npz` file from this [link](https://drive.google.com/file/d/1tLSgISiftPtt6qGkgNUfbmLkMjSBtcNm/view?usp=sharing) and put it under `/data/performance/`
- JEV vs. Laya semantic decision model evaluation from frozen features.
- Nested Lite-12 probe selection and noninferiority analysis.
- Leave-one-probe-out analysis.
- Dataset-OOD and Domain-OOD evaluation.
- Selective-routing / semantic-entropy analysis.
- Final validation-selected performance-cost protocol with GPT-5 as the fixed reference.
- Frozen JEV-Direct predictions. JEV-Direct is **not** re-queried.
- Two released CatBoost deployment routers, one fitted to JEV semantic features and one fitted to Laya semantic features.
- Three notebooks:
  1. paper reproduction,
  2. offline use of a released router on published semantic features,
  3. optional live routing of a new query through Laya or JEV.

## 1. Environment

The release is frozen for Python 3.13, and the default dependencies are pinned to exact versions in `pyproject.toml`.

```bash
uv sync --extra dev --extra notebooks
uv run pytest -q
uv run selmroute --help
```

For a pure command-line reproduction without notebooks:

```bash
uv sync --extra dev
```

The optional live Laya backend is installed separately:

```bash
uv sync --extra live-laya --extra notebooks
```

No Laya dependency is needed for paper reproduction.

## 2. Frozen-data integrity checks

Before running an experiment:

```bash
uv run selmroute validate
```

The validator requires the following performance-oriented pool:

```text
samples     11,481
datasets        15
models          20
outcomes   229,620
JEV ProbabilityMass features   40
Laya ProbabilityMass features  40
```

and the final performance-cost pool:

```text
samples     12,446
datasets        10
models          13
outcomes   161,520
fixed reference model: gpt-5
```

It also verifies sample support for both semantic backends, the GTE embedding artifact, frozen JEV-Direct predictions, and both pretrained CatBoost bundles. A SHA-256 inventory is written to:

```text
artifacts/release_validation.json
```

Do not proceed if validation fails.

## 3. Reproduce the paper from frozen data

Run:

```bash
uv run selmroute reproduce
```

The command performs no network calls and never invokes JEV or Laya. It writes all generated results under:

```text
artifacts/reproduction/
```

Important outputs include:

```text
artifacts/reproduction/
├── validation.json
├── summary.json
├── performance_grouped/
│   └── paper/
│       ├── main_table/table1_complete.csv
│       ├── representation_grouped/summary.csv
│       ├── model_ablation_grouped/model_ablation_5seed_summary.csv
│       ├── cv_grouped/probability_mass/oof_metrics.json
│       ├── cv_grouped/hard/oof_metrics.json
│       ├── laya_grouped/cv/oof_metrics.json
│       ├── nested_lite12_grouped/
│       ├── leave_one_probe_out/
│       └── stats/
├── ood/summary.csv
├── selective/
├── calibrated_domain_ood/
├── performance_cost/
├── frozen_jev_direct/metrics.json
└── profiles/
```

### Expected headline checks

Small floating-point differences can occur across hardware, but the frozen environment should reproduce the reported values closely. The main checks are:

```text
Five-seed ProbabilityMass AvgAcc     72.0781
Grouped OOF ProbabilityMass AvgAcc   72.6360
Grouped OOF Hard AvgAcc              71.8097
Grouped OOF Laya AvgAcc              70.5339
Best Single in grouped OOF           69.2267

PM - Hard macro difference           +0.8262 pp
PM - Hard permutation p              0.1099

Nested Lite-12 AvgAcc                72.4914
Full-16 OOF AvgAcc                   72.6360
Nested Lite - Full                   -0.1446 pp

Mean performance-cost PerfGain       about +2.66%
Positive PerfGain splits             5 / 5
Positive strict CostSave splits      0 / 5
```

The public reproduction recomputes the frozen JEV-Direct aggregate from its released OOF predictions. It does not make new JEV-Direct API calls because the original candidate profiles were fold-specific and the experiment is intentionally frozen.

## 4. Notebook 1: reproduce the experiments

The notebook validates the artifact bundle, runs `reproduce_all()`, displays the headline tables, and generates figures. It is equivalent to the CLI workflow and makes no network calls.

## 5. Notebook 2: use the released CatBoost router offline

This notebook demonstrates the deployment model without JEV or Laya. It takes a query already present in the published benchmark, loads its frozen semantic features, and passes them to the released CatBoost router.

The default backend is Laya:

```python
from selmroute.inference import route_published_sample

result = route_published_sample(
  sample_id="...",
  backend="laya",
  root="..",
)
```

Set `backend="jev"` to use the JEV-trained CatBoost checkpoint and the corresponding frozen JEV features.

The two checkpoints are separate because JEV and Laya produce different probability distributions even under the same semantic schema.

## 6. Notebook 3: optional live routing of a new query

Live routing is intentionally separate from the reproducibility path.

### Laya, default

Install the optional backend:

```bash
uv sync --extra live-laya --extra notebooks
```
The notebook defaults to:

```python
backend = "laya"
```

The flow is:

```text
new query
  -> live Laya semantic judgments
  -> 40 Laya ProbabilityMass features
  -> released Laya CatBoost router
  -> ranked candidate LLMs
```

The Laya model is downloaded by the Laya/Hugging Face stack when it is not already available locally. `HF_TOKEN` can be placed in `.env` if needed.

### TypeSafe JEV

Copy the example environment file:

```bash
cp .env.example .env
```

Set:

```dotenv
TYPESAFE_API_KEY=your_key_here
```

The code reads `.env` with `python-dotenv`.

In the notebook use:

```python
backend = "jev"
```

The flow becomes:

```text
new query
  -> TypeSafe JEV
  -> 40 JEV ProbabilityMass features
  -> released JEV CatBoost router
  -> ranked candidate LLMs
```

Equivalent CLI examples are:

```bash
uv run selmroute live-route \
  --backend laya \
  --query "Implement a parser that preserves comments and rejects duplicate keys."
```

or:

```bash
uv run selmroute live-route \
  --backend jev \
  --query "Implement a parser that preserves comments and rejects duplicate keys."
```

The second command requires `TYPESAFE_API_KEY` in `.env`.

## 7. Released deployment checkpoints versus paper evaluation

The two models under `models/` are trained on the complete 11,481-query performance-oriented pool. Their purpose is practical inference after publication.

They are **not evaluation checkpoints**. Paper performance is obtained exclusively by the grouped train/test and grouped OOF procedures in `selmroute reproduce`.

The released checkpoints deliberately use all available observations, including repeated router-input groups, because no held-out performance claim is made from them.

## 8. Cost definition

The performance-cost experiment follows LLMRouterBench. `cost` means candidate-model API inference cost in U.S. dollars, derived from recorded input/output token usage and model-specific prices.

The benchmark metric excludes the semantic-extraction call so that PerfGain and CostSave remain benchmark-comparable. The release separately profiles the frozen JEV input-token and observed-latency metadata from the semantic feature files.

GPT-5 is fixed as the reference model because it is the Best Single model in the final 13-model flagship pool and is the reference used by the benchmark protocol. It is not selected independently for each test split.


## 9. Reproducibility boundary

Everything in Sections 3--5 is expected to run in a fresh environment with no external model or API access once the listed release artifacts are present.

`03_live_routing.ipynb` and `selmroute live-route` are the only workflows that intentionally require an external semantic backend. They are demonstration/inference paths and are not required to reproduce the paper.

## License

Apache-2.0. See `LICENSE`.
