# Third-party benchmark data

SeLMRoute is released under Apache-2.0. That license covers SeLMRoute source code and original project material; it does **not** grant or replace rights in third-party benchmark content.

## LLMRouterBench

The experiments use frozen result artifacts from LLMRouterBench. Those artifacts aggregate multiple independently sourced benchmarks and model outputs. SeLMRoute therefore does not ship the raw benchmark row files used by the experiments.

Users obtain the applicable LLMRouterBench release themselves and run:

```bash
uv run selmroute prepare-data --bundle /path/to/bench-release.tar.gz
```

or:

```bash
uv run selmroute prepare-data --results-root /path/to/bench-release
```

The generated `.local_data/**/samples.csv` and `.local_data/**/outcomes.csv` remain local and are gitignored.

The frozen benchmark provenance used for the paper is:

- repository: `NPULH/LLMRouterBench`
- revision: `0e5af1b84bf73437a01a1849c0f1d2468baa93fc`
- bundle: `bench-release.tar.gz`
- SHA-256: `b79f8cde1a6f029c2efa663a3a3b6f7748defb22341fe59f328cebef6648c8f1`

## Constituent datasets

The performance-oriented SeLMRoute experiment uses AIME, BBH, EmoryNLP, FinQA, GPQA, HumanEval, K&K, KORBench, LiveCodeBench, MATH-500, MathBench, MBPP, MedQA, MELD, and MMLU-Pro.

The performance-cost experiment uses AIME, LiveMathBench, GPQA, HLE, LiveCodeBench, MMLU-Pro, SWE-bench, SimpleQA, Tau2, and ArenaHard.

These datasets have independent licenses, access conditions, and/or source-content rights. Users are responsible for complying with the terms that apply to the data they obtain. The existence of an open-source license on SeLMRoute or on benchmark tooling does not relicense third-party dataset content.

## SeLMRoute-derived artifacts

SeLMRoute may publish numerical artifacts produced by the research pipeline, including semantic feature matrices, embeddings, aggregate metrics, frozen routing predictions, and trained router parameters, provided the published versions contain no original benchmark question/answer/prompt text or raw model completions.

The intended public locations are `release_assets/` and `models/`. `RELEASE_ASSETS.md` lists the accepted files and migration procedure.

This file documents the project's technical distribution boundary. It is not a substitute for legal advice concerning a particular third-party dataset or model provider.
