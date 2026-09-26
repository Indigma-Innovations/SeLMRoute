from pathlib import Path

import yaml

from .schema import ProbeSet


def load_probe_set(path: str | Path) -> ProbeSet:
    with Path(path).open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    return ProbeSet.model_validate(data)


def dump_probe_set(probes: ProbeSet, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(probes.model_dump(mode="json"), fh, sort_keys=False, allow_unicode=True)
