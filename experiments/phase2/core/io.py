from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import json, csv, random

@dataclass
class RunConfig:
    seed: int
    output_dir: Path
    model_name: str
    max_new_tokens: int
    temperature: float
    top_p: float
    runs: int
    buckets: list[str]
    interventions: list[str]
    verifier_model: str
    metrics: dict

class IO:
    def __init__(self, cfg: RunConfig):
        self.cfg = cfg
        self.output_dir = cfg.output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def write_json(self, data: dict, name: str):
        path = self.output_dir / f"{name}.json"
        path.write_text(json.dumps(data, indent=2))
        return path

    def append_jsonl(self, data: dict, name: str):
        path = self.output_dir / f"{name}.jsonl"
        with path.open("a") as f:
            f.write(json.dumps(data) + "\n")
        return path

    def write_csv_rows(self, rows: list[dict], name: str):
        if not rows:
            return None
        path = self.output_dir / f"{name}.csv"
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        return path
