from __future__ import annotations
import csv
import json
import os
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

# Default SRB trace columns (used if no steps were logged)
DEFAULT_TRACE_FIELDS = [
    "step",
    "token",
    "text_so_far",
    "entropy",
    "coherence",
    "resonance",
    "d_resonance",
    "temperature",
    "top_p",
]


class TraceLogger:
    """Handles per-step trace logging and summary output for SRB runs."""

    def __init__(self, outdir: str | Path):
        """
        Initialize TraceLogger.

        Args:
            outdir (str | Path): Directory to store trace and summary files.
        """
        self.outdir = Path(outdir)
        self.outdir.mkdir(parents=True, exist_ok=True)
        self.trace_path = self.outdir / "trace.csv"
        self.summary_path_file = self.outdir / "summary.json"
        # Ensure the trace file exists with a header early so downstream plotting doesn't fail
        self.trace_path.parent.mkdir(parents=True, exist_ok=True)
        _write_csv_header_if_needed(self.trace_path, DEFAULT_TRACE_FIELDS)
        self._step_rows: List[Dict[str, Any]] = []
        self._fieldnames: List[str] = []
        self._csv_file = None
        self._csv_writer = None

    def log_step(self, **fields: Any) -> None:
        """
        Log a single step's telemetry data.

        Args:
            **fields: Arbitrary keyword telemetry fields for the step.
        """
        # Round floats to 6 decimal places for compactness
        rounded_fields = {}
        for k, v in fields.items():
            if isinstance(v, float):
                rounded_fields[k] = round(v, 6)
            else:
                rounded_fields[k] = v

        if not self._fieldnames:
            self._fieldnames = list(rounded_fields.keys())
            _write_csv_header_if_needed(self.trace_path, self._fieldnames)
            self._csv_file = open(self.trace_path, "a", newline="", encoding="utf-8")
            self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=self._fieldnames)

        self._csv_writer.writerow(rounded_fields)
        self._csv_file.flush()

    def finalize(self, summary: Dict[str, Any]) -> None:
        """
        Finalize logging by writing the summary and closing files.

        Args:
            summary (Dict[str, Any]): Summary dictionary to save.
        """
        if self._csv_file is not None:
            self._csv_file.close()
            self._csv_file = None
            self._csv_writer = None

        # Ensure a trace file with header exists even if no steps were logged
        if not self.trace_path.exists() or self.trace_path.stat().st_size == 0:
            header = self._fieldnames if self._fieldnames else DEFAULT_TRACE_FIELDS
            _write_csv_header_if_needed(self.trace_path, header)

        save_summary(summary, self.summary_path_file)

    def summary_path(self) -> str:
        """
        Returns:
            str: Path to the summary JSON file.
        """
        return str(self.summary_path_file)


def _write_csv_header_if_needed(path: Path, fieldnames: List[str]) -> None:
    """
    Create CSV header if the file does not exist or is empty.

    Args:
        path (Path): Path to the CSV file.
        fieldnames (List[str]): List of column names.
    """
    if not path.exists() or path.stat().st_size == 0:
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()


def load_trace(path: str | Path) -> List[Dict[str, Any]]:
    """
    Load a trace CSV into a list of dictionaries.

    Args:
        path (str | Path): Path to the trace CSV file.

    Returns:
        List[Dict[str, Any]]: List of step telemetry dictionaries.
    """
    path = Path(path)
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = []
        for row in reader:
            # convert numeric fields if possible
            converted = {}
            for k, v in row.items():
                if v == "":
                    converted[k] = None
                else:
                    try:
                        fval = float(v)
                        if fval.is_integer():
                            converted[k] = int(fval)
                        else:
                            converted[k] = fval
                    except ValueError:
                        converted[k] = v
            rows.append(converted)
        return rows


def save_summary(summary: Dict[str, Any], path: str | Path) -> None:
    """
    Save the summary dictionary as a pretty JSON file with a timestamp.

    Args:
        summary (Dict[str, Any]): Summary data to save.
        path (str | Path): Path to the summary JSON file.
    """
    summary_copy = dict(summary)
    summary_copy["_saved_at"] = datetime.utcnow().isoformat() + "Z"
    path = Path(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary_copy, f, indent=2, sort_keys=True)
