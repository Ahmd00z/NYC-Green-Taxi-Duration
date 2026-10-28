"""Mini Project 3, step 5: BATCH inference. Score a big Parquet file in chunks.

Batch means "no one is waiting for the answer". We read a file, score all rows, and
write a file. That lets us use big chunks, which is far cheaper per prediction than
answering one HTTP request at a time.

Memory stays flat: we never load the whole input. We read `chunk_size` rows, score them,
write them, forget them, and move on. So 1 million rows and 100 million rows need about
the same RAM.

Typical use (see reports/module-3.md for the numbers):

    # 1) build a 1.2M-row input from your real August file (for the throughput test)
    python -m prodml.batch --make-sample data/scoring/input/s.parquet --rows 1200000

    # 2) score it with the Production model from the MLflow registry
    python -m prodml.batch --input data/scoring/input/sample_1m.parquet \
        --output data/scoring/output --run-date 2019-08-31

    # or score it with the Module 1 ONNX model (no MLflow needed)
    python -m prodml.batch --input ... --output ... --source local --format onnx
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from prodml.config import settings
from prodml.logging_conf import setup_logging
from prodml.predict import DurationPredictor

logger = logging.getLogger("prodml.batch")

# An ASSUMPTION used for the cost estimate: what one small cloud VM costs per hour.
# Replace it with your own instance price (--instance-cost-per-hour).
DEFAULT_INSTANCE_COST_PER_HOUR = 0.10

# Columns we need from the raw trips file. Everything else is ignored.
INPUT_COLUMNS = ["PULocationID", "DOLocationID", "trip_distance"]


# ---------------------------------------------------------------- feature building
def features_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Raw trips -> the two model columns, for a whole chunk at once (vectorized).

    This must give EXACTLY what features.build_features gives one row at a time
    (tests/test_batch.py checks that), otherwise batch and web service would disagree.
    A missing pickup or drop-off zone becomes the text "unknown", like in the API.
    """
    # "Int64" is pandas' integer type that can hold missing values.
    pu_num = pd.to_numeric(df["PULocationID"], errors="coerce")
    do_num = pd.to_numeric(df["DOLocationID"], errors="coerce")
    # Build the text from the numbers, then replace the missing ones explicitly.
    pu = pu_num.astype("Int64").astype(str).mask(pu_num.isna(), "unknown")
    do = do_num.astype("Int64").astype(str).mask(do_num.isna(), "unknown")
    # Missing or negative distances become 0 so no model receives NaN.
    distance = pd.to_numeric(df["trip_distance"], errors="coerce").fillna(0.0)
    return pd.DataFrame(
        {
            "PU_DO": pu + "_" + do,
            "trip_distance": distance.clip(lower=0.0).astype(float),
        }
    )


# ---------------------------------------------------------------- measuring helpers
def peak_memory_mb() -> float | None:
    """Highest RAM this process has used so far, in MB (None where unsupported)."""
    try:
        import resource  # only exists on Linux and macOS
    except ImportError:
        return None
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux reports kilobytes, macOS reports bytes.
    return peak / 1024 if sys.platform != "darwin" else peak / (1024 * 1024)


def cost_per_million(
    rows_per_sec: float, instance_cost_per_hour: float = DEFAULT_INSTANCE_COST_PER_HOUR
) -> float:
    """Dollars to make 1,000,000 predictions at this speed on one paid-by-the-hour VM.

    Works for ANY serving style: pass the batch job's rows/sec, or the web service's
    requests/sec at saturation, and compare the two numbers.
    """
    if rows_per_sec <= 0:
        return float("inf")
    hours_for_a_million = (1_000_000 / rows_per_sec) / 3600
    return hours_for_a_million * instance_cost_per_hour


# ---------------------------------------------------------------- the batch job
def batch_score(
    input_path: str | Path,
    output_root: str | Path,
    run_date: str,
    chunk_size: int = 100_000,
    predictor: DurationPredictor | None = None,
    instance_cost_per_hour: float = DEFAULT_INSTANCE_COST_PER_HOUR,
) -> dict[str, Any]:
    """Score `input_path` in chunks and write
    output_root/run_date=<date>/part-00000.parquet.

    Re-running with the same run_date REPLACES that partition (it never appends), so the
    job is idempotent: Airflow can retry or backfill a day without creating duplicates.
    Returns a report with the numbers the PDF asks for (time, memory, cost per million).
    """
    predictor = predictor or DurationPredictor().load()
    partition = Path(output_root) / f"run_date={run_date}"
    partition.mkdir(parents=True, exist_ok=True)
    final_file = partition / "part-00000.parquet"
    # Write to a temporary name first, then rename: a crash never leaves a half file.
    temp_file = partition / "part-00000.parquet.tmp"

    parquet = pq.ParquetFile(str(input_path))
    total_rows = parquet.metadata.num_rows
    available = set(parquet.schema_arrow.names)
    wanted = INPUT_COLUMNS + (
        ["lpep_pickup_datetime"] if "lpep_pickup_datetime" in available else []
    )
    logger.info(
        "batch started",
        extra={"rows": total_rows, "chunk_size": chunk_size, "run_date": run_date},
    )

    writer: pq.ParquetWriter | None = None
    rows_written = 0
    start = time.perf_counter()
    # iter_batches reads the file piece by piece instead of all at once.
    for record_batch in parquet.iter_batches(batch_size=chunk_size, columns=wanted):
        chunk = record_batch.to_pandas()
        # Score the whole chunk in ONE call: this is the batch advantage.
        predictions = predictor.predict_frame(features_frame(chunk))
        chunk["predicted_duration_min"] = np.round(predictions, 4)
        chunk["run_date"] = run_date
        table = pa.Table.from_pandas(chunk, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(str(temp_file), table.schema)
        writer.write_table(table)
        rows_written += len(chunk)
    if writer is not None:
        writer.close()
    os.replace(temp_file, final_file) if writer is not None else None

    elapsed = time.perf_counter() - start
    rows_per_sec = rows_written / elapsed if elapsed > 0 else float("inf")
    report = {
        "rows": rows_written,
        "chunk_size": chunk_size,
        "run_date": run_date,
        "model_version": predictor.get_metadata().get("model_version"),
        "serving_format": predictor.serving_format,
        "wall_clock_seconds": round(elapsed, 3),
        "rows_per_second": round(rows_per_sec, 1),
        "peak_memory_mb": round(peak_memory_mb() or 0.0, 1),
        "instance_cost_per_hour_usd": instance_cost_per_hour,
        "cost_per_million_usd": round(
            cost_per_million(rows_per_sec, instance_cost_per_hour), 6
        ),
        "output": str(final_file),
    }
    # The leading underscore keeps Parquet readers from treating this file as data.
    (partition / "_report.json").write_text(json.dumps(report, indent=2) + "\n")
    logger.info("batch finished", extra=report)
    return report


# ---------------------------------------------------------------- sample data
def make_sample(
    out_path: str | Path,
    rows: int,
    source_path: str | Path | None = None,
    seed: int = 0,
) -> str:
    """Make a big input file by re-sampling the real trips (for the 1M-row test only).

    Rows are drawn WITH replacement and the distance gets a little noise, so the file
    looks like real traffic. It measures speed and memory. Its predictions mean nothing.
    """
    source_path = str(source_path or settings.data_path)
    base = pd.read_parquet(
        source_path, columns=INPUT_COLUMNS + ["lpep_pickup_datetime"]
    )
    rng = np.random.default_rng(seed)
    sample = base.iloc[rng.integers(0, len(base), rows)].reset_index(drop=True)
    sample["trip_distance"] = (
        sample["trip_distance"] * rng.uniform(0.95, 1.05, rows)
    ).round(2)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    sample.to_parquet(out_path, index=False)
    return str(out_path)


# ---------------------------------------------------------------- command line
def main(argv: list[str] | None = None) -> int:
    setup_logging()
    parser = argparse.ArgumentParser(prog="prodml.batch")
    parser.add_argument("--make-sample", help="write a re-sampled input file and exit")
    parser.add_argument(
        "--rows", type=int, default=1_200_000, help="rows for --make-sample"
    )
    parser.add_argument("--input", help="Parquet file with the trips to score")
    parser.add_argument("--output", default="data/scoring/output", help="output folder")
    parser.add_argument("--run-date", default=time.strftime("%Y-%m-%d"))
    parser.add_argument("--chunk-size", type=int, default=100_000)
    parser.add_argument("--source", choices=["registry", "local"], default=None)
    parser.add_argument(
        "--format", choices=["onnx", "pickle", "openvino"], default=None
    )
    parser.add_argument(
        "--instance-cost-per-hour", type=float, default=DEFAULT_INSTANCE_COST_PER_HOUR
    )
    args = parser.parse_args(argv)

    if args.make_sample:
        sys.stdout.write(make_sample(args.make_sample, args.rows) + "\n")
        return 0
    if not args.input:
        parser.error("--input is required (or use --make-sample)")

    if args.format:
        settings.serving_format = args.format  # type: ignore[assignment]
    predictor = DurationPredictor(source=args.source).load()
    report = batch_score(
        args.input,
        args.output,
        args.run_date,
        args.chunk_size,
        predictor,
        args.instance_cost_per_hour,
    )
    sys.stdout.write(json.dumps(report, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
