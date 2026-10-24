"""Mini Project 3, step 7b: faster inference runtimes for the Module 1 linear model.

Three ways to run the SAME fitted LinearRegression:

  pickle    scikit-learn "eager" runtime (the baseline, what FastAPI used in Module 1)
  onnx      ONNX Runtime with all graph optimizations (models/model.onnx from Module 1)
  openvino  Intel OpenVINO runtime, loaded from an IR file (models/openvino/model.xml)

Why a separate "core ops" ONNX file for OpenVINO? skl2onnx writes a linear model as one
`LinearRegressor` node, which belongs to the ONNX-ML extension. OpenVINO only
understands
the core ONNX operators, so it cannot read that file. A linear model is just
y = X @ w + b, so we write that as plain MatMul + Add nodes. Same maths, same numbers.

Commands (run after `python -m prodml.train --local` has created models/):

    python -m prodml.runtimes export       # models/model_core.onnx + models/openvino/
    python -m prodml.runtimes benchmark    # latency + accuracy table for the 3 runtimes
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from prodml.config import settings
from prodml.logging_conf import setup_logging

logger = logging.getLogger("prodml.runtimes")

# Where the OpenVINO files live, relative to the models directory.
OPENVINO_SUBDIR = "openvino"
OPENVINO_XML = "model.xml"
CORE_ONNX_NAME = "model_core.onnx"


# ---------------------------------------------------------------------------- export
def export_core_onnx(model: Any, num_features: int, path: str | Path) -> str:
    """Write a fitted LinearRegression as an ONNX graph made of core operators only.

    Graph:  float_input [batch, num_features]  ->  MatMul(W)  ->  Add(b)  ->  variable
    """
    # onnx is only needed for exporting, so it is imported here and not at the top.
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    # coef_ has shape [num_features]; MatMul wants [num_features, 1].
    weights = np.asarray(model.coef_, dtype=np.float32).reshape(num_features, 1)
    bias = np.asarray([model.intercept_], dtype=np.float32)

    # The graph input has a dynamic batch size (None) so 1 row or 1000 rows both work.
    x = helper.make_tensor_value_info(
        "float_input", TensorProto.FLOAT, [None, num_features]
    )
    y = helper.make_tensor_value_info("variable", TensorProto.FLOAT, [None, 1])
    nodes = [
        helper.make_node("MatMul", ["float_input", "W"], ["xw"]),
        helper.make_node("Add", ["xw", "b"], ["variable"]),
    ]
    graph = helper.make_graph(
        nodes,
        "linear_regression_core_ops",
        [x],
        [y],
        initializer=[
            numpy_helper.from_array(weights, "W"),
            numpy_helper.from_array(bias, "b"),
        ],
    )
    onnx_model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    onnx_model.ir_version = 9  # old enough for every runtime we use
    onnx.checker.check_model(onnx_model)
    Path(path).write_bytes(onnx_model.SerializeToString())
    return str(path)


def convert_to_openvino(
    onnx_path: str | Path, out_dir: str | Path, fp16: bool = False
) -> str:
    """Convert an ONNX file to OpenVINO IR (model.xml + model.bin) and return the xml
    path.

    fp16=True stores the weights as 16-bit floats (smaller file, tiny accuracy change).
    """
    import openvino as ov

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    xml_path = out_dir / OPENVINO_XML
    ov_model = ov.convert_model(str(onnx_path))
    ov.save_model(ov_model, str(xml_path), compress_to_fp16=fp16)
    return str(xml_path)


def export_all(models_dir: str | Path, fp16: bool = False) -> dict[str, str]:
    """Create models/model_core.onnx and models/openvino/ from models/model.pkl."""
    models_dir = Path(models_dir)
    # model.pkl holds (DictVectorizer, LinearRegression), written by prodml.train
    # --local.
    with open(models_dir / settings.pickle_name, "rb") as f:  # a file WE produced
        dv, lr = pickle.load(f)
    core = export_core_onnx(lr, len(dv.feature_names_), models_dir / CORE_ONNX_NAME)
    xml = convert_to_openvino(core, models_dir / OPENVINO_SUBDIR, fp16=fp16)
    return {"core_onnx": core, "openvino_xml": xml}


# ---------------------------------------------------------------------------- runtime
class OpenVINOSession:
    """A tiny wrapper so OpenVINO looks like the other runtimes: run(dense_float32)."""

    def __init__(self, xml_path: str | Path, device: str = "CPU") -> None:
        import openvino as ov

        core = ov.Core()
        # compile_model turns the IR into an executable for the chosen device (plugin).
        self._compiled = core.compile_model(str(xml_path), device)
        self.device = device

    def run(self, x: np.ndarray) -> np.ndarray:
        """x is a dense float32 matrix [batch, features]; returns a flat prediction
        array."""
        result = self._compiled([x])[self._compiled.output(0)]
        return np.asarray(result).ravel()


def make_ort_session(onnx_path: str | Path):
    """ONNX Runtime session with every graph optimization turned on (fuse, fold,
    ...)."""
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    # The execution provider decides WHERE the graph runs. CPU is what we have.
    # Alternatives: CUDAExecutionProvider, TensorrtExecutionProvider,
    # OpenVINOExecutionProvider, DmlExecutionProvider (Windows GPU),
    # CoreMLExecutionProvider.
    return ort.InferenceSession(
        str(onnx_path), options, providers=["CPUExecutionProvider"]
    )


# ---------------------------------------------------------------------------- benchmark
def _measure(fn, repeats: int) -> dict[str, float]:
    """Time fn() `repeats` times (after one warm-up call). Returns mean and p95 in
    ms."""
    fn()  # warm-up: first call pays one-off setup costs, so it is not counted
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        times.append((time.perf_counter() - start) * 1000)
    return {
        "mean_ms": float(np.mean(times)),
        "p95_ms": float(np.percentile(times, 95)),
    }


def benchmark_runtimes(
    models_dir: str | Path, rows: list[dict[str, Any]], repeats: int = 50
) -> dict[str, dict[str, float]]:
    """Latency and accuracy of pickle vs ONNX Runtime vs OpenVINO on the same rows.

    The vectorizing step (dicts -> dense matrix) is done ONCE outside the timing, so the
    numbers compare the runtimes only, not the feature code.
    """
    models_dir = Path(models_dir)
    with open(models_dir / settings.pickle_name, "rb") as f:  # a file WE produced
        dv, lr = pickle.load(f)
    x = dv.transform(rows).toarray().astype(np.float32)

    ort_session = make_ort_session(models_dir / settings.onnx_name)
    ort_input = ort_session.get_inputs()[0].name
    ov_session = OpenVINOSession(models_dir / OPENVINO_SUBDIR / OPENVINO_XML)

    runners = {
        "pickle": lambda: lr.predict(x),
        "onnxruntime": lambda: ort_session.run(None, {ort_input: x})[0].ravel(),
        "openvino": lambda: ov_session.run(x),
    }
    reference = np.asarray(runners["pickle"]())
    out: dict[str, dict[str, float]] = {}
    for name, fn in runners.items():
        stats = _measure(fn, repeats)
        pred = np.asarray(fn(), dtype=np.float64)
        # Accuracy delta = the biggest difference from the eager scikit-learn answer.
        stats["max_abs_diff_vs_pickle"] = float(np.max(np.abs(pred - reference)))
        stats["batch_rows"] = float(len(rows))
        out[name] = stats
    base = out["pickle"]["mean_ms"]
    for stats in out.values():
        stats["speedup_vs_pickle"] = (
            base / stats["mean_ms"] if stats["mean_ms"] else 0.0
        )
    return out


# ---------------------------------------------------------------------------- CLI
def main(argv: list[str] | None = None) -> int:
    setup_logging()
    parser = argparse.ArgumentParser(prog="prodml.runtimes")
    sub = parser.add_subparsers(dest="cmd", required=True)
    exp = sub.add_parser("export", help="write the core ONNX file and the OpenVINO IR")
    exp.add_argument(
        "--fp16", action="store_true", help="store OpenVINO weights as FP16"
    )
    bench = sub.add_parser("benchmark", help="compare pickle / onnxruntime / openvino")
    bench.add_argument("--rows", type=int, default=500, help="rows per timed call")
    bench.add_argument("--repeats", type=int, default=50)
    args = parser.parse_args(argv)

    if args.cmd == "export":
        files = export_all(settings.models_dir, fp16=args.fp16)
        sys.stdout.write(json.dumps(files, indent=2) + "\n")
        return 0

    # benchmark: use the last N validation trips of the real dataset, like export.py
    # does.
    from prodml.data import read_dataframe, split_data
    from prodml.features import dataframe_to_dicts

    _, df_val = split_data(read_dataframe(settings.data_path), settings.train_ratio)
    rows = dataframe_to_dicts(df_val.tail(args.rows))
    result = benchmark_runtimes(settings.models_dir, rows, repeats=args.repeats)
    sys.stdout.write(json.dumps(result, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
