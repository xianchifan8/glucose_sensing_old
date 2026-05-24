#!/usr/bin/env python3
"""NNI trial wrapper for tuning main.py without modifying the training entry.

This script receives sampled hyperparameters from NNI, appends them to the
known-good baseline command, runs ``main.py`` as a subprocess, parses metrics
from stdout/stderr, and reports the objective value back to NNI.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


try:
    import nni
except ImportError:  # Allows --dry_run/standalone debugging before NNI is installed.
    nni = None


BASE_COMMAND = [
    sys.executable,
    "main.py",
    "--data_source",
    "db",
    "--db_user",
    "Tao_db",
    "--model",
    "TCN",
    "--mode",
    "window",
    "--window_size",
    "50",
    "--data_fusion",
    "--fusion_stage",
    "late",
    "--fusion_method_late",
    "gated_residual",
    "--aux_feature_mode",
    "engineered",
    "--split_strategy",
    "random",
    "--no_val",
    "--train_split",
    "0.8",
    "--test_split",
    "0.2",
    "--normalize",
    "--epochs",
    "20",
    "--lr",
    "0.001",
    "--dropout",
    "0.15",
    "--batch_size",
    "32",
    "--no_wandb",
    "--weight_decay",
    "0.001",
    "--loss",
    "Huber",
    "--huber_delta",
    "0.8",
]


PARAM_TO_ARG = {
    "window_size": "--window_size",
    "epochs": "--epochs",
    "lr": "--lr",
    "dropout": "--dropout",
    "batch_size": "--batch_size",
    "weight_decay": "--weight_decay",
    "huber_delta": "--huber_delta",
}

INT_PARAMS = {"window_size", "epochs", "batch_size"}
LOWER_IS_BETTER = {"rmse", "mae", "mape", "loss", "mse", "huber"}
HIGHER_IS_BETTER = {"r2", "r_2", "accuracy", "acc"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one NNI trial for main.py.")
    parser.add_argument(
        "--target_metric",
        default=os.getenv("NNI_TARGET_METRIC", "rmse"),
        help="Metric to optimize, e.g. rmse, mae, loss, r2.",
    )
    parser.add_argument(
        "--target_scope",
        default=os.getenv("NNI_TARGET_SCOPE", "test"),
        help="Prefer metrics whose nearby text contains this scope; use 'any' to disable.",
    )
    parser.add_argument(
        "--log_dir",
        default=os.getenv("NNI_TRIAL_LOG_DIR", "logs/nni_trials"),
        help="Directory for per-trial subprocess logs.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print the generated command without starting training.",
    )
    return parser.parse_args()


def normalize_params(params: dict[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for key, value in params.items():
        if key not in PARAM_TO_ARG:
            continue
        if key in INT_PARAMS:
            normalized[key] = int(round(float(value)))
        else:
            normalized[key] = float(value)
    return normalized


def replace_arg(command: list[str], option: str, value: Any) -> None:
    text = str(value)
    if option in command:
        index = command.index(option)
        if index + 1 >= len(command):
            raise ValueError(f"Option {option} has no value in base command")
        command[index + 1] = text
    else:
        command.extend([option, text])


def build_command(params: dict[str, Any]) -> list[str]:
    command = list(BASE_COMMAND)
    for key, value in normalize_params(params).items():
        replace_arg(command, PARAM_TO_ARG[key], value)
    return command


def trial_id() -> str:
    if nni is not None:
        try:
            return str(nni.get_trial_id())
        except Exception:
            pass
    return f"standalone_{int(time.time())}"


def metric_regex(metric_name: str) -> re.Pattern[str]:
    metric = re.escape(metric_name)
    pattern = (
        rf"(?P<context>[A-Za-z0-9_ /.-]{{0,80}}?)"
        rf"(?<![A-Za-z0-9])(?P<name>{metric})(?![A-Za-z0-9])"
        rf"\s*(?:[:=]|is)?\s*"
        rf"(?P<value>[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"
    )
    return re.compile(pattern, re.IGNORECASE)


def extract_metrics(text: str, metric_name: str, target_scope: str) -> list[float]:
    pattern = metric_regex(metric_name)
    scoped: list[float] = []
    unscoped: list[float] = []
    scope = target_scope.lower()
    for match in pattern.finditer(text):
        value = float(match.group("value"))
        if not math.isfinite(value):
            continue
        unscoped.append(value)
        context = match.group("context").lower()
        if scope == "any" or scope in context:
            scoped.append(value)
    return scoped or unscoped


def extract_best_test_summary(text: str) -> dict[str, float | int] | None:
    """Extract the explicit best-test marker printed by GlucoseTrainer.train."""
    pattern = re.compile(
        r"BEST_TEST_MAE:\s*(?P<mae>[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"
        r".*?BEST_TEST_EPOCH:\s*(?P<epoch>\d+)"
        r".*?BEST_TEST_RMSE:\s*(?P<rmse>[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"
        r".*?BEST_TEST_LOSS:\s*(?P<loss>[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)",
        re.IGNORECASE | re.DOTALL,
    )
    matches = list(pattern.finditer(text))
    if not matches:
        return None
    match = matches[-1]
    return {
        "best_test_mae": float(match.group("mae")),
        "best_test_epoch": int(match.group("epoch")),
        "best_test_rmse": float(match.group("rmse")),
        "best_test_loss": float(match.group("loss")),
    }


def choose_objective(values: list[float], metric_name: str) -> float:
    metric = metric_name.lower()
    if metric in HIGHER_IS_BETTER:
        return max(values)
    if metric in LOWER_IS_BETTER:
        return min(values)
    return values[-1]


def report_intermediate(value: float) -> None:
    if nni is not None:
        nni.report_intermediate_result(value)


def report_final(value: float) -> None:
    if nni is not None:
        nni.report_final_result(value)


def append_trial_summary(
    log_dir: Path,
    trial_name: str,
    params: dict[str, Any],
    command: list[str],
    objective: float,
    metric_name: str,
    metric_source: str,
    best_summary: dict[str, float | int] | None,
    log_path: Path,
) -> None:
    """Append one compact record per hyperparameter setting."""
    summary_path = log_dir / "hyperparameter_best_results.jsonl"
    record = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "trial_id": trial_name,
        "target_metric": metric_name,
        "objective": objective,
        "metric_source": metric_source,
        "params": params,
        "normalized_params": normalize_params(params),
        "command": shlex.join(command),
        "trial_log": str(log_path),
    }
    if best_summary is not None:
        record.update(best_summary)
    with summary_path.open("a", encoding="utf-8") as summary_file:
        summary_file.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    print(f"Hyperparameter summary appended to: {summary_path}")


def main() -> int:
    args = parse_args()
    params = nni.get_next_parameter() if nni is not None else {}
    command = build_command(params)

    print("NNI parameters:", json.dumps(params, ensure_ascii=False, sort_keys=True))
    print("Training command:", shlex.join(command), flush=True)

    if args.dry_run:
        return 0

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    current_trial_id = trial_id()
    log_path = log_dir / f"{current_trial_id}.log"

    metric_values: list[float] = []
    full_output: list[str] = []
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    assert process.stdout is not None
    with log_path.open("w", encoding="utf-8") as log_file:
        for line in process.stdout:
            print(line, end="")
            log_file.write(line)
            log_file.flush()
            full_output.append(line)
            values = extract_metrics(line, args.target_metric, args.target_scope)
            if values:
                metric_values.extend(values)
                report_intermediate(choose_objective(metric_values, args.target_metric))

    return_code = process.wait()
    if return_code != 0:
        print(f"Training failed with return code {return_code}. Log: {log_path}", file=sys.stderr)
        return return_code

    joined_output = "".join(full_output)
    best_summary = extract_best_test_summary(joined_output)

    if best_summary is not None and args.target_metric.lower() == "mae" and args.target_scope.lower() in {"test", "any"}:
        objective = float(best_summary["best_test_mae"])
        metric_source = "BEST_TEST_MAE marker"
    else:
        if not metric_values:
            metric_values = extract_metrics(joined_output, args.target_metric, args.target_scope)

        if not metric_values:
            print(
                f"Could not find metric '{args.target_metric}' in training output. "
                f"Check {log_path} and set --target_metric/--target_scope if needed.",
                file=sys.stderr,
            )
            return 2

        objective = choose_objective(metric_values, args.target_metric)
        metric_source = "parsed metric stream"

    append_trial_summary(
        log_dir=log_dir,
        trial_name=current_trial_id,
        params=params,
        command=command,
        objective=objective,
        metric_name=args.target_metric,
        metric_source=metric_source,
        best_summary=best_summary,
        log_path=log_path,
    )
    print(f"NNI final objective ({args.target_metric}, scope={args.target_scope}, source={metric_source}): {objective}")
    print(f"Trial log saved to: {log_path}")
    report_final(objective)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
