#!/usr/bin/env python3
"""Run a small, reproducible QGRN-vs-Netter experiment matrix."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
QGRN_ROOT = SCRIPT_DIR.parent
REPO_ROOT = QGRN_ROOT.parent
COMPARE_SCRIPT = SCRIPT_DIR / "compare_netter.py"

DATASETS = {
    "GRN_params_100": {
        "expression": QGRN_ROOT / "QGRN/data/simulated/GRN_params_100/expr_matrix_raw.csv",
        "ground_truth": QGRN_ROOT / "QGRN/data/simulated/GRN_params_100/grn.csv",
    },
    "LCL_GRN": {
        "expression": QGRN_ROOT / "QGRN/data/simulated/LCL_GRN/expr_matrix_raw.csv",
        "ground_truth": QGRN_ROOT / "QGRN/data/simulated/LCL_GRN/grn.csv",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run QGRN once per small dataset and several Netter settings."
    )
    parser.add_argument(
        "--datasets",
        default=",".join(DATASETS),
        help="Comma-separated dataset names (default: GRN_params_100,LCL_GRN).",
    )
    parser.add_argument(
        "--netter-iterations",
        default="1,5,10",
        help="Comma-separated Netter totalIterations values (default: 1,5,10).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=QGRN_ROOT / "QGRN/Testing/plotting/results/netter_experiments",
        help="Root directory for all experiment artifacts.",
    )
    parser.add_argument("--netter-workers", type=int, default=4)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def read_dataset_metadata(dataset: str, expression: Path, ground_truth: Path, adapter_dir: Path) -> dict:
    with expression.open(newline="") as stream:
        reader = csv.reader(stream)
        header = next(reader)
        genes = [row[0] for row in reader if row and row[0]]

    with ground_truth.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    tfs = []
    for row in rows:
        tf = row.get("regulator.gene") or row.get("Gene1")
        if tf in genes and tf not in tfs:
            tfs.append(tf)

    adapter_dir.mkdir(parents=True, exist_ok=True)
    genes_path = adapter_dir / f"{dataset}_genes.json"
    tfs_path = adapter_dir / f"{dataset}_tfs.json"
    genes_path.write_text(json.dumps(genes))
    tfs_path.write_text(json.dumps(tfs))
    return {
        "source": "custom",
        "genes_in_rows": True,
        "scRNA": str(expression),
        "tfs": str(tfs_path),
        "GT_GRN": str(ground_truth),
        "genes": str(genes_path),
    }


def create_config(example_config: Path, dataset: str, dataset_info: dict, path: Path) -> None:
    with example_config.open() as stream:
        config = yaml.safe_load(stream)
    config["data"]["mode"] = "single_run"
    config["data"]["single_run"]["active_dataset"] = dataset
    config["data"]["datasets"] = {dataset: dataset_info}

    # Keep the experiment focused on the requested baseline and postprocessors.
    config["base_lines"]["random"]["mode"] = False
    config["base_lines"]["self"]["mode"] = False
    config["base_lines"]["grnboost2"]["mode"] = False
    config["base_lines"]["genie3"]["mode"] = True
    config["annealer"]["num_cores"] = 4
    config["outputs"]["save_inference_grns"]["mode"] = True
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as stream:
        yaml.safe_dump(config, stream, sort_keys=False)


def run(command: list[str], verbose: bool) -> None:
    try:
        subprocess.run(command, cwd=QGRN_ROOT, check=True, capture_output=not verbose)
    except subprocess.CalledProcessError as error:
        if error.stdout:
            print(error.stdout, file=sys.stdout, end="")
        if error.stderr:
            print(error.stderr, file=sys.stderr, end="")
        raise


def collect_summary(experiment_dir: Path, dataset: str, iterations: int) -> pd.DataFrame:
    summary = pd.read_csv(experiment_dir / "summary.csv")
    with (experiment_dir / "run_metadata.json").open() as stream:
        metadata = json.load(stream)
    with (experiment_dir / "rank_change.json").open() as stream:
        rank_change = json.load(stream)
    summary.insert(0, "dataset", dataset)
    summary.insert(1, "netter_iterations", iterations)
    for key, value in rank_change.items():
        summary[f"rank_{key}"] = value
    summary["run_dir"] = str(experiment_dir)
    summary["qgrn_config"] = metadata["qgrn_config"]
    return summary


def main() -> None:
    args = parse_args()
    datasets = [name.strip() for name in args.datasets.split(",") if name.strip()]
    iterations = [int(value.strip()) for value in args.netter_iterations.split(",")]
    unknown = sorted(set(datasets).difference(DATASETS))
    if unknown:
        raise ValueError(f"Unknown datasets: {unknown}. Choose from {sorted(DATASETS)}")
    if any(value < 1 for value in iterations):
        raise ValueError("Netter iterations must be positive integers.")

    args.output_dir = args.output_dir.resolve()
    adapter_dir = args.output_dir / "_adapters"
    config_dir = args.output_dir / "_configs"
    all_summaries = []

    for dataset in datasets:
        info = DATASETS[dataset]
        dataset_root = args.output_dir / dataset
        qgrn_run_dir = dataset_root / "qgrn_baseline"
        dataset_info = read_dataset_metadata(
            dataset, info["expression"], info["ground_truth"], adapter_dir
        )
        config_path = config_dir / f"{dataset}.yaml"
        create_config(
            QGRN_ROOT / "QGRN/configs/example.yaml",
            dataset,
            dataset_info,
            config_path,
        )

        # This is the only QGRN execution for this dataset. Every Netter run
        # below reuses its saved GENIE3 and QGRN confidence outputs.
        run(
            [
                sys.executable,
                str(COMPARE_SCRIPT),
                "--qgrn-config",
                str(config_path),
                "--output-dir",
                str(qgrn_run_dir),
                "--netter-iterations",
                str(iterations[0]),
                "--netter-workers",
                str(args.netter_workers),
            ],
            args.verbose,
        )
        saved_dir = qgrn_run_dir / "qgrn_saved_grns"

        for setting in iterations:
            run_dir = dataset_root / f"netter_iterations_{setting}"
            run(
                [
                    sys.executable,
                    str(COMPARE_SCRIPT),
                    "--qgrn-config",
                    str(config_path),
                    "--qgrn-saved-dir",
                    str(saved_dir),
                    "--skip-qgrn",
                    "--output-dir",
                    str(run_dir),
                    "--netter-iterations",
                    str(setting),
                    "--netter-workers",
                    str(args.netter_workers),
                ],
                args.verbose,
            )
            all_summaries.append(collect_summary(run_dir, dataset, setting))

    pd.concat(all_summaries, ignore_index=True).to_csv(
        args.output_dir / "all_runs.csv", index=False
    )
    print(f"Aggregate results written to: {args.output_dir / 'all_runs.csv'}")


if __name__ == "__main__":
    main()
