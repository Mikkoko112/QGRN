#!/usr/bin/env python3
"""Run QGRN and compare its saved GENIE3 baseline with Netter.

This is intentionally an external orchestration script. It does not change
QGRN's execution code or Netter's Java sources.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd
import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
QGRN_ROOT = SCRIPT_DIR.parent
REPO_ROOT = QGRN_ROOT.parent
sys.path.insert(0, str(QGRN_ROOT))
DEFAULT_NETTER_CHUNKS = 50
NETTER_PROFILES = {
    # Values from netter/jar/parameter_small.txt.
    "small": {
        "workers": 4,
        "iterations": 1,
        "modify_each_turn": 70,
        "move_each_turn": 50,
        "start_temperature": 5100.0,
        "max_iterations": 1000,
        "temperature_auto": True,
        "determination_zone": 0.1,
        "temperature_target": 0.12,
        "temperature_deviation": 0.0025,
    },
    # Values from netter/jar/parameter.txt.
    "jar": {
        "workers": 4,
        "iterations": 75,
        "modify_each_turn": 70,
        "move_each_turn": 50,
        "start_temperature": 5100.0,
        "max_iterations": 20000,
        "temperature_auto": True,
        "determination_zone": 0.1,
        "temperature_target": 0.12,
        "temperature_deviation": 0.0025,
    },
    # Values from netter/src/parameter.txt and the README example.
    "source": {
        "workers": 2,
        "iterations": 10,
        "modify_each_turn": 50,
        "move_each_turn": 50,
        "start_temperature": 10.0,
        "max_iterations": 500,
        "temperature_auto": False,
        "determination_zone": 10.0,
        "temperature_target": 0.12,
        "temperature_deviation": 10.0,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare QGRN's QUBO result with Netter applied to GENIE3."
    )
    parser.add_argument(
        "--qgrn-config",
        type=Path,
        default=QGRN_ROOT / "QGRN" / "configs" / "example.yaml",
        help="QGRN YAML configuration (default: QGRN/configs/example.yaml).",
    )
    parser.add_argument(
        "--netter-jar",
        type=Path,
        default=REPO_ROOT / "netter" / "jar" / "netter_01_1.jar",
        help="Netter jar to execute.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=QGRN_ROOT / "QGRN" / "Testing" / "plotting" / "results" / "netter_comparison",
        help="Directory for intermediate files and final metrics.",
    )
    parser.add_argument(
        "--skip-qgrn",
        action="store_true",
        help="Reuse saved QGRN CSV files instead of running QGRN first.",
    )
    parser.add_argument(
        "--qgrn-saved-dir",
        type=Path,
        default=None,
        help="Existing directory containing <dataset>/annealer.csv and genie3.csv.",
    )
    parser.add_argument(
        "--cutoff",
        type=int,
        default=None,
        help="Number of ranked GENIE3 edges Netter may rerank (default: largest valid cutoff).",
    )
    parser.add_argument(
        "--netter-chunks",
        type=int,
        default=None,
        help="Netter partition size (default: 50, or 10 for baselines under 50 edges).",
    )
    parser.add_argument(
        "--netter-profile",
        choices=sorted(NETTER_PROFILES),
        default="small",
        help="Parameter template: small, jar, or source (default: small).",
    )
    parser.add_argument(
        "--netter-iterations",
        type=int,
        default=None,
        help="Override the selected profile's totalIterations value.",
    )
    parser.add_argument(
        "--netter-workers",
        type=int,
        default=None,
        help="Override the selected profile's worker count.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show subprocess output and evaluation details.",
    )
    return parser.parse_args()


def load_config(config_path: Path) -> dict:
    with config_path.open() as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError(f"QGRN config is not a YAML mapping: {config_path}")
    return config


def dataset_name(config: dict) -> str:
    mode = config["data"]["mode"]
    if mode != "single_run":
        raise ValueError(
            "This one-dataset wrapper currently requires data.mode: single_run."
        )
    return config["data"]["single_run"]["active_dataset"]


def run_qgrn(config_path: Path, output_dir: Path, verbose: bool) -> None:
    """Run QGRN with only the switches needed to export GENIE3."""
    config = load_config(config_path)
    config.setdefault("base_lines", {}).setdefault("genie3", {})["mode"] = True
    config.setdefault("outputs", {}).setdefault("save_inference_grns", {})[
        "mode"
    ] = True

    # Keep QGRN's normal output location relative to its repository root.
    config["outputs"]["save_inference_grns"]["path"] = str(
        output_dir / "qgrn_saved_grns"
    )

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", prefix="qgrn_netter_", delete=False
    ) as config_file:
        yaml.safe_dump(config, config_file, sort_keys=False)
        temporary_config = Path(config_file.name)

    command = [
        sys.executable,
        str(QGRN_ROOT / "QGRN" / "src" / "main.py"),
        "-c",
        str(temporary_config),
    ]
    if verbose:
        command.append("-v")

    # Prefer this checkout over a different installed package named QGRN.
    qgrn_env = os.environ.copy()
    existing_pythonpath = qgrn_env.get("PYTHONPATH")
    qgrn_env["PYTHONPATH"] = os.pathsep.join(
        path for path in (str(QGRN_ROOT), existing_pythonpath) if path
    )
    try:
        subprocess.run(command, cwd=QGRN_ROOT, check=True, env=qgrn_env)
    finally:
        temporary_config.unlink(missing_ok=True)


def standardize_netter_input(genie3_csv: Path, input_path: Path) -> pd.DataFrame:
    """Create Netter's source-target ranked text file from QGRN's dataframe."""
    from QGRN.src.utils_annealing import csv_to_df_grn

    genie3 = csv_to_df_grn(genie3_csv).copy()
    required = {"regulator.gene", "regulated.gene", "regulator.effect"}
    missing = required.difference(genie3.columns)
    if missing:
        raise ValueError(f"GENIE3 output is missing columns: {sorted(missing)}")

    genie3["regulator.gene"] = genie3["regulator.gene"].astype(str)
    genie3["regulated.gene"] = genie3["regulated.gene"].astype(str)
    genie3 = genie3[genie3["regulator.gene"] != genie3["regulated.gene"]]
    genie3 = genie3.drop_duplicates(
        subset=["regulator.gene", "regulated.gene"], keep="first"
    ).reset_index(drop=True)
    genie3 = genie3[
        ["regulator.gene", "regulated.gene", "regulator.effect"]
    ]
    if genie3.empty:
        raise ValueError(f"GENIE3 output contains no usable edges: {genie3_csv}")

    # Netter uses line order as the original ranking and ignores columns after
    # the first two. Keep a rank column for reproducibility and diagnostics.
    input_path.parent.mkdir(parents=True, exist_ok=True)
    with input_path.open("w") as stream:
        for rank, row in enumerate(genie3.itertuples(index=False), start=1):
            stream.write(f"{row[0]}\t{row[1]}\t{rank}\n")
    return genie3


def write_netter_config(
    config_path: Path,
    test_dir: Path,
    output_dir: Path,
    cutoff: int,
    chunks: int,
    iterations: int,
    workers: int,
    profile: dict,
) -> None:
    values = {
        "testDir": str(test_dir),
        "preds": "genie3.txt",
        "allPreds": "false",
        "workersAvailable": str(workers),
        "totalIterations": str(iterations),
        "cutoff": str(cutoff),
        "chuncks": str(chunks),
        "chunckIncrease": "0.5",
        "modifyEachTurn": str(profile["modify_each_turn"]),
        "moveEachTurn": str(profile["move_each_turn"]),
        "startTemperature": str(profile["start_temperature"]),
        "maxIterations": str(profile["max_iterations"]),
        "temperatureAutoDetermination": str(profile["temperature_auto"]).lower(),
        "determinationZone": str(profile["determination_zone"]),
        "temperatureTargetGoal": str(profile["temperature_target"]),
        "temperatureAllowedDeviation": str(profile["temperature_deviation"]),
        "outputDirString": str(output_dir),
        "LOGGERS": "",
        "PENALTIES": "",
    }
    with config_path.open("w") as stream:
        stream.write("# Generated by scripts/compare_netter.py\n")
        for key, value in values.items():
            stream.write(f"{key}={value}\n")
        stream.write("TFSparsityPenalty=25.0=-0.001=1000=0.85=0=1.001=176\n")
        stream.write("GraphletG4Penalty=3.0=-0.001=7000=0.90=0=1.001=700\n")
        stream.write("AntiDominatePenalty=75.000000000000\n")
        stream.write("PredictionConfidencePenalty=0.000010000000\n")


def run_netter(
    jar_path: Path,
    test_dir: Path,
    output_dir: Path,
    cutoff: int,
    chunks: int,
    iterations: int,
    workers: int,
    profile: dict,
    verbose: bool,
) -> Path:
    if not jar_path.is_file():
        raise FileNotFoundError(f"Netter jar not found: {jar_path}")

    config_path = test_dir.parent / "netter_parameter.txt"
    write_netter_config(
        config_path,
        test_dir,
        output_dir,
        cutoff,
        chunks,
        iterations,
        workers,
        profile,
    )
    command = ["java", "-jar", str(jar_path), str(config_path)]
    subprocess.run(command, cwd=jar_path.parent, check=True, capture_output=not verbose)

    candidates = sorted(output_dir.rglob("*ENSEMBLE*.txt"))
    if not candidates:
        raise FileNotFoundError(
            f"Netter completed but produced no *ENSEMBLE*.txt file under {output_dir}"
        )
    return candidates[-1]


def read_netter_result(result_path: Path) -> pd.DataFrame:
    """Convert Netter's output order into QGRN confidence columns."""
    raw = pd.read_csv(result_path, sep="\t", header=None, usecols=[0, 1])
    raw.columns = ["regulator.gene", "regulated.gene"]
    raw["regulator.gene"] = raw["regulator.gene"].astype(str)
    raw["regulated.gene"] = raw["regulated.gene"].astype(str)
    raw = raw[raw["regulator.gene"] != raw["regulated.gene"]]
    raw = raw.drop_duplicates(
        subset=["regulator.gene", "regulated.gene"], keep="first"
    ).reset_index(drop=True)
    raw["regulator.effect"] = (len(raw) - raw.index).astype(float)
    return raw[["regulator.gene", "regulated.gene", "regulator.effect"]]


def evaluate(dataset_dir: Path, output_dir: Path, config: dict) -> None:
    from QGRN.src.evaluate import calculate_ap_auprc, evaluate_n_graphs
    from QGRN.src.utils_annealing import csv_to_df_grn

    ground_truth = csv_to_df_grn(dataset_dir / "ground_truth.csv")
    qgrn = csv_to_df_grn(dataset_dir / "annealer.csv")
    genie3 = csv_to_df_grn(dataset_dir / "genie3.csv")
    netter = read_netter_result(output_dir / "netter_result.txt")

    # Compare methods at the same edge budget. QGRN saves confidence scores
    # for the full candidate space, while GENIE3/Netter produce a fixed list.
    qgrn = qgrn.sort_values(
        by="regulator.effect", ascending=False
    ).head(len(genie3)).reset_index(drop=True)
    qgrn.to_csv(output_dir / "annealer_top_k.csv", index=False)
    inferred = {"genie3": genie3, "netter": netter, "annealer": qgrn}
    dataset_info = config["data"]["datasets"][dataset_name(config)]
    genes = pd.read_json(QGRN_ROOT / dataset_info["genes"], typ="series").astype(str).tolist()
    tfs = pd.read_json(QGRN_ROOT / dataset_info["tfs"], typ="series").astype(str).tolist()
    tfs = list(set(genes).intersection(tfs))

    binary = evaluate_n_graphs(
        ngenes=len(genes),
        ntfs=len(tfs),
        df_ground_truth=ground_truth,
        df_inferred_grns=inferred,
        metrics=["precision", "recall", "f1_score", "jaccard", "mcc", "npred_edges"],
    )
    confidence = pd.DataFrame.from_dict(
        {
            name: calculate_ap_auprc(df, ground_truth, tfs, genes)
            for name, df in inferred.items()
        },
        orient="index",
    )
    binary.to_csv(output_dir / "binary_metrics.csv")
    confidence.to_csv(output_dir / "confidence_metrics.csv")
    netter.to_csv(output_dir / "netter.csv", index=False)

    baseline_edges = list(zip(genie3["regulator.gene"], genie3["regulated.gene"]))
    netter_edges = list(zip(netter["regulator.gene"], netter["regulated.gene"]))
    baseline_rank = {edge: rank for rank, edge in enumerate(baseline_edges)}
    netter_rank = {edge: rank for rank, edge in enumerate(netter_edges)}
    comparable_edges = set(baseline_rank).intersection(netter_rank)
    changed_positions = sum(
        baseline_rank[edge] != netter_rank[edge] for edge in comparable_edges
    )
    rank_change = {
        "baseline_edges": len(baseline_edges),
        "netter_edges": len(netter_edges),
        "common_edges": len(comparable_edges),
        "changed_positions": changed_positions,
        "changed_fraction": changed_positions / len(comparable_edges)
        if comparable_edges
        else 0.0,
    }
    for k in (10, 50, 100, 500):
        if k <= len(baseline_edges):
            rank_change[f"top_{k}_overlap"] = len(
                set(baseline_edges[:k]).intersection(netter_edges[:k])
            )
    with (output_dir / "rank_change.json").open("w") as stream:
        json.dump(rank_change, stream, indent=2)

    summary = pd.concat(
        [
            binary.reset_index(names="method").assign(evaluation="binary"),
            confidence.reset_index(names="method").assign(evaluation="ranking"),
        ],
        ignore_index=True,
        sort=False,
    )
    summary.to_csv(output_dir / "summary.csv", index=False)

    print("\nBinary evaluation:")
    print(binary.to_string())
    print("\nRanking evaluation (Netter rank converted to descending confidence):")
    print(confidence.to_string())


def main() -> None:
    args = parse_args()
    args.qgrn_config = args.qgrn_config.resolve()
    args.netter_jar = args.netter_jar.resolve()
    args.output_dir = args.output_dir.resolve()
    if args.qgrn_saved_dir is not None:
        args.qgrn_saved_dir = args.qgrn_saved_dir.resolve()
    config = load_config(args.qgrn_config)
    dataset = dataset_name(config)
    profile = NETTER_PROFILES[args.netter_profile]
    netter_iterations = args.netter_iterations or profile["iterations"]
    netter_workers = args.netter_workers or profile["workers"]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prediction_dir = args.output_dir / "netter_input" / "predictions"
    netter_output_dir = args.output_dir / f"netter_output_{os.getpid()}"

    if not args.skip_qgrn:
        run_qgrn(args.qgrn_config, args.output_dir, args.verbose)

    wrapper_dataset_dir = args.output_dir / "qgrn_saved_grns" / dataset
    configured_save_dir = Path(
        config.get("outputs", {})
        .get("save_inference_grns", {})
        .get("path", "QGRN/outputs/saved_grns")
    )
    if not configured_save_dir.is_absolute():
        configured_save_dir = QGRN_ROOT / configured_save_dir
    if args.qgrn_saved_dir is not None:
        configured_save_dir = args.qgrn_saved_dir
    dataset_dir = (
        wrapper_dataset_dir
        if wrapper_dataset_dir.is_dir()
        else configured_save_dir / dataset
    )

    genie3_csv = dataset_dir / "genie3.csv"
    if not genie3_csv.is_file():
        raise FileNotFoundError(
            f"Expected saved GENIE3 output at {genie3_csv}. "
            "Run without --skip-qgrn or check QGRN output settings."
        )

    input_path = prediction_dir / "genie3.txt"
    genie3 = standardize_netter_input(genie3_csv, input_path)
    chunks = args.netter_chunks or (
        DEFAULT_NETTER_CHUNKS if len(genie3) >= DEFAULT_NETTER_CHUNKS else 10
    )
    if chunks < 1:
        raise ValueError("netter-chunks must be positive")
    cutoff = (
        args.cutoff
        if args.cutoff is not None
        else (len(genie3) // chunks) * chunks
    )
    if cutoff < chunks or cutoff > len(genie3) or cutoff % chunks:
        raise ValueError(
            f"cutoff must be a multiple of {chunks} between "
            f"{chunks} and {len(genie3)}"
        )

    netter_output_dir.mkdir(parents=True)
    result_path = run_netter(
        args.netter_jar,
        prediction_dir.parent,
        netter_output_dir,
        cutoff,
        chunks,
        netter_iterations,
        netter_workers,
        profile,
        args.verbose,
    )
    (args.output_dir / "netter_result.txt").write_bytes(result_path.read_bytes())
    evaluate(dataset_dir, args.output_dir, config)
    metadata = {
        "dataset": dataset,
        "qgrn_config": str(args.qgrn_config),
        "netter_jar": str(args.netter_jar),
        "netter_profile": args.netter_profile,
        "netter_iterations": netter_iterations,
        "netter_workers": netter_workers,
        "netter_cutoff": cutoff,
        "netter_chunks": chunks,
        "genie3_edges": len(genie3),
        "qgrn_saved_dir": str(dataset_dir.parent),
    }
    with (args.output_dir / "run_metadata.json").open("w") as stream:
        json.dump(metadata, stream, indent=2)
    print(f"\nArtifacts written to: {args.output_dir}")


if __name__ == "__main__":
    main()
