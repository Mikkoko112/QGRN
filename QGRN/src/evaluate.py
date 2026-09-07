#Performs the top edges Agreement Problem for n different GRNs
#assumes GRN files to have 3 columns named regulated.gene,regulator.gene,regulator.effect

import pandas as pd
import numpy as np
import math
from itertools import product
from typing import Optional


def _confidence_scores_to_df(confidence_scores: dict) -> pd.DataFrame:
    return pd.DataFrame.from_records(
        (
            {
                "regulator.gene": parsed_edge[0],
                "regulated.gene": parsed_edge[1],
                "regulator.effect": confidence,
            }
            for edge, confidence in confidence_scores.items()
            for parsed_edge in [(edge[len("edge:"):] if edge.startswith("edge:") else edge).split("->")]
        ),
        columns=["regulator.gene", "regulated.gene", "regulator.effect"],
    )


def _calculate_confidence_chunk(
    chunk_edges,
    sample_rows,
    bqm_serialized,
    lambda_energy,
):
    present_counts = {edge: 0 for edge in chunk_edges}
    weighted_present_sums = {edge: 0.0 for edge in chunk_edges}
    weighted_energy_present_sums = {edge: 0.0 for edge in chunk_edges} if lambda_energy != 0.0 else None
    weighted_delta_sums = {edge: 0.0 for edge in chunk_edges} if lambda_energy != 0.0 else None
    edge_set = set(chunk_edges)

    if lambda_energy != 0.0:
        from dimod import BinaryQuadraticModel

        bqm = BinaryQuadraticModel.from_serializable(bqm_serialized)
    else:
        bqm = None

    for sample, active_edges, ground_energy, num_occurrences, robustness_weight, energy_weight in sample_rows:
        present_edges = [edge for edge in active_edges if edge in edge_set]

        for edge in present_edges:
            present_counts[edge] += num_occurrences
            weighted_present_sums[edge] += robustness_weight

        if lambda_energy != 0.0:
            for edge in present_edges:
                flipped_sample = sample.copy()
                flipped_sample[edge] = 0
                flipped_energy = bqm.energy(flipped_sample)
                weighted_energy_present_sums[edge] += energy_weight
                weighted_delta_sums[edge] += energy_weight * (flipped_energy - ground_energy)

    return present_counts, weighted_present_sums, weighted_energy_present_sums, weighted_delta_sums

import dimod

def _edge_set(df: pd.DataFrame):
    return set(zip(df["regulated.gene"], df["regulator.gene"]))

def evaluate_2_graphs(ngenes : int,
                    ntfs : int,
                    df_ref: pd.DataFrame,
                    df_pred: pd.DataFrame,
                    verbose: bool = False):
    total_possible = ntfs * ngenes

    ref_edges = _edge_set(df_ref)
    pred_edges = _edge_set(df_pred)

    tp = len(ref_edges & pred_edges)
    fp = len(pred_edges - ref_edges)
    fn = len(ref_edges - pred_edges)
    tn = max(total_possible - tp - fp - fn, 0)
    npred_edges = tp + fp
    pred_density = npred_edges / total_possible
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    jaccard = tp / (tp + fp + fn) if (tp + fp + fn) else 0.0
    mcc_denominator = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    if mcc_denominator <= 0:
        mcc = 0.0
    else:
        mcc = (tp * tn - fp * fn) / math.sqrt(float(mcc_denominator))
    print(f"TP: {tp}, FP: {fp}, TN: {tn}, FN: {fn}") if verbose else None

    return {
        "TP": tp,
        "FP": fp,
        "TN": tn,
        "FN": fn,
        "npred_edges": npred_edges,
        "pred_density": pred_density,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "jaccard": jaccard,
        "mcc": mcc 
    }

def evaluate_n_graphs(  ngenes : int,
                        ntfs : int,
                        df_ground_truth : pd.DataFrame,
                        df_inferred_grns : dict[str, pd.DataFrame], 
                        metrics : list[str] = ["precision", "recall", "f1_score", "jaccard", "mcc"],
                        verbose: bool = False):
    """
    Parameters:
    dfs: list of dataframe representing the GRNs to compare
    labels: names given to grns to better distinguish them
    metrics: list of metrics to compute. Possible metrics are 
            "TP", "TN", "FP", "FN", "precision", "recall", "f1_score", "jaccard"
    """
    NGRAPHS = len(df_inferred_grns)
    labels = list(df_inferred_grns.keys())
    results = pd.DataFrame(
        index=labels,
        columns=metrics,
        dtype=float
    )
    
    for i in range(NGRAPHS): 
        print(f"Comparing Reference vs {labels[i]} as Inference") if verbose else None
        result = evaluate_2_graphs(
            ngenes,
            ntfs,
            df_ground_truth, 
            df_inferred_grns[labels[i]], 
            verbose = verbose
        )

        for metric in results:
            #all metrics are calulated in evaluate_2_graphs, filter out only requested ones
            if metric not in results:
                continue

            results.loc[labels[i], metric] = result[metric]

    return results

import dimod
from QGRN.src.annealing import Annealer

def calculate_confidence(
    interaction_scores: dict,
    annealing_results: dimod.SampleSet,
    bqm: dimod.BinaryQuadraticModel,
    lambda_scores: float = 0.5,
    lambda_robustness: float = 0.5,
    lambda_energy: float = 0.0,
    robustness_energy_temperature: Optional[float] = None,
    robustness_energy_temperature_min: float = 1e-3,
    verbose: bool = False,
    num_cores: int = 1,
):
    """
    Compute edge confidence from
    1) local interaction score
    2) energy-weighted robustness across sampled solutions
    3) edge indispensability via single-bit flip energy increase

    Optional
    --------
    - robustness_energy_temperature: Boltzmann temperature for converting
      sample energies into robustness weights for the robustness term.
      If None, an adaptive value based on the sample energy spread is used.
    - robustness_energy_temperature_min: lower bound used to keep the
      adaptive robustness temperature from collapsing when sample energies
      are nearly identical.

    Assumptions
    -----------
    - interaction_scores maps edge variable name -> normalized score in [0, 1]
    - annealing_results is a dimod.SampleSet
    - annealer has a .bqm attribute compatible with the returned samples
    """
    import pandas as pd

    edges = list(interaction_scores.keys())
    confidence_scores = {}
    edge_set = set(edges)

    raw_sample_rows = [
        (
            dict(row.sample),
            [
                edge for edge, value in row.sample.items()
                if edge in edge_set and value == 1
            ],
            float(row.energy),
            int(row.num_occurrences),
        )
        for row in annealing_results.data(fields=["sample", "energy", "num_occurrences"])
    ]
    total_occurrences = sum(num_occurrences for _, _, _, num_occurrences in raw_sample_rows)

    if total_occurrences == 0:
        raise ValueError("annealing_results contains no samples.")

    sample_energies = np.array([ground_energy for _, _, ground_energy, _ in raw_sample_rows], dtype=float)
    min_energy = float(sample_energies.min())
    robustness_energy_temperature_min = float(robustness_energy_temperature_min)
    if robustness_energy_temperature_min <= 0.0:
        raise ValueError("robustness_energy_temperature_min must be > 0.")

    if robustness_energy_temperature is None:
        energy_scale = max(float(sample_energies.std()), robustness_energy_temperature_min)
    else:
        energy_scale = float(robustness_energy_temperature)
        if energy_scale <= 0.0:
            raise ValueError("robustness_energy_temperature must be > 0.")
        energy_scale = max(energy_scale, robustness_energy_temperature_min)

    sample_rows = [
        (
            sample,
            active_edges,
            ground_energy,
            num_occurrences,
            float(num_occurrences) * float(np.exp(-(ground_energy - min_energy) / energy_scale)),
            float(num_occurrences) * float(np.exp(-(ground_energy - min_energy) / energy_scale)),
        )
        for sample, active_edges, ground_energy, num_occurrences in raw_sample_rows
    ]
    total_robustness_weight = sum(
        robustness_weight for _, _, _, _, robustness_weight, _ in sample_rows
    )
    if total_robustness_weight <= 0.0:
        # Numerical fallback: if all weights underflow, revert to occurrence-based robustness.
        sample_rows = [
            (
                sample,
                active_edges,
                ground_energy,
                num_occurrences,
                float(num_occurrences),
                float(num_occurrences),
            )
            for sample, active_edges, ground_energy, num_occurrences in raw_sample_rows
        ]
        total_robustness_weight = float(total_occurrences)
    total_energy_weight = sum(energy_weight for _, _, _, _, _, energy_weight in sample_rows)
    if total_energy_weight <= 0.0:
        total_energy_weight = float(total_occurrences)

    present_counts = {edge: 0 for edge in edges}
    weighted_present_sums = {edge: 0.0 for edge in edges}
    weighted_energy_present_sums = {edge: 0.0 for edge in edges} if lambda_energy != 0.0 else None
    weighted_delta_sums = {edge: 0.0 for edge in edges} if lambda_energy != 0.0 else None

    num_cores = max(1, min(num_cores, len(edges)))
    if num_cores == 1:
        chunk_results = [
            _calculate_confidence_chunk(
                edges,
                sample_rows,
                bqm.to_serializable(use_bytes=False) if lambda_energy != 0.0 else None,
                lambda_energy,
            )
        ]
    else:
        import ray

        @ray.remote
        def calculate_confidence_chunk_remote(
            chunk_edges,
            sample_rows,
            bqm_serialized,
            lambda_energy,
        ):
            return _calculate_confidence_chunk(
                chunk_edges,
                sample_rows,
                bqm_serialized,
                lambda_energy,
            )

        chunk_size = max(1, (len(edges) + num_cores - 1) // num_cores)
        edge_chunks = [edges[i:i + chunk_size] for i in range(0, len(edges), chunk_size)]
        bqm_serialized = bqm.to_serializable(use_bytes=False) if lambda_energy != 0.0 else None
        chunk_results = ray.get([
            calculate_confidence_chunk_remote.remote(
                chunk_edges,
                sample_rows,
                bqm_serialized,
                lambda_energy,
            )
            for chunk_edges in edge_chunks
        ])

    for chunk_present_counts, chunk_weighted_present_sums, chunk_weighted_energy_present_sums, chunk_weighted_delta_sums in chunk_results:
        present_counts.update(chunk_present_counts)
        weighted_present_sums.update(chunk_weighted_present_sums)
        if lambda_energy != 0.0:
            weighted_energy_present_sums.update(chunk_weighted_energy_present_sums)
            weighted_delta_sums.update(chunk_weighted_delta_sums)

    robustness_scores = {
        edge: weighted_present_sums[edge] / total_robustness_weight
        for edge in edges
    }
    if lambda_energy != 0.0:
        energy_scores = {}
        for edge in edges:
            if weighted_energy_present_sums[edge] > 0:
                energy_scores[edge] = weighted_delta_sums[edge] / weighted_energy_present_sums[edge]
            else:
                energy_scores[edge] = 0.0
    else:
        energy_scores = {edge: 0.0 for edge in edges}

    # Normalize energy scores to [0, 1] if they are used
    min_e = min(energy_scores.values())
    max_e = max(energy_scores.values())

    if max_e - min_e > 0:
        norm_energy_scores = {
            edge: (energy_scores[edge] - min_e) / (max_e - min_e)
            for edge in edges
        }
    else:
        norm_energy_scores = {edge: 0.0 for edge in edges}

    for edge in edges:
        confidence_scores[edge] = (
            lambda_scores * interaction_scores[edge]
            + lambda_robustness * robustness_scores[edge]
            + lambda_energy * norm_energy_scores[edge]
        )

    return _confidence_scores_to_df(confidence_scores)

def calculate_ap_auprc(
    inferred_df: pd.DataFrame,
    ground_truth_df: pd.DataFrame,
    tf_list: list[str],
    gene_list: list[str],
) -> dict:
    """
    Calculate AUROC and trapezoidal PR-AUC over the full TF x gene edge space.

    Parameters
    ----------
    inferred_df : pd.DataFrame
        Must contain columns ["regulator.gene", "regulated.gene", "regulator.effect"].
        Missing edges are assigned confidence 0.
    ground_truth_df : pd.DataFrame
        Must contain columns ["regulator.gene", "regulated.gene"].
    tf_list : list[str]
        Candidate transcription factors.
    gene_list : list[str]
        Candidate target genes.

    Returns
    -------
    dict
        {
            "auprc": float,
            "auroc": float,
        }
    """
    from sklearn.metrics import precision_recall_curve, roc_auc_score, auc

    # full candidate edge space
    full_edges = pd.DataFrame(
        product(tf_list, gene_list),
        columns=["regulator.gene", "regulated.gene"]
    )

    # merge inferred confidences, assign 0 to absent edges
    merged = full_edges.merge(
        inferred_df[["regulator.gene", "regulated.gene", "regulator.effect"]],
        left_on=["regulator.gene", "regulated.gene"],
        right_on=["regulator.gene", "regulated.gene"],
        how="left"
    )
    merged["Confidence"] = merged["regulator.effect"].fillna(0.0)

    # ground-truth labels
    gt_edges = set(zip(ground_truth_df["regulator.gene"], ground_truth_df["regulated.gene"]))
    merged["Label"] = [
        1 if (tf, tg) in gt_edges else 0
        for tf, tg in zip(merged["regulator.gene"], merged["regulated.gene"])
    ]

    y_true = merged["Label"].to_numpy()
    y_score = merged["Confidence"].to_numpy()

    # explicit PR curve
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)

    # trapezoidal PR-AUC
    pr_auc = auc(recall, precision)
    auroc = roc_auc_score(y_true, y_score)

    return {
        "auprc": pr_auc,
        "auroc": auroc,
    }
