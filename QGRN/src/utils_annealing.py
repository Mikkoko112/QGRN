import pandas as pd
import numpy as np
from pathlib import Path
from typing import Set

"""-----------------------------------
utils for gene selection and filtering
-----------------------------------"""

def select_random_genes(df, k):
    """
    input
    df (pd.DataFrame) : dataframe with raw scRNA data
    k (int) : number of genes to select
    """
    assert k <= df.shape[1]
    sampled_df = df.sample(n=k)
    return sampled_df

def filter_grn(df, genes):
    """
    Filter a GRN dataframe to edges where both genes are in `genes`.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with GRN edges. Must contain columns describing the
        regulator and regulated gene (e.g. 'regulated.gene', 'regulator.gene').
    genes : iterable[str]
        Iterable of gene names that should remain in the network.

    Returns
    -------
    pd.DataFrame
        Filtered dataframe that only contains edges whose regulator and
        regulated genes are both in the provided gene list.
    """

    filtered = df[
        df["regulated.gene"].isin(genes) & df["regulator.gene"].isin(genes)
    ]

    return filtered.copy()

def filter_df_by_index(df : pd.DataFrame, labels):
    """
    Return rows of `df` whose index is in `labels`.

    input
    df : pd.DataFrame
        DataFrame to filter.
    labels : list-like
        Sequence of index labels to keep.

    output
    pd.DataFrame
        Filtered DataFrame.
    """

    return df.reindex(list(labels))

"""----------------------------------------------------------------
utils for converting scRNA data and Ground Truth GRNs to dataframes
----------------------------------------------------------------"""

def unpickle_causal_graph(filepath, effect=1):
    """
    Transforms a pickled causal graph file into a GRN pandas DataFrame.
    
    Parameters
    ----------
    filepath : str or Path
        Path to the pickle file.

    Returns
    df (pandas.Dataframe) : df of grn 

            regulated.gene regulator.gene regulator.effect
    edge1
    edge2
    edge3
    ...
    """
    with open(filepath, 'rb') as f:
        import pickle as pkl
        data = pkl.load(f)

    df = pd.DataFrame(columns=["regulated.gene", "regulator.gene", "regulator.effect"])
    for gene, tfs in data.items():
        for tf in tfs:
            new_row = pd.DataFrame(
                [{"regulated.gene": gene, "regulator.gene": tf, "regulator.effect": effect}]
            )

            df = pd.concat([df, new_row], ignore_index=True)
    return df

def pickle_df_grn(df, filepath, filepath_gene_idx):
    """
    Pickles a GRN dataframe to the specified filepath. Uses the gene index table
    from groundgan to convert gene names to indices for compatibility with groundgan

    Parameters
    ----------
    df (pandas.Dataframe) : df of grn 

            regulated.gene regulator.gene regulator.effect
    edge1
    edge2
    edge3
    ...

    filepath : str or Path
        Path to save the pickle file.
    filepath_gene_idx : str or Path
        Path to corresponding the gene index mapping.
    """
    with open(filepath_gene_idx, "r") as f:
        import json
        ordered_genes = json.load(f)
    gene_to_idx = {g: i for i, g in enumerate(ordered_genes)}

    #TF set from regulator column (restricted to genes in mapping)
    tf_names = {tf for tf in df["regulator.gene"] if tf in gene_to_idx}
    tf_indices = {gene_to_idx[tf] for tf in tf_names}

    #Targets should be all non-TF genes to keep full coverage and bipartite structure
    data = {i: set() for i in range(len(ordered_genes)) if i not in tf_indices}

    for gene, tf in zip(df["regulated.gene"], df["regulator.gene"]):
        gene_idx = gene_to_idx[gene]
        tf_idx = gene_to_idx[tf]

        #enforce bipartite graph: skip TF targets, skip non-TF regulators
        if gene_idx in tf_indices or tf_idx not in tf_indices:
            continue
        data[gene_idx].add(tf_idx)

    with open(filepath, "wb") as f:
        import pickle as pkl
        pkl.dump(data, f)

def csv_to_df_scRNA(file, 
                    genes_in_rows : bool = True):
    """
    input 
    file (str)/(Path) : filepath + filename of raw data
    genes_in_rows (bool) : if True, assumes genes are in rows and cells in columns

    output
    df (pandas.Dataframe) : df of raw data indexed by gene name and genes in rows

            cell1 cell2 cell3 ...
    gene1
    gene2
    gene3
    ...

    info
    expects raw data file with gene names first row or column, assumes n_cells > n_genes
    """
    FILETYPES = {
        ".txt" : "\t",
        ".tsv" : "\t",
        ".csv" : ",",
        ".h5ad" : None
    }

    if isinstance(file, str):
        file = Path(file)
    filetype = file.suffix

    if filetype == ".h5ad":
        import anndata
        adata = anndata.read_h5ad(file)
        df = pd.DataFrame(
            adata.X.toarray() if not isinstance(adata.X, np.ndarray) else adata.X,
            index=adata.obs_names,
            columns=adata.var_names
        )
        #anndata stores genes in columns, transpose to have genes in rows
        return df.T

    if genes_in_rows:
        # Common format: first column stores gene ids.
        df = pd.read_csv(file, delimiter=FILETYPES[filetype], index_col=0)
    else:
        # Columns-first matrices (genes as columns) may not include an explicit
        # index column; forcing index_col=0 would drop the first gene (e.g. G1).
        df = pd.read_csv(file, delimiter=FILETYPES[filetype], index_col=None)
        if len(df.columns) > 0 and str(df.columns[0]).startswith("Unnamed"):
            df = df.drop(columns=df.columns[0])
        df = df.T

    return df

def csv_to_df_grn(file, header=0):
    """
    input 
    file (str)|(Path) : filepath + filename of raw data
    header (int | None) : row number of header (default: 0), or None if no header

    output
    df (pandas.Dataframe) : df of grn 

            regulated.gene regulator.gene regulator.effect
    edge1
    edge2
    edge3
    ...

    info
    expects raw data file with gene names first row 
    """
    FILETYPES = {
        ".txt" : "\t",
        ".tsv" : "\t",
        ".csv" : ",",
        ".pkl" : None
    }

    if isinstance(file, str):
        file = Path(file)
    filetype = file.suffix

    #assumes groundgan pickled format if .pkl
    if filetype == ".pkl":
        df = unpickle_causal_graph(file)
        return df
    
    df = pd.read_csv(file, delimiter=FILETYPES[filetype], header=header)

    #if regulator.effect column is present, filter out any edges with zero effect
    if "regulator.effect" in df.columns:
        df = df[df["regulator.effect"] != 0]

    return df

def grnboost2_to_df_grn(grnboost_csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(grnboost_csv_path)

    #expected GRNBoost2 columns: TF, target, importance
    df = df[["TF", "target", "importance"]].copy()
    df["TF"] = df["TF"].astype(str)
    df["target"] = df["target"].astype(str)
    df["importance"] = df["importance"].astype(float)

    #map into standard df_grn
    df_grn = pd.DataFrame(
        {
            "regulator.gene": df["TF"],
            "regulated.gene": df["target"],
            "regulator.effect": df["importance"],
        }
    )
    return df_grn

def jax_qgrn_to_df_grn(genes, theta):
    """
    
    info
    directionality of edge is impossible to know"""
    records = []
    for target_idx, target in enumerate(genes):
        for control_idx, control in enumerate(genes):
            if target_idx == control_idx:
                continue
            records.append(
                {
                    "regulated.gene": target,
                    "regulator.gene": control,
                    "regulator.effect": float(theta[target_idx, control_idx]),
                }
            )

    return pd.DataFrame.from_records(
        records, columns=["regulated.gene", "regulator.gene", "regulator.effect"]
    )

"""---------------------------------
utils for creating Ground Truth GRNs
---------------------------------"""

def create_random_grn(genes, tfs,  n_edges):
    """
    creates random grn from given dataframe

    input
    df (pd.DataFrame) : dataframe with raw scRNA data
    n_genes (int) : number of genes in grn
    n_edges (int) : number of edges in grn

    output
    df_grn (pd.DataFrame) : dataframe with grn edges
    """
    assert n_edges <= len(genes) ** 2

    possible_edges = [
        (regulator, regulated)
        for regulator in tfs
        for regulated in genes
    ]

    selected_edges = np.random.choice(
        len(possible_edges), size=n_edges, replace=False
    )

    records = []
    for idx in selected_edges:
        regulator, regulated = possible_edges[idx]
        records.append(
            {
                "regulated.gene": regulated,
                "regulator.gene": regulator,
                "regulator.effect": np.random.uniform(0, 1) #random effect between 0 and 1
            }
        )

    df_grn = pd.DataFrame.from_records(
        records, columns=["regulated.gene", "regulator.gene", "regulator.effect"]
    )

    return df_grn

def create_gt_grn_trunc_powerlaw_topk(
    df_grn_inferred: pd.DataFrame,
    genes: Set[str],
    tfs: Set[str],
    alpha: float = 2.3,
    kc: float = 50.0,
    kmin: int = 1,
    kmax: int = 20,
    seed: int = 0,
    verbose: bool = False
) -> pd.DataFrame:
    """
    Info:
    create a bioligically more meaningful GT-GRN from grnboost2 output with the following scheme:
    1. sample a number of targets k for each TF from a truncated power-law distribution (with parameters alpha, kc, kmin, kmax)
    2. rank TFs by their sum of importance scores across all targets
    3. for each TF in ranked order, pick the top-k targets from grnboost2 data according to the sampled distribution

    Parameters:
    df_grn_inferred (pandas.Dataframe) : df of full grn with importance scores (e.g. from grnboost2)
            regulated.gene regulator.gene regulator.effect
    edge1
    edge2
    edge3
    ...
    genes (set of str) : set of all gene names in the system
    tfs (set of str) : set of all TF names in the system (subset of genes)
    alpha (float) : power-law exponent for sampling target counts per TF
    kc (float) : cutoff parameter for sampling target counts per TF
    kmin (int) : minimum number of targets per TF
    kmax (int) : maximum number of targets per TF
    seed (int) : random seed for reproducibility
    verbose (bool) : whether to print out additional information about the sampled distribution and GT GRN statistics

    Returns:
    df (pandas.Dataframe) : sampled GT GRN according to the above scheme
            regulated.gene regulator.gene regulator.effect
    edge1
    edge2
    edge3
    ...
    """
    # helper function to sample target counts per TF from truncated power-law distribution
    def sample_distribution(alpha, 
                       kc, 
                       n_samples, 
                       kmin=1,
                       kmax=200,
                       seed=0):
    
        def truncated_powerlaw(x, alpha, kc):
            return (x ** (-alpha)) * np.exp(-x / kc)
    
        rng = np.random.default_rng(seed)
        ks = np.arange(kmin, kmax + 1)
        weights = np.array([truncated_powerlaw(k, alpha, kc) for k in ks])
        probablities = weights / weights.sum()
        samples = rng.choice(ks, size=n_samples, replace=True, p=probablities)
        samples = np.sort(samples)[::-1] # sort ascendingly and reverse to get descending order (more targets for more important TFs)

        return samples

    # Step 1: create sampled distribution of target counts per TF (k) from truncated power-law
    # and sort descendingly to assign more targets to more important TFs
    sampled_distribution = sample_distribution(alpha=alpha, kc=kc, n_samples=len(tfs), kmin=kmin, kmax=kmax, seed=seed)
    print(f"alpha: {alpha}, kc: {kc}, kmin: {kmin}, kmax: {kmax}") if verbose else None
    print(f"Sampled target counts per TF (k): {sampled_distribution}") if verbose else None
    print(f"number of edges in GT GRN: {sum(sampled_distribution)}") if verbose else None
    print(f"number of known TF-gene edges: {len(tfs) * len(genes)}") if verbose else None
    print(f"number of gene-gene edges: {len(genes) ** 2}") if verbose else None
    print(f"edge density in GT GRN relative to known TFs->genes edges: {sum(sampled_distribution) / (len(tfs) * len(genes)):.4f}") if verbose else None
    print(f"edge density in GT GRN relative to all gene->gene edges: {sum(sampled_distribution) / (len(genes) ** 2):.4f}") if verbose else None
    
    # Step 2: rank TFs by their sum of importance scores across all targets 
    # and assign them the corresponding k from the sampled distribution
    sum_importance = df_grn_inferred.groupby("regulator.gene")["regulator.effect"].sum().reset_index()
    sum_importance = sum_importance.rename(columns={"regulator.effect": "sum_importance"})
    sum_importance = sum_importance.sort_values(by="sum_importance", ascending=False)
    sum_importance["sampled_k"] = sampled_distribution[: len(sampled_distribution)]

    # Step 3: for each TF in ranked order, pick the top-k targets from grnboost2 data according to the sampled distribution
    df_gt_grn = pd.DataFrame(columns=df_grn_inferred.columns)
    for _, row in sum_importance.iterrows():
        regulator = row["regulator.gene"]
        k = row["sampled_k"]
        df_regulator = df_grn_inferred[df_grn_inferred["regulator.gene"] == regulator]
        df_regulator = df_regulator.sort_values(by="regulator.effect", ascending=False)
        selected_targets = df_regulator.head(k)
        df_gt_grn = pd.concat([df_gt_grn, selected_targets], ignore_index=True)

    return df_gt_grn

def create_gt_grn_topk_TF_per_Gene(df_grn_inferred: pd.DataFrame,
                                genes: Set[str],
                                tfs: Set[str],
                                k: int = 1,
                                verbose: bool = False):
    """
    Create a GT-GRN by selecting the top k TFs (regulators) for each target gene.

    Parameters
    ----------
    df_grn_inferred (pandas.DataFrame) : df of full GRN with importance scores (e.g. from grnboost2)
            regulated.gene regulator.gene regulator.effect
    edge1
    edge2
    edge3
    ...
    genes (set of str) : set of all gene names in the system
    tfs (set of str) : set of all TF names in the system (subset of genes)
    k (int) : number of top TFs to select per target gene
    verbose (bool) : whether to print out additional information

    Returns
    -------
    df_gt_grn (pandas.DataFrame) : GT GRN with top k TFs per target gene
            regulated.gene regulator.gene regulator.effect
    edge1
    edge2
    edge3
    ...
    """
    df_gt_grn = pd.DataFrame(columns=df_grn_inferred.columns)
    
    # For each target gene, select the top k TFs based on importance score
    for gene in genes:
        # Get all edges targeting this gene
        df_gene = df_grn_inferred[df_grn_inferred["regulated.gene"] == gene]
        
        # Sort by importance score (regulator.effect) in descending order
        df_gene = df_gene.sort_values(by="regulator.effect", ascending=False)
        
        # Select top k TFs for this gene
        selected_tfs = df_gene.head(k)
        
        # Add to result dataframe
        df_gt_grn = pd.concat([df_gt_grn, selected_tfs], ignore_index=True)
    
    if verbose:
        print(f"Total number of genes: {len(genes)}")
        print(f"Number of edges in GT GRN: {len(df_gt_grn)}")
        print(f"Max TFs per gene: {k}")
        print(f"Expected max edges: {len(genes) * k}")
        print(f"Actual edge density: {len(df_gt_grn) / (len(genes) * k):.4f}")
    
    return df_gt_grn

def show_grn(df):
    """
    visualizes GRN Graph
    """
