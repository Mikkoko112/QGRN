#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#imports for static typing
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame
from dimod import BinaryQuadraticModel
from typing import Optional, Callable

class Scores():
    def __init__(self, 
                 df : DataFrame,
                 genes : Optional[list] = None,
                 tfs : Optional[list] = None,
                 save_scores : bool = False,
                 verbose : bool = False):
        self.df = df
        if not genes:
            self.genes = self.df.index.to_list()
        else:
            self.genes = genes
        if not tfs:
            self.tfs = self.genes
        else:
            self.tfs = tfs
        self.save_scores = save_scores
        self.verbose = verbose

        self.scores = dict()
        self.scores_df = None

    def get_scores(self) -> dict:
        return self.scores

    def get_confidence_scores(self) -> dict:
        return {edge: -score for edge, score in self.scores.items()}

    def get_confidence_df(self) -> DataFrame:
        if self.scores_df is not None:
            return self.scores_df
        
        #convert scores dict to df for easier analysis
        import pandas as pd
        data = []
        for edge, score in self.get_confidence_scores().items():
            edge = edge.replace("edge:", "")
            tf, gene = edge.split("->")
            data.append({"regulator.gene": tf, "regulated.gene": gene, "regulator.effect": score})
        self.scores_df = pd.DataFrame(data)
        return self.scores_df
    
    def _set_scores(self, 
                    scores : dict,
                    normalize : bool = True):
        if normalize:
            scores = self._normalize_scores(scores)
        self.scores = scores
        self.scores_df = None
    
    def _normalize_scores(self, scores : dict):
        #1.clip positives scores to 0 since they are not inhibitory and we want to focus on the most confident edges
        #2.normalize scores to [-1,0] range for easier interpretation and combination with other metrics. 
        if not scores:
            return {}

        clipped_scores = {k: min(v, 0.0) for k, v in scores.items()}
        min_score = min(clipped_scores.values())
        
        if min_score == 0:
            return {k: 0.0 for k, v in clipped_scores.items()}
        return {k: v / abs(min_score) for k, v in clipped_scores.items()}

    @staticmethod
    def _normalize_df_scRNA(df : DataFrame, 
                            method : str) -> DataFrame:
        """
        normalizes the scRNA raw data according to the specified method

        Parameters
        ----------
        df (pd.DataFrame) : df with scRNA raw data (output of csv_to_df_scRNA) with genes in rows
        method (str) : method for normalizing the data. Choose from
                    "rowwise-min-max" : scales each gene expression to [0,1] per gene
                    "global-min-max" : scales all gene expressions to [0,1] globally

        Returns
        -------
        pd.DataFrame : normalized df with scRNA raw data
        """
        import numpy as np

        if method == "rowwise-min-max":
            #scale each gene expression to [0,1] per gene
            row_min = df.min(axis=1)
            row_max = df.max(axis=1)
            row_range = row_max - row_min

            # (df - min) / (max - min) with safe division
            df_norm = df.sub(row_min, axis=0).div(row_range.replace(0.0, np.nan), axis=0)
            return df_norm.replace([np.inf, -np.inf], np.nan).fillna(0.0)  #fill NaN values resulting from division by zero

        elif method == "global-min-max":
            #scale all gene expressions to [0,1] globally
            arr = df.to_numpy(dtype=np.float64, copy=False)
            global_min = np.nanmin(arr)
            global_max = np.nanmax(arr)
            denom = global_max - global_min

            df_norm = (df - global_min) / denom
            return df_norm.replace([np.inf, -np.inf], np.nan).fillna(0.0)  #fill NaN values resulting from division by zero

        else:
            raise ValueError(f"Unknown normalization method: {method}")

    def _validate_arboreto_inputs(self, df: DataFrame, method_name: str) -> None:
        if df.empty:
            raise ValueError(f"{method_name} received an empty expression matrix.")
        if len(self.genes) == 0:
            raise ValueError(f"{method_name} received an empty gene list.")
        if len(self.tfs) == 0:
            raise ValueError(
                f"{method_name} cannot run because no transcription factors overlap "
                "between the TF list and the active dataset."
            )
        if df.columns.duplicated().any():
            duplicate_genes = df.columns[df.columns.duplicated()].unique().tolist()
            raise ValueError(
                f"{method_name} cannot run with duplicate gene names: "
                f"{duplicate_genes[:10]}"
            )

        values = df.to_numpy(dtype=np.float64, copy=False)
        if not np.isfinite(values).all():
            raise ValueError(
                f"{method_name} cannot run because the expression matrix contains "
                "NaN or infinite values."
            )

    def _validate_container_backend_inputs(self, df: DataFrame, method_name: str) -> None:
        self._validate_arboreto_inputs(df.T, method_name)

        if shutil.which("docker") is None:
            raise RuntimeError(
                f"{method_name} requires Docker, but the 'docker' executable was not found."
            )

    def _prepare_external_scores_df(self, method_name: str) -> DataFrame:
        missing_genes = [gene for gene in self.genes if gene not in self.df.index]
        if missing_genes:
            raise ValueError(
                f"{method_name} cannot run because these configured genes are missing "
                f"from self.df: {missing_genes[:10]}"
            )

        df = self.df.loc[self.genes].copy()
        self._validate_container_backend_inputs(df, method_name)
        return df

    def _run_beeline_container(self, image: str, command: str, mount_dir: Path, method_name: str) -> None:
        # Dockerized BEELINE backends may run as a different user than the host process.
        # Relax permissions on the mounted temp directory so the container can read inputs
        # and write outputs regardless of the uid/gid it uses internally.
        mount_dir.chmod(0o777)
        for child in mount_dir.iterdir():
            if child.is_file():
                child.chmod(0o666)
            elif child.is_dir():
                child.chmod(0o777)

        docker_cmd = [
            "docker",
            "run",
            "--rm",
            "-v",
            f"{mount_dir.resolve()}:/work",
            image,
            "/bin/sh",
            "-lc",
            command,
        ]
        result = subprocess.run(
            docker_cmd,
            capture_output=not self.verbose,
            text=True,
        )
        if result.returncode != 0:
            stderr = result.stderr.strip() if result.stderr else "no stderr captured"
            raise RuntimeError(
                f"{method_name} Docker container failed with exit code {result.returncode}: "
                f"{stderr}"
            )
        if self.verbose and result.stdout:
            print(result.stdout)

    def _convert_undirected_output_to_scores(
        self,
        edges_df: DataFrame,
        gene1_col,
        gene2_col,
        score_col,
        method_name: str,
        absolute_score: bool = True,
    ) -> dict[str, float]:
        genes_set = set(self.genes)
        tfs_set = set(self.tfs)
        scores = dict()

        for _, row in edges_df.iterrows():
            gene_a = str(row[gene1_col])
            gene_b = str(row[gene2_col])

            if gene_a == gene_b:
                continue
            if gene_a not in genes_set or gene_b not in genes_set:
                continue

            score = float(row[score_col])
            if not np.isfinite(score):
                continue
            if absolute_score:
                score = abs(score)

            # Treat undirected associations as candidate directed TF->gene edges.
            if gene_a in tfs_set:
                scores[f"edge:{gene_a}->{gene_b}"] = -score
            if gene_b in tfs_set:
                scores[f"edge:{gene_b}->{gene_a}"] = -score

        if not scores:
            raise ValueError(
                f"{method_name} produced no usable TF->gene scores after converting "
                "the undirected output into directed candidate edges."
            )

        return scores

    def scores_conditional_activation(self,
                                    pruning_threshhold : float = float("-inf"),
                                    normalization_method : str = "global-min-max",
                                    bias_correction : bool = True) -> dict[str, float]:
        """
        computes the linear activation terms of the QUBO. For more Detail look into the paper:
        s_ij = (∑_k d_ik d_jk) / (∑_k d_ik) - (1/m) ∑_k d_jk

        where:
        i = control gene
        j = target gene
        k = cell index
        
        Parameters
        ----------
        df (pd.DataFrame) : df with scRNA raw data (output of csv_to_df_scRNA) with genes in rows
        penalty (float) : penalty term for the QUBO
        pruning_threshhold (float) : threshold for pruning edges based on linear score
        normalization_method (str) : method for normalizing the data. Choose from
                                    "rowwise-min-max" : scales each gene expression to [0,1] per gene
                                    "global-min-max" : scales all gene expressions to [0,1] globally
                                    "no-normalization" : no normalization

        Returns
        -------
        dict (str : float) : dictionary with edges (TF->Gene) as keys and linear scores as values
        """
        import numpy as np

        #normalize data
        df = self.df.copy()
        if normalization_method != "no-normalization":
            df = self._normalize_df_scRNA(df, normalization_method)

        #Convert to NumPy for fast linear algebra
        D = df.to_numpy(dtype=np.float64, copy=False)  #shape (n_genes, n_cells)
        present_tfs = [tf for tf in self.tfs if tf in df.index]
        tf_idxs = [df.index.get_loc(tf) for tf in present_tfs]
        D_tf = D[tf_idxs, :] #shape (n_tfs, n_cells)
        
        #mu_j = (1/m) ∑_k d_jk   (per target gene j)
        mu = D.mean(axis=1)  # shape (n_genes,)

        #numerator_ij = ∑_k d_ik * d_jk  (all pairs at once)
        numerator = D_tf @ D.T  # shape (n_tfs, n_genes)

        #denom_i = ∑_k d_ik   (per control gene i)
        denominator = D_tf.sum(axis=1)  # shape (n_genes,)

        #expected_ij = numerator_ij / denom_i (all pairs at once)
        #set expected_ij to 0 when denom_i == 0 to avoid division by zero
        expected = np.divide(
            numerator,
            denominator[:, None], #convert 1D np Array to column vector
            out=np.zeros_like(numerator),
            where=denominator[:, None] != 0
        )

        #s_ij = expected_ij - mu_j
        scores_mat = expected - mu[None, :]  # shape (n_tfs, n_genes)

        if bias_correction:
            if normalization_method != "global-min-max":
                raise ValueError("bias_correction on conditional scores should only be used with globally normalized scRNA data")
            from numpy import log2
            base_line = 1 / (1+mu) #normalize baseline 
            target_overexpression = -log2(mu+1e-12) #penalize TGs according to their overall expression
            mu_i = D_tf.mean(axis=1)
            TF_underexpression = mu_i / (mu_i + 0.01) #penalize TFs which are expressin in only roughly 1% of cells

            scores_mat *= (
                TF_underexpression[:, None] *
                base_line[None, :] *
                target_overexpression[None, :]
            )

        #prune edges below threshhold
        #create boolean mask of which scores to keep
        keep = scores_mat >= pruning_threshhold

        genes_in_df = df.index.tolist()

        #TODO: optimize memory usage here
        #Build {"control->target": score} output to match QUBO format
        scores_dict = dict()
        for i, tf in enumerate(present_tfs):
            for j, gene in enumerate(genes_in_df):
                if keep[i, j]:
                    scores_dict[f"edge:{tf}->{gene}"] = -float(scores_mat[i, j]) #flip sign for minimization
        
        self._set_scores(scores_dict)
        return scores_dict

    def scores_conditional_inhibition( self,
                                    pruning_threshhold : float = float("-inf"),
                                    normalization_method : str = "global-min-max",
                                    bias_correction : bool = True) -> dict[str, float]:
        """
        computes the linear inhibition terms of the QUBO. For more Detail look into the paper:
        s_ij = (∑_k d_ik (M_j - d_jk)) / (∑_k d_ik) - (M_j - (1/m) ∑_k d_jk)

        where:
        i = control gene
        j = target gene
        k = cell index
        M_j = max expression of gene j across all cells
        
        Parameters
        ----------
        df (pd.DataFrame) : df with scRNA raw data (output of csv_to_df_scRNA) with genes in rows
        penalty (float) : penalty term for the QUBO
        pruning_threshhold (float) : threshold for pruning edges based on linear score

        Returns
        -------
        dict (str : float) : dictionary with edges (TF->Gene) as keys and linear scores as values
        """
        import numpy as np

        #normalize data
        df = self.df.copy()
        if normalization_method != "no-normalization":
            df = self._normalize_df_scRNA(df, normalization_method)

        #Convert to NumPy for fast linear algebra
        D = df.to_numpy(dtype=np.float64, copy=False)  #shape (n_genes, n_cells)
        present_tfs = [tf for tf in self.tfs if tf in df.index]
        tf_idxs = [df.index.get_loc(tf) for tf in present_tfs]
        D_tf = D[tf_idxs, :] #shape (n_tfs, n_cells)
        
        #mu_j = (1/m) ∑_k d_jk   (per target gene j)
        mu = D.mean(axis=1)  # shape (n_genes,)

        M = D.max(axis=1)  # shape (n_genes,)

        #numerator_ij = ∑_k d_ik * (M_j - d_jk)  (all pairs at once)
        numerator = D_tf @ (M - D.T)  # shape (n_tfs, n_genes)
        
        #denom_i = ∑_k d_ik   (per control gene i)
        denominator = D_tf.sum(axis=1)  # shape (n_tfss,)

        #expected_ij = numerator_ij / denom_i (all pairs at once)
        #set expected_ij to 0 when denom_i == 0 to avoid division by zero
        expected = np.divide(
            numerator,
            denominator[:, None], #convert 1D np Array to column vector
            out=np.zeros_like(numerator),
            where=denominator[:, None] != 0
        )

        #s_ij = expected_ij - (M_j - mu_j)
        scores_mat = expected - (M - mu)[None, :]  # shape (n_genes, n_genes)

        if bias_correction:
            if normalization_method != "global-min-max":
                raise ValueError("bias_correction on conditional scores should only be used with globally normalized scRNA data")
            from numpy import log2
            base_line = 1 / (mu + 1e-12) #normalize baseline 
            target_overexpression = -log2(1 - mu + 1e-12) #penalize TGs according to their overall expression
            mu_i = D_tf.mean(axis=1)
            TF_underexpression = mu_i / (mu_i + 0.01) #penalize TFs which are expressin in only roughly 1% of cells

            scores_mat *= (
                TF_underexpression[:, None] *
                base_line[None, :] *
                target_overexpression[None, :]
            )

        #prune edges below threshhold
        #create boolean mask of which scores to keep
        keep = scores_mat >= pruning_threshhold

        genes_in_df = df.index.tolist()

        #TODO: optimize memory usage here
        #Build {"control->target": score} output to match QUBO format
        scores_dict = dict()
        for i, tf in enumerate(present_tfs):
            for j, gene in enumerate(genes_in_df):
                if keep[i, j]:
                    scores_dict[f"edge:{tf}->{gene}"] = -float(scores_mat[i, j]) #flip sign for minimization
        
        self._set_scores(scores_dict)
        return scores_dict

    def scores_conditional_combined(self,
                                    pruning_threshhold : float = float("-inf"),
                                    normalization_method : str = "global-min-max",
                                    bias_correction : bool = True) -> dict[str, float]:
        """
        computes the linear activation terms of the QUBO. For more Detail look into the paper:
        s_ij = (∑_k d_ik d_jk) / (∑_k d_ik) - (1/m) ∑_k d_jk

        where:
        i = control gene
        j = target gene
        k = cell index
        
        Parameters
        ----------
        df (pd.DataFrame) : df with scRNA raw data (output of csv_to_df_scRNA) with genes in rows
        penalty (float) : penalty term for the QUBO
        pruning_threshhold (float) : threshold for pruning edges based on linear score
        normalization_method (str) : method for normalizing the data. Choose from
                                    "rowwise-min-max" : scales each gene expression to [0,1] per gene
                                    "global-min-max" : scales all gene expressions to [0,1] globally
                                    "no-normalization" : no normalization

        Returns
        -------
        dict (str : float) : dictionary with edges (TF->Gene) as keys and linear scores as values
        """
        scores_act = self.scores_conditional_activation(
            pruning_threshhold=float("-inf"),
            normalization_method=normalization_method,
            bias_correction=bias_correction,
        )

        scores_inh = self.scores_conditional_inhibition(
            pruning_threshhold=float("-inf"),
            normalization_method=normalization_method,
            bias_correction=bias_correction,
        )

        scores_combined = {
            e: min(scores_act.get(e, float("inf")), scores_inh.get(e, float("inf")))
            for e in (scores_act.keys() | scores_inh.keys())
        }
        #TODO implement combined pruning
        
        self._set_scores(scores_combined)
        return scores_combined

    def scores_grnboost2(self) -> dict[str, float]:
        from arboreto.algo import grnboost2
        from distributed import Client
        df = self.df.copy().T
        self._validate_arboreto_inputs(df, "grnboost2")
        client = Client(processes=False)
        try:
            df_grnboost2 = grnboost2(
                expression_data=df,
                gene_names=self.genes,
                tf_names=self.tfs,
                client_or_address=client,
                verbose=self.verbose,
            )
        finally:
            client.close()
        scores = dict()
        for tf, gene, importance in df_grnboost2[["TF", "target", "importance"]].itertuples(index=False):
            scores[f"edge:{tf}->{gene}"] = float(-importance)
        
        self._set_scores(scores)
        return scores
    
    def scores_genie3(self) -> dict[str, float]:
        from arboreto.algo import genie3
        from distributed import Client
        df = self.df.copy().T
        self._validate_arboreto_inputs(df, "genie3")
        # Keep GENIE3 execution conservative to reduce Dask task cancellations
        # observed on some SERGIO datasets.
        client = Client(processes=False, n_workers=1, threads_per_worker=1)
        try:
            df_grnboost2 = genie3(
                expression_data=df,
                gene_names=self.genes,
                tf_names=self.tfs,
                client_or_address=client,
                verbose=self.verbose,
            )
        finally:
            client.close()
        scores = dict()
        for tf, gene, importance in df_grnboost2[["TF", "target", "importance"]].itertuples(index=False):
            scores[f"edge:{tf}->{gene}"] = float(-importance)
        
        self._set_scores(scores)
        return scores

    def scores_pidc(self) -> dict[str, float]:
        df = self._prepare_external_scores_df("pidc")
        with tempfile.TemporaryDirectory(prefix="qgrn_pidc_") as tmpdir:
            tmp_path = Path(tmpdir)
            input_path = tmp_path / "ExpressionData.csv"
            output_path = tmp_path / "outFile.txt"

            # Match the BEELINE PIDC runner input format: genes x cells, tab-separated.
            df.to_csv(input_path, sep="\t", header=True, index=True)
            self._run_beeline_container(
                image="grnbeeline/pidc:base",
                command=(
                    f"/usr/local/julia/bin/julia /runPIDC.jl "
                    f"/work/{input_path.name} /work/{output_path.name}"
                ),
                mount_dir=tmp_path,
                method_name="pidc",
            )

            if not output_path.exists():
                raise RuntimeError("pidc container finished but did not produce outFile.txt.")

            pidc_df = pd.read_csv(
                output_path,
                sep="\t",
                header=None,
                names=["Gene1", "Gene2", "EdgeWeight"],
            )
            pidc_df["EdgeWeight"] = pidc_df["EdgeWeight"].astype(float)

        scores = self._convert_undirected_output_to_scores(
            pidc_df,
            gene1_col="Gene1",
            gene2_col="Gene2",
            score_col="EdgeWeight",
            method_name="pidc",
            absolute_score=True,
        )
        self._set_scores(scores)
        self.get_confidence_df()
        return scores

    def scores_ppcor(self, pval_threshold: float = 0.01) -> dict[str, float]:
        df = self._prepare_external_scores_df("ppcor")
        with tempfile.TemporaryDirectory(prefix="qgrn_ppcor_") as tmpdir:
            tmp_path = Path(tmpdir)
            input_path = tmp_path / "ExpressionData.csv"
            output_path = tmp_path / "outFile.txt"

            # Match the BEELINE PPCOR runner input format: genes x cells, comma-separated.
            df.to_csv(input_path, sep=",", header=True, index=True)
            self._run_beeline_container(
                image="grnbeeline/ppcor:base",
                command=f"Rscript runPPCOR.R /work/{input_path.name} /work/{output_path.name}",
                mount_dir=tmp_path,
                method_name="ppcor",
            )

            if not output_path.exists():
                raise RuntimeError("ppcor container finished but did not produce outFile.txt.")

            ppcor_df = pd.read_csv(output_path, sep="\t", header=0)

        significant = ppcor_df.loc[ppcor_df["pValue"] <= float(pval_threshold)].copy()
        significant["EdgeWeight"] = significant["corVal"].abs()
        nonsignificant = ppcor_df.loc[ppcor_df["pValue"] > float(pval_threshold)].copy()
        nonsignificant["EdgeWeight"] = 0.0
        ranked_ppcor = pd.concat(
            [significant, nonsignificant],
            ignore_index=True,
        )

        scores = self._convert_undirected_output_to_scores(
            ranked_ppcor,
            gene1_col="Gene1",
            gene2_col="Gene2",
            score_col="EdgeWeight",
            method_name="ppcor",
            absolute_score=False,
        )
        self._set_scores(scores)
        self.get_confidence_df()
        return scores

    def scores_pearson_correlation(self) -> dict[str, float]:
        df = self.df.copy()
        df = df[df.sum(axis=1) != 0]
        pearson_corr = df.T.corr("pearson")
        tf_rows = pearson_corr.loc[pearson_corr.index.intersection(self.tfs)]
        scores = dict()
        for tf, row in tf_rows.iterrows():
            for gene in pearson_corr.columns:
                #TODO exclude self regulation
                scores[f"edge:{tf}->{gene}"] = float(-abs(row[gene]))
        
        self._set_scores(scores)
        return scores

    def scores_spearman_correlation(self):
        df = self.df.copy()
        df = df[df.sum(axis=1) != 0]
        pearson_corr = df.T.corr("spearman")
        tf_rows = pearson_corr.loc[pearson_corr.index.intersection(self.tfs)]
        scores = dict()
        for tf, row in tf_rows.iterrows():
            for gene in pearson_corr.columns:
                #TODO exclude self regulation
                scores[f"edge:{tf}->{gene}"] = float(-abs(row[gene]))
        
        self._set_scores(scores)
        return scores

    def scores_mutual_information(self):
        from sklearn.feature_selection import mutual_info_regression

        scores = {}
        for tf in self.df.index:
            if tf not in self.tfs:
                continue

            X = self.df.loc[tf].to_numpy().reshape(-1, 1)

            for gene in self.df.index:
                #TODO exclude self regulation

                y = self.df.loc[gene].to_numpy()
                mi = mutual_info_regression(X, y, random_state=0)[0]
                scores[f"edge:{tf}->{gene}"] = float(-mi)

        self._set_scores(scores)
        return scores

class TopologicalConstraints():
    def __init__(
        self,
        genes: list,
        tfs: Optional[list] = None,
        linear_list: Optional[list[dict[str, float]]] = None,
        quadratic_list: Optional[list[dict[tuple[str, str], float]]] = None,
        offset_list: Optional[list[float]] = None,
        verbose: bool = False,
    ):
        self.genes = genes
        self.tfs = self.genes if not tfs else tfs
        self.linear_list = [] if linear_list is None else linear_list
        self.quadratic_list = [] if quadratic_list is None else quadratic_list
        self.offset_list = [] if offset_list is None else offset_list
        self.verbose = verbose

    def get_terms(self) -> tuple[list[dict[str,float]], list[dict[tuple[str,str]], float], list[float]]:
        return (self.linear_list, self.quadratic_list, self.offset_list)

    def add_terms(self, 
                  linear : Optional[dict[str, float]] = None,
                  quadratic : Optional[dict[tuple[str,str], float]] = None,
                  offset : Optional[list[float]] = None) -> None:
        if linear:
            self.linear_list.append(linear)
        if quadratic:
            self.quadratic_list.append(quadratic)
        if offset:
            self.offset_list.append(offset)

    def enforce_sparsity(self,
                         penalty : float = 0.1) -> dict[str, float]:
        linear = dict()
        for tf in self.tfs:
            for gene in self.genes:
                linear[f"edge:{tf}->{gene}"] = penalty
        
        self.add_terms(linear=linear)
        return linear

    def penalize_two_cycles(self,
                        penalty : float = 1.0) -> dict[tuple[str,str], float]:
        
        """
        computes the quadratic penalty terms for two-cycles in the QUBO.
        Parameters
        ----------
        genes (list of str) : list of gene names
        penalty (float) : penalty term for the QUBO
        
        Returns
        -------
        dict (tuple(str,str) : float) : dictionary with edge pairs (TF1->Gene1, TF2->Gene2) as keys and quadratic penalty scores as values
        """

        print("Computing quadratic penalty terms for two-cycles...") if self.verbose else None
        quadratic = dict()
        for tf in self.tfs:
            for gene in self.genes:
                if tf == gene:
                    continue
                edge1 = f"edge:{tf}->{gene}"
                edge2 = f"edge:{gene}->{tf}"
                quadratic[(edge1, edge2)] = penalty

        print(f"Computed {len(quadratic)} quadratic penalty terms for two-cycles.") if self.verbose else None
        self.add_terms(quadratic=quadratic)
        return quadratic

    def penalize_reflexive_edges(self,
                                penalty : float = 1.0) -> dict[str, float]:
        """
        computes the linear penalty terms for self-loops in the QUBO.
        Parameters
        ----------
        genes (list of str) : list of gene names
        penalty (float) : penalty term for the QUBO
        
        Returns
        -------
        dict (str : float) : dictionary with edge (TF->Gene) as keys and linear penalty scores as values
        """
        print("Computing linear penalty terms for self-loops...") if self.verbose else None
        linear = dict()
        for gene in self.genes:
            linear[f"edge:{gene}->{gene}"] = penalty
        
        print(f"Computed {len(linear)} linear penalty terms for self-loops.") if self.verbose else None
        self.add_terms(linear=linear)
        return linear

    def enforce_exact_in_degree(self,
                                penalty : float = 0.1,
                                exact_in_degree : int = 1) -> dict[tuple[str,str], float]:
        """
        computes the quadratic penalty terms for exact in-degree constraints in the QUBO.
        Parameters
        ----------
        genes (list of str) : list of gene names
        exact_in_degree (int) : exact in-degree for each gene
        verbose (bool) : whether to print progress messages
        Returns
        -------
        dict (tuple(str,str) : float) : dictionary with edge pairs (TF1->Gene1, TF2->Gene2) as keys and quadratic penalty scores as values
        """
        print("Computing quadratic penalty terms for exact in-degree constraints...") if self.verbose else None
        linear = dict()
        quadratic = dict()

        for gene in self.genes:
            #build linear terms
            for tf in self.tfs:
                edge = f"edge:{tf}->{gene}"
                linear[edge] = penalty * (1 - 2*exact_in_degree)

            #build quadratic terms
            for i, tf1 in enumerate(self.tfs):
                for j, tf2 in enumerate(self.tfs):
                    if i >= j:
                        continue
                    edge1 = f"edge:{tf1}->{gene}"
                    edge2 = f"edge:{tf2}->{gene}"
                    quadratic[(edge1, edge2)] = 2 * penalty

        #compute constant shift
        offset = penalty * exact_in_degree ** 2 * len(self.genes)
        print(f"Computed {len(linear)} linear penalty terms for exact in-degree constraints.") if self.verbose else None
        print(f"Computed {len(quadratic)} quadratic penalty terms for exact in-degree constraints.") if self.verbose else None
        self.add_terms(linear=linear,
                       quadratic=quadratic,
                       offset=offset)
        return linear, quadratic, offset

    def enforce_exact_out_degree(self,
                                penalty : float = 0.1,
                                exact_out_degree : int = 1) -> dict[tuple[str,str], float]:
        """
        computes the quadratic penalty terms for exact out-degree constraints in the QUBO.
        Parameters
        ----------
        genes (list of str) : list of gene names
        exact_out_degree (int) : exact out-degree for each gene
        verbose (bool) : whether to print progress messages
        Returns
        -------
        dict (tuple(str,str) : float) : dictionary with edge pairs (TF1->Gene1, TF2->Gene2) as keys and quadratic penalty scores as values
        """
        print("Computing quadratic penalty terms for exact out-degree constraints...") if self.verbose else None

        linear = dict()
        quadratic = dict()
        for tf in self.tfs:
            #build linear terms
            for gene in self.genes:
                edge = f"edge:{tf}->{gene}"
                linear[edge] = penalty * (1 - 2*exact_out_degree)

            #build quadratic terms
            for i, gene1 in enumerate(self.genes):
                for j, gene2 in enumerate(self.genes):
                    if i >= j:
                        continue
                    edge1 = f"edge:{tf}->{gene1}"
                    edge2 = f"edge:{tf}->{gene2}"
                    quadratic[(edge1, edge2)] = 2 * penalty

        #compute constant shift
        offset = penalty * exact_out_degree ** 2 * len(self.tfs)
        print(f"Computed {len(linear)} linear penalty terms for exact out-degree constraints.") if self.verbose else None
        print(f"Computed {len(quadratic)} quadratic penalty terms for exact out-degree constraints.") if self.verbose else None
        self.add_terms(linear=linear,
                       quadratic=quadratic,
                       offset=offset)
        return linear, quadratic, offset

    def enforce_range_in_degree(self,
                            min_penalty: float = 0.1,
                            max_penalty: float = 0.1,
                            min_in_degree: int = 1,
                            max_in_degree: int = 10) -> dict[tuple[str, str], float]:
        """
        Computes linear/quadratic penalty terms for enforcing

            min_in_degree <= in-degree(gene) <= max_in_degree

        using:
        1. max-degree slack     S_j + s_j = K_max
        2. range restriction    s_j + t_j = K_max - K_min

        where
        S_j = sum_i x_ij
        s_j = sum_b 2^b z^s_{b,j}
        t_j = sum_b 2^b z^t_{b,j}

        Parameters
        ----------
        min_penalty : float
            Penalty for restricting the first slack variable s_j to the range
            [0, max_in_degree - min_in_degree].
        max_penalty : float
            Penalty for the max in-degree constraint S_j + s_j = max_in_degree.
        min_in_degree : int
            Minimum allowed in-degree for each gene.
        max_in_degree : int
            Maximum allowed in-degree for each gene.

        Returns
        -------
        tuple[dict, dict, float]
            linear, quadratic, offset
        """
        print("Computing penalty terms for range in-degree constraints...") if self.verbose else None
        from numpy import ceil, log2

        if min_in_degree < 0:
            raise ValueError("min_in_degree must be >= 0")
        if max_in_degree < 0:
            raise ValueError("max_in_degree must be >= 0")
        if min_in_degree > max_in_degree:
            raise ValueError("min_in_degree must be <= max_in_degree")

        def add_linear(term_dict: dict[str, float], key: str, value: float) -> None:
            term_dict[key] = term_dict.get(key, 0.0) + value

        def add_quadratic(term_dict: dict[tuple[str, str], float],
                        u: str,
                        v: str,
                        value: float) -> None:
            key = (u, v) if u < v else (v, u)
            term_dict[key] = term_dict.get(key, 0.0) + value

        linear = dict()
        quadratic = dict()
        offset = 0.0

        # -------------------------------------------------
        # Enforce max in-degree
        # -------------------------------------------------
        n_bits_s = int(ceil(log2(max_in_degree + 1))) if max_in_degree > 0 else 1

        for gene in self.genes:
            # Linear terms for edge vars x_ij
            for tf in self.tfs:
                edge = f"edge:{tf}->{gene}"
                add_linear(linear, edge, max_penalty * (1 - 2 * max_in_degree))

            # Linear terms for slack vars s_j
            for b in range(n_bits_s):
                slack_s = f"slack:in_{b}_{gene}"
                add_linear(
                    linear,
                    slack_s,
                    (2 ** b) * max_penalty * ((2 ** b) - 2 * max_in_degree)
                )

            # Quadratic terms for edge-edge pairs
            for i, tf1 in enumerate(self.tfs):
                for j, tf2 in enumerate(self.tfs):
                    if i >= j:
                        continue
                    edge1 = f"edge:{tf1}->{gene}"
                    edge2 = f"edge:{tf2}->{gene}"
                    add_quadratic(quadratic, edge1, edge2, 2 * max_penalty)

            # Quadratic terms for slack_s-slack_s pairs
            for b1 in range(n_bits_s):
                for b2 in range(b1 + 1, n_bits_s):
                    slack1 = f"slack:in_{b1}_{gene}"
                    slack2 = f"slack:in_{b2}_{gene}"
                    add_quadratic(
                        quadratic,
                        slack1,
                        slack2,
                        (2 ** (b1 + b2 + 1)) * max_penalty
                    )

            # Quadratic terms for edge-slack_s pairs
            for tf in self.tfs:
                edge = f"edge:{tf}->{gene}"
                for b in range(n_bits_s):
                    slack_s = f"slack:in_{b}_{gene}"
                    add_quadratic(
                        quadratic,
                        edge,
                        slack_s,
                        (2 ** (b + 1)) * max_penalty
                    )

        offset += max_penalty * (max_in_degree ** 2) * len(self.genes)

        # -------------------------------------------------
        # Enforce min in-degree
        # -------------------------------------------------
        D = max_in_degree - min_in_degree

        for gene in self.genes:
            # General case: introduce second slack t_j with enough bits for 0..D
            n_bits_t = int(ceil(log2(D + 1)))

            # Linear terms for existing slack vars s_j
            for b in range(n_bits_s):
                slack_s = f"slack:in_{b}_{gene}"
                add_linear(
                    linear,
                    slack_s,
                    (2 ** b) * min_penalty * ((2 ** b) - 2 * D)
                )

            # Linear terms for new slack vars t_j
            for b in range(n_bits_t):
                slack_t = f"slack:range_{b}_{gene}"
                add_linear(
                    linear,
                    slack_t,
                    (2 ** b) * min_penalty * ((2 ** b) - 2 * D)
                )

            # Quadratic terms among s_j bits
            for b1 in range(n_bits_s):
                for b2 in range(b1 + 1, n_bits_s):
                    slack1 = f"slack:in_{b1}_{gene}"
                    slack2 = f"slack:in_{b2}_{gene}"
                    add_quadratic(
                        quadratic,
                        slack1,
                        slack2,
                        (2 ** (b1 + b2 + 1)) * min_penalty
                    )

            # Quadratic terms among t_j bits
            for b1 in range(n_bits_t):
                for b2 in range(b1 + 1, n_bits_t):
                    slack1 = f"slack:range_{b1}_{gene}"
                    slack2 = f"slack:range_{b2}_{gene}"
                    add_quadratic(
                        quadratic,
                        slack1,
                        slack2,
                        (2 ** (b1 + b2 + 1)) * min_penalty
                    )

            # Cross terms between s_j and t_j bits
            for bs in range(n_bits_s):
                slack_s = f"slack:in_{bs}_{gene}"
                for bt in range(n_bits_t):
                    slack_t = f"slack:range_{bt}_{gene}"
                    add_quadratic(
                        quadratic,
                        slack_s,
                        slack_t,
                        (2 ** (bs + bt + 1)) * min_penalty
                    )

            offset += min_penalty * (D ** 2)

        print(f"Computed {len(linear)} linear penalty terms for range in-degree constraints.") if self.verbose else None
        print(f"Computed {len(quadratic)} quadratic penalty terms for range in-degree constraints.") if self.verbose else None

        self.add_terms(
            linear=linear,
            quadratic=quadratic,
            offset=offset
        )
        return linear, quadratic, offset

    def enforce_range_out_degree(self,
                            min_penalty: float = 0.1,
                            max_penalty: float = 0.1,
                            min_out_degree: int = 1,
                            max_out_degree: int = 10) -> dict[tuple[str, str], float]:
        """
        Computes linear/quadratic penalty terms for enforcing

            min_out_degree <= out-degree(TF) <= max_out_degree

        using:
        1. max-degree slack     S_i + s_i = K_max
        2. range restriction    s_i + t_i = K_max - K_min

        where
        S_i = sum_j x_ij
        s_i = sum_b 2^b z^s_{b,i}
        t_i = sum_b 2^b z^t_{b,i}

        Parameters
        ----------
        min_penalty : float
            Penalty for restricting the first slack variable s_i to the range
            [0, max_out_degree - min_out_degree].
        max_penalty : float
            Penalty for the max out-degree constraint S_i + s_i = max_out_degree.
        min_out_degree : int
            Minimum allowed out-degree for each TF.
        max_out_degree : int
            Maximum allowed out-degree for each TF.

        Returns
        -------
        tuple[dict, dict, float]
            linear, quadratic, offset
        """
        print("Computing penalty terms for range out-degree constraints...") if self.verbose else None
        from numpy import ceil, log2

        if min_out_degree < 0:
            raise ValueError("min_out_degree must be >= 0")
        if max_out_degree < 0:
            raise ValueError("max_out_degree must be >= 0")
        if min_out_degree > max_out_degree:
            raise ValueError("min_out_degree must be <= max_out_degree")

        def add_linear(term_dict: dict[str, float], key: str, value: float) -> None:
            term_dict[key] = term_dict.get(key, 0.0) + value

        def add_quadratic(term_dict: dict[tuple[str, str], float],
                        u: str,
                        v: str,
                        value: float) -> None:
            key = (u, v) if u < v else (v, u)
            term_dict[key] = term_dict.get(key, 0.0) + value

        linear = dict()
        quadratic = dict()
        offset = 0.0

        # -------------------------------------------------
        # Enforce max out-degree
        # -------------------------------------------------
        n_bits_s = int(ceil(log2(max_out_degree + 1))) if max_out_degree > 0 else 1

        for tf in self.tfs:
            # Linear terms for edge vars x_ij
            for gene in self.genes:
                edge = f"edge:{tf}->{gene}"
                add_linear(linear, edge, max_penalty * (1 - 2 * max_out_degree))

            # Linear terms for slack vars s_i
            for b in range(n_bits_s):
                slack_s = f"slack:out_{b}_{tf}"
                add_linear(
                    linear,
                    slack_s,
                    (2 ** b) * max_penalty * ((2 ** b) - 2 * max_out_degree)
                )

            # Quadratic terms for edge-edge pairs
            for i, gene1 in enumerate(self.genes):
                for j, gene2 in enumerate(self.genes):
                    if i >= j:
                        continue
                    edge1 = f"edge:{tf}->{gene1}"
                    edge2 = f"edge:{tf}->{gene2}"
                    add_quadratic(quadratic, edge1, edge2, 2 * max_penalty)

            # Quadratic terms for slack_s-slack_s pairs
            for b1 in range(n_bits_s):
                for b2 in range(b1 + 1, n_bits_s):
                    slack1 = f"slack:out_{b1}_{tf}"
                    slack2 = f"slack:out_{b2}_{tf}"
                    add_quadratic(
                        quadratic,
                        slack1,
                        slack2,
                        (2 ** (b1 + b2 + 1)) * max_penalty
                    )

            # Quadratic terms for edge-slack_s pairs
            for gene in self.genes:
                edge = f"edge:{tf}->{gene}"
                for b in range(n_bits_s):
                    slack_s = f"slack:out_{b}_{tf}"
                    add_quadratic(
                        quadratic,
                        edge,
                        slack_s,
                        (2 ** (b + 1)) * max_penalty
                    )

        offset += max_penalty * (max_out_degree ** 2) * len(self.tfs)

        # -------------------------------------------------
        # Enforce min out-degree 
        # -------------------------------------------------
        D = max_out_degree - min_out_degree

        for tf in self.tfs:
            # General case: introduce second slack t_i with enough bits for 0..D
            n_bits_t = int(ceil(log2(D + 1)))

            # Linear terms for existing slack vars s_i
            for b in range(n_bits_s):
                slack_s = f"slack:out_{b}_{tf}"
                add_linear(
                    linear,
                    slack_s,
                    (2 ** b) * min_penalty * ((2 ** b) - 2 * D)
                )

            # Linear terms for new slack vars t_i
            for b in range(n_bits_t):
                slack_t = f"slack:out_range_{b}_{tf}"
                add_linear(
                    linear,
                    slack_t,
                    (2 ** b) * min_penalty * ((2 ** b) - 2 * D)
                )

            # Quadratic terms among s_i bits
            for b1 in range(n_bits_s):
                for b2 in range(b1 + 1, n_bits_s):
                    slack1 = f"slack:out_{b1}_{tf}"
                    slack2 = f"slack:out_{b2}_{tf}"
                    add_quadratic(
                        quadratic,
                        slack1,
                        slack2,
                        (2 ** (b1 + b2 + 1)) * min_penalty
                    )

            # Quadratic terms among t_i bits
            for b1 in range(n_bits_t):
                for b2 in range(b1 + 1, n_bits_t):
                    slack1 = f"slack:out_range_{b1}_{tf}"
                    slack2 = f"slack:out_range_{b2}_{tf}"
                    add_quadratic(
                        quadratic,
                        slack1,
                        slack2,
                        (2 ** (b1 + b2 + 1)) * min_penalty
                    )

            # Cross terms between s_i and t_i bits
            for bs in range(n_bits_s):
                slack_s = f"slack:out_{bs}_{tf}"
                for bt in range(n_bits_t):
                    slack_t = f"slack:out_range_{bt}_{tf}"
                    add_quadratic(
                        quadratic,
                        slack_s,
                        slack_t,
                        (2 ** (bs + bt + 1)) * min_penalty
                    )

            offset += min_penalty * (D ** 2)

        print(f"Computed {len(linear)} linear penalty terms for range out-degree constraints.") if self.verbose else None
        print(f"Computed {len(quadratic)} quadratic penalty terms for range out-degree constraints.") if self.verbose else None

        self.add_terms(
            linear=linear,
            quadratic=quadratic,
            offset=offset
        )
        return linear, quadratic, offset

    def enforce_distribution_out_degree(
        self,
        k_max: int = 100,
        k_min: int = 1,
        alpha: float = 2.0,
        k_c: int = 50,
        one_hot_penalty: float = 1.0,
        global_penalty: float = 0.01,
        linking_penalty: float = 0.001,
        max_overlay_penalty: float = 0.0,
    ):
        """
        Computes QUBO penalty terms that encourage a truncated power-law distribution
        on TF out-degree using logarithmic bins.

        Returns
        -------
        tuple[dict[str, float], dict[tuple[str, str], float], float]
            linear, quadratic, offset

        Variable naming convention
        --------------------------
        edge variables:
            "edge:{tf}->{gene}"
        bin variables:
            "bin:{b}_{tf}"
        """
        assert k_max > 0
        assert 1 <= k_min <= k_max

        print("Computing penalty terms for truncated power-law TF out-degree distribution...")  if self.verbose else None

        from math import ceil, log2, exp

        tfs = list(self.tfs)
        genes = list(self.genes)
        n_tfs = len(tfs)

        linear: dict[str, float] = {}
        quadratic: dict[tuple[str, str], float] = {}
        offset = 0.0

        def add_linear(var: str, value: float):
            linear[var] = linear.get(var, 0.0) + value

        def add_quadratic(var1: str, var2: str, value: float):
            key = (var1, var2) if var1 <= var2 else (var2, var1)
            quadratic[key] = quadratic.get(key, 0.0) + value

        def truncated_powerlaw(k: int) -> float:
            return (k ** (-alpha)) * exp(-k / k_c)

        # ------------------------------------------------------------------
        # Prepare logarithmic bins B_b = [L_b, U_b]
        # ------------------------------------------------------------------
        raw_n_bins = int(ceil(log2(k_max + 1)))
        raw_bins = [(2 ** (b - 1), min(2 ** b - 1, k_max)) for b in range(1, raw_n_bins + 1)]
        bins = []
        for lower, upper in raw_bins:
            clipped_lower = max(lower, k_min)
            if clipped_lower <= upper:
                bins.append((clipped_lower, upper))
        n_bins = len(bins)

        #calculate representatives
        representatives = [(lower + upper) / 2.0 for lower, upper in bins]

        # ------------------------------------------------------------------
        # Expected TF counts per bin from truncated power law
        # ------------------------------------------------------------------
        bin_masses = []
        for L_b, U_b in bins:
            mass = sum(truncated_powerlaw(k) for k in range(L_b, U_b + 1))
            bin_masses.append(mass)

        z = sum(bin_masses)
        probs = [mass / z for mass in bin_masses]
        expected_tfs = [n_tfs * p for p in probs]

        #Precompute variable names
        edge_vars_by_tf: dict[str, list[str]] = {
            tf: [f"edge:{tf}->{gene}" for gene in genes] for tf in tfs
        }
        bin_vars_by_tf: dict[str, list[str]] = {
            tf: [f"bin:{b}_{tf}" for b in range(n_bins)] for tf in tfs
        }

        import time
        # ------------------------------------------------------------------
        # one-hot penalty
        # ------------------------------------------------------------------
        start = time.time()
        for tf in tfs:
            y_vars = bin_vars_by_tf[tf]

            #linear
            for y in y_vars:
                add_linear(y, -one_hot_penalty)

            #quadratic
            for b1 in range(n_bins):
                y1 = y_vars[b1]
                for b2 in range(b1 + 1, n_bins):
                    y2 = y_vars[b2]
                    add_quadratic(y1, y2, 2.0 * one_hot_penalty)

            offset += one_hot_penalty
        end = time.time()
        print(f"finished one-hot terms in {end-start}s") if self.verbose else None
        # ------------------------------------------------------------------
        # global histogram matching
        # ------------------------------------------------------------------#
        start = time.time()
        for b in range(n_bins):
            E_b = expected_tfs[b]

            #linear
            for tf in tfs:
                y = bin_vars_by_tf[tf][b]
                add_linear(y, global_penalty * (1.0 - 2.0 * E_b))

            #quadratic 
            for i in range(n_tfs):
                y1 = bin_vars_by_tf[tfs[i]][b]
                for k in range(i + 1, n_tfs):
                    y2 = bin_vars_by_tf[tfs[k]][b]
                    add_quadratic(y1, y2, 2.0 * global_penalty)

            offset += global_penalty * (E_b ** 2)
        end = time.time()
        print(f"finished global histogram matching in {end-start}s") if self.verbose else None
        # ------------------------------------------------------------------
        # linking penalty
        # ------------------------------------------------------------------
        start = time.time()
        for tf in tfs:
            x_vars = edge_vars_by_tf[tf]
            y_vars = bin_vars_by_tf[tf]

            #x-linear
            for x in x_vars:
                add_linear(x, linking_penalty)

            #x-x quadratic
            n_edges = len(x_vars)
            for j1 in range(n_edges):
                x1 = x_vars[j1]
                for j2 in range(j1 + 1, n_edges):
                    x2 = x_vars[j2]
                    add_quadratic(x1, x2, 2.0 * linking_penalty)

            #y-linear
            for b in range(n_bins):
                y = y_vars[b]
                add_linear(y, linking_penalty * (representatives[b] ** 2))

            #y-y quadratic
            for b1 in range(n_bins):
                y1 = y_vars[b1]
                k1 = representatives[b1]
                for b2 in range(b1 + 1, n_bins):
                    y2 = y_vars[b2]
                    k2 = representatives[b2]
                    add_quadratic(y1, y2, 2.0 * linking_penalty * k1 * k2)

            #x-y cross terms
            for x in x_vars:
                for b in range(n_bins):
                    y = y_vars[b]
                    add_quadratic(x, y, -2.0 * linking_penalty * representatives[b])
        end = time.time()
        print(f"finished global histogram matching in {end-start}s") if self.verbose else None
        if self.verbose:
            print(f"Bins: {bins}")
            print(f"Representatives: {representatives}")
            print(f"Expected TFs per bin: {expected_tfs}")
            print(f"Computed {len(linear)} linear penalty terms for TF out-degree distribution.")
            print(f"Computed {len(quadratic)} quadratic penalty terms for TF out-degree distribution.")

        self.add_terms(linear=linear, 
                       quadratic=quadratic, 
                       offset=offset)

        # Optional soft max overlay to anchor the realized out-degree closer to the
        # support of the target distribution without changing the main architecture.
        if max_overlay_penalty and max_overlay_penalty > 0:
            self.enforce_range_out_degree(
                min_penalty=0.0,
                max_penalty=max_overlay_penalty,
                min_out_degree=0,
                max_out_degree=k_max,
            )
        return linear, quadratic, offset
    
class Annealer():
    def __init__(
        self,
        linear_list: Optional[list[dict[str, float]]] = None,
        quadratic_list: Optional[list[dict[tuple[str, str], float]]] = None,
        offset_list: Optional[list[float]] = None,
        bqm: Optional[BinaryQuadraticModel] = None,
        verbose: bool = False,
    ):
        self.linear_list = [] if linear_list is None else linear_list
        self.quadratic_list = [] if quadratic_list is None else quadratic_list
        self.offset_list = [] if offset_list is None else offset_list
        self.verbose = verbose
        self.bqm = bqm
        self.results = None

    def set_bqm(self,
                bqm : BinaryQuadraticModel) -> None:
        self.bqm = bqm
    
    def add_terms(self,
                  linear : Optional[dict[str, float]] = None,
                  quadratic : Optional[dict[tuple[str, str], float]] = None,
                  offset : Optional[float]= None) -> None:
        self.linear_list.append(linear) if linear else None
        self.quadratic_list.append(quadratic) if quadratic else None
        self.offset_list.append(offset) if offset else None

    def get_results(self) -> DataFrame:
        return self.results
    
    def construct_qubo(self) -> BinaryQuadraticModel:
        """
        Constructs a Binary Quadratic Model (BQM) from given list linear and quadratic terms.

        Parameters
        ----------
        linear_list (list of dict): List of dictionaries representing linear terms.
        {"edge:TF1->Gene1": value,
        "edge:TF2->Gene2": value,
        "slack:0_Gene1": value,
        "slack:1_Gene1": value,
        ...}
        
        quadratic_list (list of dict): List of dictionaries representing quadratic terms.
        {("edge:TF1->Gene1", "edge:TF2->Gene2"): value,
        ("edge:TF3->Gene3", "edge:TF4->Gene4"): value,
        ("slack:0_Gene1", "slack:1_Gene1"): value,
        ("edge:TF1->Gene1", "slack:0_Gene1"): value,
        ...}

        offset (float): Constant offset for the BQM.
        verbose (bool): Whether to print progress messages.

        Returns
        -------
        BinaryQuadraticModel: The constructed BQM.
        """
        from dimod import BinaryQuadraticModel

        total_offset = sum(self.offset_list)
        bqm = BinaryQuadraticModel({}, {}, total_offset, vartype='BINARY')

        for linear in self.linear_list:
            bqm.add_linear_from(linear)

        for quadratic in self.quadratic_list:
            bqm.add_quadratic_from(quadratic)

        print(f"Number of Variables in BQM: {len(bqm.variables)}") if self.verbose else None
        print(f"Minimal linear score in BQM: {min(bqm.linear.values())}") if self.verbose else None

        self.set_bqm(bqm)
        return bqm

    def anneal( self,
               mode = "SA",
                num_reads : int = 100,
                num_cores : Optional[int] = None) -> DataFrame:
        """
        Parameters
        ----------
        mode (str) : which annealer to execute
            SA : simulated annealer
            SQA : simulated quantum annealer
        num_reads (int) : number of reads for the annealer
        num_cores (int) : number of cores to use for parallel annealing

        Returns
        -------
        df (pd.DataFrame) : df with linear scores for each edges in form
        control target score of the top k highest scores

        info
        this implementation simplifies the calculation of a QUBO with constant penalty
        terms, as the solutions are simply the top k edges with the highest score
        """
        if not self.bqm:
            raise ValueError("BinaryQuadraticModel is not initialized. Call construct_qubo() before anneal or pass bqm in Annealer constructor")
        #use Simulated Annealing to solve QUBO with constant penalty terms
        from dwave.samplers import PathIntegralAnnealingSampler, SimulatedAnnealingSampler
        from dimod import BinaryQuadraticModel, concatenate
        import ray

        #check available cores if not manually specified
        if num_cores is None:
            total_cpus = int(ray.cluster_resources().get("CPU", 1))
            num_cores = max(1, total_cpus)
        
        print(f"Using {num_cores} cores for annealing.") if self.verbose else None

        # Ray can fail to pickle cython-backed BQM objects on some systems.
        # Send a serializable payload and reconstruct BQM in the worker.
        bqm_payload = self.bqm.to_serializable(use_bytes=False)

        @ray.remote
        def run_SA(bqm_serialized, num_reads):
            bqm_local = BinaryQuadraticModel.from_serializable(bqm_serialized)
            sampler = SimulatedAnnealingSampler()
            response = sampler.sample(bqm_local, num_reads=num_reads)
            return response
        
        @ray.remote
        def run_SQA(bqm_serialized, num_reads):
            bqm_local = BinaryQuadraticModel.from_serializable(bqm_serialized)
            sampler = PathIntegralAnnealingSampler()
            response = sampler.sample(bqm_local, num_reads=num_reads)
            return response
        
        reads_per_core = max(1, num_reads // num_cores)
        print("Running Annealer...") if self.verbose else None
        if mode == "SA":
            futures = [run_SA.remote(bqm_payload, num_reads=reads_per_core) for _ in range(num_cores)]
            results = ray.get(futures)
        elif mode == "SQA":
            futures = [run_SQA.remote(bqm_payload, num_reads=reads_per_core) for _ in range(num_cores)]
            results = ray.get(futures)
        else:
            raise ValueError(f"Invalid Annealer '{mode}' was selected")
        
        print("Annealing complete.") if self.verbose else None
        combined_results = concatenate(results).aggregate()

        self.results = combined_results
        return combined_results
        
    def get_best_solution(self):
        if self.results is None:
            raise ValueError("No results available. Call anneal() before get_best_solution().")
        best_sample = dict(self.results.first.sample)
        selected_edges = [
            var for var, value in best_sample.items()
            if value == 1 and var.startswith("edge:")
        ]
        selected_edges = [edge[len("edge:"):] for edge in selected_edges] #remove "edge:" prefix
        
        df_top_edges = DataFrame(
            [edge.split("->") for edge in selected_edges],
            columns=["regulator.gene", "regulated.gene"]
        )
        return df_top_edges
    
    def to_grn(self):
        """
        save as grn.csv output

        input
        df (pd.DataFrame) : df with linear scores for each edges in form
        control target score of the top k highest scores (Output from annealer)

        output
        None
        """
        #TODO
