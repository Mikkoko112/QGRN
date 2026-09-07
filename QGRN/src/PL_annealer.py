import pandas as pd
from pathlib import Path

from QGRN.src.ConfigParser import ConfigParser
from QGRN.src.evaluate import evaluate_n_graphs
from QGRN.src.utils_annealing import csv_to_df_scRNA, csv_to_df_grn, create_random_grn
from QGRN.src.annealing import Scores, TopologicalConstraints, Annealer

def execute_annealer(config_parser : ConfigParser, run_name, dataset_info):
    VERBOSE = config_parser.get_value("verbose", default=False)
    #---------------------------------------------------------------------------
    #build data frame from raw data
    #---------------------------------------------------------------------------
    root_path = Path(config_parser.get_value("root_path")) #gets set in main.py
    active_dataset = run_name

    required_dataset_fields = ["scRNA", "tfs", "GT_GRN", "genes", "genes_in_rows", "source"]
    missing_fields = [
        field for field in required_dataset_fields
        if field not in dataset_info
    ]
    if missing_fields:
        raise ValueError(
            f"dataset: {active_dataset} is missing required fields: "
            f"{', '.join(missing_fields)}"
        )

    scRNA_data_path = root_path / Path(dataset_info["scRNA"])
    transcription_factors_path = root_path / Path(dataset_info["tfs"])
    ground_truth_grn_path = root_path / Path(dataset_info["GT_GRN"])
    ordered_genes_path = root_path / Path(dataset_info["genes"])
    genes_in_rows = dataset_info.get("genes_in_rows")
    data_source = dataset_info.get("source")
    df_ref_grn = csv_to_df_grn(ground_truth_grn_path)
    df_raw = csv_to_df_scRNA(scRNA_data_path, genes_in_rows=genes_in_rows)

    #extract gene and tf information
    genes = pd.read_json(ordered_genes_path, typ="series").astype(str).tolist()
    ngenes = len(genes)
    full_tfs = pd.read_json(transcription_factors_path, typ="series").astype(str).tolist()
    tfs = list(set(genes).intersection(set(full_tfs)))
    print(f"tfs in beginning: {len(full_tfs)}, tfs in data: {len(tfs)}") if VERBOSE else None
    ntfs = len(tfs)

    if data_source == "GRouNdGAN":
        #map indices to gene names
        genes_idx_to_name_str = {str(idx): name for idx, name in enumerate(genes)}
        genes_idx_to_name_int = {idx: name for idx, name in enumerate(genes)}

        #scRNA index from h5ad var names is stringified indices (e.g., "0", "1", ...)
        df_raw.index = df_raw.index.map(genes_idx_to_name_str)

        #ground-truth pickle uses integer node ids; keep a string fallback for robustness
        df_ref_grn["regulated.gene"] = (
            df_ref_grn["regulated.gene"]
            .map(genes_idx_to_name_int)
            .fillna(df_ref_grn["regulated.gene"].astype(str).map(genes_idx_to_name_str))
        )
        df_ref_grn["regulator.gene"] = (
            df_ref_grn["regulator.gene"]
            .map(genes_idx_to_name_int)
            .fillna(df_ref_grn["regulator.gene"].astype(str).map(genes_idx_to_name_str))
        )


    #---------------------------------------------------------------------------
    # build qubo and perform annealing
    #---------------------------------------------------------------------------
    """
    expression scores are calculates as (s_{ij}=P(x_j=1|x_i=1)-P(x_j=1))
    inhibition scores are calculates as (s_{ij}=P(x_j=0|x_i=1)-P(x_j=0))
    where j is the target gene and i is the control gene
    """
    scores = Scores(df=df_raw,
                genes=genes,
                tfs=tfs,
                verbose=VERBOSE)
    constraints = TopologicalConstraints(genes=genes,
                                        tfs=tfs,
                                        verbose=VERBOSE)
    annealer = Annealer(verbose=VERBOSE)

    ntopedges = df_ref_grn.shape[0]
    inference_dataframes = dict()

    #SCORES

    scores_conditional_mode = config_parser.get_value("scores.conditional.mode", default=None)
    if scores_conditional_mode != None:
        pruning_threshold = config_parser.get_value("constraints.pruning.threshold", default=0.0)
        normalization_mode = config_parser.get_value("scores.conditional.normalization", default="global-min-max")
        bias_correction = config_parser.get_value("scores.conditional.bias_correction", default=True)
        if scores_conditional_mode == "combined":
            scores.scores_conditional_combined(pruning_threshhold=pruning_threshold,
                                                normalization_method=normalization_mode,
                                                bias_correction=bias_correction)
        elif scores_conditional_mode == "activation":
            scores.scores_conditional_activation(pruning_threshhold=pruning_threshold, 
                                                normalization_method=normalization_mode,
                                                bias_correction=bias_correction)
        elif scores_conditional_mode == "inhibition":
            scores.scores_conditional_inhibition(pruning_threshhold=pruning_threshold,
                                                normalization_method=normalization_mode,
                                                bias_correction=bias_correction)
        else:
            raise ValueError(
                f"edge_score_mode: '{scores_conditional_mode}' is not valid"
            )

    if config_parser.get_value("scores.grnboost2.mode", default=False):
        scores.scores_grnboost2()
    
    if config_parser.get_value("scores.genie3.mode", default=False):
        scores.scores_genie3()

    if config_parser.get_value("scores.pidc.mode", default=False):
        scores.scores_pidc()

    if config_parser.get_value("scores.ppcor.mode", default=False):
        scores.scores_ppcor(
            pval_threshold=config_parser.get_value("scores.ppcor.pval_threshold", default=0.01)
        )

    if config_parser.get_value("scores.mutual_information.mode", default=False):
        scores.scores_mutual_information()
    
    if config_parser.get_value("scores.pearson.mode", default=False):
        scores.scores_pearson_correlation()

    if config_parser.get_value("scores.spearman.mode", default=False):
        scores.scores_spearman_correlation()
    
    #CONSTRAINTS

    #sparsity
    if config_parser.get_value("constraints.sparsity.mode", default=True):
        penalty_sparsity = config_parser.get_value("constraints.sparsity.penalty", default=1e-1)
        constraints.enforce_sparsity(penalty=penalty_sparsity)
    
    #self-regulation penalty
    if config_parser.get_value("constraints.reflexive_edges.mode", default=True):
        penalty_reflexive_edges = config_parser.get_value("constraints.reflexive_edges.penalty", default=1.0)
        constraints.penalize_reflexive_edges(penalty=penalty_reflexive_edges)

    #reciprocal regulation penalty
    if config_parser.get_value("constraints.two_cycles.mode", default=False):
        penalty_two_cycles = config_parser.get_value("constraints.two_cycles.penalty", default=1.0)
        constraints.penalize_two_cycles(penalty=penalty_two_cycles)
    
    #in-degree constraints
    in_degree_mode = config_parser.get_value("constraints.in_degree.mode", default=None)
    if in_degree_mode != None:
        penalty_in_degree = config_parser.get_value("constraints.in_degree.penalty", default=1e-1)
        if in_degree_mode == "exact":
            exact_in_degree = config_parser.get_value("constraints.in_degree.target_exact", default=5)
            constraints.enforce_exact_in_degree(penalty=penalty_in_degree,
                                                exact_in_degree=exact_in_degree)
        elif in_degree_mode == "range":
            range_degree = config_parser.get_value("constraints.in_degree.target_range", default=(0,5))
            min_penalty = config_parser.get_value(
                "constraints.in_degree.min_penalty",
                default=penalty_in_degree,
            )
            max_penalty = config_parser.get_value(
                "constraints.in_degree.max_penalty",
                default=penalty_in_degree,
            )
            constraints.enforce_range_in_degree(
                min_penalty=min_penalty,
                max_penalty=max_penalty,
                min_in_degree=range_degree[0],
                max_in_degree=range_degree[1],
            )

    #out-degree constraints
    out_degree_mode = config_parser.get_value("constraints.out_degree.mode", None)
    if out_degree_mode != None:
        penalty_out_degree = config_parser.get_value("constraints.out_degree.penalty", default=1e-1)
        if out_degree_mode == "exact":
            exact_out_degree = config_parser.get_value("constraints.out_degree.target_exact", default=5)
            constraints.enforce_exact_out_degree(penalty=penalty_out_degree,
                                                exact_out_degree=exact_out_degree)
        elif out_degree_mode == "range":
            range_degree = config_parser.get_value("constraints.out_degree.target_range", default=(0,5))
            min_penalty = config_parser.get_value(
                "constraints.out_degree.min_penalty",
                default=penalty_out_degree,
            )
            max_penalty = config_parser.get_value(
                "constraints.out_degree.max_penalty",
                default=penalty_out_degree,
            )
            constraints.enforce_range_out_degree(
                min_penalty=min_penalty,
                max_penalty=max_penalty,
                min_out_degree=range_degree[0],
                max_out_degree=range_degree[1],
            )
        elif out_degree_mode == "distribution":
            constraints.enforce_distribution_out_degree(
                k_max=config_parser.get_value("constraints.out_degree.kmax", default=50),
                k_min=config_parser.get_value("constraints.out_degree.kmin", default=1),
                alpha=config_parser.get_value("constraints.out_degree.alpha", default=2.0),
                k_c=config_parser.get_value("constraints.out_degree.kc", default=25),
                one_hot_penalty=config_parser.get_value("constraints.out_degree.one_hot_penalty", default=1.0),
                global_penalty=config_parser.get_value("constraints.out_degree.global_penalty", default=0.001),
                linking_penalty=config_parser.get_value("constraints.out_degree.linking_penalty", default=0.001),
                max_overlay_penalty=config_parser.get_value("constraints.out_degree.max_overlay_penalty", default=0.0),
            )

    #ANNEALER

    score_terms = scores.get_scores()
    linear_terms, quadratic_terms, offset_terms = constraints.get_terms()
    linear_terms.append(score_terms)
    annealer = Annealer(linear_list=linear_terms,
                        quadratic_list=quadratic_terms,
                        offset_list=offset_terms,
                        verbose=VERBOSE)
    annealer.construct_qubo()

    num_reads = config_parser.get_value("annealer.num_reads", default=1000)
    num_cores = config_parser.get_value("annealer.num_cores", default=None)
    annealer_mode = config_parser.get_value("annealer.mode", default="SA")
    annealer.anneal(mode=annealer_mode, 
                    num_reads=num_reads, 
                    num_cores=num_cores)
    inference_dataframes["annealer"] = annealer.get_best_solution()

    #---------------------------------------------------------------------------
    # execute baseline methods
    #---------------------------------------------------------------------------

    nprededges = inference_dataframes["annealer"].shape[0]
    if config_parser.get_value("base_lines.random.mode", default=True):
        limit_mode = config_parser.get_value("base_lines.random.limit")
        if limit_mode == None:
            limit = ntfs * ngenes
        elif limit_mode == "n_GT_edges":
            limit = ntopedges
        elif limit_mode == "n_annealer_edges":
            limit = nprededges
        else:
            raise ValueError(
                f"limit_mode: '{limit_mode}' is not valid"
            )
        df_random = create_random_grn(genes, tfs, limit)
        inference_dataframes["random"] = df_random

    if config_parser.get_value("base_lines.self.mode", default=True):
        limit_mode = config_parser.get_value("base_lines.self.limit")
        if limit_mode == None:
            limit = ntfs * ngenes
        elif limit_mode == "n_GT_edges":
            limit = ntopedges
        elif limit_mode == "n_annealer_edges":
            limit = nprededges
        else:
            raise ValueError(
                f"limit_mode: '{limit_mode}' is not valid"
            )
        inference_dataframes["self"] = scores.get_confidence_df().sort_values(by="regulator.effect", ascending=False).head(limit) 

    if config_parser.get_value("base_lines.grnboost2.mode", default=True):
        from QGRN.src.baselines import baseline_GRNBoost2
        limit_mode = config_parser.get_value("base_lines.grnboost2.limit")
        if limit_mode == None:
            limit = ntfs * ngenes
        elif limit_mode == "n_GT_edges":
            limit = ntopedges
        elif limit_mode == "n_annealer_edges":
            limit = nprededges
        else:
            raise ValueError(
                f"limit_mode: '{limit_mode}' is not valid"
            )
        inference_dataframes["grnboost2"] = baseline_GRNBoost2(
            df_raw=df_raw,
            genes=genes,
            tfs=tfs,
            limit=limit,
            verbose=VERBOSE
        )
    
    if config_parser.get_value("base_lines.genie3.mode", default=False):
        from QGRN.src.baselines import baseline_GENIE3
        limit_mode = config_parser.get_value("base_lines.genie3.limit")
        if limit_mode == None:
            limit = ntfs * ngenes
        elif limit_mode == "n_GT_edges":
            limit = ntopedges
        elif limit_mode == "n_annealer_edges":
            limit = nprededges
        else:
            raise ValueError(
                f"limit_mode: '{limit_mode}' is not valid"
            )
        inference_dataframes["genie3"] = baseline_GENIE3(
            df_raw=df_raw,
            genes=genes,
            tfs=tfs,
            limit=limit,
            verbose=VERBOSE
        )

    #---------------------------------------------------------------------------
    # evaluate results
    #---------------------------------------------------------------------------
    
    evaluation_results = {"binary_classification":pd.DataFrame(), 
                          "confidence_evaluation":pd.DataFrame()}
    if config_parser.get_value("evaluation.binary_classification.mode", default=True):
        binary_metrics = list()
        possible_metrics = config_parser.get_value("evaluation.binary_classification.metrics", default={}).keys()
        for metric in possible_metrics:
            if config_parser.get_value(f"evaluation.binary_classification.metrics.{metric}", default=True):
                binary_metrics.append(metric)
                
        if len(binary_metrics) == 0:
            raise ValueError("At least one evaluation metric must be selected in the configuration.")
        
        binary_evaluation_results = evaluate_n_graphs(ngenes=ngenes,
                                                      ntfs=ntfs,
                                                      df_ground_truth=df_ref_grn, 
                                                      df_inferred_grns=inference_dataframes, 
                                                      metrics=binary_metrics,
                                                      verbose=VERBOSE)
        evaluation_results["binary_classification"] = binary_evaluation_results

    if config_parser.get_value("evaluation.confidence_evaluation.mode", default=False):
        #calculate confidence scores for annealer results
        interaction_scores = scores.get_confidence_scores()
        annealing_results = annealer.get_results()
        
        from QGRN.src.evaluate import calculate_confidence, calculate_ap_auprc
        print("Calculating confidence scores for annealer results...") if VERBOSE else None
        annealer_confidence_df = calculate_confidence(
                interaction_scores=interaction_scores,
                annealing_results=annealing_results,
                bqm=annealer.bqm,
                lambda_scores=config_parser.get_value("evaluation.confidence_evaluation.lambda_scores", default=0.5),
                lambda_robustness=config_parser.get_value("evaluation.confidence_evaluation.lambda_robustness", default=0.5),
                lambda_energy=config_parser.get_value("evaluation.confidence_evaluation.lambda_energy", default=0.0),
                verbose=VERBOSE,
                num_cores=config_parser.get_value("annealer.num_cores", default=1),
            )
        print("Annealer confidence scores calculated.") if VERBOSE else None
        inference_dataframes["annealer"] = annealer_confidence_df

        #get selected metrics for confidence evaluation
        confidence_metrics = list()
        possible_metrics = config_parser.get_value("evaluation.confidence_evaluation.metrics", default={}).keys()
        for metric in possible_metrics:
            if config_parser.get_value(f"evaluation.confidence_evaluation.metrics.{metric}", default=True):
                confidence_metrics.append(metric)
                
        if len(confidence_metrics) == 0:
            raise ValueError("At least one evaluation metric must be selected in the configuration.")

        #calculate confidence evaluation metrics for all inference dataframes (including annealer and baselines)
        confidence_rows = {}
        for inference_name, df_inference in inference_dataframes.items():
            results = calculate_ap_auprc(
                inferred_df=df_inference,
                ground_truth_df=df_ref_grn,
                tf_list=tfs,
                gene_list=genes
            )
            confidence_rows[inference_name] = {
                metric: results[metric] for metric in possible_metrics
            }

        confidence_evaluation_results = pd.DataFrame.from_dict(
            confidence_rows,
            orient="index",
        )
        evaluation_results["confidence_evaluation"] = confidence_evaluation_results

    if config_parser.get_value("outputs.save_inference_grns.mode", default=False):
        save_path = root_path / Path(
            config_parser.get_value(
                "outputs.save_inference_grns.path",
                default="QGRN/outputs/saved_grns",
            )
        )
        dataset_dir = save_path / active_dataset
        dataset_dir.mkdir(parents=True, exist_ok=True)

        df_ref_grn.to_csv(dataset_dir / "ground_truth.csv", index=False)

        for method_name, df_inference in inference_dataframes.items():
            df_inference.to_csv(dataset_dir / f"{method_name}.csv", index=False)

    return evaluation_results
