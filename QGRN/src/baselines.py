from distributed import Client


def _run_arboreto_baseline(algo, df_raw, genes, tfs, limit, verbose=False):
    # Avoid Arboreto's default LocalCluster(processes=True) path, which can emit
    # noisy nanny shutdown errors in this environment.
    client = Client(processes=False)
    try:
        df_inferred = algo(
            expression_data=df_raw.copy().T,
            gene_names=genes,
            tf_names=tfs,
            client_or_address=client,
            limit=limit,  # change to ntopedges to limit edges
            verbose=verbose,
        )
    finally:
        client.close()

    df_inferred.rename(
        columns={
            "TF": "regulator.gene",
            "target": "regulated.gene",
            "importance": "regulator.effect",
        },
        inplace=True,
    )
    return df_inferred


def baseline_GRNBoost2(df_raw,
                    genes,
                    tfs,
                    limit,
                    verbose = False):
    from arboreto.algo import grnboost2
    print("Running GRNBoost2...") if verbose else None
    df_grnboost2 = _run_arboreto_baseline(
        algo=grnboost2,
        df_raw=df_raw,
        genes=genes,
        tfs=tfs,
        limit=limit,
        verbose=verbose,
    )
    print("GRNBoost2 finished.") if verbose else None
    return df_grnboost2

def baseline_GENIE3(df_raw,
                    genes,
                    tfs,
                    limit,
                    verbose = False):
    from arboreto.algo import genie3
    print("Running GENIE3...") if verbose else None
    df_genie3 = _run_arboreto_baseline(
        algo=genie3,
        df_raw=df_raw,
        genes=genes,
        tfs=tfs,
        limit=limit,
        verbose=verbose,
    )
    print("GENIE3 finished.") if verbose else None
    return df_genie3

def baseline_ppcor():
    pass

def baseline_pidc():
    pass
