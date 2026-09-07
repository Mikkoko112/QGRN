import json

class DataParser():
    '''
    Input: 
    config_parser (ConfigParser)

    Output:
    dict: {run_name1:
    {
        scRNA_data_path,
        genes_in_rows (bool),
        GT_GRN_path
        genes.json,
        tfs.json
    },
    ...}

    Info:
    If files are not in original data, they are created here according to custom 
    procedure

    DataParser reads a config file an does the following:
        1. Check if single run or multi run is chosen
        if single run:
            2. return file paths to single run files
        else:
            2. preprocess data into file formats if needed
            3. save them into data/prepared/<Source>/run_namex
            4. return file paths to multi run files

    '''
    def __init__(self, config_parser):
        self.config_parser = config_parser
        self.root_path = self.config_parser.get_value("root_path")
        self.prepared_root = self.root_path / "QGRN" / "data" / "prepared"
        self.prepared_root.mkdir(parents=True, exist_ok=True)

    def parse(self):
        data_mode = self.config_parser.get_value("data.mode")

        if data_mode == 'single_run':
            active_dataset = self.config_parser.get_value("data.single_run.active_dataset")
            dataset_info = self.config_parser.get_value(f"data.datasets.{active_dataset}")
            return {active_dataset: dataset_info}
        
        elif data_mode == 'multi_run':
            source = self.config_parser.get_value("data.multi_run.source")
            prepared = self.config_parser.get_value("data.multi_run.prepared")
            runs_path = self.root_path / "QGRN" / "data" / "prepared" / source / "runs.json"
            preprocessors = {
                "DREAM5": self.DREAM5_preprocess,
                "DREAM4-Multifactorial": self.DREAM4_multifactorial_preprocess,
                "BEELINE-Curated": self.BEELINE_curated_preprocess,
                "BEELINE-Synthetic": self.BEELINE_synthetic_preprocess,
                "SERGIO100": self.SERGIO100_preprocess,
            }

            if not prepared:
                if source not in preprocessors:
                    raise ValueError(
                        f"Invalid source: {source}. Unknown sources must be "
                        "prepared manually and configured with data.multi_run.prepared: true."
                    )
                preprocessors[source](source)

            if not runs_path.exists():
                raise FileNotFoundError(
                    f"Could not find prepared run metadata for source '{source}': "
                    f"{runs_path}"
                )

            with open(runs_path, "r") as f:
                return json.load(f)
        else:
            raise ValueError(f"Invalid mode: {data_mode}. Expected 'single_run' or 'multiple_runs'.")
        
    def BEELINE_curated_preprocess(self, source):
        '''
        Implement the preprocessing logic for BEELINE dataset here.
        This function should read the raw data, preprocess it, and save the prepared data to prepared_data_path.
        It should Additionally save a JSON file with all data paths.
        '''
        from QGRN.src.utils_annealing import csv_to_df_grn
        curated_root = self.root_path / "QGRN" / "data" / "BEELINE-data" / "inputs" / "Curated"
        datasets = ["GSD", "GSD-q50", "GSD-q70", 
                    "HSC", "HSC-q50", "HSC-q70", 
                    "mCAD", "mCAD-q50", "mCAD-q70", 
                    "VSC", "VSC-q50", "VSC-q70"]
        
        run_paths = dict()
        for dataset in datasets:
            dataset_path = curated_root / dataset
            ground_truth_path = dataset_path / "GroundTruthNetwork.csv"
            df_GT = csv_to_df_grn(ground_truth_path).rename(
                columns={
                    "Gene1": "regulator.gene",
                    "Gene2": "regulated.gene",
                    "Type": "regulator.effect",
                }
            )
            tfs = df_GT[df_GT["regulator.effect"] != 0]["regulator.gene"].unique().tolist()
            genes = list(set(df_GT["regulated.gene"].unique().tolist()).union(set(tfs)))

            target_dir = self.prepared_root / source / dataset
            target_dir.mkdir(parents=True, exist_ok=True)

            target_path_genes = target_dir / "genes.json"
            target_path_tfs = target_dir / "tfs.json"
            target_path_GT = target_dir / "GT_GRN.csv" 
            
            with open(target_path_genes, "w") as f:
                json.dump(genes, f)

            with open(target_path_tfs, "w") as f:
                json.dump(tfs, f)
            
            df_GT.to_csv(target_path_GT, sep=",", index=False)

            #extract only the first run of the batch, as is used in the example from BEELINE
            runs = [run for run in dataset_path.iterdir() if run.is_dir()]
            first_run = sorted(runs)[0]
            scRNA = dataset_path / first_run / "ExpressionData.csv"
            target_path_scRNA = target_dir / "scRNA.csv"
            with open(scRNA, "r") as f_in, open(target_path_scRNA, "w") as f_out:
                header = f_in.readline()
                f_out.write(header)
                for line in f_in:
                    f_out.write(line)
                
            run_key = f"{dataset}_{first_run.name}"
            run_paths[run_key] = {
                "genes" : str(target_path_genes),
                "tfs" : str(target_path_tfs),
                "GT_GRN" : str(target_path_GT),
                "scRNA" : str(target_path_scRNA),
                "genes_in_rows": True,
                "source": source
            }
        
        runs_path = self.root_path / "QGRN" / "data" / "prepared" / source / "runs.json"
        with open(runs_path, "w") as f:
            json.dump(run_paths, f, indent=4)

    def BEELINE_synthetic_preprocess(self, source):
        '''
        Implement the preprocessing logic for BEELINE synthetic dataset here.
        This function should read the raw data, preprocess it, and save the prepared data to prepared_data_path.
        It should Additionally save a JSON file with all data paths.
        This takes the 6 synthetic datasets from BEELINE with 200 cells each and the first run of the batch,
        as is used in the example from BEELINE
        '''
        from QGRN.src.utils_annealing import csv_to_df_grn
        curated_root = self.root_path / "QGRN" / "data" / "BEELINE-data" / "inputs" / "Synthetic"
        datasets = ["dyn-BF", "dyn-BFC","dyn-CY","dyn-LI","dyn-LL","dyn-TF"]
        
        run_paths = dict()
        for dataset in datasets:
            dataset_path = curated_root / dataset
            ground_truth_path = dataset_path / "GroundTruthNetwork.csv"
            df_GT = csv_to_df_grn(ground_truth_path).rename(
                columns={
                    "Gene1": "regulator.gene",
                    "Gene2": "regulated.gene",
                    "Type": "regulator.effect",
                }
            )
            tfs = df_GT[df_GT["regulator.effect"] != 0]["regulator.gene"].unique().tolist()
            genes = list(set(df_GT["regulated.gene"].unique().tolist()).union(set(tfs)))

            target_dir = self.prepared_root / source / dataset
            target_dir.mkdir(parents=True, exist_ok=True)

            target_path_genes = target_dir / "genes.json"
            target_path_tfs = target_dir / "tfs.json"
            target_path_GT = target_dir / "GT_GRN.csv" 
            
            with open(target_path_genes, "w") as f:
                json.dump(genes, f)

            with open(target_path_tfs, "w") as f:
                json.dump(tfs, f)

            df_GT.to_csv(target_path_GT, sep=",", index=False)

            dataset_path = dataset_path / f"{dataset}-200"
            runs = [run for run in dataset_path.iterdir() if run.is_dir()]
            first_run = sorted(runs)[0]
            scRNA = dataset_path / first_run / "ExpressionData.csv"
            target_path_scRNA = target_dir / "scRNA.csv"
            with open(scRNA, "r") as f_in, open(target_path_scRNA, "w") as f_out:
                header = f_in.readline()
                f_out.write(header)
                for line in f_in:
                    f_out.write(line)
                
            run_key = f"{dataset}_{first_run.name}"
            run_paths[run_key] = {
                "genes" : str(target_path_genes),
                "tfs" : str(target_path_tfs),
                "GT_GRN" : str(target_path_GT),
                "scRNA" : str(target_path_scRNA),
                "genes_in_rows": True,
                "source": source
            }
        
        runs_path = self.root_path / "QGRN" / "data" / "prepared" / source / "runs.json"
        with open(runs_path, "w") as f:
            json.dump(run_paths, f, indent=4)

    def DREAM5_preprocess(self, source):
        '''
        Implement the preprocessing logic for DREAM5 dataset here.
        This function should read the raw data from raw_data_path, preprocess it,
        and save the prepared data to prepared_data_path.
        '''
        from QGRN.src.utils_annealing import csv_to_df_grn
        dataset_root = self.root_path / "QGRN" / "data" / "DREAM5_network_inference_challenge"
        datasets = ["Network1", "Network2", "Network3", "Network4"]
        
        run_paths = dict()
        for i, dataset in enumerate(datasets):
            dataset_path = dataset_root / dataset
            if dataset == "Network2": #Network2 has .txt ground truth file instead of .tsv
                ground_truth_path = dataset_path / "gold standard" / f"DREAM5_NetworkInference_GoldStandard_{dataset}.txt"
            else:
                ground_truth_path = dataset_path / "gold standard" / f"DREAM5_NetworkInference_GoldStandard_{dataset}.tsv"
            df_GT = csv_to_df_grn(ground_truth_path, header=None).rename(
                columns={
                    0 : "regulator.gene",
                    1 : "regulated.gene",
                    2 : "regulator.effect",
                }
            )
            tfs_path = dataset_path / "input data" / f"net{i+1}_transcription_factors.tsv"
            with open(tfs_path, "r") as f:
                tfs = [line.strip() for line in f]
            genes = list(set(df_GT["regulated.gene"].unique().tolist()).union(set(tfs)))

            target_dir = self.prepared_root / source / dataset
            target_dir.mkdir(parents=True, exist_ok=True)

            target_path_genes = target_dir / "genes.json"
            target_path_tfs = target_dir / "tfs.json"
            target_path_GT = target_dir / "GT_GRN.csv" 
            
            with open(target_path_genes, "w") as f:
                json.dump(genes, f)

            with open(target_path_tfs, "w") as f:
                json.dump(tfs, f)

            df_GT.to_csv(target_path_GT, sep=",", index=False)
            scRNA = dataset_path / "input data" / f"net{i+1}_expression_data.tsv"
            target_path_scRNA = target_dir / "scRNA.tsv"
            with open(scRNA, "r") as f_in, open(target_path_scRNA, "w") as f_out:
                header = f_in.readline()
                f_out.write(header)
                for line in f_in:
                    f_out.write(line)
                
            run_paths[dataset] = {
                "genes" : str(target_path_genes),
                "tfs" : str(target_path_tfs),
                "GT_GRN" : str(target_path_GT),
                "scRNA" : str(target_path_scRNA),
                "genes_in_rows": False,
                "source": source
            }

        runs_path = self.root_path / "QGRN" / "data" / "prepared" / source / "runs.json"
        with open(runs_path, "w") as f:
            json.dump(run_paths, f, indent=4)
        
    def DREAM4_multifactorial_preprocess(self, source):
        '''
        Implement the preprocessing logic for DREAM4 dataset here.
        This function should read the raw data from raw_data_path, preprocess it,
        and save the prepared data to prepared_data_path.
        '''

        from QGRN.src.utils_annealing import csv_to_df_grn
        dataset_root = self.root_path / "QGRN" / "data" / "DREAM4 in-silico challenge" / "Size 100 multifactorial"
        datasets = ["1", "2", "3", "4", "5"]
        
        run_paths = dict()
        for dataset in datasets:
            ground_truth_path = dataset_root / "DREAM4 gold standards" / f"insilico_size100_multifactorial_{dataset}_goldstandard.tsv"
            df_GT = csv_to_df_grn(ground_truth_path, header=None).rename(
                columns={
                    0 : "regulator.gene",
                    1 : "regulated.gene",
                    2 : "regulator.effect",
                }
            )
            tfs = df_GT[df_GT["regulator.effect"] != 0]["regulator.gene"].unique().tolist()
            genes = list(set(df_GT["regulated.gene"].unique().tolist()).union(set(tfs)))

            target_dir = self.prepared_root / source / dataset
            target_dir.mkdir(parents=True, exist_ok=True)

            target_path_genes = target_dir / "genes.json"
            target_path_tfs = target_dir / "tfs.json"
            target_path_GT = target_dir / "GT_GRN.csv" 
            
            with open(target_path_genes, "w") as f:
                json.dump(genes, f)

            with open(target_path_tfs, "w") as f:
                json.dump(tfs, f)

            df_GT.to_csv(target_path_GT, sep=",", index=False)
            scRNA = dataset_root / "DREAM4 training data" / f"insilico_size100_{dataset}_multifactorial.tsv"
            target_path_scRNA = target_dir / "scRNA.tsv"
            with open(scRNA, "r") as f_in, open(target_path_scRNA, "w") as f_out:
                header = f_in.readline()
                f_out.write(header)
                for line in f_in:
                    f_out.write(line)
                
            run_paths[dataset] = {
                "genes" : str(target_path_genes),
                "tfs" : str(target_path_tfs),
                "GT_GRN" : str(target_path_GT),
                "scRNA" : str(target_path_scRNA),
                "genes_in_rows": False,
                "source": source
            }

        runs_path = self.root_path / "QGRN" / "data" / "prepared" / source / "runs.json"
        with open(runs_path, "w") as f:
            json.dump(run_paths, f, indent=4)
        
    def SERGIO100_preprocess(self, source):
        '''
        Implement the preprocessing logic for SERGIO dataset here.
        This function should read the raw data from raw_data_path, preprocess it,
        and save the prepared data to prepared_data_path.
        '''

        from QGRN.src.utils_annealing import csv_to_df_grn, csv_to_df_scRNA
        import pandas as pd
        dataset_root = self.root_path / "QGRN" / "data" / "SERGIO"
        datasets = ["De-noised_100G_3T_300cPerT_dynamics_8_DS8",
                    "De-noised_100G_3T_300cPerT_dynamics_9_DS4",
                    "De-noised_100G_4T_300cPerT_dynamics_10_DS5",
                    "De-noised_100G_6T_300cPerT_dynamics_7_DS6",
                    "De-noised_100G_7T_300cPerT_dynamics_11_DS7",
                    "De-noised_100G_9T_300cPerT_4_DS1"]
        
        run_paths = dict()
        for dataset in datasets:
            ground_truth_path = dataset_root / dataset / "gt_GRN.csv"
            df_GT = csv_to_df_grn(ground_truth_path, header=None).rename(
                columns={
                    0 : "regulator.gene",
                    1 : "regulated.gene",
                }
            )
            df_GT["regulator.gene"] = "G" + df_GT["regulator.gene"].astype(str)
            df_GT["regulated.gene"] = "G" + df_GT["regulated.gene"].astype(str)
            tfs = df_GT["regulator.gene"].unique().tolist()
            genes = list(set(df_GT["regulated.gene"].unique().tolist()).union(set(tfs)))

            target_dir = self.prepared_root / source / dataset
            target_dir.mkdir(parents=True, exist_ok=True)

            target_path_genes = target_dir / "genes.json"
            target_path_tfs = target_dir / "tfs.json"
            target_path_GT = target_dir / "GT_GRN.csv" 
            
            with open(target_path_genes, "w") as f:
                json.dump(genes, f)

            with open(target_path_tfs, "w") as f:
                json.dump(tfs, f)

            df_GT.to_csv(target_path_GT, sep=",", index=False)
            
            #different name for static dataset
            if dataset == "De-noised_100G_9T_300cPerT_4_DS1":
                scRNA = dataset_root / dataset / f"simulated_noNoise_0.csv" 
            else:
                scRNA = dataset_root / dataset / f"simulated_noNoise_S_0.csv" 
            df_scRNA = csv_to_df_scRNA(scRNA)
            df_scRNA.index = "G" + df_scRNA.index.astype(str)
            df_scRNA.columns = df_scRNA.columns.astype(str)
        
            target_path_scRNA = target_dir / "scRNA.tsv"
            with open(target_path_scRNA, "w") as f:
                df_scRNA.to_csv(f, sep="\t", index=True)
                
            run_paths[dataset] = {
                "genes" : str(target_path_genes),
                "tfs" : str(target_path_tfs),
                "GT_GRN" : str(target_path_GT),
                "scRNA" : str(target_path_scRNA),
                "genes_in_rows": True,
                "source": source
            }

        runs_path = self.root_path / "QGRN" / "data" / "prepared" / source / "runs.json"
        with open(runs_path, "w") as f:
            json.dump(run_paths, f, indent=4)
        
