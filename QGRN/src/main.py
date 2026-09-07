from QGRN.src.ConfigParser import ConfigParser
from QGRN.src.DataParser import DataParser
import argparse
from pathlib import Path
import time

from QGRN.src import PL_annealer
import ray, os
import traceback

def parser():
    p = argparse.ArgumentParser()

    p.add_argument(
        '-c', '--config',
        type=str,
        default="template.yaml",
        help="Define the configuration file"
    )

    p.add_argument(
        '-v', '--verbose',
        action='store_true',
        help="Enable verbose output"
    )

    p.add_argument(
        '-t', '--exectime',
        action='store_true',
        help="Print execution times"
    )

    return p.parse_args()

def main():
    #fetch arguments
    args = parser()
    root_path = Path(__file__).parent.parent.parent #QGRN-git root

    #initialize config parser
    config_parser = ConfigParser(load_from_file=args.config)
    config_parser.set_value("root_path", root_path)

    #initialize data parser
    data_parser = DataParser(config_parser=config_parser)
    runs = data_parser.parse()

    if args.verbose:
        config_parser.set_value("verbose", True)

    if args.exectime:
        start_time = time.time()
        print("Execution started at: ", time.ctime(start_time))
    
    ray_tmp = os.path.expanduser("~/ray_tmp")
    os.makedirs(ray_tmp, exist_ok=True)
    ray.init(ignore_reinit_error=True,
            _temp_dir=ray_tmp)
    try:
        for run_name, dataset_info in runs.items():
            print(f"Running annealer for dataset: {run_name}")
            evaluation = PL_annealer.execute_annealer(
                config_parser=config_parser,
                run_name=run_name,
                dataset_info=dataset_info,
            )
            if config_parser.get_value("evaluation.binary_classification.mode", default=False):
                print("Binary evaluation results:")
                print(evaluation["binary_classification"])
            if config_parser.get_value("evaluation.confidence_evaluation.mode", default=False):
                print("Confidence evaluation results:")
                print(evaluation["confidence_evaluation"])
    except Exception:
        traceback.print_exc()
        raise
    finally:
        ray.shutdown()

    if args.exectime:
        end_time = time.time()
        print("Execution ended at: ", time.ctime(end_time))
        print("Total execution time: ", end_time - start_time, " seconds")

if __name__ == "__main__":
    main()
