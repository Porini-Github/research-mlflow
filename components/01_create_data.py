import argparse 
from pathlib import Path
import json
from utils.mlflowmanager import MlFlowManager
from utils.datasetmanager import DatasetManager

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_name", type=str)
    parser.add_argument("--input_file_path", type=str)
    parser.add_argument("--data_asset_name", type=str)
    parser.add_argument("--create_output_path", type=str)

    args = parser.parse_args()

    mf = MlFlowManager()
    dm = DatasetManager()

    mf.setExperimentName(args.experiment_name)
    mf.setStartRun("write_run")
    
    version = dm.RegisterDataset(args.input_file_path, args.data_asset_name)
    output_dict = {'data_asset_name': args.data_asset_name, 
                   'data_asset_version': version,
                   'experiment_name': args.experiment_name}
    
    (Path(args.create_output_path) / "01_create_output.txt").write_text(json.dumps(output_dict))
    mf.setEndRun()

if __name__ == "__main__":
    main()