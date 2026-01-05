from typing import List

import data.load_data as ld

from experiments.shared import EvalSet, run_experiment

def _paper_eval_sets(test_log_names, folder_name: str) -> List[EvalSet]:
    """Build evaluation datasets specific to the paper experiment."""
    return [
        ("paper test", lambda: (test_log_names, folder_name)),
        ("bent conical", lambda: ld.get_conical_logs(bent=True)),
        ("anatomical", ld.get_anat_logs),
        ("conical no clot", lambda: ld.get_conical_logs(with_clot=False)),
    ]


def paper_data_experiment(data_cfg_path):
    run_experiment(
        experiment_name="paper",
        data_cfg_path=data_cfg_path,
        train_logs_fn=ld.get_paper_logs,
        eval_sets_builder=_paper_eval_sets,
    )

def paper_data_experiments_all():
    data_cfg_paths = [ #"configs/data/base_forward.yml", # simplest model with only forward data
                      #"configs/data/forward_freq.yml",  # add frequency components
                      #"configs/data/forward_image.yml", # use only images as input
                      #"configs/data/complete_forward.yml", # use complete forward data (signal + freq + image)
                      #"configs/data/complete_backward.yml", # use complete backward data
                       "configs/data/complete_both.yml" # use complete data of both directions
    ]
    for data_cfg_path in data_cfg_paths:
        paper_data_experiment(data_cfg_path)

    print("="*10 + "All paper data experiments completed."+ "="*10)