from typing import List

import data.load_data as ld

from experiments.shared import EvalSet, run_experiment

def _twist_eval_sets(test_log_names, folder_name: str) -> List[EvalSet]:
    """Build evaluation datasets specific to the twist experiment."""
    return [
        ("twist test", lambda: (test_log_names, folder_name)),
        ("paper", ld.get_paper_logs),
        ("conical non-bent", lambda: ld.get_conical_logs(bent=False)),
        ("anatomical twist", lambda: ld.get_anat_logs(twist=True, bent=True)),
        ("conical no clot", lambda: ld.get_conical_logs(with_clot=False)),
    ]


def twist_data_experiment(data_cfg_path):
    run_experiment(
        experiment_name="twist",
        data_cfg_path=data_cfg_path,
        train_logs_fn=lambda: ld.get_conical_logs(twist=True, bent=True),
        eval_sets_builder=_twist_eval_sets,
    )


def twist_data_experiments_all():
    data_cfg_paths = [ #"configs/data/base_forward.yml", # simplest model with only forward data
                      #"configs/data/forward_freq.yml",  # add frequency components
                      #"configs/data/forward_image.yml", # use only images as input
                      #"configs/data/complete_forward.yml", # use complete forward data (signal + freq + image)
                      #"configs/data/complete_backward.yml", # use complete backward data
                       "configs/data/complete_both.yml" # use complete data of both directions
    ]
    for data_cfg_path in data_cfg_paths:
        twist_data_experiment(data_cfg_path)

