from typing import List

import data.load_data as ld

from experiments.shared import EvalSet, run_experiment

def _eval_sets(test_log_names, folder_name: str) -> List[EvalSet]:
    """Build evaluation datasets specific to the paper experiment."""
    return [
        ("full test", lambda: (test_log_names, folder_name)),
        ("paper", ld.get_paper_logs),
    ]


def full_data_experiment(data_cfg_path):
    run_experiment(
        experiment_name="full",
        data_cfg_path=data_cfg_path,
        train_logs_fn=ld.get_extra_logs,
        eval_sets_builder=_eval_sets,
    )

def full_data_experiments_all():
    data_cfg_paths = [ #"configs/data/base_forward.yml", 
                      "configs/data/base_both.yml",
    ]
    for data_cfg_path in data_cfg_paths:
        full_data_experiment(data_cfg_path)

    print("="*10 + "All full data experiments completed."+ "="*10)