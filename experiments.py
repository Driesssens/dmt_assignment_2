from base_experiment import BaseExperiment
from srch_id_standardized_experiment import SrchIdStandardizedExperiment
from srch_id_normalized_experiment import SrchIdNormalizedExperiment
from user_history_experiment import UserHistoryExperiment
from user_history_normalized_experiment import UserHistoryNormalizedExperiment


def run_some_experiments():
    BaseExperiment().run_full_experiment()
    SrchIdNormalizedExperiment().run_full_experiment()
    SrchIdStandardizedExperiment().run_full_experiment()
    UserHistoryNormalizedExperiment().run_full_experiment()
    UserHistoryExperiment().run_full_experiment()


run_some_experiments()
