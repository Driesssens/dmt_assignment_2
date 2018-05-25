from base_experiment import BaseExperiment
from srch_id_standardized_experiment import SrchIdStandardizedExperiment
from srch_id_normalized_experiment import SrchIdNormalizedExperiment
from user_history_experiment import UserHistoryExperiment
from user_history_normalized_experiment import UserHistoryNormalizedExperiment
from one_feature_experiment import OneFeatureExperiment
from missing_values_test_experiment import MissingValuesTestExperiment


def run_some_experiments():
    # MissingValuesTestExperiment().run_mini_experiment(reset_data=True)
    SrchIdStandardizedExperiment().run_mini_experiment(reset_data=True)
    SrchIdNormalizedExperiment().run_mini_experiment(reset_data=True)


run_some_experiments()
