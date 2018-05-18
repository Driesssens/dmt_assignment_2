from base_experiment import BaseExperiment
from srch_id_standardized_experiment import SrchIdStandardizedExperiment
from srch_id_normalized_experiment import SrchIdNormalizedExperiment


def run_some_experiments():
    BaseExperiment().run_development_experiment()
    SrchIdStandardizedExperiment().run_development_experiment()
    SrchIdNormalizedExperiment().run_development_experiment()


run_some_experiments()
