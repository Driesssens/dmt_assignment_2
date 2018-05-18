from abstract_experiment import AbstractExperiment
from configuration import make_configuration


class BaseExperiment(AbstractExperiment):
    split_identifier = "spl_20180518114037"

    experiment_name = "BaseExperiment"
    experiment_description = """The most basic experiment. Ignores 'date_time', includes all other raw features without doing any preprocessing. Uses 0.000 for missing values."""

    ignored_features = ['date_time']

    def feature_engineering(self, raw_data_frame):
        return raw_data_frame


BaseExperiment().run_mini_experiment()
