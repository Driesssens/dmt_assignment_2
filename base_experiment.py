from abstract_experiment import AbstractExperiment
from configuration import make_configuration


class BaseExperiment(AbstractExperiment):
    split_identifier = "spl_20180508153924"

    experiment_name = "BaseExperiment"
    experiment_description = """The most basic experiment. Ignores 'date_time', includes all other raw features without doing any preprocessing. Uses 0.000 for missing values."""

    ignored_features = ['date_time']

    def missing_value_default(self, feature_name, feature_value):
        return '0.000000'

    def feature_engineering(self, raw_data_frame):
        return raw_data_frame


BaseExperiment().run_mini_experiment()

short_experiment = make_configuration(epochs=10)
BaseExperiment().run_development_experiment(configuration=short_experiment)
