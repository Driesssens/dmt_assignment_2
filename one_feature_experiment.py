from abstract_experiment import AbstractExperiment


class OneVariableExperiment(AbstractExperiment):
    split_identifier = "spl_20180518114037"

    experiment_name = "OneVariableExperiment"

    the_single_feature = 'prop_'
    experiment_description = """Like the BaseExperiment, but includes only one feature: """

    ignored_features = ['date_time']

    def feature_engineering(self, raw_data_frame):
        return raw_data_frame
