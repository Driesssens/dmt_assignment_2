from abstract_experiment import AbstractExperiment


class CompetitorAggregationExperiment(AbstractExperiment):
    split_identifier = "spl_20180518114037"

    experiment_name = "CompetitorAggregationExperiment"
    experiment_description = """Removes """

    ignored_features = ['date_time']

    def feature_engineering(self, raw_data_frame):
        return raw_data_frame
