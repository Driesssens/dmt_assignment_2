from abstract_experiment import AbstractExperiment


class MissingValuesTestExperiment(AbstractExperiment):
    split_identifier = "spl_20180518114037"

    experiment_name = "MissingValuesTestExperiment"
    experiment_description = """Quick test if the new missing values system works."""

    ignored_features = ['date_time']
    general_missing_value_filler = 7.4
    column_specific_missing_value_fillers = {
        'comp2_rate_percent_diff': 9,
        'comp3_rate_percent_diff': 7.1
    }

    def feature_engineering(self, raw_data_frame):
        return raw_data_frame
