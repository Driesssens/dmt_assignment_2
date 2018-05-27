from deployment_experiment import DeploymentExperiment

class BaseExperiment(DeploymentExperiment):
    split_identifier = "spl_20180518114037"

    experiment_name = "BaseExperimentwithimprovedleafs"
    experiment_description = """The most basic experiment. Ignores 'date_time', includes all other raw features without doing any preprocessing. Uses 0.000 for missing values."""

    ignored_features = ['date_time']

    def feature_engineering(self, raw_data_frame):
        return raw_data_frame



# BaseExperiment().run_mini_experiment()
# if no run Identifier is given the model will automatically take the last generated model from the location of the experiment name.
BaseExperiment().run_deployment(training_CHECK=False, run_identifier=None, reset_data=False, relevance_score_testing=True)
