from deployment_experiment import DeploymentExperiment

class BaseExperiment(DeploymentExperiment):
    split_identifier = "spl_20180518114037"

    experiment_name = "BaseExperimentwithimprovedleafs"
    experiment_description = """The most basic experiment. Ignores 'date_time', includes all other raw features without doing any preprocessing. Uses 0.000 for missing values."""

    ignored_features = ['date_time']

    def feature_engineering(self, raw_data_frame):
        return raw_data_frame


# BaseExperiment().run_mini_experiment()

BaseExperiment().run_deployment(run_identifier="run_mini_20180524212118",reset_data=False)