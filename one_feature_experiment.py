from abstract_experiment import AbstractExperiment
from abstract_experiment import NON_FEATURE_COLUMNS

the_single_feature = 'prop_starrating'


def create_extra_file(variables):
    open(variables['output_folder'] + '/1-feature-is-{}'.format(the_single_feature), 'w+')


class OneFeatureExperiment(AbstractExperiment):
    split_identifier = "spl_20180518114037"

    experiment_name = "OneFeatureExperiment"

    experiment_description = """Like the BaseExperiment, but includes only one feature: """ + the_single_feature

    ignored_features = []

    def feature_engineering(self, raw_data_frame):
        return raw_data_frame[NON_FEATURE_COLUMNS[1:] + [the_single_feature]]

    def run_mini_experiment(self, configuration=None, reset_data=False, add_to_leaderboard=True, extra_instructions=None):
        self.run_experiment("mini", configuration, reset_data, add_to_leaderboard, extra_instructions=create_extra_file, missing_values_old_style=False)

    def run_development_experiment(self, configuration=None, reset_data=False, add_to_leaderboard=True, extra_instructions=None):
        self.run_experiment("development", configuration, reset_data, add_to_leaderboard, extra_instructions=create_extra_file, missing_values_old_style=False)

    def run_medium_experiment(self, configuration=None, reset_data=False, add_to_leaderboard=True, extra_instructions=None):
        self.run_experiment("medium", configuration, reset_data, add_to_leaderboard, extra_instructions=create_extra_file, missing_values_old_style=False)

    def run_full_experiment(self, configuration=None, reset_data=False, add_to_leaderboard=True, extra_instructions=None):
        self.run_experiment("full", configuration, reset_data, add_to_leaderboard, extra_instructions=create_extra_file, missing_values_old_style=False)

    # I thought that this might be more time efficient, but it turned out not to be
    # def store_data_frame_as_svm_light(self, data_frame, file_name):
    #     data_frame['relevance'] = data_frame.apply(lambda row: 5 if row['booking_bool'] is 1 else row['click_bool'], axis=1)
    #
    #     dump_svmlight_file(data_frame[self.the_single_feature].values.reshape(-1, 1), data_frame['relevance'], file_name, query_id=data_frame['srch_id'])
