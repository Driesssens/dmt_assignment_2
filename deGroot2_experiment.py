from abstract_ized_experiment import AbstractIzedExperiment
from configuration import make_configuration
from deployment_experiment import DeploymentExperiment
# own imports
import numpy


class DeGroot2Experiment(AbstractIzedExperiment):
    split_identifier = "spl_final_20180526151122"

    experiment_name = "full_data_standardized_missing_values"
    grouped_attributes = ["srch_id"]
    ization = "standardized"
    experiment_description = "Missing values imputed"+ AbstractIzedExperiment.make_description(ization, grouped_attributes)

    ignored_features = ['date_time']

    def missing_value_default(self, feature_name, feature_value):

        return '0.000000'

    def feature_engineering(self, raw_data_frame):

        raw_data_frame = AbstractIzedExperiment.add_normalized_attributes(self, raw_data_frame, self.grouped_attributes)
        # finding max value
        raw_data_frame.loc[:, 'max_rate_percent_diff'] = raw_data_frame[['comp1_rate_percent_diff', 'comp2_rate_percent_diff',
                                                                         'comp3_rate_percent_diff', 'comp4_rate_percent_diff',
                                                                         'comp5_rate_percent_diff', 'comp6_rate_percent_diff',
                                                                         'comp7_rate_percent_diff', 'comp8_rate_percent_diff']
        ].max(axis=0)

        # Clipping to a maximum of 100
        raw_data_frame.loc[:, 'max_rate_percent_diff'] = raw_data_frame['max_rate_percent_diff'].clip(lower=0.0, upper=100.0)

        # Finding minimal values
        raw_data_frame.loc[:, 'min_rate_percent_diff'] = raw_data_frame[['comp1_rate_percent_diff', 'comp2_rate_percent_diff',
                                                                         'comp3_rate_percent_diff', 'comp4_rate_percent_diff',
                                                                         'comp5_rate_percent_diff', 'comp6_rate_percent_diff',
                                                                         'comp7_rate_percent_diff', 'comp8_rate_percent_diff']
        ].min(axis=0)

        # Count higher lower
        raw_data_frame.loc[:, 'count_available'] = 0.0
        raw_data_frame.loc[:, 'count_higher_rate'] = 0.0
        raw_data_frame.loc[:, 'count_lower_rate'] = 0.0
        for elem in range(1, 9):
            raw_data_frame.loc[:, 'count_lower_rate'] += numpy.where(raw_data_frame['comp' + str(elem) + '_rate'] == -1, 1, 0)
            raw_data_frame.loc[:, 'count_higher_rate'] += numpy.where(raw_data_frame['comp' + str(elem) + '_rate'] == 1, 1, 0)
            raw_data_frame.loc[:, 'count_available'] += numpy.where(raw_data_frame['comp' + str(elem) + '_inv'] == 1, 1, 0)
            raw_data_frame = raw_data_frame.drop(labels=['comp' + str(elem) + '_rate', 'comp' + str(elem) + '_inv', 'comp' + str(elem) + '_rate_percent_diff'], axis=1)


        raw_data_frame.loc[:,'visitor_hist_starrating'] = raw_data_frame['visitor_hist_starrating'].fillna(value=-1.00)
        raw_data_frame.loc[:,'visitor_hist_adr_usd'] = raw_data_frame['visitor_hist_adr_usd'].fillna(value=-1.00)
        raw_data_frame.loc[:,'prop_review_score'] = raw_data_frame['prop_review_score'].fillna(value=1.00)
        raw_data_frame.loc[:,'prop_location_score2'] = raw_data_frame['prop_location_score2'].fillna(value=0.30)
        raw_data_frame.loc[:,'srch_query_affinity_score'] = raw_data_frame['srch_query_affinity_score'].fillna(value=-20.4513)        
        raw_data_frame.loc[:,'orig_destination_distance'] = raw_data_frame['orig_destination_distance'].fillna(value=325.00)

        return raw_data_frame

# BaseExperiment().run_mini_experiment()
# if no run Identifier is given the model will automatically take the last generated model from the location of the experiment name.

DeGroot2Experiment().run_deployment(training_CHECK=True, run_identifier=None, reset_data=False, relevance_score_testing=True)