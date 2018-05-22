from abstract_experiment import AbstractExperiment

# own imports
import shutil
import numpy


class DeGrootExperiment(AbstractExperiment):
    split_identifier = "spl_20180518114037"

    experiment_name = "deGrootonlymissingExperiment"
    experiment_description = """Basic experiment only adding in some features for the missing values."""

    ignored_features = ['date_time']

    def missing_value_default(self, feature_name, feature_value):

        if feature_name == 'visitor_hist_starrating':
            return '3.374334'

        if feature_name == 'visitor_hist_adr_usd':
            return '0.000000'

        if feature_name == 'prop_review_score':
            return '0.000000'

        if feature_name == 'prop_location_score2':
            return '0.000000'

        if feature_name == 'srch_query_affinity_score':
            return '-24.14641'

        if feature_name == 'orig_destination_distance':
            return '0.000000'

        else:
            return '0.000000'

    def feature_engineering(self, raw_data_frame):

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

        return raw_data_frame



DeGrootExperiment().run_mini_experiment(reset_data=False)
DeGrootExperiment().run_development_experiment(reset_data=False)
DeGrootExperiment().run_medium_experiment(reset_data=False)
DeGrootExperiment().run_full_experiment(reset_data=False)

