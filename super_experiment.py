from abstract_experiment import AbstractExperiment
from abstract_ized_experiment import AbstractIzedExperiment, add_standardized_attributes
import numpy as np
from datetime import datetime


class SuperExperiment(AbstractExperiment):
    split_identifier = "spl_20180518114037"

    experiment_name = "SuperExperiment"
    experiment_description = """Does basically everything we know might be good. Except for missing values. Check the class."""

    ignored_features = ['date_time']

    def feature_engineering(self, raw_data_frame):
        ################## COMPS ##################
        comp_rate_features = ["comp{}_rate".format(n + 1) for n in range(8)]
        percent_diff_features = ["comp{}_rate_percent_diff".format(n + 1) for n in range(8)]

        for n in range(8):
            raw_data_frame[(percent_diff_features[n] + '|signed')] = raw_data_frame[percent_diff_features[n]] * raw_data_frame[comp_rate_features[n]]

        raw_data_frame.drop(percent_diff_features, axis=1)

        raw_data_frame['comp_rate|sum'] = raw_data_frame[comp_rate_features].sum(axis=1)
        raw_data_frame['comp_rate|sum|positive'] = raw_data_frame[comp_rate_features].clip_lower(0).sum(axis=1)
        raw_data_frame['comp_rate|sum|negative'] = raw_data_frame[comp_rate_features].clip_upper(0).sum(axis=1)

        ################## USER HISTORY ##################
        raw_data_frame['prop_starrating|personal_difference'] = np.abs(raw_data_frame['prop_starrating'] - raw_data_frame['visitor_hist_starrating'])
        raw_data_frame['price_usd|personal_difference'] = np.abs(raw_data_frame['price_usd'] - raw_data_frame['visitor_hist_adr_usd'])

        ################## TIME ##################
        dates = raw_data_frame['date_time'].apply(lambda date: datetime.strptime(date, "%Y-%m-%d %H:%M:%S"))
        raw_data_frame['day_part'] = dates.apply(lambda dt: (dt.hour % 24 + 6) // 6)
        raw_data_frame['season'] = dates.apply(lambda dt: (dt.month % 12 + 3) // 3)

        ################## STANDARDIZATION ##################
        existing_features = ['prop_starrating',
                             'prop_review_score',
                             'prop_location_score1',
                             'prop_location_score2',
                             'prop_log_historical_price',
                             'price_usd',
                             'srch_query_affinity_score',
                             'orig_destination_distance']

        new_features = ['comp_rate|sum',
                        'comp_rate|sum|positive',
                        'comp_rate|sum|negative',
                        'prop_starrating|personal_difference',
                        'price_usd|personal_difference']

        standardized_df = add_standardized_attributes(raw_data_frame, ['srch_id'], existing_features + new_features)

        return standardized_df
