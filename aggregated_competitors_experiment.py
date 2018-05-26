from abstract_experiment import AbstractExperiment


class AggregatedCompetitorsExperiment(AbstractExperiment):
    split_identifier = "spl_20180518114037"

    experiment_name = "AggregatedCompetitorsExperiment"
    experiment_description = """"""

    ignored_features = ['date_time']

    def feature_engineering(self, raw_data_frame):
        comp_rate_features = ["comp{}_rate".format(n + 1) for n in range(8)]
        percent_diff_features = ["comp{}_rate_percent_diff".format(n + 1) for n in range(8)]
        comp_inv_features = ["comp{}_inv".format(n + 1) for n in range(8)]

        for n in range(8):
            raw_data_frame[(percent_diff_features[n] + '_new')] = raw_data_frame[percent_diff_features[n]] * raw_data_frame[comp_rate_features[n]]

        raw_data_frame['comp_rate|sum'] = raw_data_frame[comp_rate_features].sum(axis=1)
        raw_data_frame['comp_rate|sum|positive'] = raw_data_frame[comp_rate_features].clip_lower(0).sum(axis=1)
        raw_data_frame['comp_rate|sum|negative'] = raw_data_frame[comp_rate_features].clip_upper(0).sum(axis=1)

        return raw_data_frame
