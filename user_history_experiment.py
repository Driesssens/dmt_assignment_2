from abstract_experiment import AbstractExperiment
import numpy as np

class UserHistoryExperiment(AbstractExperiment):
    split_identifier = "spl_20180518114037"

    experiment_name = "UserHistoryExperiment"
    experiment_description = """Adds as new features |prop_starrating - visitor_hist_starrating|, same for price_usd. Intuition: higher values for these features means the booking deviates from the visitor's usual taste. Is 0 when there is no visitor history (reflecting the idea that in that case, all hotels in the srch_id should be fine)."""

    ignored_features = ['date_time']

    def feature_engineering(self, raw_data_frame):
        raw_data_frame['prop_starrating|personal_difference'] = np.abs(raw_data_frame['prop_starrating'] - raw_data_frame['visitor_hist_starrating'])
        raw_data_frame['price_usd|personal_difference'] = np.abs(raw_data_frame['price_usd'] - raw_data_frame['visitor_hist_adr_usd'])
        return raw_data_frame
