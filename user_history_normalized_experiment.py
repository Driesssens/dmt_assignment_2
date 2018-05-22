from abstract_experiment import AbstractExperiment
import numpy as np
from abstract_ized_experiment import add_normalized_attributes
from abstract_ized_experiment import AbstractIzedExperiment


class UserHistoryNormalizedExperiment(AbstractExperiment):
    split_identifier = "spl_20180518114037"

    experiment_name = "UserHistoryNormalizedExperiment"
    experiment_description = """Adds as new features |prop_starrating - visitor_hist_starrating|, same for price_usd. Intuition: higher values for these features means the booking deviates from the visitor's usual taste. Is 0 when there is no visitor history (reflecting the idea that in that case, all hotels in the srch_id should be fine). In addition, also normalizes features: these new ones, as well as the ones that SrchIdNormalizedExperiment also normalizes."""

    ignored_features = ['date_time']

    def feature_engineering(self, raw_data_frame):
        raw_data_frame['prop_starrating|personal_difference'] = np.abs(raw_data_frame['prop_starrating'] - raw_data_frame['visitor_hist_starrating'])
        raw_data_frame['price_usd|personal_difference'] = np.abs(raw_data_frame['price_usd'] - raw_data_frame['visitor_hist_adr_usd'])
        new_data_frame = add_normalized_attributes(raw_data_frame, ['srch_id'], AbstractIzedExperiment.ized_attributes + ['prop_starrating|personal_difference', 'price_usd|personal_difference'])
        return new_data_frame
