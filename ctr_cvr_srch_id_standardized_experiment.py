from abstract_ized_experiment import AbstractIzedExperiment
from abstract_ized_experiment import add_standardized_attributes
import json
from scipy import mean
from collections import defaultdict


class CtrCvrSrchIdStandardizedExperiment(AbstractIzedExperiment):
    split_identifier = "spl_20180518114037"
    uses_ctr_and_cvr = True

    experiment_name = "CtrCvrSrchIdStandardizedExperiment"
    experiment_description = """Adds the CTR (click through rate) and CVR (conversion rate) of the prop_id to every row, as well as srch_id-standardized versions of them (and all other normal variables)."""

    grouped_attributes = ["srch_id"]
    ization = "standardized"

    def feature_engineering(self, df, ctr, cvr):
        df['ctr'] = df['prop_id'].map(ctr)
        df['cvr'] = df['prop_id'].map(cvr)

        return add_standardized_attributes(df, self.grouped_attributes, self.ized_attributes + ['ctr', 'cvr'])
