from abstract_ized_experiment import AbstractIzedExperiment


class SrchIdNormalizedExperiment(AbstractIzedExperiment):
    split_identifier = "spl_20180518114037"

    experiment_name = "SrchIdNormalizedExperiment"

    grouped_attributes = ["srch_id"]
    ization = "normalized"
    experiment_description = AbstractIzedExperiment.make_description(ization, grouped_attributes)

    def feature_engineering(self, df):
        return AbstractIzedExperiment.add_normalized_attributes(self, df, self.grouped_attributes)