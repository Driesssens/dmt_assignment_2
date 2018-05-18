from abstract_ized_experiment import AbstractIzedExperiment


class SrchIdStandardizedExperiment(AbstractIzedExperiment):
    split_identifier = "spl_20180518114037"

    experiment_name = "SrchIdStandardizedExperiment"

    grouped_attributes = ["srch_id"]
    ization = "standardized"
    experiment_description = AbstractIzedExperiment.make_description(ization, grouped_attributes)

    def feature_engineering(self, df):
        return AbstractIzedExperiment.add_standardized_attributes(self, df, self.grouped_attributes)