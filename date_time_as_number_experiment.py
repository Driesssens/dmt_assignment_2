from abstract_experiment import AbstractExperiment
import pandas


class DateTimeAsNumberExperiment(AbstractExperiment):
    split_identifier = "spl_20180518114037"

    experiment_name = "DateTimeAsNumberExperiment"
    experiment_description = """Includes all raw features, except that it converts date_time to a number. For instance, 02/02/2013 15:27 becomes 201302021527. The purpose of this class is not really to do actual experiments, but to compute the univariate nDCGs of the raw dataset."""

    ignored_features = []

    def feature_engineering(self, raw_data_frame):
        raw_data_frame['date_time'] = pandas.to_datetime(raw_data_frame['date_time']).apply(lambda dt: int(dt.strftime('%Y%m%d%H%M%S')))
        return raw_data_frame
