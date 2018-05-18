from abstract_experiment import AbstractExperiment
from configuration import make_configuration


def add_srch_id_normalized_attributes(df, attributes):
    min_max = df.groupby('srch_id')[attributes].agg(['min', 'max']).reset_index()
    min_max.columns = ['{}{}'.format(attr, '|' + aggr if aggr else '') for attr, aggr in min_max.columns]
    merged = df.merge(min_max, on='srch_id')

    for attr in attributes:
        merged[attr + '|srch_id_normalized'] = (merged[attr] - merged[attr + '|min']) / (merged[attr + '|max'] - merged[attr + '|min'])
        merged = merged.drop([attr + '|max', attr + '|min'], axis=1)

    return merged


class SrchIdNormalizedExperiment(AbstractExperiment):
    split_identifier = "spl_20180516171649"

    experiment_name = "SrchIdNormalizedExperiment"

    normalized_attributes = ['price_usd', 'prop_starrating']

    experiment_description = """Here, some of the attributes also get a version that is normalized within the srch_id, meaning that for all hotels with that srch_id, for a given attribute, the hotel with the highest value for that attribute gets an additional feature with value 1, the lowest a new feature with value 0, and the others in between. The attributes are: """ + str(normalized_attributes)

    ignored_features = ['date_time']

    def missing_value_default(self, feature_name, feature_value):
        return '0.000000'

    def feature_engineering(self, raw_data_frame):
        return add_srch_id_normalized_attributes(raw_data_frame, self.normalized_attributes)


SrchIdNormalizedExperiment().run_mini_experiment(reset_data=True)
