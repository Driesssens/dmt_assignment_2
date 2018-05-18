from abstract_experiment import AbstractExperiment
from configuration import make_configuration


def add_normalized_attributes(df, groups, attributes):
    for group in groups:
        min_max = df.groupby(group)[attributes].agg(['min', 'max']).reset_index()
        min_max.columns = ['{}{}'.format(attr, '|' + aggr if aggr else '') for attr, aggr in min_max.columns]
        df = df.merge(min_max, on=group)

        for attr in attributes:
            df[attr + '|{}_normalized'.format(group)] = (df[attr] - df[attr + '|min']) / (df[attr + '|max'] - df[attr + '|min'])
            df = df.drop([attr + '|max', attr + '|min'], axis=1)

    return df


def add_standardized_attributes(df, groups, attributes):
    for group in groups:
        mean_std = df.groupby(group)[attributes].agg(['mean', 'std']).reset_index()
        mean_std.columns = ['{}{}'.format(attr, '|' + aggr if aggr else '') for attr, aggr in mean_std.columns]
        df = df.merge(mean_std, on=group)

        for attr in attributes:
            df[attr + '|{}_standardized'.format(group)] = (df[attr] - df[attr + '|mean']) / (df[attr + '|std'])
            df = df.drop([attr + '|mean', attr + '|std'], axis=1)

    return df


class AbstractIzedExperiment(AbstractExperiment):
    normalization_method = "the hotel with the highest value for that attribute gets an additional feature with {method} value 1, the lowest a new feature with value 0, and the others in between"

    standardization_method = "it gets an additional feature that is divided by the standard variation after the mean is subtracted"

    ized_attributes = ['prop_starrating', 'prop_review_score', 'prop_location_score1', 'prop_location_score2', 'price_usd', 'orig_destination_distance']
    ignored_features = ['date_time']

    @classmethod
    def make_description(cls, ization, grouped_attributes):
        return """Here, some of the attributes also get a version that is {ization} within the group of rows of the same {groups}, meaning that for all hotels within that group, for a given attribute, {method}. The {ization} attributes are: {ized_attributes}""".format(
            ization=ization,
            groups=' / '.join(grouped_attributes),
            method=cls.normalization_method if ization == "normalization" else cls.standardization_method,
            ized_attributes=cls.ized_attributes
        )

    def add_normalized_attributes(self, df, groups):
        return add_normalized_attributes(df, groups, self.ized_attributes)

    def add_standardized_attributes(self, df, groups):
        return add_standardized_attributes(df, groups, self.ized_attributes)
