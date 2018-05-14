import pandas
import random
import numpy
import os
import pickle
import logging
from rankpy.queries import Queries
from rankpy.models import LambdaMART
from datetime import datetime
from configuration import standard_configuration
from configuration import make_model
from pprint import pprint

NON_FEATURE_COLUMNS = ['Index',
                       'srch_id',
                       'booking_bool',
                       'click_bool',
                       'gross_bookings_usd',
                       'position']

MINI = "mini"
DEVELOPMENT = "development"
MEDIUM = "medium"
FULL = "full"


def standard_hyperparameters():
    return LambdaMART(metric='nDCG',
                      max_leaf_nodes=7,
                      shrinkage=0.1,
                      estopping=50,
                      n_jobs=-1,
                      min_samples_leaf=50,
                      random_state=42)


class AbstractExperiment:
    experiment_name = None
    experiment_description = None
    ignored_features = None
    split_identifier = None

    def missing_value_default(self, feature_name, feature_value):
        # NOT IMPLEMENTED
        pass

    def feature_engineering(self, raw_data_frame):
        # NOT IMPLEMENTED
        pass

    def convert_pandas_row_to_svm_light_format(self, pandas_row):
        qid = pandas_row.srch_id
        relevance_grade = 5 if pandas_row.booking_bool is 1 else pandas_row.click_bool

        svm_light_string = "{} qid:{} ".format(relevance_grade, qid)

        feature_number = 1

        for feature_name, feature_value in pandas_row._asdict().iteritems():
            if feature_name not in NON_FEATURE_COLUMNS + self.ignored_features:
                if numpy.isnan(feature_value):
                    sanitized_feature_value = self.missing_value_default(feature_name, feature_value)
                else:
                    sanitized_feature_value = feature_value

                svm_light_string += '{}:{} '.format(feature_number, sanitized_feature_value)
                feature_number += 1

        return svm_light_string

    def store_data_frame_as_svm_light(self, data_frame, file_name):
        with open(file_name, 'w') as output_file:
            for row in data_frame.itertuples():
                output_file.write(self.convert_pandas_row_to_svm_light_format(row) + '\n')

    def run_mini_experiment(self, configuration=None, reset_data=False):
        self.run_experiment(MINI, configuration, reset_data)

    def run_development_experiment(self, configuration=None, reset_data=False):
        self.run_experiment(DEVELOPMENT, configuration, reset_data)

    def run_medium_experiment(self, configuration=None, reset_data=False):
        self.run_experiment(MEDIUM, configuration, reset_data)

    def run_full_experiment(self, configuration=None, reset_data=False):
        self.run_experiment(FULL, configuration, reset_data)

    def run_experiment(self, experiment_size, configuration, reset_data):
        data_set_location = "data/{}/{}/{}".format(self.split_identifier, self.experiment_name, experiment_size)
        data_set_name = "{}-{}-{}".format(self.experiment_name, self.split_identifier, experiment_size)
        if reset_data or not os.path.exists(data_set_location):
            print "Data set {} had not yet been generated. Will generate now...".format(data_set_name)
            full_training_set = pandas.read_csv('data/training_set_VU_DM_2014.csv')
            print "Loaded full data set."
            os.makedirs(data_set_location)
            print "Created folder."

            for set_name in ["training", "validation", "test"]:
                print "Generating the {} set...".format(set_name)
                with open('splits/{}/{}/{}_qids.pkl'.format(self.split_identifier, experiment_size, set_name), 'rb') as fp:
                    qids = pickle.load(fp)

                sample_rows = full_training_set[full_training_set.srch_id.isin(qids)]
                data_set = self.feature_engineering(sample_rows)
                data_set_path = data_set_location + '/' + set_name
                self.store_data_frame_as_svm_light(data_set, data_set_path)
        else:
            print "Data set {} was already generated.".format(data_set_name)

        print "Through with the experiment."

        run_identifier = 'run_{}_{}'.format(experiment_size, datetime.now().strftime('%Y%m%d%H%M%S'))

        # Turn on logging.
        logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.INFO)

        # Load the data sets.
        training_data = Queries.load_from_text(data_set_location + '/' + 'training')
        validation_data = Queries.load_from_text(data_set_location + '/' + 'validation')
        test_data = Queries.load_from_text(data_set_location + '/' + 'test')

        logging.info('================================================================================')

        if configuration is None:
            configuration = standard_configuration()

        model = make_model(configuration)

        model.fit(training_data, validation_queries=validation_data)

        logging.info('================================================================================')

        test_set_performance = model.evaluate(test_data, n_jobs=-1)
        logging.info('%s on the test queries: %.8f' % (model.metric, test_set_performance))

        output_folder = 'output/{}/{}/{}'.format(self.split_identifier, self.experiment_name, run_identifier)
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        model.save(output_folder + '/trained_model')
        open(output_folder + '/nDCG_{}'.format(str(test_set_performance)), 'w+')

        with open(output_folder + '/details.txt', "w+") as df:
            df.write("Details of run {} of {} on the {} set.\n".format(run_identifier, self.experiment_name, experiment_size))
            df.write("About {}: {}\n".format(self.experiment_name, self.experiment_description))
            df.write("Used split: {}\n".format(self.split_identifier))
            df.write("Result: nDCG {} on test set after {} epochs.\n".format(test_set_performance, model.n_estimators))

            df.write("\nConfiguration (hyperparameters):\n")
            df.write("{}\n".format(str(dict(configuration._asdict()))))

        with open(output_folder + '/training_performance_per_epoch.txt', "w+") as tf:
            tf.write(str(model.training_performance.tolist()))

        with open(output_folder + '/validation_performance_per_epoch.txt', "w+") as vf:
            vf.write(str(model.validation_performance.tolist()))
