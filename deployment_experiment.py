import pandas
import numpy
import os
import pickle
import logging
from rankpy.queries import Queries
from rankpy.models import LambdaMART
from datetime import datetime
from configuration import standard_configuration
from configuration import make_model
import json
from shutil import copyfile
from abstract_experiment import AbstractExperiment

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


def log(message, start_time, since=None):
    print '{date:%Y-%m-%d %H:%M:%S} @{minute}M: {message}{since}'.format(
        date=datetime.now(),
        message=message,
        minute=int(minutes_passed(start_time)),
        since=" (took {} minutes)".format(minutes_passed(since)) if since is not None else "")


def minutes_passed(starting_time):
    return (datetime.now() - starting_time).total_seconds() / 60


class DeploymentExperiment(AbstractExperiment):
    experiment_name = None
    experiment_description = None
    ignored_features = None
    split_identifier = None


    def convert_pandas_row_to_svm_light_format_deployment(self, pandas_row):
        qid = pandas_row.srch_id
        relevance_grade = 0

        svm_light_string = "{} qid:{} ".format(relevance_grade, qid)

        feature_number = 1

        for feature_name, feature_value in pandas_row._asdict().iteritems():
            if feature_name not in NON_FEATURE_COLUMNS + self.ignored_features:
                if numpy.isnan(feature_value) or numpy.isinf(feature_value):
                    sanitized_feature_value = self.missing_value_default(feature_name, feature_value)
                else:
                    sanitized_feature_value = feature_value

                svm_light_string += '{}:{} '.format(feature_number, sanitized_feature_value)
                feature_number += 1

        return svm_light_string

    def store_data_frame_as_svm_light_deployment(self, data_frame, file_name):
        with open(file_name, 'w') as output_file:
            for row in data_frame.itertuples():
                output_file.write(self.convert_pandas_row_to_svm_light_format_deployment(row) + '\n')


    def run_deployment(self, deployment_set_location='data/test_set_VU_DM_2014.csv', run_identifier="run_mini_20180524212118", training_CHECK=True, experiment_size=MINI, reset_data=False):

        # Turn on logging.
        logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.INFO)

        # Starting the data tim
        starting_time = datetime.now()

        logging.info('================================================================================')
        log("TEST START", starting_time)


        ## Setting the locations of the data and name

        store_svm_light_loc_VU = "data/{}/{}/{}/".format(self.split_identifier, self.experiment_name, "deployment")
        VU_test_set_name = "VU_test_set"


        # Check if the model has a second set that can be used to validate if the model works and if the model shows the correct results.
        if training_CHECK:
            data_valid_location = 'data/training_set_VU_DM_2014.csv'
            store_svm_light_loc_valid = "data/{}/{}/{}/".format(self.split_identifier, self.experiment_name, "valid_deployment")
            valid_test_set_name = "valid_test_set" + experiment_size

        
        # Using the model indentifier to locate the model that is used to run the model.
        model_location = "output/{}/{}/{}/".format(self.split_identifier, self.experiment_name, run_identifier)

        # Model location is equal to the location in which the output of the dataset is stored.
        output_folder = model_location

        logging.info('================================================================================')
        ## Loading in the VU test data
        # Load data
        if reset_data or not os.path.exists(store_svm_light_loc_VU):

            # Setting the data timer.
            data_loading_timer = datetime.now()
            log("Data set {} had not yet been generated (or needs to be regenerated). Will generate now...".format(VU_test_set_name), starting_time)

            # Loading in the dataset.
            full_training_set = pandas.read_csv(deployment_set_location)
            log("Loaded full data set.", starting_time, data_loading_timer)

            #Set the data generation timeer.
            data_generation_timer = datetime.now()

            # Generate a place to store the data that is converted to a svm light frame
            if not os.path.exists(store_svm_light_loc_VU):
                os.makedirs(store_svm_light_loc_VU)
                log("Created folder.", starting_time)
            else:
                log("Folder already existed.", starting_time)


            # Convert data
            test_set = self.feature_engineering(full_training_set)


            # Store converted data            
            test_set_path = store_svm_light_loc_VU + VU_test_set_name
            self.store_data_frame_as_svm_light_deployment(test_set, test_set_path)

            # Retrieve the correct list index for pandas later
            prop_loc_id = [x for x in list(test_set.columns) if x not in NON_FEATURE_COLUMNS + self.ignored_features].index("prop_id")


            with open(store_svm_light_loc_VU + 'prop_loc_id', 'w+') as idstorage:
                idstorage.write(str(prop_loc_id))

            log("Generated the {} set!".format("test_set_complete"), starting_time, data_generation_timer)

        else:
            log("Data set {} was already generated.".format(VU_test_set_name), starting_time)


        logging.info('================================================================================')
        ## Loading in the validation split data
        # Load data BUT ONLY when it is specified in the arguments
        if reset_data or not os.path.exists(store_svm_light_loc_valid) and training_CHECK:

            # Setting the data loading timer for the validation dataset
            data_loading_timer = datetime.now()
            log("Data set {} had not yet been generated (or needs to be regenerated). Will generate now...".format(valid_test_set_name), starting_time)

            # loading the dataset used for the validation dataset
            full_training_set = pandas.read_csv(data_valid_location)
            log("Loaded full data set.", starting_time, data_loading_timer)

            # Setting the data generation timer.
            data_generation_timer = datetime.now()

            # Generate the map to store the new dataframe.
            if not os.path.exists(store_svm_light_loc_valid):
                os.makedirs(store_svm_light_loc_valid)
                log("Created folder.", starting_time)
            else:
                log("Folder already existed.", starting_time)

            # Loading only a subset of the queries
            log("Generating the {} set...".format("validation set"), starting_time)
            with open('splits/{}/{}/{}_qids.pkl'.format(self.split_identifier, experiment_size, "test"), 'rb') as fp:
                qids = pickle.load(fp)

            # Find the correct queries.
            sample_rows = full_training_set[full_training_set.srch_id.isin(qids)]

            # Convert data
            test_set = self.feature_engineering(sample_rows)

            # Store converted data            
            test_set_path = store_svm_light_loc_valid + valid_test_set_name
            self.store_data_frame_as_svm_light_deployment(test_set, test_set_path)

            log("Generated the {} set!".format("test_set_complete"), starting_time, data_generation_timer)

        elif not(training_CHECK):
            log("Validation Data set does not need to be generated.", starting_time)

        else:
            log("Data set {} was already generated.".format("validation set"), starting_time)

        logging.info('================================================================================')
        ## Loading in the model on which we test
        # Load model

        with open(model_location + 'trained_model.pkl') as model_fn:
            model = pickle.load(model_fn)
        model_timer = datetime.now()
        log("Model {} is loaded.".format(model_location.split("/")[-2]), model_timer)
        # model = LambdaMART.load(location_model + 'trained_model')

        # Load location of the propID
        with open(store_svm_light_loc_VU + 'prop_loc_id', 'r') as idstorage:
            prop_loc_id = int(idstorage.read())

        ## Testing the model on the validation set
        # Load Queries
        if training_CHECK:
            valid_data = Queries.load_from_text(store_svm_light_loc_valid + valid_test_set_name)


            # Test model
            valid_set_performance = model.predict_rankings(valid_data, n_jobs=-1)


            # Storing test results
            i = 0
            with open(output_folder + 'valid_set_values.csv', "w+") as tf:
                tf.write("SearchId ,PropertyId \n")
                for qid in valid_data.query_ids:
                    propId = valid_data.get_query(qid).get_feature_vectors(0)[:,prop_loc_id]
                    for elem in valid_set_performance[i]:
                        tf.write(str(qid) + ", " + str(int(propId[elem])) + '\n' )
                    i += 1



        logging.info('================================================================================')
        ## Testing the model on the VU test data
        # Load Queries
        test_data = Queries.load_from_text(store_svm_light_loc_VU + VU_test_set_name)
       
        # Test model
        test_data_set_performance = model.predict_rankings(test_data, n_jobs=-1)

        # Storing test results
        i = 0
        with open(output_folder + 'test_set_values.csv', "w+") as tf:
            tf.write("SearchId ,PropertyId \n")
            for qid in test_data.query_ids:
                propId = test_data.get_query(qid).get_feature_vectors(0)[:,prop_loc_id]
                for elem in test_data_set_performance[i]:
                    tf.write(str(qid) + ", " + str(int(propId[elem])) + '\n' )
                i += 1

        logging.info('================================================================================')
        ## Finalizing the results.

        output_timer = datetime.now()
        log("All output done!", starting_time, output_timer)

