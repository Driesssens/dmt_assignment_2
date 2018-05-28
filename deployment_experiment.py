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


def dcg_at_k(r, k, method=0):
    """Score is discounted cumulative gain (dcg)
    Relevance is positive real values.  Can use binary
    as the previous methods.
    Example from
    http://www.stanford.edu/class/cs276/handouts/EvaluationNew-handout-6-per.pdf
    3.0
    3.0
    5.0
    4.2618595071429155
    9.6051177391888114
    9.6051177391888114
    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
        k: Number of results to consider
        method: If 0 then weights are [1.0, 1.0, 0.6309, 0.5, 0.4307, ...]
                If 1 then weights are [1.0, 0.6309, 0.5, 0.4307, ...]
    Returns:
        Discounted cumulative gain
    """
    r = numpy.asfarray(r)[:k]
    if r.size:
        if method == 0:
            return r[0] + numpy.sum(r[1:] / numpy.log2(numpy.arange(2, r.size + 1)))
        elif method == 1:
            return numpy.sum(r / numpy.log2(numpy.arange(2, r.size + 2)))
        else:
            raise ValueError('method must be 0 or 1.')
    return 0.


def ndcg_at_k(r, k, method=1):
    """Score is normalized discounted cumulative gain (ndcg)
    Relevance is positive real values.  Can use binary
    as the previous methods.
    Example from
    1.0
    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
        k: Number of results to consider
        method: If 0 then weights are [1.0, 1.0, 0.6309, 0.5, 0.4307, ...]
                If 1 then weights are [1.0, 0.6309, 0.5, 0.4307, ...]
    Returns:
        Normalized discounted cumulative gain
    """
    dcg_max = dcg_at_k(sorted(r, reverse=True), k, method)
    if not dcg_max:
        return 0.
    return dcg_at_k(r, k, method) / float(dcg_max)


class DeploymentExperiment(AbstractExperiment):
    experiment_name = "BaseExperiment"
    experiment_description = None
    ignored_features = ['date_time']

    def feature_engineering(self, raw_data_frame):
        return raw_data_frame

    split_identifier = None

    def convert_pandas_row_to_svm_light_format_deployment(self, pandas_row, missing_values_old_style=False):
        qid = pandas_row.srch_id
        relevance_grade = 0

        svm_light_string = "{} qid:{} ".format(relevance_grade, qid)

        feature_number = 1

        for feature_name, feature_value in pandas_row._asdict().iteritems():
            if feature_name not in NON_FEATURE_COLUMNS + self.ignored_features:
                if missing_values_old_style:
                    if numpy.isnan(feature_value) or numpy.isinf(feature_value):
                        feature_value = self.missing_value_default(feature_name, feature_value)

                svm_light_string += '{}:{} '.format(feature_number, feature_value)
                feature_number += 1

        return svm_light_string

    def store_data_frame_as_svm_light_deployment(self, data_frame, file_name, missing_values_old_style=False):
        with open(file_name, 'w') as output_file:
            for row in data_frame.itertuples():
                output_file.write(self.convert_pandas_row_to_svm_light_format_deployment(row, missing_values_old_style) + '\n')

    def run_deployment(self, deployment_set_location='data/test_set_VU_DM_2014.csv', run_identifier=None, training_CHECK=True, experiment_size=MINI, reset_data=False, relevance_score_testing=False):
        """
        Args:
            deployment_set_location: Location of the dataset is remodeled given the model
            run_identifier: location of the model of the run that is used. If no run_indentifier is specified the model will use the last model that is generated.
            training_CHECK: uses the test set to generate an comparison dataset to check if the model is doing it's work correctly
            experiment_size: The experiment size of the trainingscheck. If not yet generated will be generated now.
            reset_data: will reset the data of the experiments and rebuild the experiments.
        Returns:
            several files in the output folder.
        """

        # Turn on logging.
        logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.INFO)

        # Starting the data tim
        starting_time = datetime.now()

        logging.info('================================================================================')
        log("TEST START", starting_time)

        # self.experiment_name += "jaspertestdingen"
        print(self.experiment_name)

        ## Setting the locations of the data and name
        self.split_identifier = "spl_20180518114037"
        store_svm_light_loc_VU = "data/{}/{}/{}/".format(self.split_identifier, self.experiment_name, "mini")
        VU_test_set_name = "VU_test_set"

        # Check if the model has a second set that can be used to validate if the model works and if the model shows the correct results.
        data_valid_location = 'data/training_set_VU_DM_2014.csv'
        valid_test_set_name = "test"

        store_svm_light_loc_valid = "data/{}/{}/{}/".format(self.split_identifier, self.experiment_name, experiment_size)

        # Using the model indentifier to locate the model that is used to run the model.
        identifier = 0
        if run_identifier == None:
            list_of_dir = os.listdir("output/{}/{}/".format(self.split_identifier, self.experiment_name))
            for elem in list_of_dir:
                if int(elem.split("_")[-1]) > identifier:
                    run_identifier = elem
        if run_identifier == None:
            raise ValueError('Run identifier can not find a run able to use as input for the model')

        model_location = "output/{}/{}/{}/".format(self.split_identifier, self.experiment_name, run_identifier)

        # Model location is equal to the location in which the output of the dataset is stored.
        output_folder = model_location

        logging.info('================================================================================')
        ## Loading in the VU test data
        # Load data
        if False and reset_data or not os.path.exists(store_svm_light_loc_VU):

            # Setting the data timer.
            data_loading_timer = datetime.now()
            log("Data set {} had not yet been generated (or needs to be regenerated). Will generate now...".format(VU_test_set_name), starting_time)

            # Loading in the dataset.
            full_training_set = pandas.read_csv(deployment_set_location)
            log("Loaded full data set.", starting_time, data_loading_timer)

            # Set the data generation timeer.
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
            # deployment and other version
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
                log("Folder already existed XXXX.", starting_time)

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

            if relevance_score_testing:
                self.store_data_frame_as_svm_light(test_set, test_set_path)
            else:
                self.store_data_frame_as_svm_light_deployment(test_set, test_set_path)

            log("Generated the {} set!".format("test_set_complete"), starting_time, data_generation_timer)

            # Retrieve the correct list index for pandas later
            prop_loc_id = [x for x in list(test_set.columns) if x not in NON_FEATURE_COLUMNS + self.ignored_features].index("prop_id")

            with open(store_svm_light_loc_VU + 'prop_loc_id', 'w+') as idstorage:
                idstorage.write(str(prop_loc_id))

        elif not (training_CHECK):
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

            # Storing test results
            # testing with relevance_score
            if relevance_score_testing:
                valid_data = Queries.load_from_text(store_svm_light_loc_valid + valid_test_set_name, has_sorted_relevances=True)

                # Test model
                valid_set_performance = model.predict_rankings(valid_data, n_jobs=-1)
                i = 0
                qid_ndcg = []
                with open(output_folder + 'valid_set_values_with_rel.csv', "w+") as tf:
                    tf.write("SearchId ,PropertyId, relevance_score \n")
                    for qid in valid_data.query_ids:
                        propId = valid_data.get_query(qid).get_feature_vectors(0)[:, prop_loc_id]
                        relevance_score = valid_data.get_query(qid).relevance_scores
                        relevance_score_sorted = []
                        for elem in valid_set_performance[i]:
                            tf.write(str(qid) + ", " + str(int(propId[elem])) + ", " + str(int(relevance_score[elem])) + '\n')
                            relevance_score_sorted.append(int(relevance_score[elem]))
                        i += 1
                        qid_ndcg.append(ndcg_at_k(relevance_score_sorted, len(relevance_score_sorted)))
                logging.info('%s on the test queries according to rankpy metrics: %.8f' % (model.metric, model.evaluate(valid_data, n_jobs=-1)))
                logging.info('%s on the test queries according to own metrics: %.8f' % ("nDCG", sum(qid_ndcg) / float(len(qid_ndcg))))
            else:
                # Testing witout relevance score
                valid_data = Queries.load_from_text(store_svm_light_loc_valid + valid_test_set_name, has_sorted_relevances=True)

                # Test model
                valid_set_performance = model.predict_rankings(valid_data, n_jobs=-1)
                print "THE SCORE IS!!!!!! {}".format(model.evaluate(valid_data, n_jobs=1))
                i = 0
                with open(output_folder + 'valid_set_values_without_rel.csv', "w+") as tf:
                    tf.write("SearchId ,PropertyId \n")
                    for qid in valid_data.query_ids:
                        propId = valid_data.get_query(qid).get_feature_vectors(0)[:, prop_loc_id]
                        for elem in valid_set_performance[i]:
                            tf.write(str(qid) + ", " + str(int(propId[elem])) + '\n')
                        i += 1

        logging.info('================================================================================')
        ## Testing the model on the VU test data
        # Load Queries
        test_data = Queries.load_from_text(store_svm_light_loc_VU + VU_test_set_name, has_sorted_relevances=True)

        # Test model
        test_data_set_performance = model.predict_rankings(test_data, n_jobs=-1)

        # Storing test results
        i = 0
        with open(output_folder + 'VU_prediction_results_group_106.csv', "w+") as tf:
            tf.write("SearchId ,PropertyId \n")
            for qid in test_data.query_ids:
                propId = test_data.get_query(qid).get_feature_vectors(0)[:, prop_loc_id]
                for elem in test_data_set_performance[i]:
                    tf.write(str(qid) + ", " + str(int(propId[elem])) + '\n')
                i += 1

        logging.info('================================================================================')
        ## Finalizing the results.

        output_timer = datetime.now()
        log("All output done!", starting_time, output_timer)


DeploymentExperiment().run_deployment(run_identifier="run_mini_20180526041336", reset_data=True)
