from abstract_experiment import NON_FEATURE_COLUMNS, MINI, DEVELOPMENT, MEDIUM, FULL, log, store_data_frame_as_svm_light
from datetime import datetime
import pickle
import logging
import pandas
from rankpy.queries import Queries
from rankpy.models import LambdaMART
from datetime import datetime
from configuration import standard_configuration
from configuration import make_model
import os
import shutil
from one_feature_experiment import OneFeatureExperiment
from date_time_as_number_experiment import DateTimeAsNumberExperiment


def compute_univariate_ndcg(data_frame, verbose=False, starting_time=None, size=FULL, save_to_file=False, file_name="", message="", ):
    split_identifier = "spl_20180518114037"

    if starting_time is None:
        starting_time = datetime.now()

    log("EXPERIMENT START", starting_time)

    computation_identifier = 'computation_{}'.format(datetime.now().strftime('%Y%m%d%H%M%S'))

    columns = data_frame.columns.values
    features = [column for column in columns if column not in NON_FEATURE_COLUMNS]
    ndcgs = []

    temporary_files_location = "temp/{}/".format(computation_identifier)
    os.makedirs(temporary_files_location)

    for feature in features:
        feature_starting_time = datetime.now()
        log("Starting feature '{}'...".format(feature), starting_time)

        feature_data_frame = data_frame[['srch_id', 'booking_bool', 'click_bool', feature]]

        for set_name in ["training", "validation", "test"]:
            timer = datetime.now()

            if verbose: log("{} | Generating the {} set...".format(feature, set_name), starting_time)

            with open('../splits/{}/{}/{}_qids.pkl'.format(split_identifier, size, set_name), 'rb') as fp:
                qids = pickle.load(fp)

            sample_rows = feature_data_frame[feature_data_frame.srch_id.isin(qids)]

            data_set_path = temporary_files_location + '/' + feature + '_' + set_name
            store_data_frame_as_svm_light(sample_rows, data_set_path)

            if verbose: log("{} | Generated the {} set!".format(feature, set_name), starting_time, timer)

        if verbose:
            logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.INFO)
        else:
            logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.WARN)

        # Load the data sets.
        training_data = Queries.load_from_text(temporary_files_location + '/' + feature + '_' + 'training')
        validation_data = Queries.load_from_text(temporary_files_location + '/' + feature + '_' + 'validation')
        test_data = Queries.load_from_text(temporary_files_location + '/' + feature + '_' + 'test')

        configuration = standard_configuration()

        model_fitting_timer = datetime.now()
        model = make_model(configuration)
        model.fit(training_data, validation_queries=validation_data)
        if verbose: log("{} | Model fitted.".format(feature), starting_time, model_fitting_timer)

        performance_testing_timer = datetime.now()
        test_set_performance = model.evaluate(test_data, n_jobs=-1)

        if verbose: log("{} | Performance tested.".format(feature), starting_time, performance_testing_timer)
        ndcgs.append(test_set_performance)
        log("Finished feature '{}'!".format(feature), starting_time, feature_starting_time)

    if save_to_file:
        if not os.path.exists("output/"):
            os.makedirs("output/")

        with open("output/" + file_name + ".txt", "w+") as df:
            df.write("Message: {}\n".format(message))
            df.write("Split: {}\n".format(split_identifier))
            df.write("Size: {}\n".format(size))

            df.write("\n")

            for i, feature in enumerate(features):
                df.write("{}: {}\n".format(feature, ndcgs[i]))

            df.write("\n")

            df.write("Feature list: {}\n\n".format(features))

            df.write("nDCGs list: {}\n\n".format(ndcgs))

    shutil.rmtree(temporary_files_location)
    return features, ndcgs


def test_this_thing():
    now = datetime.now()
    log("Starting.", now)

    data = DateTimeAsNumberExperiment().make_data_set(pandas.read_csv('../data/training_set_VU_DM_2014.csv'))
    print compute_univariate_ndcg(data, verbose=True, starting_time=now, size=MINI, save_to_file=True, file_name="test_all_mini", message="Uses DateTimeAsNumberExperiment to test all features.")

    print compute_univariate_ndcg(data, verbose=True, starting_time=now, size=DEVELOPMENT, save_to_file=True, file_name="test_all_development", message="Uses DateTimeAsNumberExperiment to test all features.")

    print compute_univariate_ndcg(data, verbose=True, starting_time=now, size=FULL, save_to_file=True, file_name="test_all_full", message="Uses DateTimeAsNumberExperiment to test all features.")


test_this_thing()
