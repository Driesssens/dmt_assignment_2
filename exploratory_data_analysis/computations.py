from abstract_experiment import NON_FEATURE_COLUMNS, MINI, DEVELOPMENT, MEDIUM, FULL, log, store_data_frame_as_svm_light
import pickle
import logging
import pandas
from rankpy.queries import Queries
from datetime import datetime
from configuration import standard_configuration
from configuration import make_model
import os
import shutil
from date_time_as_number_experiment import DateTimeAsNumberExperiment
from plots import plot_correlations
from ctr_cvr_srch_id_standardized_experiment import CtrCvrSrchIdStandardizedExperiment
from super_experiment import SuperExperiment


def compute_univariate_correlation_booking_bool(data_frame, save_to_file=False, file_name="", message=""):
    return compute_univariate_correlation(data_frame, 'booking_bool', save_to_file, file_name, message)


def compute_univariate_correlation_click_bool(data_frame, save_to_file=False, file_name="", message=""):
    return compute_univariate_correlation(data_frame, 'click_bool', save_to_file, file_name, message)


def compute_univariate_correlation(data_frame, target, save_to_file=False, file_name="", message=""):
    features = [column for column in data_frame.columns.values if column not in NON_FEATURE_COLUMNS]

    print features

    correlations = data_frame.corr()[target][features].values.tolist()

    print correlations

    sorted_correlations, sorted_features = (list(t) for t in zip(*sorted(zip(correlations, features), key=lambda tup: abs(tup[0]), reverse=True)))

    print correlations

    absolute_correlations = [abs(correlation) for correlation in correlations]

    print absolute_correlations

    if save_to_file:
        if not os.path.exists("output/"):
            os.makedirs("output/")

        with open("output/" + file_name + ".txt", "w+") as df:
            df.write("Message: {}\n".format(message))

            df.write("Largest correlation: {}\n".format(max(absolute_correlations)))
            df.write("Smallest correlation: {}\n".format(min(absolute_correlations)))
            df.write("Average absolute correlation: {}\n".format(sum(absolute_correlations) / float(len(absolute_correlations))))

            df.write("\n")

            for i, feature in enumerate(sorted_features):
                df.write("{}: {}\n".format(feature, sorted_correlations[i]))

            df.write("\n")

            df.write("Feature list: {}\n\n".format(features))

            df.write("Correlations list: {}\n\n".format(correlations))

    return features, correlations


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

            df.write("Highest nDCG: {}\n".format(max(ndcgs)))
            df.write("Lowest nDCG: {}\n".format(min(ndcgs)))
            df.write("Average nDCG: {}\n".format(sum(ndcgs) / float(len(ndcgs))))

            df.write("\n")

            for i, feature in enumerate(features):
                df.write("{}: {}\n".format(feature, ndcgs[i]))

            df.write("\n")

            df.write("Feature list: {}\n\n".format(features))

            df.write("nDCGs list: {}\n\n".format(ndcgs))

    shutil.rmtree(temporary_files_location)
    return features, ndcgs


def test_univariate_ndcg():
    now = datetime.now()
    log("Starting.", now)

    data = DateTimeAsNumberExperiment().make_data_set(pandas.read_csv('../data/training_set_VU_DM_2014.csv'))

    features, ndcgs = compute_univariate_ndcg(data, verbose=True, starting_time=now, size=FULL, save_to_file=True, file_name="test_all_full", message="Uses DateTimeAsNumberExperiment to test all features.")


def test_correlation():
    original_features = ['site_id', 'visitor_location_country_id', 'visitor_hist_starrating', 'visitor_hist_adr_usd', 'prop_country_id', 'prop_id', 'prop_starrating', 'prop_review_score', 'prop_brand_bool', 'prop_location_score1', 'prop_location_score2', 'prop_log_historical_price', 'price_usd', 'promotion_flag', 'srch_destination_id', 'srch_length_of_stay', 'srch_booking_window', 'srch_adults_count', 'srch_children_count', 'srch_room_count', 'srch_saturday_night_bool', 'srch_query_affinity_score', 'orig_destination_distance', 'random_bool', 'comp1_rate', 'comp1_inv', 'comp1_rate_percent_diff', 'comp2_rate', 'comp2_inv', 'comp2_rate_percent_diff', 'comp3_rate', 'comp3_inv', 'comp3_rate_percent_diff', 'comp4_rate', 'comp4_inv', 'comp4_rate_percent_diff', 'comp5_rate', 'comp5_inv', 'comp5_rate_percent_diff', 'comp6_rate', 'comp6_inv', 'comp6_rate_percent_diff', 'comp7_rate', 'comp7_inv', 'comp7_rate_percent_diff', 'comp8_rate', 'comp8_inv', 'comp8_rate_percent_diff']

    data = SuperExperiment().make_data_set(pandas.read_csv('../data/training_set_VU_DM_2014.csv')).drop(original_features, axis=1)
    features, correlations = compute_univariate_correlation_booking_bool(data, save_to_file=True, file_name="correlations_SuperExperiment", message="Uses SuperExperiment to check correlation of all features except the original features.")


def test_comp_dingen():
    data = CtrCvrSrchIdStandardizedExperiment().make_data_set(pandas.read_csv('../data/training_set_VU_DM_2014.csv'))
    features, correlations = compute_univariate_correlation_booking_bool(data, save_to_file=True, file_name="correlations_CtrCvrSrchIdStandardizedExperiment", message="CtrCvrSrchIdStandardizedExperiment correlations")
    plot_correlations(features, correlations, vertical=False)


test_correlation()
