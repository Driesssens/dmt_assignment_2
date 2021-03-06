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
from configuration import boosted_configuration
import json
from shutil import copyfile
from scipy import mean
from collections import defaultdict

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


def store_data_frame_as_svm_light(data_frame, file_name):
    with open(file_name, 'w') as output_file:
        for row in data_frame.itertuples():
            output_file.write(convert_pandas_row_to_svm_light_format(row) + '\n')


def convert_pandas_row_to_svm_light_format(pandas_row):
    qid = pandas_row.srch_id
    relevance_grade = 5 if pandas_row.booking_bool is 1 else pandas_row.click_bool

    svm_light_string = "{} qid:{} ".format(relevance_grade, qid)

    feature_number = 1

    for feature_name, feature_value in pandas_row._asdict().iteritems():
        if feature_name not in NON_FEATURE_COLUMNS:
            svm_light_string += '{}:{} '.format(feature_number, feature_value)
            feature_number += 1

    return svm_light_string


class AbstractExperiment:
    experiment_name = None
    experiment_description = None
    ignored_features = None
    split_identifier = None
    uses_ctr_and_cvr = False

    # Explanation: column_specific_missing_value_fillers can be a dict from column name to value. After
    # feature engineering, each missing value in that specific column will be filled with that specific
    # value. All columns that you don't specify in column_specific_missing_value_fillers, their missing
    # values will be filled by general_missing_value_filler regardless of what column it is. If you simply
    # keep column_specific_missing_value_fillers at None, every missing value in the whole data frame
    # will be general_missing_value_filler. So the order of application is like this:
    # feature_engineering --> column_specific_missing_value_fillers --> general_missing_value_filler

    column_specific_missing_value_fillers = None
    general_missing_value_filler = 0.0

    def missing_value_default(self, feature_name, feature_value):
        ###############################################################################################
        ############ DEPRECATED. USE missing_value_filler OR missing_value_fillers INSTEAD ############
        ############ IF YOU STILL USE THIS, ADD missing_values_old_style=True TO run_experiment() #####
        ###############################################################################################

        # Can be overwritten in child classes - but should probably be done in feature_engineering instead.
        # This function handles missing values even after the feature engineering. Also handles inf, nan, etc.

        raise Exception("This method is deprecated. If you really need to use it, implement it in your child class.")
        return '0.000000'

    def feature_engineering(self, raw_data_frame):
        # NOT IMPLEMENTED
        pass

    def OLD_convert_pandas_row_to_svm_light_format(self, pandas_row):
        qid = pandas_row.srch_id
        relevance_grade = 5 if pandas_row.booking_bool is 1 else pandas_row.click_bool

        svm_light_string = "{} qid:{} ".format(relevance_grade, qid)

        feature_number = 1

        for feature_name, feature_value in pandas_row._asdict().iteritems():
            if feature_name not in NON_FEATURE_COLUMNS + self.ignored_features:
                if numpy.isnan(feature_value) or numpy.isinf(feature_value):
                    feature_value = self.missing_value_default(feature_name, feature_value)

                svm_light_string += '{}:{} '.format(feature_number, feature_value)
                feature_number += 1

        return svm_light_string

    def OLD_store_data_frame_as_svm_light(self, data_frame, file_name):
        with open(file_name, 'w') as output_file:
            for row in data_frame.itertuples():
                output_file.write(self.OLD_convert_pandas_row_to_svm_light_format(row) + '\n')

    def run_mini_experiment(self, configuration=None, reset_data=False, add_to_leaderboard=True, extra_instructions=None, missing_values_old_style=False):
        self.run_experiment(MINI, configuration, reset_data, add_to_leaderboard, extra_instructions, missing_values_old_style)

    def run_development_experiment(self, configuration=None, reset_data=False, add_to_leaderboard=True, extra_instructions=None, missing_values_old_style=False):
        self.run_experiment(DEVELOPMENT, configuration, reset_data, add_to_leaderboard, extra_instructions, missing_values_old_style)

    def run_medium_experiment(self, configuration=None, reset_data=False, add_to_leaderboard=True, extra_instructions=None, missing_values_old_style=False):
        self.run_experiment(MEDIUM, configuration, reset_data, add_to_leaderboard, extra_instructions, missing_values_old_style)

    def run_full_experiment(self, configuration=None, reset_data=False, add_to_leaderboard=True, extra_instructions=None, missing_values_old_style=False):
        self.run_experiment(FULL, configuration, reset_data, add_to_leaderboard, extra_instructions, missing_values_old_style)

    def get_ctr_and_cvr(self, experiment_size):
        location = "splits/{}/{}/".format(self.split_identifier, experiment_size)

        with open(location + "ctr.json") as js:
            ctr = {int(k): v for k, v in json.load(js).items()}

        with open(location + "cvr.json") as js:
            cvr = {int(k): v for k, v in json.load(js).items()}

        average_ctr = mean(ctr.values())
        average_cvr = mean(cvr.values())

        ctr = defaultdict(lambda: average_ctr, ctr)
        cvr = defaultdict(lambda: average_cvr, cvr)

        return ctr, cvr

    def maybe_create_ctr_and_cvr(self, data_frame, experiment_size, qids):
        location = "splits/{}/{}/".format(self.split_identifier, experiment_size)

        if not (os.path.isfile(location + "ctr.json") and os.path.isfile(location + "cvr.json")):
            print "CREATING CTR AND CVR FOR {}...".format(experiment_size)

            if experiment_size == FULL:
                sample_rows = data_frame[~data_frame.srch_id.isin(qids['test'])]
            else:
                sample_rows = data_frame[~data_frame.srch_id.isin(qids['validation'] | qids['test'] | qids['validation'])]

            prop_ids = sample_rows['prop_id'].unique()

            ctr = {}
            cvr = {}

            for i, prop_id in enumerate(prop_ids):
                relevant_rows = sample_rows.loc[sample_rows['prop_id'] == prop_id]
                clicks = relevant_rows['click_bool']
                bookings = relevant_rows['booking_bool']

                ctr[str(prop_id)] = clicks.mean()
                cvr[str(prop_id)] = bookings.mean()

            with open(location + "ctr.json", 'w+') as fp:
                json.dump(ctr, fp)

            with open(location + "cvr.json", 'w+') as fp:
                json.dump(cvr, fp)

    def make_data_set(self, data_frame, experiment_size=None):
        if self.uses_ctr_and_cvr:
            ctr, cvr = self.get_ctr_and_cvr(experiment_size)
            feature_engineered_data_frame = self.feature_engineering(data_frame, ctr, cvr)
        else:
            feature_engineered_data_frame = self.feature_engineering(data_frame)

        if self.column_specific_missing_value_fillers is not None:
            column_specific_filled_data_frame = feature_engineered_data_frame.fillna(self.column_specific_missing_value_fillers)
        else:
            column_specific_filled_data_frame = feature_engineered_data_frame

        fully_filled_data_frame = column_specific_filled_data_frame.fillna(self.general_missing_value_filler)
        ignored_features_data_frame = fully_filled_data_frame.drop(self.ignored_features, axis=1)

        return ignored_features_data_frame

    def run_boosted_experiment(self, reset_data=False, add_to_leaderboard=True, extra_instructions=None, missing_values_old_style=False):
        self.experiment_name += "Boosted"
        self.run_full_experiment(configuration=boosted_configuration(),
                                 reset_data=reset_data,
                                 add_to_leaderboard=add_to_leaderboard,
                                 extra_instructions=extra_instructions,
                                 missing_values_old_style=missing_values_old_style)

    def run_test_experiment(self, reset_data=False, add_to_leaderboard=True, extra_instructions=None, missing_values_old_style=False):
        self.experiment_name += "Testing"
        self.run_mini_experiment(configuration=boosted_configuration(),
                                 reset_data=reset_data,
                                 add_to_leaderboard=add_to_leaderboard,
                                 extra_instructions=extra_instructions,
                                 missing_values_old_style=missing_values_old_style)

    def run_experiment(self, experiment_size, configuration, reset_data, add_to_leaderboard, extra_instructions, missing_values_old_style):
        starting_time = datetime.now()

        log("EXPERIMENT START", starting_time)

        data_set_location = "data/{}/{}/{}".format(self.split_identifier, self.experiment_name, experiment_size)
        data_set_name = "{}-{}-{}".format(self.experiment_name, self.split_identifier, experiment_size)

        if reset_data or not os.path.exists(data_set_location):

            data_generation_timer = datetime.now()
            data_loading_timer = datetime.now()

            log("Data set {} had not yet been generated (or needs to be regenerated). Will generate now...".format(data_set_name), starting_time)

            full_training_set = pandas.read_csv('data/training_set_VU_DM_2014.csv')
            log("Loaded full data set.", starting_time, data_loading_timer)

            if not os.path.exists(data_set_location):
                os.makedirs(data_set_location)
                log("Created folder.", starting_time)
            else:
                log("Folder already existed.", starting_time)

            qids = {}

            for set_name in ["training", "validation", "test"]:
                with open('splits/{}/{}/{}_qids.pkl'.format(self.split_identifier, experiment_size, set_name), 'rb') as fp:
                    qids[set_name] = pickle.load(fp)

            if self.uses_ctr_and_cvr:
                self.maybe_create_ctr_and_cvr(full_training_set, experiment_size, qids)

            for set_name in ["training", "validation", "test"]:
                timer = datetime.now()
                log("Generating the {} set...".format(set_name), starting_time)
                # with open('splits/{}/{}/{}_qids.pkl'.format(self.split_identifier, experiment_size, set_name), 'rb') as fp:
                #     qids = pickle.load(fp)

                sample_rows = full_training_set[full_training_set.srch_id.isin(qids[set_name])]

                pandas.set_option('mode.use_inf_as_null', True)

                data_set_path = data_set_location + '/' + set_name

                if missing_values_old_style:
                    data_set = self.feature_engineering(sample_rows)
                    self.OLD_store_data_frame_as_svm_light(data_set, data_set_path)
                else:
                    data_set = self.make_data_set(sample_rows, experiment_size)
                    store_data_frame_as_svm_light(data_set, data_set_path)

                if set_name == 'training':
                    data_set.head(250).to_csv(data_set_location + '/example_data.csv')

                log("Generated the {} set!".format(set_name), starting_time, timer)

            log("All data generated.".format(minutes_passed(starting_time)), starting_time, data_generation_timer)

        else:
            log("Data set {} was already generated.".format(data_set_name), starting_time)

        log("Through with the experiment.", starting_time)

        run_identifier = 'run_{}_{}'.format(experiment_size, datetime.now().strftime('%Y%m%d%H%M%S'))

        # Turn on logging.
        logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.INFO)

        data_loading_timer = datetime.now()

        # Load the data sets.
        load_timer = datetime.now()
        training_data = Queries.load_from_text(data_set_location + '/' + 'training')
        log("Training queries loaded.".format(minutes_passed(starting_time)), starting_time, load_timer)

        load_timer = datetime.now()
        validation_data = Queries.load_from_text(data_set_location + '/' + 'validation')
        log("Validation queries loaded.".format(minutes_passed(starting_time)), starting_time, load_timer)

        load_timer = datetime.now()
        test_data = Queries.load_from_text(data_set_location + '/' + 'test')
        log("Test queries loaded.".format(minutes_passed(starting_time)), starting_time, load_timer)

        log("ALL queries loaded.".format(minutes_passed(starting_time)), starting_time, data_loading_timer)

        if configuration is None:
            configuration = standard_configuration()

        model_fitting_timer = datetime.now()
        model = make_model(configuration)
        model.fit(training_data, validation_queries=validation_data)
        validation_performance = model.best_performance[1][0]
        log("Model fitted.", starting_time, model_fitting_timer)

        logging.info('================================================================================')

        performance_testing_timer = datetime.now()
        test_set_performance = model.evaluate(test_data, n_jobs=-1)

        log("Performance tested.", starting_time, performance_testing_timer)

        output_timer = datetime.now()

        logging.info('%s on the test queries: %.8f' % (model.metric, test_set_performance))

        output_folder = 'output/{}/{}/{}'.format(self.split_identifier, self.experiment_name, run_identifier)
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        model.save(output_folder + '/trained_model')

        with open(output_folder + '/trained_model.pkl', 'wb') as secondstorage:
            pickle.dump(model, secondstorage)

        open(output_folder + '/nDCG_test_{}'.format(str(test_set_performance)), 'w+')
        open(output_folder + '/nDCG_validation_{}'.format(str(validation_performance)), 'w+')

        with open(output_folder + '/details.txt', "w+") as df:
            df.write("Details of run {} of {} on the {} set.\n".format(run_identifier, self.experiment_name, experiment_size))
            df.write("Experiment took {} minutes.\n".format(minutes_passed(starting_time)))
            df.write("Using old missing values system: {}.\n".format(missing_values_old_style))
            df.write("About {}: {}\n".format(self.experiment_name, self.experiment_description))
            df.write("Used split: {}\n".format(self.split_identifier))
            df.write("Result: nDCG {} on test set after {} epochs.\n".format(test_set_performance, model.n_estimators))
            df.write("Result: nDCG {} on validation set.\n".format(validation_performance))

            df.write("\nConfiguration (hyperparameters):\n")
            df.write("{}\n".format(str(dict(configuration._asdict()))))

        with open(output_folder + '/training_performance_per_epoch.txt', "w+") as tf:
            tf.write(str(model.training_performance.tolist()))

        with open(output_folder + '/validation_performance_per_epoch.txt', "w+") as vf:
            vf.write(str(model.validation_performance.tolist()))

        split_folder = 'output/{}'.format(self.split_identifier)

        if add_to_leaderboard:
            with open(split_folder + '/leaderboard.json', "a+") as json_file:
                try:
                    json_file.seek(0)
                    leaderboard = json.load(json_file)
                except ValueError:
                    leaderboard = {}

                if experiment_size not in leaderboard:
                    leaderboard[experiment_size] = []

                leaderboard[experiment_size].append((test_set_performance, self.experiment_name, run_identifier))
                leaderboard[experiment_size] = sorted(leaderboard[experiment_size], key=lambda tup: tup[0], reverse=True)
                json_file.truncate(0)
                json.dump(leaderboard, json_file)

                with open(split_folder + '/leaderboard.txt', "w") as txt_file:
                    for exp_size in leaderboard:
                        txt_file.write(exp_size + '\n')
                        largest_experiment_name = max([len(row[1]) for row in leaderboard[exp_size]])
                        for i, row in enumerate(leaderboard[exp_size]):
                            txt_file.write("   {ind}. {ndcg:1.6f} {name:<{len}} {run}\n".format(
                                ind='[' + str(i + 1) + ']', ndcg=row[0], name=row[1], run=row[2], len=largest_experiment_name)
                            )
                        txt_file.write('\n')

        copyfile(data_set_location + '/example_data.csv', output_folder + '/example_data.csv')

        log("All output done!", starting_time, output_timer)

        if extra_instructions is not None:
            extra_instructions({
                'output_folder': output_folder,
            })

            log("Performed extra instructions.", starting_time, output_timer)

    def run_final_experiment(self, missing_values_old_style):
        final_split_identifier = "spl_final_20180526151122"
        experiment_size = "final"
        self.experiment_name += "Final"

        starting_time = datetime.now()

        log("EXPERIMENT START", starting_time)

        data_set_location = "data/{}/{}/{}".format(final_split_identifier, self.experiment_name, experiment_size)
        data_set_name = "{}-{}-{}".format(self.experiment_name, final_split_identifier, experiment_size)

        # if reset_data or not os.path.exists(data_set_location):
        if True:
            data_generation_timer = datetime.now()
            data_loading_timer = datetime.now()

            log("Data set {} had not yet been generated (or needs to be regenerated). Will generate now...".format(data_set_name), starting_time)

            full_training_set = pandas.read_csv('data/training_set_VU_DM_2014.csv')
            log("Loaded full data set.", starting_time, data_loading_timer)

            if not os.path.exists(data_set_location):
                os.makedirs(data_set_location)
                log("Created folder.", starting_time)
            else:
                log("Folder already existed.", starting_time)

            qids = {}

            for set_name in ["training", "validation"]:
                with open('splits/{}/{}/{}_qids.pkl'.format(final_split_identifier, experiment_size, set_name), 'rb') as fp:
                    qids[set_name] = pickle.load(fp)

            for set_name in ["training", "validation"]:
                timer = datetime.now()
                log("Generating the {} set...".format(set_name), starting_time)

                sample_rows = full_training_set[full_training_set.srch_id.isin(qids[set_name])]

                pandas.set_option('mode.use_inf_as_null', True)

                data_set_path = data_set_location + '/' + set_name

                if missing_values_old_style:
                    data_set = self.feature_engineering(sample_rows)
                    self.OLD_store_data_frame_as_svm_light(data_set, data_set_path)
                else:
                    data_set = self.make_data_set(sample_rows, experiment_size)
                    store_data_frame_as_svm_light(data_set, data_set_path)

                if set_name == 'training':
                    data_set.head(250).to_csv(data_set_location + '/example_data.csv')

                log("Generated the {} set!".format(set_name), starting_time, timer)

            log("All data generated.".format(minutes_passed(starting_time)), starting_time, data_generation_timer)

        else:
            log("Data set {} was already generated.".format(data_set_name), starting_time)

        log("Through with the experiment.", starting_time)

        run_identifier = 'run_{}_{}'.format(experiment_size, datetime.now().strftime('%Y%m%d%H%M%S'))

        # Turn on logging.
        logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.INFO)

        data_loading_timer = datetime.now()

        # Load the data sets.
        load_timer = datetime.now()
        training_data = Queries.load_from_text(data_set_location + '/' + 'training')
        log("Training queries loaded.".format(minutes_passed(starting_time)), starting_time, load_timer)

        load_timer = datetime.now()
        validation_data = Queries.load_from_text(data_set_location + '/' + 'validation')
        log("Validation queries loaded.".format(minutes_passed(starting_time)), starting_time, load_timer)

        log("ALL queries loaded.".format(minutes_passed(starting_time)), starting_time, data_loading_timer)

        configuration = boosted_configuration()

        model_fitting_timer = datetime.now()
        model = make_model(configuration)
        model.fit(training_data, validation_queries=validation_data)
        validation_performance = model.best_performance[1][0]
        log("Model fitted.", starting_time, model_fitting_timer)

        logging.info('================================================================================')

        output_timer = datetime.now()

        output_folder = 'output/{}/{}/{}'.format(final_split_identifier, self.experiment_name, run_identifier)
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        model.save(output_folder + '/trained_model')

        with open(output_folder + '/trained_model.pkl', 'wb') as secondstorage:
            pickle.dump(model, secondstorage)

        open(output_folder + '/nDCG_validation_{}'.format(str(validation_performance)), 'w+')

        with open(output_folder + '/details.txt', "w+") as df:
            df.write("Details of run {} of {} on the {} set.\n".format(run_identifier, self.experiment_name, experiment_size))
            df.write("Experiment took {} minutes.\n".format(minutes_passed(starting_time)))
            df.write("Using old missing values system: {}.\n".format(missing_values_old_style))
            df.write("About {}: {}\n".format(self.experiment_name, self.experiment_description))
            df.write("Used split: {}\n".format(final_split_identifier))
            df.write("Result: nDCG {} on validation set after {} epochs.\n".format(validation_performance, model.n_estimators))

            df.write("\nConfiguration (hyperparameters):\n")
            df.write("{}\n".format(str(dict(configuration._asdict()))))

        with open(output_folder + '/training_performance_per_epoch.txt', "w+") as tf:
            tf.write(str(model.training_performance.tolist()))

        with open(output_folder + '/validation_performance_per_epoch.txt', "w+") as vf:
            vf.write(str(model.validation_performance.tolist()))

        copyfile(data_set_location + '/example_data.csv', output_folder + '/example_data.csv')

        log("All output done!", starting_time, output_timer)
