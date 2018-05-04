import pandas
import random
from input_output import output_data_frame_as_letor_file


def make_small_train_and_test_sets(number, file_name_prefix=""):
    full_training_set = pandas.read_csv('data/full_training.csv')
    srch_ids = set(full_training_set.srch_id.unique())
    print len(srch_ids)
    train_sample_ids = set(random.sample(srch_ids, number))
    test_sample_ids = set(random.sample(srch_ids - train_sample_ids, number))
    train_sample_rows = full_training_set[full_training_set.srch_id.isin(train_sample_ids)]
    test_sample_rows = full_training_set[full_training_set.srch_id.isin(test_sample_ids)]

    train_sample_rows.to_csv("{}small_train_{}.csv".format(file_name_prefix, str(number)))
    test_sample_rows.to_csv("{}small_test_{}.csv".format(file_name_prefix, str(number)))
    output_data_frame_as_letor_file(train_sample_rows, "{}small_train_{}.txt".format(file_name_prefix, str(number)))
    output_data_frame_as_letor_file(test_sample_rows, "{}small_test_{}.txt".format(file_name_prefix, str(number)))


def make_multiple_downsamples(name_size_tuples, file_name_prefix=""):
    full_training_set = pandas.read_csv('data/full_training.csv')
    srch_ids = set(full_training_set.srch_id.unique())

    for name, size in name_size_tuples:
        if size is -1:
            sample_ids = srch_ids
        else:
            sample_ids = set(random.sample(srch_ids, size))
            srch_ids -= sample_ids

        sample_rows = full_training_set[full_training_set.srch_id.isin(sample_ids)]
        name = "{}_{}_{}".format(file_name_prefix, name, str(size))
        sample_rows.to_csv("{}.csv".format(name))
        output_data_frame_as_letor_file(sample_rows, "{}.txt".format(name))


make_multiple_downsamples([('train', 50000), ('validation', 10000), ('test', 50000)], "50k")
