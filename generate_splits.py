from collections import namedtuple
from datetime import datetime
import random
import pickle
import pandas
import os

Split = namedtuple('Split', 'name unit training validation test')

SPLITS = [
    Split('mini', 'qids', 1000, 1000, 1000),
    Split('development', 'qids', 10000, 10000, 10000),
    Split('medium', 'qids', 50000, 10000, 50000),
    Split('full', '%', 70, 10, 20)
]


def cin(file_path):
    # 'cin' stands for 'create if necessary'
    if not os.path.exists(os.path.dirname(file_path)):
        os.makedirs(os.path.dirname(file_path))

    return file_path


def generate_splits():
    full_training_set = pandas.read_csv('data/training_set_VU_DM_2014.csv')
    all_qids = set(full_training_set.srch_id.unique())

    split_identifier = 'spl_{}'.format(datetime.now().strftime('%Y%m%d%H%M%S'))
    with open(cin("splits/{}/description.txt".format(split_identifier)), "w+") as description_file:
        description_file.write("Description of split {}\n".format(split_identifier))

        for split in SPLITS:
            description_file.write("{0} set: training {2} {1}, validation {3} {1}, test {4} {1}\n"
                                   .format(split.name, split.unit, split.training, split.validation, split.test))

    for split in SPLITS:
        if split.unit == 'qids':
            training_qids = set(random.sample(all_qids, split.training))
            validation_qids = set(random.sample(all_qids - training_qids, split.validation))
            test_qids = set(random.sample(all_qids - training_qids - validation_qids, split.test))
        else:
            training_qids = set(random.sample(all_qids, len(all_qids) * split.training / 100))
            validation_qids = set(random.sample(all_qids - training_qids, len(all_qids) * split.validation / 100))
            test_qids = all_qids - training_qids - validation_qids

        # should output right amounts of qids
        print len(training_qids)
        print len(validation_qids)
        print len(test_qids)

        # should output correct percentages
        print len(training_qids) * 100 / (len(training_qids) + len(validation_qids) + len(test_qids))
        print len(validation_qids) * 100 / (len(training_qids) + len(validation_qids) + len(test_qids))
        print len(test_qids) * 100 / (len(training_qids) + len(validation_qids) + len(test_qids))

        # should output empty
        print training_qids.intersection(validation_qids)
        print training_qids.intersection(test_qids)
        print validation_qids.intersection(test_qids)

        for qid_set, set_name in [(training_qids, "training"), (validation_qids, "validation"), (test_qids, "test")]:
            with open(cin("splits/{}/{}/{}_qids".format(split_identifier, split.name, set_name)), 'wb+') as fp:
                pickle.dump(qid_set, fp)


generate_splits()
