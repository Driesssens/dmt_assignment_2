import logging
from rankpy.queries import Queries
from rankpy.models import LambdaMART
import pickle


def test_pretrained_model(experiment_name, split_identifier, run_title, experiment_size):
    model_file_path = "output/{}/{}/{}/trained_model.pkl".format(split_identifier, experiment_name, run_title)

    with open(model_file_path) as model_file:
        model = pickle.load(model_file)

    # model = LambdaMART.load(model_file_path)

    data_set_location = "data/{}/{}/{}".format(split_identifier, experiment_name, experiment_size)
    logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.INFO)
    test_data = Queries.load_from_text(data_set_location + '/' + 'test')
    test_set_performance = model.evaluate(test_data, n_jobs=-1)
    logging.info('%s on the test queries: %.8f' % (model.metric, test_set_performance))


test_pretrained_model("SuperExperiment", "spl_20180518114037", "run_full_20180526060437", "full")
