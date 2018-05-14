from abstract_experiment import AbstractExperiment
from configuration import make_configuration


class BaseExperiment(AbstractExperiment):
    split_identifier = "spl_20180511180820"

    experiment_name = "BaseExperiment"
    experiment_description = """The most basic experiment. Ignores 'date_time', includes all other raw features without doing any preprocessing. Uses 0.000 for missing values."""

    ignored_features = ['date_time','visitor_hist_starrating', 'visitor_hist_adr_usd']


    def missing_value_default(self, feature_name, feature_value):

    	if feature_name == 'visitor_hist_starrating':
        	return '3.374334'
        if feature_name == 'visitor_hist_adr_usd':
        	return '0.000000'

    	if feature_name == 'prop_review_score':
        	return '0.000000'

     	if feature_name == 'prop_location_score2':
        	return '0.000000'
        	

    	if feature_name == 'srch_query_affinity_score':
        	return '-24.14641'


    	if feature_name == 'orig_destination_distance':
        	return '0.000000'

        for elem in range(1,9)
	    	if feature_name == "comp"+str(elem)+"_rate":
	        	return '0.000000'

	    	if feature_name == "comp"+str(elem)+"_inv":
	        	return '0.000000'

	    	if feature_name == "comp"+str(elem)+"_rate_percent_diff":
	        	return '0.000000'
      


        else:
        	return '0.000000'

    def feature_engineering(self, raw_data_frame):


    	self.data_frame_max = raw_data_frame.max()
		self.data_frame_min = raw_data_frame.min()
		self.data_frame_mean = raw_data_frame.mean()

        return raw_data_frame


# BaseExperiment().run_mini_experiment()

short_experiment = make_configuration(epochs=10)
BaseExperiment().run_development_experiment(configuration=short_experiment, reset_data=True)
