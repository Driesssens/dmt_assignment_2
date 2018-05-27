import pickle
import numpy
import pandas
from rankpy.queries import Queries
from rankpy.models import LambdaMART
from configuration import standard_configuration
from configuration import make_model
from plots import plot_correlations

with open("../output/spl_20180518114037/BaseExperiment7/run_medium_20180525204426/" + 'trained_model.pkl') as model_fn:
    model = pickle.load(model_fn)

# locate the experiment headers of the columns Needed for the features.
NON_FEATURE_COLUMNS = ['Index',
                       'srch_id',
                       'booking_bool',
                       'click_bool',
                       'gross_bookings_usd',
                       'position']
ignored_features = ['date_time']

ALL_FEATURES = ["srch_id","date_time","site_id","visitor_location_country_id","visitor_hist_starrating","visitor_hist_adr_usd","prop_country_id",
				"prop_id","prop_starrating","prop_review_score","prop_brand_bool","prop_location_score1","prop_location_score2","prop_log_historical_price",
				"position","price_usd","promotion_flag","srch_destination_id","srch_length_of_stay","srch_booking_window","srch_adults_count",
				"srch_children_count","srch_room_count","srch_saturday_night_bool","srch_query_affinity_score","orig_destination_distance",
				"random_bool","comp1_rate","comp1_inv","comp1_rate_percent_diff","comp2_rate","comp2_inv","comp2_rate_percent_diff","comp3_rate",
				"comp3_inv","comp3_rate_percent_diff","comp4_rate","comp4_inv","comp4_rate_percent_diff","comp5_rate","comp5_inv","comp5_rate_percent_diff",
				"comp6_rate","comp6_inv","comp6_rate_percent_diff","comp7_rate","comp7_inv","comp7_rate_percent_diff","comp8_rate","comp8_inv",
				"comp8_rate_percent_diff","click_bool","gross_bookings_usd","booking_bool"]
for elem in NON_FEATURE_COLUMNS + ignored_features:
	try:
		ALL_FEATURES.remove(elem)
	except:
		pass


numpy_array_feat = numpy.zeros([len(model.estimators),len(model.estimators[0].feature_importances_)])

i = 0

for tree in model.estimators:

	numpy_array_feat[i,:] = tree.feature_importances_
	i += 1

panda_summary = pandas.DataFrame(data=numpy_array_feat, index=None, columns=ALL_FEATURES)


plot_correlations(panda_summary.mean(axis=0).sort_values(ascending=False).index,panda_summary.mean(axis=0).sort_values(ascending=False).values)
