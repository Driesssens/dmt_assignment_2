from abstract_experiment import AbstractExperiment
from abstract_ized_experiment import AbstractIzedExperiment
from configuration import make_configuration
from sklearn import preprocessing
import datetime
import pandas as pd

class HustinxExperimentE15(AbstractIzedExperiment):
    split_identifier = "spl_20180518114037"

    experiment_name = "HustinxExperimentE15"
    experiment_description = """Experiment that uses the hour of the day that a search is done (note, this doesnt include time zone). Ignores 'date_time', includes all other raw features without doing any preprocessing. Uses 0.000 for missing values."""

    ignored_features = ['date_time', 'gross_booking_usd', 'booking_bool', 'click_bool', 'position',
        'orig_destination_distance', 'srch_saterday_night_bool', 'srch_children_count', 'srch_adults_count', 'visitor_location_country_id']
        #'comp1_rate', 'comp1_inv', 'comp1_rate_percent_diff',
        #'comp2_rate', 'comp2_inv', 'comp2_rate_percent_diff',
        #'comp3_rate', 'comp3_inv', 'comp3_rate_percent_diff',
        #'comp4_rate', 'comp4_inv', 'comp4_rate_percent_diff',
        #'comp5_rate', 'comp5_inv', 'comp5_rate_percent_diff',
        #'comp6_rate', 'comp6_inv', 'comp6_rate_percent_diff',
        #'comp7_rate', 'comp7_inv', 'comp7_rate_percent_diff',
        #'comp8_rate', 'comp8_inv', 'comp8_rate_percent_diff']
        
    grouped_attributes = ["srch_id"]
    ization = "normalized"    
    

    def missing_value_default(self, feature_name, feature_value):

        comps = ['comp1_rate', 'comp1_inv', 'comp1_rate_percent_diff', 
        'comp2_rate', 'comp2_inv', 'comp2_rate_percent_diff',
        'comp3_rate', 'comp3_inv', 'comp3_rate_percent_diff',
        'comp4_rate', 'comp4_inv', 'comp4_rate_percent_diff',
        'comp5_rate', 'comp5_inv', 'comp5_rate_percent_diff',
        'comp6_rate', 'comp6_inv', 'comp6_rate_percent_diff',
        'comp7_rate', 'comp7_inv', 'comp7_rate_percent_diff',
        'comp8_rate', 'comp8_inv', 'comp8_rate_percent_diff']
    
        if(feature_name == 'prop_review_score' or 
        feature_name == 'prop_location_score1' or
        feature_name == 'prop_location_score2' or 
        feature_name == 'srch_query_affinity_score'):
            feature_value = '1.000000'
        elif(feature_name in comps):
            feature_value = '0.000000'
        else:
            feature_value = '-1.000000'
        
        return feature_value

    def feature_engineering(self, raw_data_frame):
        df = raw_data_frame
        # df_dates = df['date_time']
        # df_dates = df_dates.apply(lambda date: datetime.datetime.strptime(date, "%Y-%m-%d %H:%M:%S"))
        # hours = df_dates.apply(lambda dt: (dt.hour))
        
        # df['hour'] = hours
        
        df_dates = df['date_time']
        df_dates = df_dates.apply(lambda date: datetime.datetime.strptime(date, "%Y-%m-%d %H:%M:%S"))
        day_part = df_dates.apply(lambda dt: (dt.hour%24+6)//6)

        df['day_part'] = day_part
        
        df_dates = df['date_time']
        df_dates = df_dates.apply(lambda date: datetime.datetime.strptime(date, "%Y-%m-%d %H:%M:%S"))
        seasons = df_dates.apply(lambda dt: (dt.month%12 + 3)//3)

        df['season'] = seasons
        
        # df_dates = df['date_time']
        # df_dates = df_dates.apply(lambda date: datetime.datetime.strptime(date, "%Y-%m-%d %H:%M:%S"))
        # months = df_dates.apply(lambda dt: (dt.month))
        
        # df['month'] = months
        
        df['diff_usd'] = abs(df['visitor_hist_adr_usd'] - df['price_usd'])
        
        df['diff_starrating'] = abs(df['visitor_hist_starrating'] - df['prop_starrating'])
        
        return AbstractIzedExperiment.add_normalized_attributes(self, df, self.grouped_attributes)

    def addSeason(df):
        df_dates = df['date_time']
        df_dates = df_dates.apply(lambda date: datetime.datetime.strptime(date, "%Y-%m-%d %H:%M:%S"))
        seasons = df_dates.apply(lambda dt: (dt.month%12 + 3)//3)

        df['season'] = seasons
        return df
        
    def addDayPart(df):
        df_dates = df['date_time']
        df_dates = df_dates.apply(lambda date: datetime.datetime.strptime(date, "%Y-%m-%d %H:%M:%S"))
        day_part = df_dates.apply(lambda dt: (dt.hour%24+6)//6)

        df['day_part'] = day_part
        return df
    
    def addMonth(df):
        df_dates = df['date_time']
        df_dates = df_dates.apply(lambda date: datetime.datetime.strptime(date, "%Y-%m-%d %H:%M:%S"))
        months = df_dates.apply(lambda dt: (dt.month))
        
        df['month'] = months
        return df
        
    def addHour(df):
        df_dates = df['date_time']
        df_dates = df_dates.apply(lambda date: datetime.datetime.strptime(date, "%Y-%m-%d %H:%M:%S"))
        hours = df_dates.apply(lambda dt: (dt.hour))
        
        df['hour'] = hours
        return df

#HustinxExperimentE2().run_mini_experiment(reset_data=True)
HustinxExperimentE15().run_full_experiment(reset_data=True)

#short_experiment = make_configuration(epochs=10)
#HustinxExperimentE().run_development_experiment(configuration=short_experiment)
