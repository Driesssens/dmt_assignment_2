from abstract_experiment import AbstractExperiment
from abstract_ized_experiment import AbstractIzedExperiment
from configuration import make_configuration
from sklearn import preprocessing
import datetime
import pandas as pd

class HustinxExperimentTop(AbstractIzedExperiment):
    split_identifier = "spl_20180518114037"

    experiment_name = "HustinxExperimentTop"
    experiment_description = """Experiment that uses the hour of the day that a search is done (note, this doesnt include time zone). Ignores 'date_time', includes all other raw features without doing any preprocessing. Uses 0.000 for missing values."""

    ignored_features = ['date_time', 'gross_booking_usd', 'booking_bool', 'click_bool', 'position',
        'orig_destination_distance', 'srch_saterday_night_bool', 'srch_children_count', 'srch_adults_count', 'visitor_location_country_id',
        "srch_booking_window"]
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
        
        num_features_prop = ['prop_starrating', 'prop_review_score', 'prop_location_score1',
                'prop_location_score2', 'prop_log_historical_price', 'price_usd',
                'comp1_rate', 'comp1_inv', 'comp1_rate_percent_diff', 
                'comp2_rate', 'comp2_inv', 'comp2_rate_percent_diff',
                'comp3_rate', 'comp3_inv', 'comp3_rate_percent_diff',
                'comp4_rate', 'comp4_inv', 'comp4_rate_percent_diff',
                'comp5_rate', 'comp5_inv', 'comp5_rate_percent_diff',
                'comp6_rate', 'comp6_inv', 'comp6_rate_percent_diff',
                'comp7_rate', 'comp7_inv', 'comp7_rate_percent_diff',
                'comp8_rate', 'comp8_inv', 'comp8_rate_percent_diff']
                
        ## mean/median/std features per prop_id
        #for feature in num_features_prop:
        #    df[feature+'_mean'] = df.groupby('prop_id')[feature].transform('mean')
        #    df[feature+'_median'] = df.groupby('prop_id')[feature].transform('median')
        #    df[feature+'_std'] = df.groupby('prop_id')[feature].transform('std')     
        
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
        
        # df_dates = df['date_time']
        # df_dates = df_dates.apply(lambda date: datetime.datetime.strptime(date, "%Y-%m-%d %H:%M:%S"))
        # df_book = df['srch_booking_window'].apply(lambda dt: datetime.timedelta(days=dt))
        # df['booking_month'] = (df_dates + df_book).apply(lambda dt: dt.month)
        
        ## Round prop location scores
        #df['prop_location_score1'] = (df['prop_location_score1']*10).round()/10
        #df['prop_location_score2'] = (df['prop_location_score2']*10).round()/10
        
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
# HustinxExperimentTop().run_medium_experiment(reset_data=True, missing_values_old_style=True)

#short_experiment = make_configuration(epochs=10)
#HustinxExperimentE().run_development_experiment(configuration=short_experiment)
