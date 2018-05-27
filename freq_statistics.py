# -*- coding: utf-8 -*-
"""
Created on Tue May 23 18:16:11 2018

@author: alexander
"""

import numpy as np
import pandas
import datetime

from matplotlib import pyplot as plt
from matplotlib import cm as cm

# Value assigned to missing values
missing_val = -11111.11

## Functions for statistical analysis

# Function that returns statistics of the dataset, more specifically, the relation to booking_bool and click_bool
#group_by: string that equals to one of the columns in the dataset
#filter_missing_values: bool that when true removes all the missing value rows of group_by
#normalize: bool that when true returns the answer normalized over the booking_bool and click_bool datasets respectively
#percentage: bool that when true returns answer as percentages (basically *100)
#deep_copy: bool that when true does not make any changes to the original datasets
#round_decimals: int that when bigger than -1 rounds the group_by values to the given amount of decimals
#dist_range: int that when bigger than -1 groups the group_by values to the given ranges (e.g. 50 will round everything to 0-50, 51-100, ... etc)
# returns statistics: [stats_booking, stats_click]
def get_freq_statistics(group_by, filter_missing_values=False, normalize=False, percentage=False, deep_copy=False, round_decimals=-1, dist_range=-1):
    if deep_copy:
        d = dataset.copy()
        db = dataset_book.copy()
        dc = dataset_click.copy()
    else:
        d = dataset
        db = dataset_book
        dc = dataset_click
    
    if filter_missing_values:
        d = d[d[group_by] > missing_val]
        db = db[db[group_by] > missing_val]
        dc = dc[dc[group_by] > missing_val]
    
    if round_decimals > -1:
        d[group_by] = d[group_by].round(decimals=round_decimals)
        db[group_by] = db[group_by].round(decimals=round_decimals)
        dc[group_by] = dc[group_by].round(decimals=round_decimals)
    
    # to group indices will be rounded to fit into a given range
    # e.g. dist_range = 500 means all values between 0-500 will be grouped together
    if dist_range > -1:
        d[group_by] = (d[group_by]//dist_range) *dist_range
        db[group_by] = (db[group_by]//dist_range) *dist_range
        dc[group_by] = (dc[group_by]//dist_range) *dist_range 
    
    d = d.shape[0]
    db = db.groupby(by=group_by).size()
    dc = dc.groupby(by=group_by).size()
    
    statistics = [db/d, dc/d]
    
    if normalize:
        statistics = [statistics[0]/statistics[0].sum(), 
                      statistics[1]/statistics[1].sum()]
        
    if percentage:
        statistics[0] = statistics[0] * 100
        statistics[1] = statistics[1] * 100
    
    return statistics
    
def fill_missing_values(df, fill_with=missing_val):
    replace_df = df.fillna(fill_with)
    return replace_df
    
## Dataset objects

#Load whole dataset
dataset = pandas.read_csv("data\\training_set_VU_DM_2014.csv")
dataset = fill_missing_values(dataset, missing_val)
dataset_size = dataset.shape[0]

#Create dataset containing only booked rows
dataset_book = dataset[dataset['booking_bool'] == True]
dataset_book_size = dataset_book.shape[0]

#Create dataset containing only clicked rows
dataset_click = dataset[dataset['click_bool'] == True]
dataset_click_size = dataset_click.shape[0]

## Missing data statistics








