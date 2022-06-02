#!/usr/bin/env python3

import os
import argparse

import numpy as np
import pandas as pd
import math

import pickle
import joblib

# import scikit-related packages
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

import argparse




class scaler:
    
    def __init__(self, xtal_id_, year_, period_):

        self.xtal_id = xtal_id_ 
        self.period = period_
        self.year = year_
        self.scaler = None
        
    def set_period(self, period_):
        self.period = period_
    
    def load_scaler(self, scaler_filename):  
        '''
        load the scaler from a file
        '''
        self.scaler = joblib.load(scaler_filename) 
        
        #combined_array = pd.read_csv(filename, header=None, delimiter=' ')
        #self.scaler.fit_transform(combined_array)       

    def get_data_matrix(self):
        '''
        Function to prepare data with a given period
        '''
        tmp = self.df.copy()
        ntotal = len(self.df.index)

        # create target values by shifting the calibration values with PERIOD
        tmp['target'] = self.df['calibration'].shift(periods=-self.period, fill_value=0.0)

        # For a given period, make an array of luminosities
        for i in range(1, self.period):
            tmp['delta_lumi+'+str(i)] = self.df['delta_lumi'].shift(periods=-i, fill_value=0.0).to_numpy()
            tmp['deltat+'+str(i)] = self.df['deltat'].shift(periods=-i, fill_value=0.0).to_numpy()
            tmp['calibration+'+str(i)] = self.df['calibration'].shift(periods=-i, fill_value=0.0).to_numpy()

        tmp_lumi = tmp.filter(regex=("delta_lumi")).to_numpy()
        tmp_dt = tmp.filter(regex=("deltat")).to_numpy()
        tmp_calibration = tmp.filter(regex=("calibration")).to_numpy()

        input_np = np.concatenate((tmp_calibration.reshape(ntotal, self.period, 1),
                                   tmp_lumi.reshape(ntotal, self.period, 1),
                                   tmp_dt.reshape(ntotal, self.period, 1)), axis=2)

        #convert to np array (train)
        self.train_y = tmp['target'].to_numpy()[0:-self.period]

        # the input is given as the number 2D array (luminosity, time gaps)
        self.train_x = input_np[0:-self.period]
    
    def prepare_dataset_from_csv(self, path_to_input):
        
        #---------------------------------------------------------------------------------------------------
        # Import the DataFrames contaning 2017 calibrations
        #---------------------------------------------------------------------------------------------------

        input_file = path_to_input+'/df_skimmed_xtal_{}_{}.csv'.format(self.xtal_id, self.year)

        xtal_array = pd.read_csv(input_file)
        xtal_array = xtal_array[xtal_array['calibration']>0.5]

        # get the time difference between two measurements
        xtal_array['deltat'] = [x for x in np.diff(pd.to_datetime(xtal_array['time']))]+[0]
        xtal_array['deltat'] = (xtal_array['deltat']*1e-9).astype(float)
        
        # remove the entries with ~ 0 luminosities
        if self.year==2016:
            xtal_array = xtal_array[pd.to_datetime(xtal_array['time'])>pd.to_datetime('2016-04-22 20:26:33')] 
        elif self.year==2017:
            xtal_array = xtal_array[pd.to_datetime(xtal_array['time'])>pd.to_datetime('2017-05-23 12:10:45')] 
        elif self.year==2018:
            xtal_array = xtal_array[pd.to_datetime(xtal_array['time'])>pd.to_datetime('2018-04-12 23:50:04')]

        # get the difference between the absolute integrated luminosity
        # irradiated luminosity between two time intervals
        xtal_array['delta_lumi'] = [x for x in xtal_array['int_deliv_inv_ub'].diff()[1:]] + [0.0]

        self.timestamps = pd.to_datetime(xtal_array['time']).to_numpy()

        # scale the dataset features
        self.tr_input = self.scaler.transform(xtal_array[['calibration', 'delta_lumi', 'deltat']])
        self.df = pd.DataFrame(self.tr_input, columns=['calibration', 'delta_lumi', 'deltat']) 
        self.get_data_matrix()

