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
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

class lstm_data_processor:
    
    def __init__(self, xtal_id_, year_, period_):

        self.period = period_
        self.year = int(year_)
        self.xtal_id = xtal_id_


        self.scaler = None
        self.df = None
        
        self.train_x = None
        self.train_y = None
        self.tr_input = None
        
        self.timestamps = None
        self.mape = None
    
    def __enter__(self):
        return self 

    def __exit__(self, exc_type, exc_val, exc_tb):
        return 

    def set_period(self, period_):
        self.period = period_
      

    def get_data_matrix(self, scaler_filename=''):
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
    
    def prepare_dataset_from_csv(self, path_to_input, scaler_filename=''):
        
        #---------------------------------------------------------------------------------------------------
        # Import the DataFrames contaning 2017 calibrations
        #---------------------------------------------------------------------------------------------------

        input_file = path_to_input+'/df_skimmed_xtal_{}_{}.csv'.format(self.xtal_id, self.year)

        xtal_array = pd.read_csv(input_file)
        xtal_array = xtal_array[xtal_array['calibration']>0.5]

        # get the time difference between two measurements
        xtal_array['deltat'] = [x for x in (np.diff(pd.to_datetime(xtal_array['laser_datetime']))/np.timedelta64(1,'m'))]+[0]
        xtal_array['deltat'] = xtal_array['deltat'].astype(float)
        
        # remove the entries with ~ 0 luminosities
        '''
        if self.year==2016:
            xtal_array = xtal_array[pd.to_datetime(xtal_array['laser_datetime'])>pd.to_datetime('2016-03-22 20:14:33')] 
        elif self.year==2017:
            xtal_array = xtal_array[pd.to_datetime(xtal_array['laser_datetime'])>pd.to_datetime('2017-04-23 12:10:45')] 
        elif self.year==2018:
            xtal_array = xtal_array[pd.to_datetime(xtal_array['laser_datetime'])>pd.to_datetime('2018-04-16 18:39:04')]
        '''

        # get the difference between the absolute integrated luminosity
        # irradiated luminosity between two time intervals
        xtal_array['delta_lumi'] = [x for x in xtal_array['int_deliv_inv_ub'].diff()[1:]/1e6] + [0.0]

        self.timestamps = pd.to_datetime(xtal_array['laser_datetime']).to_numpy()

        # scale the dataset features
        
        if os.path.exists(scaler_filename):
            print('Usign scaler from : %s' %scaler_filename)
            self.scaler = joblib.load(scaler_filename) 
            self.tr_input = self.scaler.transform(xtal_array[['calibration', 'delta_lumi', 'deltat']])
        else:
            print('Using self-scaling ...')
            #self.scaler = MinMaxScaler(feature_range=(0,1.0))
            self.scaler = StandardScaler()
            self.scaler.fit_transform(xtal_array[['calibration', 'delta_lumi', 'deltat']][((xtal_array['delta_lumi']>0) & (xtal_array['deltat']<60))])
            self.tr_input = self.scaler.transform(xtal_array[['calibration', 'delta_lumi', 'deltat']])
        
        self.df = pd.DataFrame(self.tr_input, columns=['calibration', 'delta_lumi', 'deltat']) 
        self.get_data_matrix()
        
    
    def save_to_pickle(self, path_to_output='data/processed/'):
        '''
        Save arrays in pickle format
        '''

        pickle_map = {
                'train_x': self.train_x,
                'train_y': self.train_y,
                'tr_input': self.tr_input,
                'timestamps': self.timestamps
                }

        for pickle_ in pickle_map:
            filename = '{}/xtal_{}_{}_{}.pickle'.format(path_to_output, 
                                                        self.xtal_id,
                                                        self.year,
                                                        pickle_)
            with open(filename, 'wb') as file_:    
                pickle.dump(pickle_map[pickle_], file_)

class seq2seq_data_processor():
    
    def __init__(self, xtal_id_, year_,
            encoder_period_, decoder_period_):

        self.encoder_period = encoder_period_
        self.decoder_period = decoder_period_
        self.year = year_
        self.xtal_id = xtal_id_

        self.scaler = None
        self.df = None
        
        self.train_encoder_x = None
        self.train_decoder_x = None
        self.train_y = None
        self.tr_input = None
        
        self.timestamps = None
        self.mape = None

    def set_period(self, encoder_period_, decoder_period_):
        self.encoder_period = encoder_period_
        self.decoder_period = decocer_period_
    
    def load_scaler(self, scaler_filename):  
        '''
        load the scaler from a file
        '''
        self.scaler = joblib.load(scaler_filename) 
        
        #combined_array = pd.read_csv(filename, header=None, delimiter=' ')
        #self.scaler.fit_transform(combined_array)       

    def get_data_matrix(self):
        '''
        Function to prepare inputs for encoder and decoder
        with given periods for each of the networks
        '''

        # prepare input data for encoder
        tmp_encoder = self.df.copy()
        ntotal = len(self.df.index)

        max_period = max(self.encoder_period, self.decoder_period)

        # For a given period, make an array of luminosities, deltat and calibration for encoder
        for i in range(1, max_period):
            tmp_encoder['delta_lumi+'+str(i)] = self.df['delta_lumi'].shift(periods=-i, fill_value=0.0).to_numpy()
            tmp_encoder['deltat+'+str(i)] = self.df['deltat'].shift(periods=-i, fill_value=0.0).to_numpy()
            tmp_encoder['calibration+'+str(i)] = self.df['calibration'].shift(periods=-i, fill_value=0.0).to_numpy()

        tmp_encoder_lumi = tmp_encoder.filter(regex=("delta_lumi")).to_numpy()
        tmp_encoder_dt = tmp_encoder.filter(regex=("deltat")).to_numpy()
        tmp_encoder_calibration = tmp_encoder.filter(regex=("calibration")).to_numpy()

        input_encoder_np = np.concatenate((tmp_encoder_calibration.reshape(ntotal, self.encoder_period, 1),
                                   tmp_encoder_lumi.reshape(ntotal, self.encoder_period, 1),
                                   tmp_encoder_dt.reshape(ntotal, self.encoder_period, 1)), axis=2)

        # create target values by shifting the calibration values with PERIOD
        tmp_encoder['target'] = self.df['calibration'].shift(periods=-(self.encoder_period), fill_value=0.0)
        for i in range(1, self.decoder_period):
            tmp_encoder['target+'+str(i)] = self.df['calibration'].shift(periods=-(self.encoder_period+i), fill_value=0.0)

        #convert to np array (train)
        self.train_y = tmp_encoder.filter(regex=("target")).to_numpy()[0:-(self.decoder_period+self.encoder_period)]

        # prepare encoder and decoder inputs
        self.train_encoder_x = input_encoder_np[0:-(self.encoder_period+self.decoder_period),0:self.encoder_period,:]
        self.train_decoder_x = input_encoder_np[self.encoder_period: -self.decoder_period,0:self.decoder_period,1:]
    
    def prepare_dataset_from_csv(self, path_to_input):
        
        #---------------------------------------------------------------------------------------------------
        # Import the DataFrames contaning 2017 calibrations
        #---------------------------------------------------------------------------------------------------

        input_file = path_to_input+'/df_skimmed_xtal_{}_{}.csv'.format(self.xtal_id, self.year)

        xtal_array = pd.read_csv(input_file)
        xtal_array = xtal_array[xtal_array['calibration']>0.5]
        
        # remove the entries with ~ 0 luminosities
        if self.year==2016:
            xtal_array = xtal_array[pd.to_datetime(xtal_array['laser_datetime'])>pd.to_datetime('2016-04-23 20:26:33')] 
        elif self.year==2017:
            xtal_array = xtal_array[pd.to_datetime(xtal_array['laser_datetime'])>pd.to_datetime('2017-05-23 12:10:45')] 
        elif self.year==2018:
            xtal_array = xtal_array[pd.to_datetime(xtal_array['laser_datetime'])>pd.to_datetime('2018-04-14 23:50:04')]

        # get the time difference between two measurements
        xtal_array['deltat'] = [x for x in (np.diff(pd.to_datetime(xtal_array['laser_datetime']))/np.timedelta64(1,'m'))]+[0]
        xtal_array['deltat'] = xtal_array['deltat'].astype(float)
        
        # get the difference between the absolute integrated luminosity
        # irradiated luminosity between two time intervals
        xtal_array['delta_lumi'] = [x for x in xtal_array['int_deliv_inv_ub'].diff()[1:]/1e6] + [0.0]

        self.timestamps = pd.to_datetime(xtal_array['laser_datetime']).to_numpy()

        # scale the dataset features
        self.tr_input = self.scaler.transform(xtal_array[['calibration', 'delta_lumi', 'deltat']])
        self.df = pd.DataFrame(self.tr_input, columns=['calibration', 'delta_lumi', 'deltat']) 
        self.get_data_matrix()
        
    
    def save_to_pickle(self, path_to_output='data/processed/'):
        '''
        Save arrays in pickle format
        '''

        pickle_map = {
                'train_encoder_x': self.train_encoder_x,
                'train_decoder_x': self.train_decoder_x,
                'train_y': self.train_y,
                'tr_input': self.tr_input,
                'timestamps': self.timestamps
                }

        for pickle_ in pickle_map:
            filename = '{}/xtal_{}_{}_{}.pickle'.format(path_to_output, 
                                                        self.xtal_id,
                                                        self.year,
                                                        pickle_)
            with open(filename, 'wb') as file_:    
                pickle.dump(pickle_map[pickle_], file_)
