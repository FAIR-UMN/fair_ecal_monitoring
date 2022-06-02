#!/usr/bin/env python3
import argparse
import pandas as pd
import numpy as np

import pickle

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from src.data_processor import lstm_data_processor

def prepare_scaler_from_csv(xtals, years, path_to_input, path_to_output):
    
    #---------------------------------------------------------------------------------------------------
    # Import the DataFrames contaning 2017 calibrations
    #---------------------------------------------------------------------------------------------------

    # add years and crystals
    xtal_array = None
    xtal_str = ''
    year_str = ''

    for iyr, year_ in enumerate(years):
        year_str = year_str + str(year_)
        if iyr!=len(years)-1: year_str+='_'
        for jxtal, xtal_ in enumerate(xtals):

            input_file = path_to_input+'/df_skimmed_xtal_{}_{}.csv'.format(xtal_, year_)

            if iyr==0 and jxtal==0:
                xtal_str = str(xtal_) + '-'
                xtal_array = pd.read_csv(input_file)
                # remove the entries with ~ 0 luminosities
                '''
                if year_==2016:
                    xtal_array = xtal_array[pd.to_datetime(xtal_array['time'])>pd.to_datetime('2016-04-22 20:26:33')] 
                elif year_==2017:
                    xtal_array = xtal_array[pd.to_datetime(xtal_array['time'])>pd.to_datetime('2017-05-23 12:10:45')] 
                elif year_==2018:
                    xtal_array = xtal_array[pd.to_datetime(xtal_array['time'])>pd.to_datetime('2018-04-12 23:50:04')]
                '''
            elif iyr==0 and jxtal!=0:
                xtal_str = xtal_str + str(xtal_) + '-'
            else:
                tmp_array = pd.read_csv(input_file)
                # remove the entries with ~ 0 luminosities
                '''
                if year_==2016:
                    tmp_array = tmp_array[pd.to_datetime(tmp_array['time'])>pd.to_datetime('2016-04-22 20:26:33')] 
                elif year_==2017:
                    tmp_array = tmp_array[pd.to_datetime(tmp_array['time'])>pd.to_datetime('2017-05-23 12:10:45')] 
                elif year_==2018:
                    tmp_array = tmp_array[pd.to_datetime(tmp_array['time'])>pd.to_datetime('2018-04-12 23:50:04')]
                '''
                xtal_array = pd.concat([xtal_array, tmp_array], axis=0)

    if xtal_str[-1]=='-': xtal_str = xtal_str[0:-1]
    if year_str[-1]=='-': year_str = year_str[0:-1]

    xtal_array = xtal_array[xtal_array['calibration']>0.5]

    # get the time difference between two measurements
    xtal_array['deltat'] = [x for x in (np.diff(pd.to_datetime(xtal_array['laser_datetime']))/np.timedelta64(1,'m'))]+[0]
    xtal_array['deltat'] = xtal_array['deltat'].astype(float)
    

    # get the difference between the absolute integrated luminosity
    # irradiated luminosity between two time intervals
    xtal_array['delta_lumi'] = [x for x in xtal_array['int_deliv_inv_ub'].diff()[1:]/1e6] + [0.0]
    
    scaler = StandardScaler()
    scaler.fit_transform(xtal_array[['calibration', 'delta_lumi', 'deltat']][((xtal_array['delta_lumi']>0) & (xtal_array['deltat']<60))])
    
    with open(path_to_output+'scaler_xtal_{}_{}.pickle'.format(xtal_str, year_str), 'wb') as file_:
        pickle.dump(scaler, file_)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-x', '--xtals', type=int, nargs="+", default=[54000], help='Input XTAL Id; default = 54000')
    parser.add_argument('-y', '--years', type=int, nargs="+", default=[2018],
    help='List of the years used for making predictions; default = 2018')
    parser.add_argument('-i', '--input_folder', type=str, default='../data/interim',
    help='Path to the input folder; default=\'data/interim\'')
    parser.add_argument('-o', '--output_folder', type=str, default='../src/preprocessing/',
    help='Path to output for storing the plots; default=data/processed')
    args = parser.parse_args()

    prepare_scaler_from_csv(args.xtals, args.years, args.input_folder, args.output_folder)

if __name__=='__main__':
    main()
