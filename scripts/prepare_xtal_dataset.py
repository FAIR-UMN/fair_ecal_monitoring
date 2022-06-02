#!/usr/bin/env python3
import pandas as pd
import numpy as np

path_to_dataframe = '/home/rusack/joshib/ECAL_calib/dataframes/'

def skim_year(filename, year):

    xtal_df = pd.read_csv(filename)
    column_list = xtal_df.keys()

    id_label = ''

    # add xtal_id column to the dataframe
    for col_ in column_list:
        if 'Unnamed' in col_ and xtal_df[col_].iloc()[0]!=0: id_label = col_

    
    xtal_df['xtal_id'] = xtal_df[[id_label]]
    xtal_list = xtal_df.xtal_id.unique()

    skimmed_columns = ['xtal_id', 'iov_idx', 'fill', 'temperature', 't1', 'seq_datetime',
                       'inst_lumi', 'start_ts', 'stop_ts', 'laser_datetime', 'good',
                       'calibration', 'int_inst_lumi', 'p1', 'p2', 'p3', 'time',
                       'ls', 'beamstatus', 'int_deliv_inv_ub']
    xtal_df = xtal_df[skimmed_columns]
    xtal_df = xtal_df[~np.isnan(xtal_df['calibration'])] # remove nan values
    xtal_df = xtal_df[xtal_df['calibration']>0.5] # remove bad measurements

    xtal_df = xtal_df[( (xtal_df['good']==1) & (xtal_df['beamstatus']!='ADJUST') & (xtal_df['beamstatus']!='SQUEEZE'))]
    xtal_df['inst_lumi'] = xtal_df['inst_lumi'].apply(lambda x: 0 if np.isnan(x) else x) # remove nan values of inst. lumi

    print('Iterating over crystals ...')

    for xtal_ in xtal_list:
        print('\txtal: {}'.format(xtal_))
        outputfile = path_to_dataframe+'df_skimmed_xtal_{}_{}.csv'.format(xtal_, year)
        make_skimmed_dataframe(xtal_df[xtal_df['xtal_id']==xtal_], outputfile)


def make_skimmed_dataframe(xtal_df, filename):
    
    # get the time difference between two measurements
    xtal_df['deltat'] = [x for x in np.diff(pd.to_datetime(xtal_df['laser_datetime']))]+[0]
    xtal_df['deltat'] = (xtal_df['deltat']*1e-9).astype(float)
    
    # get the difference between the absolute integrated luminosity
    # irradiated luminosity between two tmie intervals
    xtal_df['delta_lumi'] = [x for x in xtal_df['int_deliv_inv_ub'].diff()[1:]] + [0.0]

    final_column_list = ['xtal_id', 'time', 'laser_datetime', 'int_deliv_inv_ub',
                         'deltat', 'delta_lumi', 'calibration']
    xtal_df = xtal_df[final_column_list]
    xtal_df.to_csv(filename)


def combine_xtal_dataframes(lst):
   # function takes a list of dataframes to be
   # merged and combines the csv files

   combiend_df = pd.DataFrame()

   for file_ in lst:
       tmp_frame = pd.read_csv(file_)
       combined_df += tmp_frame
   
   return combined_df


def skim_all(): 
    for year_ in [2016, 2017, 2018]:
        for ring_ in [1, 66, 85]:
            path_to_csv = path_to_dataframe
            path_to_csv += 'df_xtals_ieta_{}_{}.csv'.format(ring_, year_)
            print('skimming {} ...'.format(path_to_csv))
            skim_year(path_to_csv, year_)

def combine_year():
    for year_ in [2016, 2017, 2018]:
        skimmed_files = [ f for f in os.listdir('{}/') if ]

def main():
    t0 = time()
    '''
    function
    '''
    print('Task executed in {}'.format(time()-t0))

if __name__=='__main__':
    main()
