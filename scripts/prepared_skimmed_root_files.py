#!/usr/bin/env python3
import pandas as pd
import numpy as np
import array
import uproot3

path_to_dataframe = '/home/rusack/joshib/ECAL_calib/data/raw'
path_to_output = path_to_dataframe+'/skimmed_root/'

def merge_years():
    
    # prepare root file
    filename = path_to_output+'ecal_crystal_response.root'
    print('opening %s' % filename)
    
    with uproot3.recreate(filename) as rfile:

        for year in ['2016', '2017', '2018']:
            for ring in [1, 66, 85]:

                treename ='year{}_ring_{}' .format(year, ring)
                rfile[treename] = uproot3.newtree({'xtal_id':int,
                    'seq_datetime':np.int_,
                    'laser_datetime':np.int_,
                    'lumi_datetime':np.int_,
                    'start_ts':np.int_,
                    'stop_ts':np.int_,
                    'int_deliv_inv_ub':float,
                    'laser_response':float})
                
                #tree_map[treename] = TTree(treename, treename)
                #tmptree = tree_map[treename]

                print('Filling %s' % treename)

                df_filename = '{}/df_xtals_ieta_{}_{}.csv'.format(path_to_dataframe, ring, year)
        
                xtal_df = pd.read_csv(df_filename)
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

                xtal_df = xtal_df.drop(columns=['iov_idx', 'fill', 'temperature', 't1', 'inst_lumi', 'good', 'int_inst_lumi', 'p1', 'p2', 'p3', 'ls', 'beamstatus'])
                xtal_df.to_csv('{}/{}.csv'.format(path_to_output, treename))

                rfile[treename].extend({
                    'xtal_id': xtal_df['xtal_id'].to_numpy(),
                    'seq_datetime': pd.to_datetime(xtal_df['seq_datetime']).values,
                    'laser_datetime': pd.to_datetime(xtal_df['laser_datetime']).values,
                    'lumi_datetime': pd.to_datetime(xtal_df['time']).values,
                    'start_ts': xtal_df['start_ts'].to_numpy(),
                    'stop_ts': xtal_df['stop_ts'].to_numpy(),
                    'int_deliv_inv_ub': xtal_df['int_deliv_inv_ub'].to_numpy(),
                    'laser_response': xtal_df['calibration'].to_numpy()
                    })

                print('Saved tree ...')

merge_years()
