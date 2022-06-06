#!/usr/bin/env python3

import os
#os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices' # enable gpu utilization

import pandas as pd
import numpy as np

from src.python.tf.data_processor import lstm_data_processor

# import tf packages
from tensorflow.keras.models import load_model

# import scikit-related packages
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

import pickle

# import matplotlib
from src.utils.pltutils import plt 

class lstm_predictor():
     
   def __init__(self, xtal_id_, year_, period_):
        
        self.model = None
        self.xtal_id = xtal_id_
        self.year = year_
        self.period = period_

        self.scaler = None 
        self.train_x = None
        self.train_y = None
        self.timestamps = None
        self.tr_input = None
   
   def load_scaler(self, scaler_filename):  
        '''
        load the scaler from a file
        '''
        #self.scaler = joblib.load(scaler_filename) 
        with open(scaler_filename, 'rb') as file_:
            self.scaler = pickle.load(file_)
   
   def load_dataset(self, scaler_filename='None', path_to_dataset='../data/interim', start_entry=0):
       
       prsr = lstm_data_processor(self.xtal_id, self.year, self.period)
       prsr.prepare_dataset_from_csv(path_to_dataset, scaler_filename)

       self.scaler = prsr.scaler

       self.train_x = prsr.train_x[start_entry:]
       self.train_y = prsr.train_y[start_entry:]
       self.tr_input = prsr.tr_input[start_entry:]
       self.timestamps = prsr.timestamps[start_entry:]
   
   def load_from_pickle(self, path_to_dataset='../data/processed'):
       
       pickle_list = [
                'train_x',
                'train_y',
                'tr_input',
                'timestamps'
                ]
       print(self.year) 
       for pickle_ in pickle_list:
           filename = ''
           with open('{}/xtal_{}_{}_{}.pickle'.format(path_to_dataset, self.xtal_id, self.year, pickle_), 'rb') as file_:
              setattr(self, pickle_, pickle.load(file_))

    
   def load_model(self,
                   path_to_model='training_folder/lstm_ring_01_12_2021__03_45/lstm_10_inst_lumi_deltat_epochs_15_batch_1000'):
        self.model = load_model(path_to_model)

   def get_iterative_predictions(self):

        # iterative prediction
        prediction_sequence = list(self.train_x[0,:,0].flatten())
        
        for i in range(len(self.train_x)):
            next_train = np.array([ x for x in self.train_x[i] ])
            next_train[:,0] = np.array([ x for x in prediction_sequence[i:] ])
            if i%100==1:
                print('Generating {}/{} prediction ...'.format(i, len(self.train_x)))
            output_ = self.model.predict(next_train.reshape(1, self.period, 3))
            prediction_sequence.append(output_[:,0][0])
        
        return np.array(prediction_sequence)
   
   def get_inst_predictions(self):

       # instantaneous predictions
       arr = self.model.predict(self.train_x)[:,0]
       arr = np.concatenate((self.train_x[0,:,0], arr))
       return arr
    
   def plot_predictions_vs_target(self, pred_type='iterative', path_to_output=''):
        
        prediction_sequence = None
        if pred_type=='iterative': prediction_sequence = self.get_iterative_predictions()
        elif pred_type=='instantaneous': prediction_sequence = self.get_inst_predictions()
        else:
            print('Illegal type of predictions')
            return

        plt.clf()
        plt.figure(figsize=(13.2, 6.6))
        
        target_ = np.concatenate((self.train_x[0,:,0].flatten(), self.train_y))
        
        # inverse transform the predictions
        pred_size = len(prediction_sequence)

        
        #------------------------------------------------------------------------------------------------
        iter_prediction_df = np.concatenate((prediction_sequence.reshape(pred_size, 1),
                                            self.tr_input[0:pred_size,1:].reshape(pred_size, 2)), axis=1)
        iter_original_df = np.concatenate((target_.reshape(pred_size, 1),
                                            self.tr_input[0:pred_size,1:].reshape(pred_size, 2)), axis=1)
        iter_prediction_df = self.scaler.inverse_transform(iter_prediction_df)
        iter_original_df = self.scaler.inverse_transform(iter_original_df)
        #------------------------------------------------------------------------------------------------
        
        # calculate and return MAPE
        diff = np.abs(iter_prediction_df[:, 0]-iter_original_df[:,0])
        diff = diff/np.abs(iter_original_df[:,0])
        self.mape = sum(diff)*100/pred_size
        
        prediction_plot = plt.plot(self.timestamps[:pred_size], iter_prediction_df[:,0], color='r', label='prediction')
        target_plot = plt.plot(self.timestamps[:pred_size], iter_original_df[:,0], color='b', label='target')
        plt.ylabel('Laser Response', fontsize=14)
        plt.xlabel('Time', fontsize=14)
        ymax = max(iter_prediction_df[:,0])*1.05
        ymin = min(iter_prediction_df[:,0])
        plt.xlim(pd.to_datetime(self.timestamps[0]), pd.to_datetime(self.timestamps[-1]))
        plt.ylim(0.7, ymax)
        plt.legend(loc='upper right', fontsize=14)
        plt.savefig(path_to_output+'/predictions_vs_target.png')
        
        return self.mape, target_plot, prediction_plot

