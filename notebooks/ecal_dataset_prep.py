# This script is used to prepare the dataset we need
import numpy as np
import torch
import pandas as pd
from sklearn.preprocessing import MinMaxScaler,StandardScaler
import matplotlib
import matplotlib.pyplot as plt
import os

matplotlib.rcParams.update({'font.size': 17})


# This class is used to generate the ECAL-Dataset
# And its input is a csv file name
class ECAL_Dataset_Prep:
    def __init__(self, csv_file, input_len, output_len, stride, fig_name_cali, fig_name_scaled_cali, verbose=False, plt_show=True):
        self.csv_file = csv_file
        self.il = input_len
        self.ol = output_len
        self.stride = stride  # to ensure there is no overlap in prediction, please set the stride = output_window
        self.fig_name_cali = fig_name_cali
        self.fig_name_scaled_cali = fig_name_scaled_cali
        self.verbose = verbose # True: print information; False: will not print information
        self.plt_show = plt_show # True: show plot; False: will not show plot

        # the scalers
        self.scaler_cali = StandardScaler()
        self.scaler_lumi = StandardScaler()
        # the dataframe version
        self.df_cali = None
        self.df_lumi = None
        # the numpy version
        self.np_cali = None
        self.np_lumi = None
        # the tensor/torch version
        self.torch_cali = None
        self.torch_lumi = None
        # the numpy version of input and target samples
        self.np_X = None
        self.np_Y = None
        # the tensor/torch version of input and target samples
        self.torch_X = None
        self.torch_Y = None

    def start_processing(self):
        # now, we will call functions one by one to update the values
        self.get_df()
        self.normalize_dataset()
        self.sequence_dataset()
        self.visualize_data_samples()

    # get the dataframe and split the dataset into two dataframes:
    # one for calibration
    # one for luminosity diff
    def get_df(self):
        self.df = pd.read_csv(self.csv_file, index_col=0)
        self.df.index = pd.to_datetime(self.df['laser_datetime'])
        try:
            self.df_cali = self.df[['calibration']].copy()
            self.df_lumi = self.df[['delta_lumi']].copy()
        except:
            assert False, "We except the csv should at least include ['calibration', 'delta_lumi'] (even they are empty columns)!"

    # normalize the dataset and also get other numpy and torch version
    def normalize_dataset(self):
        # normalize calibration
        if len(self.df_cali) !=0:
            self.scaler_cali.fit(self.df_cali[['calibration']])
            self.df_cali['calibration_scaled'] = None
            self.df_cali.loc[:,'calibration_scaled'] = self.scaler_cali.transform(self.df_cali[['calibration']])
            if self.verbose:
                print(self.df_cali.describe())
            self.np_cali = self.df_cali['calibration_scaled'].to_numpy()
            self.np_cali = self.np_cali.reshape(-1, 1)

        # normalize luminosity diff
        if len(self.df_lumi) !=0:
            self.scaler_lumi.fit(self.df_lumi[['delta_lumi']])
            self.df_lumi['delta_lumi_scaled'] = None
            self.df_lumi.loc[:,'delta_lumi_scaled'] = self.scaler_lumi.transform(self.df_lumi[['delta_lumi']])
            if self.verbose:
                print(self.df_lumi.describe())
            self.np_lumi = self.df_lumi['delta_lumi_scaled'].to_numpy()
            self.np_lumi = self.np_lumi.reshape(-1, 1)

    def sequence_dataset(self):
        # please note that: we arrange features in this order:
        # the first feature is lumi_diff, which is the "luminosity delta"
        # the second feature is cali, which is the "calibration"
        num_lumi = self.np_lumi.shape[0]
        num_cali = self.np_cali.shape[0]

        #case: we have the same number of cali & lumi
        if num_lumi == num_cali and num_lumi!=0:
            num_samples = (num_lumi- self.il - self.ol) // self.stride + 1
            #here, we want to combine the luminosity and calibration as our input
            y1_combined = np.hstack((self.np_lumi, self.np_cali))
            num_features_combined = y1_combined.shape[1]
            X = np.zeros([self.il, num_samples, num_features_combined])
            Y = np.zeros([self.ol, num_samples, num_features_combined])

            #processing X---input samples
            for ii in np.arange(num_samples):
                start_x = self.stride * ii
                end_x = start_x + self.il
                X[:, ii, :] = y1_combined[start_x:end_x, :]
            self.np_X = X
            self.torch_X = torch.from_numpy(X).type(torch.Tensor)

            ### processing Y---target samples
            for ii in np.arange(num_samples):
                start_y = self.stride * ii + self.il
                end_y = start_y + self.ol
                Y[:, ii, :] = y1_combined[start_y:end_y, :]
            self.np_Y = Y
            self.torch_Y = torch.from_numpy(Y).type(torch.Tensor)

        elif num_lumi>0 and num_cali==0:
            num_samples = (num_lumi- self.il - self.ol) // self.stride + 1
            Y = np.zeros([self.ol, num_samples, 1])
            ### processing Y---target samples
            for ii in np.arange(num_samples):
                start_y = self.stride * ii + self.il
                end_y = start_y + self.ol
                Y[:, ii, :] = self.np_lumi[start_y:end_y, :]
            self.np_Y = Y
            self.torch_Y = torch.from_numpy(Y).type(torch.Tensor)

        else:
            assert False, 'We only support two cases: 1) Cali and Lumi. have the same length; 2) We only have Lumi.!'


    def visualize_data_samples(self):
        plt.figure(figsize=(18, 6))
        plt.plot(self.df_cali.index, self.df_cali['calibration'], color='k', linewidth=2)
        plt.xlabel('Time')
        plt.ylabel('Calibration')
        plt.title(
            'Before normalization: Mean={}; Std={}'.format(round( self.df_cali['calibration'].mean(), 3), round( self.df_cali['calibration'].std(), 3)))
        plt.savefig(self.fig_name_scaled_cali, dpi=300)
        if self.plt_show:
            plt.show()
        plt.close()

        plt.figure(figsize=(18, 6))
        plt.plot(self.df_cali.index, self.df_cali['calibration_scaled'], color='k', linewidth=2)
        plt.xlabel('Time')
        plt.ylabel('Calibration')
        plt.title('After normalization: Mean={}; Std={}'.format(round(self.df_cali['calibration_scaled'].mean(), 3),
                                           round(self.df_cali['calibration_scaled'].std(), 3)))
        plt.savefig(self.fig_name_scaled_cali, dpi=300)
        if self.plt_show:
            plt.show()
        plt.close()
