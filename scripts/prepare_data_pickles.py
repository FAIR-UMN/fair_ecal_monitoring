#!/usr/bin/env python3
import argparse

from src.data_processor import lstm_data_processor

parser = argparse.ArgumentParser()
parser.add_argument('-x', '--xtal', type=int, default=54000, help='Input XTAL Id; default = 54000')
parser.add_argument('-y', '--year', type=int, default=2018,
        help='List of the years used for making predictions; default = 2018')
parser.add_argument('--period', type=int, default=35,
        help='Period of the LSTMs used for making predictions; default=35')
parser.add_argument('-i', '--input_folder', type=str, default='./data/interim',
        help='Path to the input folder; default=\'data/interim\'')
parser.add_argument('-o', '--output_folder', type=str, default='./data/processed',
        help='Path to output for storing the plots; default=data/processed')
parser.add_argument('-s', '--data_scaler', type=str, default='./src/preprocessing/scaler_2017_all_xtals.pickle',
        help='Pickle file containing scaler object.')

args = parser.parse_args()

XTAL = args.xtal
YEAR = args.year
PERIOD = args.period
SCALER = args.data_scaler

prsr = lstm_data_processor(XTAL, YEAR, PERIOD)
prsr.prepare_dataset_from_csv(args.input_folder, SCALER)
prsr.save_to_pickle(args.output_folder)
