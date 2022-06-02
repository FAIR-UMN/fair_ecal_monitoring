#!/usr/bin/env python3
import argparse

from src.data_processor import seq2seq_data_processor

parser = argparse.ArgumentParser()
parser.add_argument('--xtal', type=int, default=54000, help='Input XTAL Id; default = 54000')
parser.add_argument('--year', type=int, default=2018,
        help='List of the years used for making predictions; default = 2018')
parser.add_argument('--encoder_period', type=int, default=15,
        help='Period of the Encoder LSTMs; default=15')
parser.add_argument('--decoder_period', type=int, default=5,
        help='Period of the Decoder LSTMs; default=5')
parser.add_argument('-i', '--input_folder', type=str, default='../data/interim',
        help='Path to the input folder; default=\'data/interim\'')
parser.add_argument('-o', '--output_folder', type=str, default='../data/processed',
        help='Path to output for storing the plots; default=data/processed')
parser.add_argument('-s', '--data_scaler', type=str, default='../src/preprocessing/scaler_2017_all_xtals.pickle',
        help='Pickle file containing scaler object.')

args = parser.parse_args()

XTAL = args.xtal
YEAR = args.year
EN_PERIOD = args.encoder_period
DE_PERIOD = args.decoder_period
SCALER = args.data_scaler

prsr = seq2seq_data_processor(XTAL, YEAR, EN_PERIOD, DE_PERIOD)
prsr.load_scaler(SCALER)
prsr.prepare_dataset_from_csv(args.input_folder)
prsr.save_to_pickle(args.output_folder)
