#!/usr/bin/env python3
import argparse
import pickle

from src.predictor import predictor

parser = argparse.ArgumentParser()
parser.add_argument('--load_pickles', action='store_true', help='Specify this option to load pickle files.')
parser.add_argument('--xtal', type=str, default=54000, help='Input XTAL Id; default = 54000')
parser.add_argument('--year', type=str, default=2018,
        help='List of the years used for making predictions; default = 2018')
parser.add_argument('--period', type=int, default=35,
        help='Period of the LSTMs used for making predictions; default=35')
parser.add_argument('-i', '--input_folder', type=str, default='../data/processed',
        help='Path to the input folder; default=\'../data/processed\'')
parser.add_argument('-o', '--output_folder', type=str, default='../results/test',
        help='Path to output for storing the plots; default=../results/test')
parser.add_argument('-s', '--data_scaler', type=str, default='../src/preprocessing/scaler_2017_all_xtals.pickle',
        help='Pickle file containing scaler object.')
parser.add_argument('-t', '--prediction_type', type=str, default='iterative')
parser.add_argument('-m', '--model', type=str, default='training_folder/')
parser.add_argument('-se', '--start_entry', type=int, default=0)

args = parser.parse_args()

XTAL = args.xtal
YEAR = args.year
SCALER = args.data_scaler
PERIOD = args.period

prd = predictor(XTAL, YEAR, PERIOD)
prd.load_model(args.model)
print(SCALER)
if (args.load_pickles):
    prd.load_scaler(SCALER)
    prd.load_from_pickle(args.input_folder)
else:
    prd.load_dataset(args.data_scaler, args.input_folder, args.start_entry)

mape, target, predictions = prd.plot_predictions_vs_target(args.prediction_type, args.output_folder)
with open(args.output_folder+'/predictions.pickle', 'wb') as file0:
    pickle.dump(predictions, file0)
with open(args.output_folder+'/target.pickle', 'wb') as file0:
    pickle.dump(target, file0)

print("MAPE: ", mape)
