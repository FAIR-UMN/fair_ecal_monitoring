#!/usr/bin/env python3
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', type=str, help='Path to the log containing history of the training.',
        default='../training_folder/lstm_21_02_2022__13_56/train_lstm_20_period_35_calib_inst_lumi_deltat_epochs_3000_batch_500_SEED_17.log')
args = parser.parse_args()

with open(args.input) as file_:
    lines = file_.readlines()

epochs = []
training_loss = []
validation_loss = []

for line in lines:
    line = line.strip('\n\r')
    if 'Epoch' in line:
        line = line.split(' ')[1]
        epochs.append(int(line.split('/')[0]))

    if 'loss:' in line:
        line = line.split(' ')
        training_loss.append(float(line[5]))
        validation_loss.append(float(line[8]))

print(epochs)
print(training_loss)
print(validation_loss)
