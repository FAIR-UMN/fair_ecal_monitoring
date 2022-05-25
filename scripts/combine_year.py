#!/usr/bin/env python3

# combine prediction from different years
import numpy as np
import pandas as pd
import pickle
from matplotlib import pyplot as plt

import argparse

# set plotting parameters
plt.rcParams['axes.linewidth'] = 1.4
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['ytick.minor.size'] = 2.5
plt.rcParams['xtick.minor.size'] = 4.0
plt.rcParams['ytick.minor.visible'] = True
plt.rcParams['ytick.labelsize'] = 'large'
plt.rcParams['xtick.labelsize'] = 'large'

parser = argparse.ArgumentParser()
parser.add_argument('-x', '--xtal', type=str, default="54000")
parser.add_argument('-r', '--ring', type=str, default="66")
parser.add_argument('-f', '--path_to_pickles', type=str, default='../training_folder/lstm_18_05_2022__18_46/lstm_20_period_35_calib_inst_lumi_deltat_xtal_54000_epochs_5_batch_1_seed_9/')
args = parser.parse_args()

RING = args.ring
XTAL=args.xtal
path_to_pickles = args.path_to_pickles 
arrays = {}
for YEAR in [2016, 2017, 2018]:
    target_file = path_to_pickles+'{}_{}_instantaneous_se0/target.pickle'.format(XTAL, YEAR)
    prd_file = path_to_pickles+'{}_{}_instantaneous_se0/predictions.pickle'.format(XTAL, YEAR)
    arrays[YEAR] = {}
    with open(target_file, 'rb') as f:
        arrays[YEAR]['target'] = pickle.load(f)
    with open(prd_file, 'rb') as f:
        arrays[YEAR]['predictions'] = pickle.load(f)

plt.clf()
plt.figure(figsize=(13.2, 3.3))
T = np.concatenate((arrays[2016]['predictions'][0]._xorig, arrays[2017]['predictions'][0]._xorig, arrays[2018]['predictions'][0]._xorig))
Y = np.concatenate((arrays[2016]['predictions'][0]._y, arrays[2017]['predictions'][0]._y, arrays[2018]['predictions'][0]._y))
plt.plot(T, Y, 'r-', label='predictions')
T = np.concatenate((arrays[2016]['target'][0]._xorig, arrays[2017]['target'][0]._xorig, arrays[2018]['target'][0]._xorig))
Y = np.concatenate((arrays[2016]['target'][0]._y, arrays[2017]['target'][0]._y, arrays[2018]['target'][0]._y))
plt.plot(T, Y, 'b-', label='target')
plt.ylabel('Laser APD/PN', fontsize=14)
plt.xlabel('Time', fontsize=14)
plt.xlim(T[0], T[-1])
ymin = 0.75
ymax = 1.0
plt.ylim(ymin-5.0e-4, ymax)
plt.legend(loc='upper right', fontsize=14)
plt.text(s='Ring: {}, xtal: {}'.format(RING, XTAL), x=T[0], y=ymax*1.005, fontsize=14)
plt.savefig(path_to_pickles+'xtal_{}_combined_runII.png'.format(XTAL), bbox_inches='tight')
