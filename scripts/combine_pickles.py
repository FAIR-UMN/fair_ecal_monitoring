#!/usr/bin/env python3

import argparse
import pickle
import numpy as np

def combine_pickles(pickle_list, filepath):
    combined_array = None
    for ip, pkl in enumerate(pickle_list):
        with open(pkl, 'rb') as file_:
            if ip==0: combined_array = pickle.load(file_)
            else: combined_array = np.concatenate((combined_array, pickle.load(file_)), axis=0)

    with open(filepath, 'wb') as file_:
        pickle.dump(combined_array, file_)

def combine_years(xtal, year_list):
    pickles = ['timestamps', 'train_x', 'train_y', 'tr_input']
    for pkl in pickles:
        lst = []
        for year in year_list:
            filename = 'data/processed/xtal_{}_{}_{}.pickle'.format(xtal, year, pkl)
            lst.append(filename)
        yrstr = '_'.join(str(y_) for y_ in year_list)
        outputfile = 'data/processed/xtal_{}_{}_{}.pickle'.format(xtal, yrstr, pkl)
        combine_pickles(lst, outputfile)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-x', '--xtal', type=int, default=54000)
    parser.add_argument('-y', '--years', nargs='+', type=int, default=2017)
    args = parser.parse_args()
    combine_years(args.xtal, args.years)

if __name__=="__main__":
    main()
