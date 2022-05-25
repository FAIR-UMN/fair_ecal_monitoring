#!/usr/bin/env python3
import sys
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('-f', '--fcrop', type=float, default=0.5)
parser.add_argument('--xtal', type=int, default=54300, help='Input XTAL Id; default = 54000')
parser.add_argument('--year', type=int, default=2018,
        help='List of the years used for making predictions; default = 2018')
parser.add_argument('-i', '--input_folder', type=str, default='../data/processed/',
        help='Path to the input folder; default=data/processed')
parser.add_argument('-o', '--output_folder', type=str, default='../data/processed/cropped',
        help='Path to output for storing the plots; default=data/cropped')
args = parser.parse_args()

if (args.f>1 or args.f<0):
    print("The fractions must be within [0, 1]!!")
    sys.exit(2)



