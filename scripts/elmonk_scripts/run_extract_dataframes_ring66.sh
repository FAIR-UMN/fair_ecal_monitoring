#!/bin/bash -l

#SBATCH --time=04:00:00
#SBATCH --ntasks=4
#SBATCH --mem=100g
#SBATCH -p amd2tb
#SBATCH --job-name="extract_ecal_calibration_dataframes"
#SBATCH --mail-type=ALL
#SBATCH --mail-user=joshib@umn.edu

#SBATCH --output=/home/rusack/joshib/ECAL_calib/slurm_logs/run_extract_real_2.log

export PYTHONUNBUFFERED=1

module load cmake
module load gcc
module load python3
module load cuda/10.1
module load graphviz

conda activate /home/rusack/shared/.conda/env/tf
source /home/rusack/joshib/ECAL_calib/setenv.sh

./run_it.py --year 2016 --xtal_slice ieta==66 --out_file ../dataframes/df_xtals_ieta_66_2016.csv --max_xtal 10000
./run_it.py --year 2017 --xtal_slice ieta==66 --out_file ../dataframes/df_xtals_ieta_66_2017.csv --max_xtal 10000
./run_it.py --year 2018 --xtal_slice ieta==66 --out_file ../dataframes/df_xtals_ieta_66_2018.csv --max_xtal 10000
