#/usr/bin/env bash

dirlist=$(ls -1d ../../training_folder/lstm_01_03_2022*/*/)

for dir in $dirlist;
   do
      #./autopass.sh Dunkindonu+s99 "scp $"
      str1=$(echo $dir | cut -d'/' -f 4)
      str2=$(echo $dir | cut -d'/' -f 5)
      tag=$str1\_$str2
      ./autopass.sh Dunkindonu+s99 "scp $dir/2017/predictions_vs_target.png bjoshi@lxplus.cern.ch:/eos/user/b/bjoshi/www/ECAL_xtal_calibration/RANDOMISATION_TEST_21_02_22/$tag\_predictions_vs_target_2017.png"
      ./autopass.sh Dunkindonu+s99 "scp $dir/2018/predictions_vs_target.png bjoshi@lxplus.cern.ch:/eos/user/b/bjoshi/www/ECAL_xtal_calibration/RANDOMISATION_TEST_21_02_22/$tag\_predictions_vs_target_2018.png"
   done
