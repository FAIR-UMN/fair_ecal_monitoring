for xtal in {54000..54360}; do
   ./make_scaler.py --xtal $xtal --years 2016
   ./make_scaler.py --xtal $xtal --years 2017
   ./make_scaler.py --xtal $xtal --years 2018
  done
