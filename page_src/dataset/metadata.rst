*******************
Metadata
*******************

Check this `GitHub Repository <https://github.com/FAIR-UMN/fair_ecal_monitoring/blob/master/metadata.json>`_ for our Metadata.

::
    
    {
        "xtal_id": "Crystal Identification number within ECAL ranging from [0, 75848]",

        "start_ts": "Start of interval of validity (IOV)",

        "stop_ts": "End of IOV",

        "laser_datetime": "Timestamp of the measurement for a given crystal within an IOV",

        "calibration": "APD/PD ratio taken at laser_datetime",

        "time": "Time corresponding to the luminosity measurement (obtained from BRIL) closest to the laser_datetime",

        "int_deliv_inv_ub": "Approximate integrated luminosity delivered up to the measurement in the units of micro barn inverse"
    }