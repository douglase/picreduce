#gen hdf5 datsets
data_directory = "~/projects/PICTURE/data/photo_diode_calibration_20140528_test2/"
f = h5py.File(data_directory+'data.hdf5','w')
datasets=["gsedata.idl.05282014.62641",
"gsedata.idl.05282014.62852",
"gsedata.idl.05282014.63068",
"gsedata.idl.05282014.65331",
"gsedata.idl.05282014.65616",
"gsedata.idl.05282014.65988",
"gsedata.idl.05282014.74945",
"gsedata.idl.05282014.75276"]
for dset in datasets:
    PICTURE_IDL_to_HDF5.run(f,data_directory,dset)
f.close()


data_directory = "/Users/edouglas/projects/PICTURE/data/flight_sequence_20140530/"
datasets=[["gsedata.idl.05302014.49253","After fixing timing bug in JPL flight code, weather station LED was ON, Diode: 40db, 740mV, +6mV offset"],
["gsedata.idl.05302014.49628","After fixing timing bug in JPL flight code, weather station LED was ON. Diode: 40db, 740mV, +6mV offset"],
["gsedata.idl.05302014.49935","After fixing timing bug in JPL flight code, weather station LED was ON. Diode: 40db, 740mV, +6mV offset"],
["gsedata.idl.05302014.51843","Weather station LED off, Diode: 40db, 746mV, +6mV offset"],
["gsedata.idl.05302014.52110","Weather station LED off, Diode: 40db, 746mV, +6mV offset"],
["gsedata.idl.05302014.52326","Weather station LED off, Diode: 40db, 746mV, +6mV offset"],
["gsedata.idl.05302014.57522","After moving tertiary, poked up bottom rows of DM, ithresh=200,Diode: 40db, 755mV, +6mV offset"],
["gsedata.idl.05302014.57787","After moving tertiary, poked up bottom rows of DM, ithresh=200,Diode: 40db, 755mV, +6mV offset"],
["gsedata.idl.05302014.58129","After moving tertiary, poked up bottom rows of DM, ithresh=200,Diode: 40db, 755mV, +6mV offset"],
["gsedata.idl.05302014.58384","ithresh = 700,Diode: 40db, 755mV, +6mV offset"],
["gsedata.idl.05302014.58642","ithresh = 700,Diode: 40db, 755mV, +6mV offset"],
["gsedata.idl.05302014.58866","ithresh = 700,Diode: 40db, 755mV, +6mV offset"]]
f = h5py.File(data_directory+'data.hdf5','w')
for dset in datasets:
    PICTURE_IDL_to_HDF5.run(f,data_directory,dset[0])
    if len(dset)==3:
        PICTURE_IDL_to_HDF5.run(f,data_directory,dset[1])
f.close()

