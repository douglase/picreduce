data_directories=["../../../data/picture_data_wsmr_heliostat_36225/rigel/","../../../data/picture_data_wsmr_heliostat_36225/eeri/"]
for data_directory in data_directories:
    f = h5py.File(data_directory+'data.hdf5','w')
    PICTURE_IDL_to_HDF5.run(f,data_directory,"bugse")
    PICTURE_IDL_to_HDF5.run(f,data_directory,"jplgse")
    PICTURE_IDL_to_HDF5.run(f,data_directory,"atfull")
    f.close()
