'''
Utilities for handling the conversion between IDL save files and HDF5
'''

import glob
import scipy.io
import numpy as np
import h5py
import matplotlib.mlab
import os.path
import astropy.io.fits as fits

def get_dsets(sequence_dir):
    return [path.split('/')[-2] for path in glob.glob(sequence_dir+"/*/")]

def load_or_create(data_directory,
                   dset_list=[],
                   fname='/data.hdf5',
                   readwrite='r'):
    r"""Function for loading a HDF5 dataset generated from PICTURE IDL save files.
    Checks if `fname exists`, and if not, creates a file, `fname`,
    in `data_directory` with datasets from `dset_list`
    (corresponding to subdirectories inside data_directory)
    
    .. warning:: if dset_list is not set, the new file will not be created.

    Parameters
    ----------
    data_directory : string
    
    dset_list : list
       list of datasets (e.g. jplgse.20071101.11111)
    fname : string, optional
        filename of the output HDF5 file
         defaults to creating `data.hdf5`  in `data_directory` when ommited.
    Returns
    ----------
    f:
        An h5py file object.
    
    Raises
    ----------
    
    ValueError
    forgot to pass a list?
    
    See Also
    ----------
    jplgse_to_HDF5 : the heavy lifting parsing function.
    
    Notes
    ----------

    References
    ----------
    5py documentation:
    
    Examples
    ----------
    
    >>> data_directory=expanduser('~')+'/projects/picture/data/wcs_multi_color_nulling/'
    >>> dset_list=['jplgse.20141202.63381','jplgse.20141202.63531','jplgse.20141202.63737','jplgse.20141202.63960','jplgse.20141202.64119','jplgse.20141202.64294','jplgse.20141202.64497']
    >>> IDL_to_HDF5.load_or_create(data_directory,dset_list=dset_list)
    
    """
    print("Checking for hdf5 file:"+data_directory+fname)

    if not os.path.isfile(data_directory+fname):
        with h5py.File(data_directory+fname,'w') as f:
            if dset_list:
                for dset in dset_list:
                    print("creating: "+dset+" in "+fname+" in "+ data_directory)
                    jplgse_to_HDF5(f,data_directory,dset)
                #f.close()
            else:
                dset_list=get_dsets(data_directory)
                for dset in dset_list:
                    print("creating: "+dset+" in "+fname+" in "+ data_directory)
                    jplgse_to_HDF5(f,data_directory,dset)
                #raise ValueError("invalid dset list: "+str(dset_list))
    else:
        print("Found hdf5 file:"+data_directory+fname)

    f = h5py.File(data_directory+fname,readwrite)
    return f

def jplgse_to_HDF5(f,base_dir,sub_dir):
    '''
    Function for parsing a directory idl save files produced by jplgse.

    .. warning:: currently data.d is not added to the HDF5 file, additional parsing needs to be added.
    Parameters
    ----------
    f : an HDF5 file or group
    base_dir  : the path to the data directory
    sub_dir :  the the subdirectory the files of interest are stored in
     within `base_dir,` this will also be the name of the HDF data group added to `f`.
    
    Returns
    ----------
    f : the input HDF5 file or group.
    
    Examples
    ----------
    
    >>> #MULTIPLE SUBDIRECTORIES in a single file:
    >>> data_directory="~/projects/PICTURE/data/todays_run"
    >>> datasets=[["gsedata.idl.20140703.59076","showat.20140703.59067"," #test 1 bright, white light"],
    ["gsedata.idl.20140703.59453","showat.20140703.59431","test 2: dim, white light"]]

    >>> f = h5py.File(data_directory+'data.hdf5','w')
    >>> for dset in datasets:
            PICTURE_IDL_to_HDF5.jplgse_to_HDF5(f,data_directory,dset[0])
            if len(dset)==3:
                PICTURE_IDL_to_HDF5.jplgse_to_HDF5(f,data_directory,dset[1])
    >>> #process data
    >>>f.close()

    
    >>> f = h5py.File(data_directory+'data.hdf5','w')
    >>>for dset in datasets:
           PICTURE_IDL_to_HDF5.jplgse_to_HDF5(f,data_directory,dset[0])
           if len(dset)==3:
               PICTURE_IDL_to_HDF5.jplgse_to_HDF5(f,data_directory,dset[1])
    >>> f.close()
    
    '''
    grp = f.create_group(str(sub_dir))
    directory=base_dir+"/"+sub_dir


    #first look for science camera frames:
    sci_files=glob.glob(directory+"/*.s.idl")
    sci_files.sort()

    if len(sci_files)>1:
        try:
            sci_1st=scipy.io.readsav(sci_files[0])
            sciheader=sci_1st["header"]
            scidateframes=sci_1st["data"]
            sci_frame=sci_1st["data"]
        except Exception, err:
            print("error finding SCI camera files")
            print(err)
    
        for i,sci_sav in enumerate(sci_files[1:]):
            try:
                sci=scipy.io.readsav(sci_sav)
            except Exception,err:
                print(err)
                print('readsave error')
                continue
            sciheader=np.vstack([sciheader,sci['header']])
            sci_frame=np.dstack([sci_frame,sci['data']])
        
        grp.create_dataset("sci", data=sci_frame,compression="gzip",fletcher32=True,track_times=True)
        grp.create_dataset("sci_header", data=sciheader,compression="gzip",fletcher32=True,track_times=True)
    #now try to find wavefront sensor data
    wfs_types = ["phase.i.idl","phase.v.idl","phase.r.idl","phase.m.idl",
                 "phase.p.idl","phase.u.idl","frame.a.idl","data.d.idl",
                 "frame.b.idl","frame.c.idl","frame.d.idl"]
    for wfs_extension in wfs_types:
        wfs_files=glob.glob(directory+"/*"+wfs_extension)
        wfs_files.sort()
        if len(wfs_files)>1:
            if wfs_extension == "data.d.idl":
                datad_grp=grp.create_group(wfs_extension+".data")
            try:
                wfs_1st=scipy.io.readsav(wfs_files[0])
                wfs_header=wfs_1st["header"]
                wfsdateframes=wfs_1st["data"]
                wfs_frame=wfs_1st["data"]
                if wfs_extension == "data.d.idl":
                    for field in wfs_frame.dtype.fields:
                        datad_grp.create_group(field)

                    
            except Exception, err:
                print("error finding WFS files")
                print(err)
            for i,wfs_sav in enumerate(wfs_files[1:]):
                try:
                    wfs = scipy.io.readsav(wfs_sav)
                    wfs_frame=np.dstack([wfs_frame,wfs['data']])
                    wfs_header=np.vstack([wfs_header,wfs['header']])
                except Exception,err:
                    print(err)
                    print('readsave error')
                    continue
                if wfs_extension == "data.d.idl":
                    #because some of the data.d arrays are numpy objects they need to be an extra layer
                    #this currently only works for 1D data
                    wfs_data=wfs['data']
                    for field in wfs_data.dtype.fields:
                        #break object fields up
                        if (wfs_data[field].dtype == np.dtype(object)):
                            for inner_field in wfs_data[field][0].dtype.fields:
                                field_name = field+"-"+inner_field
                                inner_array = np.array(wfs_data[field][0][inner_field][0],
                                             dtype = wfs_data[field][0][inner_field][0].dtype)
                                #print([field_name,inner_array])
                                #add data to appropriate dataset:
                                _update_data_d(wfs_files,datad_grp,field,field_name,inner_array,i)
                        else:
                            field_name=field
                            new_array = wfs_data[field]
                            _update_data_d(wfs_files,datad_grp,field,field_name,new_array,i)

            grp.create_dataset(wfs_extension+".header", data=wfs_header,compression="gzip",fletcher32=True,track_times=True)
            if wfs_extension != "data.d.idl":
                grp.create_dataset(wfs_extension+".data", data=wfs_frame,compression="gzip",fletcher32=True,track_times=True)

    bu_gse_files = glob.glob(directory+"/bugse.*.idl")
    if len(bu_gse_files) > 1:
        try:
            bugse_first=scipy.io.readsav(bu_gse_files[0])
            #if scipy.io.readsav puts the image inside an object that h5py can't save, then break it ou
            bugse_frame=bugse_first['data']["IMAGE"][0]
            bugse_header=matplotlib.mlab.rec_drop_fields(bugse_first['data'],["IMAGE"])     #http://stackoverflow.com/a/15577562/2142498
            bugse_filename=np.array([bu_gse_files[0]])
            bugse_temp_frame=bugse_first['data']["TEMPSENSORS"][0]
            bugse_header=matplotlib.mlab.rec_drop_fields(bugse_header,["TEMPSENSORS"])     #http://stackoverflow.com/a/15577562/2142498                                                                   

        except Exception, err:
            print("Error finding files.")
            print(err)
            print(err)
        for i,sav in enumerate(bu_gse_files[1:]):
            try:
                data = scipy.io.readsav(sav)
                #if scipy.io.readsav puts the image inside an object that h5py can't save, then break it out:                                                                                                    
                bugse_frame = np.dstack([bugse_frame,data['data']["IMAGE"][0]])
                bugse_temp_frame = np.dstack([bugse_temp_frame,data['data']["TEMPSENSORS"][0]])
                header = matplotlib.mlab.rec_drop_fields(data['data'],["IMAGE"])
                header = matplotlib.mlab.rec_drop_fields(header,["TEMPSENSORS"])
                bugse_header = np.vstack([bugse_header,header])
                bugse_header = np.vstack([bugse_header,header])
                bugse_filename=np.vstack([bugse_filename,[sav]])

            except Exception,err:
                print("BU GSE data parsing error in frame:"+str(sav))
                print(err)
                continue
        grp.create_dataset("bugse", data=bugse_frame,compression="gzip",fletcher32=True,track_times=True)
        grp.create_dataset("bugse_temp", data=bugse_temp_frame,compression="gzip",fletcher32=True,track_times=True)
        grp.create_dataset("bugse_header", data=bugse_header,compression="gzip",fletcher32=True,track_times=True)
        grp.create_dataset("bugse_filename", data=bugse_filename,compression="gzip",fletcher32=True,track_times=True)



    #finally, try looking for angle tracker data:
    at_files=glob.glob(directory+"/atfull.*.idl")
    if len(at_files)>1:
        #at_header,at_frame=collect_data_and_headers(at_files)
        try:
            at_first=scipy.io.readsav(at_files[0])
            #if scipy.io.readsav puts the image inside an object that h5py can't save, then break it out:
            if at_first['imagepkt']["IMAGE"].dtype==np.dtype('O'):
                if len(at_first['imagepkt']["IMAGE"]) ==1:
                    at_frame=at_first['imagepkt']["IMAGE"][0]
                    at_header=matplotlib.mlab.rec_drop_fields(at_first['imagepkt'],["IMAGE"])     #http://stackoverflow.com/a/15577562/2142498
            else:
                at_frame=at_first["imagepkt"]
    
        except Exception, err:
            print("error finding files")
            print(err)
        for i,sav in enumerate(at_files[1:]):
            try:
                data=scipy.io.readsav(sav)
                #if scipy.io.readsav puts the image inside an object that h5py can't save, then break it out:
                at_frame=np.dstack([at_frame,data['imagepkt']["IMAGE"][0]])
                at_header=np.vstack([at_header,matplotlib.mlab.rec_drop_fields(data['imagepkt'],["IMAGE"])])
            except Exception,err:
                print("angle tracker frame stacking problem")
                print(err)
                continue
        grp.create_dataset("at", data=at_frame,compression="gzip",fletcher32=True,track_times=True)
        grp.create_dataset("at_header", data=at_header,compression="gzip",fletcher32=True,track_times=True)

def _update_data_d(wfs_files,datad_grp,field,field_name,new_array,index):
        '''
        function to write 1D data to a dset within a group, and create dataset if it does not yet exist.
        '''
        try:
            dset = datad_grp[field][field_name]
            dset[index,:] = new_array
            #if new_array.size > 1:
            #print([field],[field_name],dset[index,:],new_array)
        except KeyError, err:
            print("Adding missing key. (" + str(err)+")")
            datad_grp[field].create_dataset(field_name,
                                            shape=(len(wfs_files),new_array.shape[0]),
                                            compression="gzip",
                                            fletcher32=True,
                                            track_times=True,
                                            maxshape=(None,None))
            datad_grp[field][field_name][index,:] = new_array

def collect_data_and_headers(globbed_list):
    ''' this function should replace seperate sections for WFS and sci

    '''
    try:
        first=scipy.io.readsav(globbed_list[0])
        header=first["header"]
        frame=first["data"]
    except Exception, err:
        print("error finding files")
        print(err)
    for i,sav in enumerate(globbed_list[1:]):
        try:
            data=scipy.io.readsav(sav)
            header=np.vstack([header,data['header']])
            frame=np.dstack([frame,data['data']])
        except Exception,err:
            print(err)
            continue
   
    return [header,frame]





def header_to_FITS_header(inputHeader,fmt='hdf5',hdu=None):
    '''
    Takes input header from one format and parses it into a FITS header.

    keywords:
    'hdf5' structured numpy array  from a custom PICTURE hdf5 file. 
    'idlsave' header from a custom IDLSAV file opened with scipy.io
    
    returns:
    ----------
    astropy.io.fits.PrimaryHDU
    
    Examples
    ----------

    >>> hdu =  header_to_FITS_header(f[im_dset]['sci_header'][0],input='hdf5')

    Raises
    ----------

    ValueError
    ----------
    
    '''
    if hdu == None:
        hdu=fits.PrimaryHDU()
    elif not isinstance(hdu,fits.PrimaryHDU):
        raise ValueError("unexpected object, "+str(type(hdu)))

    header=hdu.header

    if fmt == 'hdf5':
        for field in inputHeader.dtype.fields.keys():
            #print([field[0],input[field[0]][0]])
            header[str(field)]=inputHeader[field][0]

    if fmt == 'idlsave':
        raise ValueError("not yet implemented")
    #else:
    #     raise ValueError("set a valid format type.")
 
    return hdu

def attribute_to_FITS_header(attrs,hdu=None):
    '''
    Takes HDF5 dsets attributes and parses it's attributes into a FITS header

    returns:
    ----------

    astropy.io.fits.PrimaryHDU
    
    Examples:
    ----------

    >>> hdu =  header_to_FITS_header(f[im_dset]['sci_header'][0], input='hdf5')

    '''
    if hdu==None:
        HDUout=fits.PrimaryHDU()
    elif not isinstance(hdu,fits.PrimaryHDU):
        raise ValueError("unexpected object, "+str(type(HDUout)))
    
    keys=attrs.keys()
    
    if len(keys) == 0:
        print("no attributes")
        return hdu
    
    for attrib in keys:
        attrib_val=attrs[attrib]
        
        #make ASCII for fits compatiblity:
        #print(attrib,type(attrib_val))
        if (isinstance(attrib_val,np.string_)) or  (isinstance(attrib_val,str)):
             attrib_val =   attrib_val.encode('utf-8').decode('ascii', 'replace').replace('\n', ' ')

        hdu.header[str(attrib)]= attrib_val
    
    return hdu


def get_dsets(sequence_dir):
    '''
    Find all the the subdirectories in `sequence_dir` and return as a sorted list
    
    '''
    dsets=[path.split('/')[-2] for path in glob.glob(sequence_dir+"/*/")]
    dsets.sort()
    return dsets


def split_dset(f, obs, timestamp_end_target_1, timestamp_begin_target_2):
    '''
    split an observation data group and retain all ancillary info.
    
    creates two new groups within the original datafile.

    It would be nice to be able to do this with pointers. right now they are duplicates.

    ..warning: implementation assumes data arrays are 2D.
    
    '''

    split_time = 1
    # 0. require empty groups for each target in the dataset
    grp1 = f.require_group(obs + "_target1")
    grp2 = f.require_group(obs + "_target2")

    if timestamp_begin_target_2 < timestamp_end_target_1:
        raise ValueError("the second target should be after the first.")
        
    for dset in f[obs].keys():
        #print(dset)
        
        # 2. find that timestamp in headers for the science and wfs data
        # 3. copy the portions of each dataset into the new groups by slicing the
        # datasets using indices that correspond to the header time stamps which define the split


        if dset.find(".header") != -1:
            data_str = dset.replace(".header",".data")
        elif dset.find("sci_header") != -1:
            data_str = "sci"
        else:
            continue
        print(data_str)
        start_index = (f[obs][dset][...]["TIMESTAMP"]  < timestamp_end_target_1).flatten()
        end_index = (f[obs][dset][...]["TIMESTAMP"] >= timestamp_begin_target_2).flatten()
        print([f[obs][dset].shape,end_index])


        grp1data= f[obs][data_str][:,:,start_index]
        grp2data= f[obs][data_str][:,:,end_index]
        grp2header= f[obs][dset][end_index,0]
        grp1header= f[obs][dset][start_index,0]



        try:
            grp1.require_dataset(data_str, grp1data.shape, grp1data.dtype, exact=False, compression = "gzip", fletcher32 = True, track_times = True)
            grp2.require_dataset(data_str, grp2data.shape, grp2data.dtype, exact=False, compression = "gzip", fletcher32 = True, track_times = True)
            
            grp1.require_dataset(dset, grp1header.shape, grp1header.dtype, exact=False, compression = "gzip", fletcher32 = True, track_times = True)
            grp2.require_dataset(dset, grp2header.shape, grp2header.dtype, exact=False, compression = "gzip", fletcher32 = True, track_times = True)


            grp1[data_str][...] = grp1data
            grp2[data_str][...] = grp2data
            grp1[dset][...] = grp1header
            grp2[dset][...] = grp2header


            
        except RuntimeError, err:
            print(err)
