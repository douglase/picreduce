import numpy as np


import os
import shutil
import glob
import subprocess
import numpy as np
import logging
_log = logging.getLogger('picb')

def get_nulled_frames(f,dset,
                      null_state=34,
                      n_skip=3,
                      avgfunction=np.median,
                      delete_saturated=False,
                      sat_val=4095):
    
    '''
    example:
    #note! the first 3 values are skipped because they usually aren't really nulling

    '''
    nulled=np.where(f[dset][u'sci_header']['STATE'].flatten()==null_state)[0]
    nulled=nulled[n_skip:-1]
    good_sci_data = f[dset][u'sci'][:,:138,nulled]
    original_shape=good_sci_data.shape
    if delete_saturated:
        saturated_frames=np.where(good_sci_data.max(axis=0).max(axis=0) >= sat_val)[0]
        _log.warn("shape of original array"+str(np.shape(good_sci_data)))
        _log.warn("saturated frames: "+str(saturated_frames))

        good_sci_data=np.delete(good_sci_data,saturated_frames,2)
        _log.warn("deleted %g frames of %g "%(len(saturated_frames),original_shape[2]))

        #print(np.delete(saturated_frames,saturated_frames,2).shape)

    #null_diagnostic_plot(good_sci_data,oc,name=dset,xlo=45,xhi=93,ylo=45,yhi=93)#,grid=outer_grid)
    median=avgfunction(good_sci_data,axis=2)#-np.median(f[dset][u'sci'][0:10,0:10,nulled[n_skip]:nulled[-1]])
    std=np.std(good_sci_data,axis=2)

    #null_diagnostic_plot(good_sci_data,oc,name=dset,xlo=45,xhi=93,ylo=45,yhi=93)#,grid=outer_grid)
    
    '''good_sci_data= good_sci_data.swapaxes(0,2)
    good_sci_data = PynPoint._Util.mk_resizerecent(good_sci_data[:,:138,:],2,1)
    median=np.median(good_sci_data,axis=0)'''
            
    #median = recenter(median,(68, 68),boxsize=10)
    return good_sci_data,median,std





def create_randomized_folders(source_dir,num_frames=None,ext='fits'):
    '''
    Takes a directory of files and randomly splits symbolic links to them between two new folders. 

    keywords:
    'num_frames' default None, the number of files in each directory.
    'ext' defaults to _fits_ for Flexible Image Transport System
    
    
    Examples
    ----------

    >>> picreduce.picbslices.create_randomized_folders('path_to_fits')

    Raises
    ----------

    ValueError
    ----------

    '''

    files=glob.glob(source_dir+"/*."+ext)
    #shuffle list of files
    np.random.shuffle(files)
    
    #make subdirectories
    Adir=source_dir+"randomA"+str(num_frames)
    Bdir=source_dir+"randomB"+str(num_frames)
    try:
        os.mkdir(Adir)
    except OSError,err:
        print(err)
        print("problem creating directory, trying to remove and recreate")
        try:
            shutil.rmtree(Adir)
            os.mkdir(Adir)
        except OSError,err:
            print(err)
            raise ValueError 
    try:
        os.mkdir(Bdir)
    except OSError,err:
        print(err)
        print("problem creating directory, trying to remove and recreate")
        try:
            shutil.rmtree(Bdir)
            os.mkdir(Bdir)
        except OSError,err:
            print("failed to create directory")
            print(err)
            raise ValueError 

    #split deck
    if num_frames is None:
        num_frames=int(np.floor(np.size(files)/2.0 -1))
    list1=files[0:num_frames]
    list2=files[num_frames+1:(num_frames+1)+num_frames]
    for fpath in list1:
        fname=fpath.split("/")[-1]
        try:
            cmd="ln -s "+fpath+" /"+Adir+"/"+fname
            subprocess.call(cmd.split(" "))
        except OSError,err:
            print(err)     

    for fpath in list2:
        fname=fpath.split("/")[-1]
        try:
            cmd="ln -s "+fpath+" /"+Bdir+"/"+fname
            subprocess.call(cmd.split(" "))
        except OSError,err:
            print(err)
            print(cmd)


            
    

def get_nulled_frame_headers(f,dset,null_state=34,n_skip=5):
    '''
    example:
    #note! the first 3 values are skipped because they usually aren't really nulling

    '''
    nulled=np.where(f[dset][u'sci_header']['STATE'].flatten()==null_state)[0]
    nulled=nulled[n_skip:-1]
    headers=f[dset][u'sci_header'][...][nulled]
    return headers


