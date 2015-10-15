import poppy
import scipy.ndimage
import astropy.io.fits as fits
import numpy as np
import scipy.optimize
import PICTURE_IDL_to_HDF5
import astropy.convolution as conv

def recenter(in_array,newcntr,verbose=True,**kwargs):
    '''
    uses the POPPY centroid measuring routine to find the center of an PSF,
    and then move that to the new coordinate specified by newcntr.
    
    '''
    cntry,cntrx=poppy.measure_centroid(fits.HDUList(fits.PrimaryHDU(in_array)),verbose=verbose,**kwargs)
    if verbose:
        print(cntrx,cntry)
    return scipy.ndimage.shift(in_array,(newcntr[0]-cntry,newcntr[1]-cntrx),mode='nearest')

def shift_and_sub(yx,image1,image2,metric=np.std):
    '''
    shift image 2 by the value of xy tuple, equal to (y,x), and subtract image 1 from it,
     return the value of the function metric, which defaults to standard deviation.
     alternatives could include rms() or np.mean() or median or...
     
    '''
    im2shift=scipy.ndimage.shift(image2,(yx))
    #metric=np.sqrt(np.mean(im2shift-image1)**2) #alternate metric
    val=metric(im2shift-image1)
    return val

def scale_and_sub(c,image1,image2,metric=np.std):
    '''
    scale image 2 by the value of c, and subtract image 1 from it,
     return the value of the function metric, which defaults to standard deviation.
     alternatives could include rms() or np.mean() or median or...
     
    '''
    #im2shift=scipy.ndimage.shift(image2,(yx))
    #metric=np.sqrt(np.mean(im2shift-image1)**2) #alternate metric
    val=metric(c*image2-image1)
    return val

def rms(array):
    return np.sqrt(np.mean(array**2))

def opt_shift(image1,image2,return_array=True,globalmin=True,x0=(0,0), method='Powell', niter=1000,stepsize=0.1,**kwargs):
    '''
    Find the optimal shift of `image2` that minimizes the residual when `image1` is  subtracted.
    
    Parameters
    ----------
    
    globalmin: bool
        if True use Monte Carlo basinhopping to jump between local minima, otherwise, use method alone.
    method:
        e.g. 'Nelder-Mead','Powell'
    x0: tuple
        starting offset in x and y.
    
    return_array: bool
        if True returns a shifted copy of array 2
               
    Returns
    ----------
    the output of the optimization, unless return array keyword is set True (the default).
            
    '''
    if not globalmin:
        opt=scipy.optimize.minimize(shift_and_sub,[x0],args=(image1,image2),
                                    method=method,options={"maxiter":1000})
    else:
        minimizer_kwargs = {"args":(image1,image2),
                            "method": method, 
                            }
        opt = scipy.optimize.basinhopping(shift_and_sub, x0,minimizer_kwargs=minimizer_kwargs,
                   niter=niter,stepsize=stepsize,**kwargs)
    print(opt)
    print(opt["x"])
    if return_array:
        return scipy.ndimage.shift(image2,opt["x"])
    else:
        return opt

def general_opt(image1,image2,function,globalmin=True,x0=1.0, method='Powell', niter=1000,stepsize=0.1,**kwargs):
    '''
    a general function for optimizing a function with two images as the arguments.

    Parameters
    ----------
    globalmin:
        Use MonteCarlo (basinhopping) to jump between local minima found, otherwise, use method alone.
    method:
        e.g. 'Nelder-Mead','Powell'
    x0: tuple
        starting offset
    return_array: bool
        if True returns the optimized copy of array 2
    Returns:
        the output of the optimization, unless return array keyword is set True (the default).

    '''
    if not globalmin:
        opt=scipy.optimize.minimize(function,[x0],args=(image1,image2),
                                    method=method,options={"maxiter":1000})
    else:
        minimizer_kwargs = {"args":(image1,image2),
                            "method": method, 
                            }
        opt = scipy.optimize.basinhopping(function, x0,minimizer_kwargs=minimizer_kwargs,
                   niter=niter,stepsize=stepsize,**kwargs)
    print(opt)
    return opt
    
def ZoomSameShape(input_array,zoom,oversample=1,**kwargs):
    '''
    Rescales input_array by a factor zoom, centered on the center of the array, 
    and returns the scaled array with the same dimensions as the input

    '''
    #rescale an array without changing it's size:
    if oversample !=1:
        input_array=scipy.ndimage.zoom(input_array,oversample)
    
    grid_x, grid_y = np.mgrid[0:1:input_array.shape[0]*1j, 0:1:input_array.shape[1]*1j]
    points=np.array([grid_x.flatten(),grid_y.flatten()])

    new_points=(points-points[-1][-1]/2)*1.01+points[-1][-1]/2

    grid_z0 = scipy.interpolate.griddata(new_points.T, input_array.flatten(), (grid_x, grid_y),**kwargs)
    if oversample !=1:
        grid_z0=scipy.ndimage.zoom(grid_z0,1.0/oversample)
    
    return grid_z0




def optimally_shift_and_save(f,
                             dset1,
                             dset2,
                             null_state=34,
                             n_skip=5,
                             sequence_dir='',
                             para_angle=0,
                             kernel=None,
                             dark_fits_name=None,
                             subtract_corner=True,
                             **kwargs):
    '''
    Align the frames of `dset2` to `dset1` using `opt_shift()` and save in a subfolder of `dset1`

    Parameters
    ----------
    f: HDF5 file object
        input hdf5 file
    dset1: string
        first hdf5 dataset.
    dset2: string
        first hdf5 dataset.
    
    null_state: int
        the PICTURE FSM state indicating the instrument is nulling
    n_skip: int
        the number of nulling frames to skip at the beginning of the data cube.
    sequence_dir: string
        where outputs go, should be the parent diretory of the file.
    para_angl: float
        rotation angle of both datasets, defaults to zero.
    kernel: astropy.convolution.kernel
         astropy smoothing kernel, default is None.
    subtract_corner: bool
         if set to True the mean value of a 10x10 pixel box in the corner of the all the images will be subtracted as a background signal
    dark_fits_name: None or string
         Default is none, otherwise a fits file which will be subtracted from each frame prior to shifting.
    **kwargs:
         keyword arguments passed to opt_shift().
    Examples
    ----------
    '''
    from os import mkdir
    from os.path import exists, isdir,expanduser

    if para_angle != 0:
        raise ValueError("De-Rotation not implemented")

    good_sci_data1,median1,std1 = get_nulled_frames(f,dset1,null_state=null_state,n_skip=n_skip)
    good_sci_data2,median2,std2 = get_nulled_frames(f,dset2,null_state=null_state,n_skip=n_skip)

    headers2 = get_nulled_frame_headers(f, dset2,null_state=null_state,n_skip=n_skip)
    frame_numbers2=headers2['FRAME_NUMBER']

    if dark_fits_name is not None:
        dark = fits.open(dark_fits_name)[0].data
        median1 -= dark
        median2 -= dark

    if subtract_corner:
        median1 -= np.mean(median1[0:10,0:10])
        median2 -= np.mean(median2[0:10,0:10])

    
    if kernel is not None:
        print("smoothing with: "+str(kernel))
        conv.convolve_fft(median1,kernel,normalize_kernel=True, ignore_edge_zeros=True)
        conv.convolve_fft(median2,kernel,normalize_kernel=True, ignore_edge_zeros=True)
        

    opt = opt_shift(median1,median2,return_array=False,**kwargs)
    
    if (frame_numbers2.shape[0] != good_sci_data2.shape[-1]):
        raise ValueError("lengths: %i, %i, do not match"%(len(frame_numbers2), len(good_sci_data2)))

    datadir = sequence_dir+'/'+dset1+'/shifted'
    if not exists(datadir):
        try:
            mkdir(datadir)
        except OSError as e:
            if not exists(datadir):
                raise
    subdir = datadir+'/'+dset2
    if not exists(subdir):
        try:
            mkdir(subdir)
        except OSError as e:
            if not exists(subdir):
                raise
    for i,frame_num in enumerate(frame_numbers2):
        shifted2 = scipy.ndimage.shift(good_sci_data2[:,:,i],opt["x"])
        
 
        #print("output max:%.4e"%(good_sci_data2[:,:,i].max()))
        FITS_HDU=PICTURE_IDL_to_HDF5.header_to_FITS_header(headers2[i],fmt='hdf5')

        PICTURE_IDL_to_HDF5.attribute_to_FITS_header(f[dset2]['sci_header'].attrs,hdu=FITS_HDU)
        FITS_HDU.header["NEW_PARA"]=para_angle
        
        FITS_HDU.header['HISTORY']="Shift optimization output"+str(opt).encode('utf-8').decode('ascii', 'replace').replace('\n', ' ')

        if dark_fits_name is not None:
            dark = fits.open(dark_fits_name)[0].data
            shifted2 =shifted2 - dark
            FITS_HDU.header['DARKFILE']=str(dark_fits_name)
            
        if subtract_corner:
            corner=np.mean(shifted2[0:10,0:10])
            shifted2 = shifted2- corner
            FITS_HDU.header['HISTORY']="subtracted mean of 100 pixels in corner: %.3g"%(corner)

   
        
        if kernel is not None:
            print("smoothing")
            shifted2 = conv.convolve_fft(shifted2,kernel,normalize_kernel=True, ignore_edge_zeros=True)
            print(type(shifted2))
            FITS_HDU.header["HISTORY"]="Applied convolution:"+str(kernel.model).replace('\n','')
            
        HDU = fits.HDUList([fits.PrimaryHDU(data=shifted2, header=FITS_HDU.header)])
        HDU.writeto(subdir+'/'+dset2+str(frame_num[0]).zfill(8)+'.shifted.sci.s.fits', clobber=True)
        
def get_nulled_frames(f,dset,null_state=34,n_skip=5,over_clock=2):
    '''
    example:
    #note! the first 3 values are skipped because they usually aren't really nulling

    '''
            
    nulled=np.where(f[dset][u'sci_header']['STATE'].flatten()==null_state)[0]
    good_sci_data=f[dset][u'sci'][:,:-over_clock,nulled[n_skip]:nulled[-1]]
    dims=np.shape(good_sci_data)
    if dims[0] !=dims[1]:
        raise ValueError("output not square,  applying %i overclock to second dimension (x), resulting dimension:"+str())
    median=np.array(np.median(good_sci_data,axis=2),dtype=np.float64)
    std=np.array(np.std(good_sci_data,axis=2),dtype=np.float64)

    return good_sci_data,median,std
    

def get_nulled_frame_headers(f,dset,null_state=34,n_skip=5):
    '''
    example:
    #note! the first 3 values are skipped because they usually aren't really nulling

    '''
    nulled=np.where(f[dset][u'sci_header']['STATE'].flatten()==null_state)[0]
    headers=f[dset][u'sci_header'][nulled[n_skip]:nulled[-1]]
    return headers


