'''
tools for optimally plotting PICTURE datasets.
'''

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, SymLogNorm  # for log scaling of images, with automatic colorbar support
import numpy as np
import poppy
import astropy.io.fits as fits
from ..utils import max_cen_phot
import poppy.fwcentroid


import matplotlib.gridspec
from scipy import signal
cmap=matplotlib.cm.get_cmap(name='gnuplot', lut=None)
import astropy.convolution as conv
import matplotlib.patches as patches
import scipy.stats
def null_diagnostic_plot(cube,
                         background,
                         wfs_cube=None,
                         exp_time=1.0,
                         name='',
                         xlo=0,
                         xhi=-1,
                         ylo=0,
                         yhi=-1,
                         radius_pixels=5,
                         boxsize=5,
                         max_bright=1,
                         **kwargs):
    '''
    show times series and power spectra of the brightest pixel,
    the central star photometry and the mean vs time.
    Plot the mean and median image.

    Parameters
    ----------
    xlo,xhi,ylo,yhi :int
       define the region where statistics are performed, NOT plotting limits
    '''
        #    if fig is None:
    plt.figure(figsize=[22,4.5])
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))

    gs = matplotlib.gridspec.GridSpec(3, 4)
    ax1 = plt.subplot(gs[0, 0])
    ax2 = plt.subplot(gs[1,0])
    ax3 = plt.subplot(gs[-1,0])
    ax1b = plt.subplot(gs[0, 1])
    ax2b = plt.subplot(gs[1,1])
    ax3b = plt.subplot(gs[-1,1])
    ax5 = plt.subplot(gs[:,2])
    ax4 = plt.subplot(gs[:,-1])
    #plt.subplot(131)
    statscube=cube[xlo:xhi,ylo:yhi,:]
    print(np.shape(statscube))
    max_vs_time = (np.max(np.max(statscube,axis=0),axis=0))
    mean_vs_time = (np.mean(np.mean(statscube,axis=0),axis=0))
    median_vs_time = (np.median(np.median(statscube,axis=0),axis=0))
    if wfs_cube is not None:
        if np.shape(wfs_cube)[2] != np.shape(statscube)[2]:
            raise ValueError("data cubes are not same length")
            return 
        wfs_median_vs_time = (np.median(np.median(wfs_cube,axis=0),axis=0))

    x=np.arange(max_vs_time.size)*exp_time

    #for i in range(statscube.shape[2]):
    #    plt.figure()
    #    plt.imshow(statscube[:,:,i])
    center = poppy.fwcentroid.fwcentroid(np.median(statscube,axis=2))
    phot = np.array([max_cen_phot(statscube[:,:,i],
                                radius_pixels = radius_pixels,
                                fixed_center = center,
                                verbose = True,
                                boxsize = boxsize,
                                **kwargs)
                   for i in range(statscube.shape[2])])
    ax1.plot(x,max_vs_time/max_bright,label='max')
    ax1.set_xticklabels([]) #remove y axis labels
    ax1.set_ylabel("Max")#remove y axis labels
    ax1.set_title("Contrast (Max Bright =%.3g)"%max_bright)
    freq, Pxx_den = signal.periodogram(np.float64(max_vs_time)/max_vs_time.max(), fs=1.0/exp_time)
    #plt.ylim([1e-7, 1e2])
    #plt.xlabel('frequency [Hz]')
    
    ax1b.plot(freq, Pxx_den)#, freq, sp.imag)
    ax1b.set_title('PSD [V**2/Hz]')

    ax2.plot(x,median_vs_time/max_bright,'--',label='mean')
    if wfs_cube is not None:
        ax2.plot(x,wfs_median_vs_time/max_bright,'--',label='mean')
    medianstart=np.mean(median_vs_time[0:10])
    medianfold=x[np.where(median_vs_time>medianstart*np.exp(1))]
    print(medianstart,medianfold)

    if np.size(medianfold)>0:
        ax2.plot([medianfold[0],medianfold[0]],[np.min(median_vs_time)/max_bright,np.max(median_vs_time)/max_bright])
        ax2.annotate('e^1', (medianfold[0],np.max(median_vs_time)/2./max_bright),fontsize=8)

    ax2.set_xticklabels([]) #remove y axis labels
    ax2.set_ylabel("median")#remove y axis labels
    
    freq, Pxx_den = signal.periodogram(np.float64(median_vs_time)/median_vs_time.max(), fs=1.0/exp_time)
    ax2b.plot(freq, Pxx_den)#, freq, sp2.imag)
    
    ax3.plot(x,phot/max_bright,label='photometry')
    #calculate e^1 length:
    start=np.mean(phot[0:10])
    
    fold=x[np.where(phot>start*np.exp(1))]
    print(start,fold)
    if np.size(fold)>0:
        ax3.plot([fold[0],fold[0]],[np.min(phot)/max_bright,np.max(phot)/max_bright])
        ax3.annotate('e^1', (fold[0],np.max(phot)/2./max_bright),fontsize=8)

    ax3.set_ylabel("Photometry")

    freq, Pxx_den = signal.periodogram(np.float64(phot)/phot.max(), fs=1.0/exp_time)
    ax3b.plot(freq, Pxx_den)#, freq, sp3.imag)

    ax3b.set_xlabel("hz")
    ''' for ax in [ax1b,ax2b,ax3b]:
        #ax.set_xlim([np.min(freq),np.max(freq)])
        ax.set_xscale('symlog')
        ax.set_yscale('symlog')
    '''
    for ax in [ax1,ax2,ax1b,ax2b]:
            ax.set_xticklabels([]) #remove y axis labels

    #ax1.set_title("<max>=%.4g,  median(max)=%.4g, $\sigma$=%.4g,"%(np.mean(max_vs_time),np.median(max_vs_time),np.std(max_vs_time)))
    median_max=np.median(max_vs_time)
    std_max=np.std(max_vs_time)
    #ax1.legend()
    ax3.set_xlabel("seconds, $t_{exp}$=%.3g"%(exp_time))
    #plt.subplot(132)
    frame=np.mean(cube,axis=2)-background
    norm=SymLogNorm(frame.max()/max_bright/4095., vmax=frame.max()/max_bright)#LogNorm(vmin=1, vmax=4095)
    im5=ax5.imshow(frame/max_bright,interpolation='none',norm=norm,origin='lower',cmap=cmap)
    plt.colorbar(im5,ax=ax5,use_gridspec=True)
    ax5.set_title("mean")

    #plt.subplot(133)
    medianframe=np.median(cube,axis=2)-background
    im4=ax4.imshow(medianframe/max_bright,interpolation='none',norm=norm,origin='lower',cmap=cmap)
    plt.colorbar(im4,ax=ax4,use_gridspec=True)
    ax4.set_title("median")
    #plt.tight_layout()
    plt.suptitle(name)
    

def plot_contrast(raw_array,ax=None,PIXELSCL=0.158,center=None):
    '''
    
    Parameters
    ----------
    raw_array: np.array 
        2-D of contrast values 
    ax:
        plotting axis
    PIXESCL: float
        plate scale, (as/pixel)
    center:
        (x,y) coordinates of central pixel.
    '''
    if ax == None:
        fig=plt.figure()
        #gs1 = gridspec.GridSpec(1, 4)
        ax = fig.add_subplot(111)
    raw_hdulist=fits.HDUList(fits.PrimaryHDU(raw_array))
    if center == None:
        center=poppy.measure_centroid(raw_hdulist,verbose=True)[::-1]
    raw_hdulist[0].header["PIXELSCL"]=PIXELSCL
    raw_avg = poppy.utils.radial_profile(raw_hdulist,center=center)
    raw_std = poppy.utils.radial_profile(raw_hdulist,center=center,stddev=True)
    ax.plot(raw_avg[0],raw_avg[1],'--',
             label='Diff Spike intensity contrast',linewidth=2.4)
    ax.plot(raw_avg[0],raw_std[1],'-',
             label='Diff Spike $\sigma$ contrast',linewidth=2.4)
    ax.set_yscale('log')





def convergence(f,dset,
                plot_modes=[5,4,3],
                skip_frames=7,
                fine_only=True,
                fine_mode=5,
                lost=0,
                findp_mode=1,
                nulling_mode=34,
                baseline=None,
                logscale=True,
                min_V=.8,
                min_I_bg_scale=2.5,
                rms=True,
                mean=False,
                legend=True,
                re_gen_mask=True,
                hist_ax=None,
                hist_skip_frames=0,
                ax=None):
    '''
    make a time series plot of the wavefront error for a particular dataset.
    
    -----------
    
    Parameters
    -----------
        f: HDF5 file object
        dset: string
             indicating the PICTURE dataset (i.e. jplgse.20141216.51805)
        plot_modes: Boolean
             not implemented. (http://stackoverflow.com/a/2361991)
        fine_only: Boolean
            if True, then don't plot wavefront error measured in any other modes.
        lost: Integer
            indicated lost mode
        fine_mode: integer
            indicating fine mode
        nulling_mode: integer
            indicating nulling mode
        findp_mode: integer
            indicating find packet mode
        rms: Boolean
            if True, plot the root-mean-square of the phase in addition to the standard deviation.

    '''

    
    fig=plt.figure(figsize=[6,3],dpi=320)
    dset_wfs_shape=f[dset]['phase.u.idl.data'].shape
    

    '''seperated_plot_modes=[]
    for k, g in itertools.groupby(enumerate(plot_modes), lambda (i,x):i-x):
        #find consecutive modes: http://stackoverflow.com/a/2361991
        seperated_plot_modes.append(map(operator.itemgetter(1), g))
    if len(seperated_plot_modes)>1:
        raise ValueError("plot modes are not consecutive. Are you sure you want to plot them?")
    '''
    not_nulling_all = np.where((f[dset]['phase.u.idl.header']['STATE'] == 5) |
                          (f[dset]['phase.u.idl.header']['STATE'] == 4) |
                           (f[dset]['phase.u.idl.header']['STATE'] == 3) |
                             (f[dset]['phase.u.idl.header']['STATE'] == 2))[0]
    if not fine_only:
        in_fine_mode = not_nulling_all
    else:
        in_fine_mode=np.where(f[dset]['phase.u.idl.header']['STATE']==fine_mode)[0]
    
    if (fine_only) and (np.max(in_fine_mode[1:-1]-in_fine_mode[0:-2])>1):
        raise ValueError("multiple periods of fine mode in selected dataset.")
    #clip first 4 frames of fine mode:
    in_fine_mode=in_fine_mode[skip_frames:]
    
    bg_intensity =  np.mean(f[dset]['phase.i.idl.data'][0:10,0:10,in_fine_mode])#,axis=2)
    intensity = np.mean(f[dset]['phase.i.idl.data'][:,:,in_fine_mode],axis=2) - bg_intensity
    #print(np.mean(intensity),bg_intensity,bg_intensity*min_I_bg_scale)

    if re_gen_mask:
        twoDmask = (np.mean(f[dset]['phase.v.idl.data'][:,:,in_fine_mode],axis=2)<min_V) | (intensity< bg_intensity*min_I_bg_scale)
        print("generating a new mask")
    else:
        twoDmask=f[dset]['phase.m.idl.data'][:,:,0]==0
        print("using flight mask")
        
    twoDmask = twoDmask.reshape(dset_wfs_shape[0],dset_wfs_shape[1],1).repeat(dset_wfs_shape[2],2)
    fine_mode_mask=twoDmask[:,:,in_fine_mode]
    masked_phase=np.ma.masked_array(data=f[dset]['phase.u.idl.data'][:,:,in_fine_mode]*675.0/(2*np.pi),mask=fine_mode_mask)
    masked_int=np.ma.masked_array(  data=f[dset]['phase.i.idl.data'][:,:,in_fine_mode],mask=fine_mode_mask)
    masked_V=np.ma.masked_array(    data=f[dset]['phase.v.idl.data'][:,:,in_fine_mode],mask=fine_mode_mask)
    ax1=plt.subplot(121)
    plt.title("Intensity, masked")
    masked_int_plot=ax1.imshow(np.mean(masked_int,axis=2))
    plt.colorbar(masked_int_plot)

    ax2=plt.subplot(122)

    masked_plot=ax2.imshow(np.mean(masked_phase,axis=2))
    plt.title("masked phase")
    plt.suptitle("mask  OK?")
    plt.colorbar(masked_plot)
    plt.tight_layout()

    if ax is None:
        fig=plt.figure(figsize=[3.5,3])
        ax=plt.subplot(111)
    wfs_exp_t=f[dset]['frame.a.idl.header'][0]['EXPTIME'][0]
    t_series = 4* wfs_exp_t*np.arange(masked_phase.shape[2])
    print("WARNING, wfs_t_exp assumes no overhead")
    print(masked_phase.shape)
    flat_ims=masked_phase.reshape(masked_phase.shape[0]*masked_phase.shape[1],masked_phase.shape[2])
    print(flat_ims.shape)
    wfe_std=np.std(flat_ims,axis=0)
    wfe_mean=np.mean(flat_ims,axis=0)
    wfe_rms=np.sqrt(wfe_mean**2+(wfe_std**2))
    print(np.mean(wfe_std),np.std(wfe_rms))


    #else:
    ax.plot(t_series,wfe_std,linewidth=2.5,label="STDEV, $\sigma(\Delta\phi)$",color='gray')

    if rms==True:
        ax.plot(t_series,wfe_rms,'.',linewidth=2.5,label="RMS, $\sqrt{<\Delta\phi^2>}$",color='black')
    if mean==True:
        ax.plot(t_series,wfe_mean,'-',linewidth=1.5,label="$<\Delta\phi>}$",color='black')

        #plt.ylabel("$\mathrm{RMS({\mathrm{WFE}}})$ [nm]")
    ax.set_ylabel("nm")

    if baseline is not None:
        ax.plot(t_series,baseline*np.ones(len(t_series)),'--')
    #if fine_only:
    #    plt.title("Fine Mode")
    # else:
    #    plt.plot([t_series[not_nulling[0]-in_fine_mode[0]],t_series[not_nulling[0]-in_fine_mode[0]]],np.array([0,1000]),'--')
    #    plt.annotate('Fine Mode', (t_series[not_nulling[0]-in_fine_mode[0]]+.1,np.array([90])),fontsize=8)
    #    print("CHECK THAT FINE MODE ANNOTATION IS IN RIGHT PLACE!")
    #plt.xlim(t_series[in_fine_mode[0]],t_series[in_fine_mode[-1]])
    ax.set_xlabel("Time [s]")
    if logscale:
        ax.set_yscale('log')
    #plt.ylim([1,150])
    if legend:
        ax.legend(fontsize=10,numpoints=1)
    if hist_ax is not None:
        bins=np.arange(-20,20,2)
        hist_ax.hist(wfe_std[hist_skip_frames:],
                     bins=bins,
                     linewidth=2.5,
                     label="$\sigma(\Delta\phi)$",color='gray')        
        hist_ax.hist(wfe_mean[hist_skip_frames:],
                     bins=bins,
                     histtype=u'step',
                     label="$<\Delta\phi>}$",
                     alpha=0.5,color='blue')
        hist_ax.hist(wfe_rms[hist_skip_frames:],bins=bins,
                     linewidth=2.5,
                     label="$RMS, \sqrt{<\Delta\phi^2>}$",
                     alpha=0.7,color='red')


    #plt.tight_layout()
    return np.array([t_series,wfe_mean])

def get_finemode_index(f,
                        dset,
                        skip_frames=20,
                        fine_mode=5,
                        lost=0,
                        findp_mode=1,
                        nulling_mode=34):
    '''
    Returns a dictionary with the indices corresponding to not_nulling and in_fine_mode, with frame skips included.
    '''
    in_fine_mode=np.where(f[dset]['phase.u.idl.header']['STATE']==fine_mode)[0]
    not_nulling=np.where((f[dset]['phase.u.idl.header']['STATE'] == 5) |
                          (f[dset]['phase.u.idl.header']['STATE'] == 4) |
                           (f[dset]['phase.u.idl.header']['STATE'] == 3) )[0]
    
    if np.max(in_fine_mode[1:-1]-in_fine_mode[0:-2])>1:
        raise ValueError("multiple periods of fine mode in selected dataset.")
    #clip first 4 frames of fine mode:
    in_fine_mode=in_fine_mode[skip_frames:]
    return {"not_nulling":not_nulling, "in_fine_mode":in_fine_mode}

def fine_mode_character(f,
                        dset,
                        boxcarwidth=2,
                        skip_frames=20,
                fine_mode=5,
                lost=0,
                findp_mode=1,
                nulling_mode=34,
                G_e_per_count=4,
                min_V=.8,
                min_I_bg_scale=2.5,
                re_gen_mask=True,
                skip_mask=False,
                measurement='phase.u.idl.data',
                figsize=[2,4],
                units="nm",
                kernel=conv.Box2DKernel,
                scaling=675.0/(2*np.pi),
                ignore_edge_zeros =True,
                hist_ax=None,
                kernel_mode='linear_interp',
                **kwargs):
    '''
    plot wfs phase measurements while in fine mode and compare performance to spatial frequencies below boxcar width via convolution with a boxcar function
    kwargs are passed to conv_fft

    Parameters
    -----------

    '''
    fig=plt.figure(figsize=figsize,dpi=320)
    dset_wfs_shape=f[dset]['phase.u.idl.data'].shape
    
    in_fine_mode=np.where(f[dset]['phase.u.idl.header']['STATE']==fine_mode)[0]
    not_nulling=np.where((f[dset]['phase.u.idl.header']['STATE'] == 5) |
                          (f[dset]['phase.u.idl.header']['STATE'] == 4) |
                           (f[dset]['phase.u.idl.header']['STATE'] == 3) )[0]
    
    if np.max(in_fine_mode[1:-1]-in_fine_mode[0:-2])>1:
        raise ValueError("multiple periods of fine mode in selected dataset.")
    #clip first 4 frames of fine mode:
    in_fine_mode=in_fine_mode[skip_frames:]

    if len(in_fine_mode) < 2:
        print("Fewer than two fine mode frames, skip_frames and dataset "+ dset)
        return
    print(in_fine_mode)
    wfs_exp_t=f[dset]['frame.a.idl.header'][0]['EXPTIME'][0]

    bg_intensity =  np.mean(f[dset]['phase.i.idl.data'][0:10,0:10,not_nulling])#,axis=2)

    intensity = np.mean(f[dset]['phase.i.idl.data'][:,:,in_fine_mode],axis=2) - bg_intensity
    
    if not skip_mask:
        if re_gen_mask:
            twoDmask = (np.mean(f[dset]['phase.v.idl.data'][:,:,in_fine_mode],axis=2)<min_V) | (intensity< bg_intensity*min_I_bg_scale)
            print("generating a new mask")
        else:
            twoDmask=f[dset]['phase.m.idl.data'][:,:,0]==0
            print("using flight mask")
        threeDmask = twoDmask.reshape(dset_wfs_shape[0],dset_wfs_shape[1],1).repeat(dset_wfs_shape[2],2)
        fine_mode_mask = threeDmask[:,:,in_fine_mode]
    
    else:
        fine_mode_mask = False
        twoDmask = False
    masked_phase = np.ma.masked_array(data=f[dset][measurement][:,:,in_fine_mode]*scaling, mask=fine_mode_mask)
        
    masked_int =np.ma.masked_array(  data=f[dset]['phase.i.idl.data'][:,:,in_fine_mode], mask=fine_mode_mask)
    masked_V = np.ma.masked_array(    data=f[dset]['phase.v.idl.data'][:,:,in_fine_mode], mask=fine_mode_mask)
    smoothed_tseries = [conv.convolve_fft(masked_phase[:,:,k],conv.Box2DKernel(4)) for k in range(len(in_fine_mode))]

    npix = np.size(masked_phase[masked_phase.mask==False])
    ax1 = plt.subplot(211)
    ax1.set_xticklabels("")
    ax1.set_yticklabels("")
    ax1.annotate("A",[0.08,.82], color='black', xycoords='axes fraction')

    if  measurement == 'phase.v.idl.data':
        masked_phase=np.sqrt(masked_phase/100.0)*100.0
        
    mean_phase = np.median(masked_phase,axis=2)

    phase_im = plt.imshow(mean_phase,vmin=mean_phase.min(),vmax=mean_phase.max(),interpolation='none')
    plt.title("%.4g$\pm$%.2g "%(np.mean(masked_phase),
                                               np.std(masked_phase)),fontsize=12)
    cax1 = fig.add_axes([0.95, 0.575, 0.1, 0.3]) #left,bottom,width,height
    cax1.set_title(units,size=12)
    cax1.tick_params(labelsize=10)
    plt.colorbar(phase_im,cax=cax1)#,orientation='horizontal')
    
    ax2 = plt.subplot(212)
    nan_mean_phase = np.ma.filled(mean_phase,fill_value=np.nan)
    phaseconvolved = np.ma.masked_array(conv.convolve_fft(nan_mean_phase,
                                                        kernel(boxcarwidth,mode=kernel_mode),
                                                        interpolate_nan=True,
                                                        ignore_edge_zeros=ignore_edge_zeros,
                                                        **kwargs),
                                                        mask=twoDmask,
                                                        )
    smooth_phase = np.ma.masked_array(phaseconvolved,mask=phaseconvolved==0)
        
    #print(smooth_phase.shape,mean_phase.shape)
    phase_smoth_im = plt.imshow(smooth_phase,interpolation='none')#,vmin=mean_phase.min(),vmax=mean_phase.max())

    ax2.set_xticklabels("")
    ax2.set_yticklabels("")
    ax2.annotate("B",[0.08,.82],color='black',xycoords='axes fraction')

    plt.title("%.4g$\pm$%.2g "%(np.mean(smooth_phase),
                                               np.std(smooth_phase)),fontsize=12)
    cax2 = fig.add_axes([0.95, 0.1, 0.1, 0.30]) #left,bottom,width,height
    cax2.set_title(units,size=12)
    cax2.tick_params(labelsize=10)
    plt.colorbar(phase_smoth_im, cax=cax2)#,orientation='horizontal')
    #hatch the masked regions:
    #https://stackoverflow.com/questions/18390068/hatch-a-nan-region-in-a-contourplot-in-matplotlib
    #(Isn't working yet, only hatches ax2?)
    #p = patches.Rectangle([0,0],1000, 1000, hatch='/',  zorder=-10,fill=None,)
    #ax1.add_patch(p)
    #ax2.add_patch(p)

    plt.tight_layout()
    wfe_std = np.std(smooth_phase)
    wfe_rms = np.sqrt(wfe_std**2+np.mean(smooth_phase)**2)

    return {"stddev smoothed":wfe_std,
            "rms":wfe_rms,
            "mean smoothed": np.mean(smooth_phase),
            "raw mean":np.mean(masked_phase),
            "raw std":np.std(masked_phase),
            "bayes_mvs":scipy.stats.bayes_mvs(smooth_phase),
            "e-/p/sec":np.mean(masked_int)*G_e_per_count/wfs_exp_t/npix,
            "array_shape":dset_wfs_shape,
            "wfs_t_exp":wfs_exp_t,
            "smooth_array":phaseconvolved,
            "raw_array":mean_phase,
            "median_tseries":masked_phase,
            "smoothed_tseries":np.ma.array(np.array(smoothed_tseries).T,mask=fine_mode_mask)
            }


def std_weighted(array, weights):
    '''
    returns the weighted standard deviation.
    from: http://stackoverflow.com/a/2415343/2142498
    '''
    weighted_avg=np.average(array,weights=weights)
    return np.sqrt(np.average((array- weighted_avg)**2, weights=weights))
        
