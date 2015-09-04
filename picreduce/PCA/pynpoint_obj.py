# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

#%matplotlib inline
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap 
#https://stackoverflow.com/questions/24997926/making-a-custom-colormap-using-matplotlib-in-python
import matplotlib.cm
import PynPoint
import numpy as np

import h5py
import poppy
import astropy.io.fits
import glob
import sys,os
sys.path.insert(0, '../../analysis_tools/')
#import radial_profile
import max_cen_phot


import astropy.convolution as conv

from copy import deepcopy
import matplotlib.style
matplotlib.style.use("grayscale")
import matplotlib as mpl
from matplotlib.colors import SymLogNorm  # for log scaling of images, with automatic colorbar support

mpl.rcParams['image.interpolation'] = 'none'
mpl.rcParams['image.origin']='lower'

# <codecell>

#ls '/home/douglase/projects/picture/data/flight_sequence_20140530/gsedata.idl.05302014.58866/shifted/gsedata.idl.05302014.58866/'

# <codecell>


# <codecell>

#PynPoint Requires the paralytic angle be stored in the FITS header, set to zero so it doesn't try to derotate
'''for filen in glob.glob(data_dir+'*'):
    hdulist=fits.open(filen,mode='update')
    
    print(filen)
    print(hdulist[0].header)
    hdulist[0].header['NEW_PARA']=0

    hdulist.flush()
    hdulist.close(verbose=True)
'''

# <markdowncell>

# 
# 

# <codecell>

class observation_sets:
    def __init__(self,image_dir,
                basis_dir,
                ran_sub=None,
                stackave=1,
                cent_size=0.00,
                prep_data=True,
                recent=False,
                resize=True,
                smooth_kernel=None):
            self.image_dir=image_dir
            self.basis_dir=basis_dir
            self.basis_images=PynPoint.images.create_wdir(basis_dir, 
                                                ran_sub=ran_sub,
                                                stackave=stackave,cent_size=cent_size,
                                                prep_data=prep_data,recent=recent,resize=resize)
            self.images = PynPoint.images.create_wdir(image_dir,ran_sub=ran_sub,
                                                stackave=stackave,cent_size=cent_size,
                                                prep_data=prep_data,recent=recent,resize=resize)
            #need another basis instance, because operations happen in-place on basis set:
            self.basis = PynPoint.basis.create_wdir(basis_dir,ran_sub=ran_sub,
                                        stackave=stackave,cent_size=cent_size,
                                        prep_data=prep_data,recent=recent,resize=resize)
            self.smooth_kernel=smooth_kernel
            self.convolution_counter = 0
    def conv_frames(self,kernel,**kwargs):
        '''
        Description:
        This function smooths each image frame by the input kernel via FFT convolution,
         it can dramatically improve median subtraction but doesn't seem to be helping PCA,
          it may also be that it's not being applied at the correct point in the PCA workflow,
          because of the normalization that occurs in Pynpoint,
         test would be how it works to have this smoothing applied on the raw fits files first.

          Parameters:
        astropy.convolution.kernel

        
        '''
        if self.convolution_counter > 0:
            print("WARNING: already convolved %3g times"%self.convolution_counter)
        for i_num in range(self.basis_images.im_arr.shape[0]):
            self.basis_images.im_arr[i_num,:,:] =  conv.convolve_fft(self.basis_images.im_arr[i_num,:,:] ,kernel, **kwargs)
            
        for i_num in range(self.images.im_arr.shape[0]):
            self.images.im_arr[i_num,:,:] =  conv.convolve_fft(self.images.im_arr[i_num,:,:], kernel, **kwargs)
            
        for i_num in range(self.basis.im_arr.shape[0]):
            self.basis.im_arr[i_num,:,:] =  conv.convolve_fft(self.basis.im_arr[i_num,:,:], kernel, **kwargs)
        self.convolution_counter =  self.convolution_counter + 1 
    def pynpoint_contrast(self,
                      plate_scale=0.158,
                      n_pix=128,
                      max_coeff=100,
                      plot_all_pca=False,
                      smooth_option=None,
                      high_color = 'orange',
                      low_color = 'black',
                      angular_units='arcsec'):
        '''
    
        angular_units:
            'arcsec' or 'lambdaD', changes xlabel.
        ran_sub:
             None or false gives glob() ordered set, -1 shuffles, N<N_files gives smaller set of images
        smooth_option:
             values: None, 'before', 'after', smooth with the bright_psf_kernel before or after the PCA analysis.
        '''
        self.smooth_option=smooth_option
        zero_color = 'white'
        semi_sym_cm = LinearSegmentedColormap.from_list('my cmap', [low_color, zero_color,high_color])
        vmin=-1e-2
        vmax=1e-2
        norm=SymLogNorm(5e-6,vmin=vmin,vmax=vmax)#, vmax=5e-3)
        tick_levels = [-1e-3, -1e-4, -1e-5, 0.0, 1e-5, 1e-4, 1e-3]

  
        first_file=self.images.files[0]
        print(first_file)
        max_bright = astropy.io.fits.open(first_file)[0].header['MAXBRITE']
        self.max_bright=max_bright
        print("Using  MAXBRITE header keyword from first FITS file in image_dir for contrast:"+str(max_bright))
        res = PynPoint.residuals.create_winstances(self.images, self.basis)
        self.res=res
        file_hdf5 = PynPoint._Util.filename4mdir(self.basis_dir)
        
        raw=astropy.io.fits.HDUList([astropy.io.fits.PrimaryHDU(np.median(self.images.im_arr.T*self.images.im_norm[np.newaxis,:],axis=2).T)])
        raw_basis=astropy.io.fits.HDUList([astropy.io.fits.PrimaryHDU(np.median(self.basis_images.im_arr.T*self.basis_images.im_norm[np.newaxis,:],axis=2).T)])
        if self.smooth_option == 'after' and self.smooth_kernel is not None: 
            raw[0].data = conv.convolve_fft(raw[0].data, self.smooth_kernel,normalize_kernel=True)
            raw_basis[0].data = conv.convolve_fft(raw_basis[0].data, self.smooth_kernel,normalize_kernel=True)

        self.raw=raw
        self.raw_basis=raw_basis

        
        #(above line uses broadcasting, see: https://stackoverflow.com/questions/7096371/)
        print("max of raw %.3g"%np.max(raw[0].data))


        #find center,     #default is y,x so reversing to keep brain from hurting    
        center=poppy.measure_centroid(HDUlist_or_filename=raw,boxsize=10)[::-1]


        phot1=max_cen_phot.max_cen_phot(raw[0].data,radius_pixels=5,verbose=True,boxsize=5)
        phot2=max_cen_phot.max_cen_phot(raw_basis[0].data,radius_pixels=5,verbose=True,boxsize=5)
  
        mean_subbed_contrast = astropy.io.fits.HDUList([astropy.io.fits.PrimaryHDU((raw[0].data - raw_basis[0].data*phot1/phot2)/max_bright)])
        # if self.smooth_option == 'after' and self.smooth_kernel is not None: 
        #    mean_subbed_contrast[0].data = conv.convolve_fft(mean_subbed_contrast[0].data,self.smooth_kernel)
        self.mean_subbed_contrast=mean_subbed_contrast
        #adjust platescale to match resized array:
        new_plate_scale=plate_scale*n_pix/self.images.im_size[0]
    
        fig_A=plt.figure(figsize=[12,3.5])
        
        ax2= fig_A.add_subplot(132)
        im=ax2.imshow(np.array(np.nan_to_num(mean_subbed_contrast[0].data)),norm=norm,cmap=semi_sym_cm)
        ax2.autoscale(False)
        ax2.axes.get_xaxis().set_visible(False)
        ax2.axes.get_yaxis().set_visible(False)
        #bar2=plt.colorbar(im, format='%.1e')
        # Set some suitable fixed "logarithmic" colourbar tick positions.
        #bar2.set_ticks(tick_levels)
        #ax2.plot(center[0],center[1],'x',markersize=200,color='w')
        #ax2.set_title("<data>-<library>")#, centroid:"+str(np.round(center,2)))
        ax2.annotate("B",[0.05,.05],color=high_color,xycoords='axes fraction')
        #----
        ax1=fig_A.add_subplot(131)
        ax1.annotate("A",[0.05,.05],color=high_color,xycoords='axes fraction')
        ax1.axes.get_xaxis().set_visible(False)
        ax1.axes.get_yaxis().set_visible(False)

        raw_contrast=np.nan_to_num(raw[0].data)/max_bright
        if self.smooth_option == 'after' and self.smooth_kernel is not None: 
            raw_contrast = conv.convolve_fft(raw_contrast,self.smooth_kernel,normalize_kernel=True)
        raw_im=plt.imshow(raw_contrast,norm=norm,cmap=semi_sym_cm)
        ax1.autoscale(False) #http://stackoverflow.com/a/9120929

        #ax1.plot(center[0],center[1],'x', markersize=200,color='w')
    
        #bar1=plt.colorbar(raw_im,format='%.1e')
        #bar1.set_ticks(tick_levels)

        #----
        plt.tight_layout()
    
        ax3=fig_A.add_subplot(133)
        #cb.formatter.set_powerlimits((0, 0)) 
        #https://stackoverflow.com/questions/25362614/force-use-of-scientific-style-for-basemap-colorbar-labels
        #cb.update_ticks()

        #conv=h5py.File(file_hdf5)
        #image_norms=conv['im_arr'][:].sum(axis=1).sum(axis=1)
        reslist=[]
    
        coeff_range=range(1,np.min([self.basis_images.num_files,max_coeff]),10)
        if plot_all_pca:
            for coeff in coeff_range:
                #renorm to contrast units
                res_contrast=np.nan_to_num(PynPoint.PynPlot.plt_res(res,coeff,imtype='mean',returnval=True))*np.median(self.images.im_norm)/max_bright
                if self.smooth_option == 'after' and self.smooth_kernel is not None: 
                    res_contrast = conv.convolve_fft(res_contrast,self.smooth_kernel,normalize_kernel=True)
                resfits=astropy.io.fits.PrimaryHDU(res_contrast)
                resfits.header["smoothing?"]=str( self.smooth_option)+str(type(self.smooth_kernel))
                resfits.header["MAXBRITE"]=max_brightafter
                reslist.append([astropy.io.fits.HDUList(resfits)])
                plt.plot(center[0],center[1],'x',markersize=200,color='k')
        else:
                coeff=coeff_range[-1]
                res_contrast=np.nan_to_num(PynPoint.PynPlot.plt_res(res,coeff,imtype='mean',returnval=True))*np.median(self.images.im_norm)/max_bright
                
                if self.smooth_option == 'after' and self.smooth_kernel is not None: 
                    res_contrast = conv.convolve_fft(res_contrast,self.smooth_kernel,normalize_kernel=True)
                resfits=astropy.io.fits.PrimaryHDU(res_contrast)
                resfits.header["smoothing?"]=str( self.smooth_option)+str(type(self.smooth_kernel))
                resfits.header["MAXBRITE"]=max_bright
                reslist.append([astropy.io.fits.HDUList(resfits)])
                plt.plot(center[0],center[1],'x',markersize=200,color='k')
        
        self.reslist=reslist
        self.resfits=resfits
        #plt.figure(figsize=[5,5])
        #plt.imshow(resfits.data[100:150,100:150])
        #plt.plot(center[0],center[1],'x',markersize=200,color='k')
        #plt.colorbar()
        mean_subbed_contrast[0].header['PIXELSCL']=new_plate_scale
        raw[0].header['PIXELSCL']=new_plate_scale
        plt.savefig(self.image_dir+"/PCA_residual.pdf")
        plt.savefig(self.image_dir+"/PCA_residual.eps")

        #plot histograms:
        plt.figure(figsize=[4,4])
        plt.subplot(211)
        #plt.title("Histogram of Normalized Int")
        plt.ylabel("Pixels")

        plt.hist(raw_contrast.flatten(),bins=10,histtype='stepfilled', label="Raw",log=True,alpha=0.3,color='black',linewidth=.5)
        plt.ylim(1e-1,1e7)
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0),useOffset=False)#


        plt.legend(fontsize=8)

        plt.subplot(212)
        bins=np.arange(-5e-5,5e-5,5e-6)
        plt.hist(mean_subbed_contrast[0].data.flatten(),bins=bins,linewidth=1.5,
             label="Median",log=True,histtype='step',color=high_color)
        plt.hist(reslist[-1][0][0].data.flatten(),bins=bins,
             label="PCA",log=True,color=high_color,alpha=1,histtype='step',linestyle='dotted')
        #https://stackoverflow.com/questions/26693642/histogram-linestyles-in-matplotlib?lq=1
  
        plt.xlabel("Contrast")
        plt.ylabel("Pixels")
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0),useOffset=False)#
        
        plt.ylim(1e-1,1e6)
        plt.legend(fontsize=10)
        plt.tight_layout()
        plt.savefig(self.image_dir+"/subtraction_comparo_histogram.pdf")
        plt.savefig(self.image_dir+"/subtraction_comparo_histogram.eps")

        #plot radial averages:
        radstd_mean_subbed = poppy.utils.radial_profile(mean_subbed_contrast,stddev=True,center=center)#,)#,center=[72,68])#center[::-1])
        #raw_avg_std_minmax =  radial_profile.radial_profile(raw, stddev=True,minmax=True,center=center)#,)# center=[72,68],

        raw_avg = poppy.utils.radial_profile(raw,center=center)
        raw_std = poppy.utils.radial_profile(raw,center=center,stddev=True)

        fig1=plt.figure(figsize=[4,4])
        ax=fig1.gca()
        #sigma_poisson = sqrt(N_e [electrons/exp]*N_exp)/N_exp
        poisson_noise=np.sqrt(raw_avg[1]*np.float(self.images.num_files))/np.float(self.images.num_files)
        
        ax.plot(raw_avg[0],raw_avg[1]/max_bright,'--',
             label='Dark Fringe Intensity',linewidth=2.4)
        '''plt.errorbar(raw_avg_std_minmax[0],np.mean(self.images.im_norm)*raw_avg_std_minmax[2]/max_bright,
                 yerr=[np.mean(self.images.im_norm)*raw_avg_std_minmax[3]/max_bright,
                       np.mean(self.images.im_norm)*raw_avg_std_minmax[4]/max_bright],
             label=r'$1\sigma$ contrast',linewidth=2.4,alpha=.2,color='k')'''
        ax.plot(raw_std[0],raw_std[1]/max_bright,
             label=r'$1\sigma$ Raw Dark Fringe',linewidth=2.4,alpha=.9,color='k')
        ax.plot(radstd_mean_subbed[0],radstd_mean_subbed[1],
             '-.',label='$1\sigma$ Median Subtracted',linewidth=3)
        
        ax.plot(raw_avg[0],poisson_noise/max_bright,'--',
             label=r'$1\sigma$ Poisson Noise',linewidth=4.) #not including photon noise in the PCA Basis.'''
        #plot all the PCA residuals:
        if plot_all_pca:
            for i,res_fits in enumerate(reslist):    
                res_fits[0][0].header['PIXELSCL']=new_plate_scale
                radial_res=poppy.utils.radial_profile(res_fits[0],center=center,stddev=True)#center[::-1])
                ax.plot(radial_res[0],radial_res[1],
                     label=r'$1\sigma$ PCA subtracted',#, n='+str(coeff_range[i]),
                     alpha=1-.8/np.sqrt(coeff_range[i]),linewidth=0.5,color='k')
        else:
            res_fits= reslist[-1]
            res_fits[0][0].header['PIXELSCL']=new_plate_scale
            radial_res=poppy.utils.radial_profile(res_fits[0],center=center,stddev=True)#center[::-1])
            ax.plot(radial_res[0],radial_res[1],
                 label=r'1$\sigma$ PCA subtracted',#, n='+str(coeff_range[-1]),
                 alpha=1-.8/np.sqrt(coeff_range[-1]),linewidth=0.5,color='k')
        #ax.yaxis._set_scale('log10')
        ax.set_yscale('log', nonposy='clip')
        ax.set_ylim([1e-8,1.0])
        #plt.errorbar(std_measured[0],std_measured[1],yerr=std_measured[1],label="measured")
        #ax.plot(np.ones(2),[0,2e-4/8.],linewidth=10,alpha=0.5,color='k')
        ax.grid(b=True, which='major')# color='0.65',linestyle='-')
        ax.set_title("Images: "+ self.image_dir[-15:-1]+",\n Basis:"+self.basis_dir[-15:-1])
        ax.set_xlim([0,35])
        #ax.legend()
        if angular_units == 'arcsecs':
            ax.plot(np.ones(2)*.49,np.arange(2),'--') #plot IWA
            ax.set_xlabel("arcsec")
        elif angular_units == 'lambda/D':
            ax.set_xlabel("$\lambda/D$")
            ax.plot(np.ones(2)*1.7,np.arange(2),'--') #plot IWA    
        ax.set_ylabel("Contrast")
        fig1.savefig(self.image_dir+"PCA_contrast.pdf")
        fig1.savefig(self.image_dir+"PCA_contrast.eps")
    
    
        residual_im=ax3.imshow(resfits.data,norm=norm,cmap=semi_sym_cm)
        ax3.annotate("%2i PCA Bases"%(coeff),[0.05,.85],color=high_color,xycoords='axes fraction')
        ax3.annotate("C",[0.05,.05],color=high_color,xycoords='axes fraction')
        bar3=fig_A.colorbar(residual_im,format="%.1e")#,format="%.1e")
        bar3.set_ticks(tick_levels)
        ax3.axes.get_xaxis().set_visible(False)
        ax3.axes.get_yaxis().set_visible(False)
        
        fig_A.savefig(self.image_dir+"/median_minus_mean.pdf", bbox_inches='tight')
        fig_A.savefig(self.image_dir+"/median_minus_mean.eps", bbox_inches='tight')


        #globals().update(locals()) #DANGER, but so handy,https://stackoverflow.com/questions/11543297/make-all-variables-in-a-python-function-global
        return {'residual':resfits,'radial_res':radial_res,'ax':ax,'center':center}

    
