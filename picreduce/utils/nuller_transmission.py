import numpy as np	

def trans_func(arcsec,wave=6.75e-7,shear=0.15):
    '''
    creates a transmission function for a shearing nuller
    inputs:
        wave : wavelength (meters)
        shear : pupil shear (meters)
    '''
    return 1-np.cos(2*np.pi*shear*np.sin(arcsec/206265.0)/wave)


def trans_2d(detx,dety,PIXELSCL,transpose=False,**kwargs):
    '''
    makes a square transmission function array.

    inputs:
       detx : length of x axis
       dety : length of y axis
       PIXELSCL : detector platescale in arcseconds
       **kwargs : keywords arguments passed to trans_funct
    '''
    trans_array=np.empty([detx])
    center=len(trans_array)/2.0-.5
    for i in range(detx):
        arcsec=(center-i)*PIXELSCL#*inFITS[0].header['PIXELSCL']
        trans_array[i] = trans_func(np.abs(arcsec),**kwargs)
    trans2d= trans_array.repeat(dety).reshape([detx,dety])
    if transpose:
        return trans2d.T
    else:
        return  trans2d


def symm_gauss_2d(x_dim,y_dim,sigma,truncation_radius=None):
    '''an array consisting of a centered  (symmetrical) gaussian distribution, set to zero beyond the (optional) truncation_radius'''
    m_gauss=np.zeros([x_dim,y_dim])
    a = x_dim/2.0# - 0.5
    b = y_dim/2.0# - 0.5
    params=[a,b,sigma]

    print("center x,y, sigma",a,b,sigma)
    for row in range(x_dim):
        for col in range(y_dim):
            r = np.sqrt((col+0.5-a)**2 + (row+0.5-b)**2)
            if truncation_radius is not None:
                if  r < truncation_radius:
                    #print(r,truncation_radius)
                    m_gauss[row,col] = np.exp(-r**2/(2.0*sigma**2))
            else:
                m_gauss[row,col] = np.exp(-r**2/(2.0*sigma**2))


    #m_gauss=m_gauss/m_gauss.sum()
    return (m_gauss,params)

def symm_circle(x_dim,y_dim,sigma):
    '''a symmetrical circle'''
    m_box=np.zeros([x_dim,y_dim])
    a = x_dim/2.0# - 0.5
    b = y_dim/2.0# - 0.5
    params=[a,b,sigma]
    print("center x,y,half width,"+str(params))
    for row in range(x_dim):
        for col in range(y_dim):
            r = np.sqrt((col+0.5-a)**2 + (row+0.5-b)**2)
            if r <sigma:
                m_box[row,col] = 1.0
    #normalize:
    #m_box=m_box/m_box.sum()
    return (m_box,params)
