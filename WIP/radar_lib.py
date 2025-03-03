import baseband
import pylab as pl
import matplotlib.colors as mcolors
from baseband_tasks.fourier import fft_maker
from baseband import vdif
import numpy as np
import astropy.units as u
from scipy.fft import fft,fftfreq
from scipy.signal import correlate,windows,convolve
from astropy.time import Time, TimeDelta
import scipy.interpolate as interpol
from   scipy.io   import  loadmat
import pickle
from tqdm import tqdm
import time
import gc
import imageio
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


################################################
### Helper functions
################################################

### Basic Info

def fhead(fname='',mode='rb',seekto=0):
    fh,ref_mjd = open_file(fname,mode,seekto)
    fh.close()
    
 
### File I/O
def open_file(fname='',mode='rb',seekto=0):
    """
    mode='rb' is in units of frames
    mode='rs' is in units of samples
    https://baseband.readthedocs.io/en/stable/vdif/index.html

    seekto = integer : nframes
             string : isot time string : 2024-10-30T15:15:00.000000000
    """
    fh = vdif.open(fname,mode)
    fin = fh.info()
    print(fin)

    n4k = int(fin['samples_per_frame']/4000)
    fbuf = int(n4k*1000 + 32)
    ## fbuf = 1032  ## 1032 for rtvlba files.  5032 for disk-copied files
    

    print('Samples per frame : %d'%(fin['samples_per_frame']) )
    print('Frame size : %d bytes '%(fbuf))
    print('Sample rate : %s'%(str(fin['sample_rate'])) )
    print('Sample shape : %s'%(str(fin['sample_shape'])))

    print('Each frame is %3.7f sec long'%(fin['samples_per_frame']/fin['sample_rate'].value) )
    
    d2s = 24*60*60

    if type(seekto) == str:
        file_start = fin['start_time']  ## Time object
        seek_time = Time(seekto)  ## Create Time object from isot string

        time_diff = (seek_time.mjd - file_start.mjd)*d2s
        seek_nframes = int( time_diff * fin['sample_rate'].value /fin['samples_per_frame'] )
        seek_loc = int( fbuf * seek_nframes )
        
    else:
        seek_loc = int( fbuf * seekto )

    ## Seek to the chosen location
    print('\nSeek to loc %d (nframes from start = %2.2f)'%(seek_loc, seek_loc/fbuf) )
    fh.seek(seek_loc)
            
    ## Read 1 frame (to get the time) and re-seek.
    #atime = fh.read_frame().header.time
    #fh.seek(seek_loc) ## Go back 1 frame

    atime = read_time(fh)
    
    ref_mjd = atime.mjd

    print('Location tell() %d and time %s (Ref MJD = %3.8f)'%(fh.tell(),atime.isot,ref_mjd) )
    
    return fh, ref_mjd


def read_time(fh=None,vb=False):
    cloc = fh.tell()
    atime = fh.read_frame().header.time
    if vb==True:
        print('Current Location %d  and  Time = %s'%(cloc, atime.isot))
    fh.seek(cloc) ## Go back 1 frame
    return atime

def read_frame(fh=None):
    try:
        frame = fh.read_frameset([0])
        samples = frame.data[:,0,0]  ## N samples per frame
        atime = frame.header0.time ## time of first sample
        return atime, samples.squeeze()
    except Exception as qq:
        print('Exception in read_frame : Caught!'+str(qq))
        return None, None


def read_stream(fh=None, data=None, times=None, ref_mjd=0):
    '''
    Read nframes-worth of samples into a 1D array. Generate timesteps as well. 
    These arrays will later be shaped into a 2D array for either SPEC or DDM processing
    '''
    if fh==None:
        print('Need a valid file handle')
        return

    d2s = 24*60*60

    fin = fh.info()
    fsize = fin['samples_per_frame']
    srate = fin['sample_rate'].value

    ## Check the number of frames that can fit in 'data' 
    nsamples = len(data)
    nframes = int(nsamples/fsize)
    start_time = times[0] * 1e6  # usec
    end_time = times[-1] * 1e6
#    print("Reading nframes = %d"%(nframes) )

    now_time = start_time
    
    while now_time < end_time: 
        atime, adata = read_frame(fh)
        ##print(atime,adata.shape)

        ## Found a sample outside of the entire time range. Stop reading.
        ## This can happen if there are dropped frames at the end of the array
        if atime != None and  (atime.mjd - ref_mjd)*d2s * 1e6 > end_time:
            break

        ## If there is a valid time, fill the data. 
        if atime != None:
            now_time = (atime.mjd - ref_mjd)*d2s * 1e6 ## (usec)
            now_index = int( now_time / (1e6/srate) ) 
            ##print('Sample time : %3.7f usec -- matches index %d the time is %3.7f usec'%(now_time,now_index,times[now_index]*1e6))
            lost_data=0
            if now_index+fsize>nsamples:
                lost_data = ((now_index + fsize) - nsamples)
            data[now_index : now_index + fsize] = adata[0:fsize-lost_data]

    return


def read_frame_set_2(fh=None, data=None, nframes=1,fsize=4000,
                     fix_drops=True,vb=True,frame_time=None,ref_mjd=0.0):
    atime = None
    fcnt = 0
    d2s = 24*60*60

    pbar = tqdm(desc='Reading',total=nframes)
    
    prev_time=None
    this_time = None
    while fcnt < nframes:
        atime,oneframe = read_frame(fh)
        pbar.update(1)

        this_time = atime.mjd - ref_mjd

        if prev_time != None:
            tdiff =  ( this_time - prev_time )*d2s/(frame_time) ## For adjacent frames, this wil be 1.0 ( some error bar? )
            if tdiff<0.0:
                ### TODO : Handle by going back and filling - there must've been a gap just before! Or DROP it ! 
                print('Frame %d (pri=%d, off=%d) : tdiff = %3.7f --> Out-of-order frame ! Dropping it. '%(fcnt,int(fcnt/32),np.mod(fcnt,32),tdiff))
                continue
            if int(np.round(np.abs(tdiff))) > 1 :  ## It can be only 1,2,3... 
                nframes_dropped = int(np.round(np.abs(tdiff))) - 1
                if vb==True:
                    print("Frame %d (pri=%d, off=%d) --   Prev time : %3.7f s This time : %3.7f s  --- tdiff : %3.7f  ==  %d frames"%(fcnt,int(fcnt/32),np.mod(fcnt,32) , prev_time*d2s, this_time*d2s, tdiff, nframes_dropped) )

                if fix_drops==True:
                    for i in range(nframes_dropped):
                        if fcnt < nframes:
                            data[fcnt*fsize : (fcnt+1)*fsize] = 0.0
                            fcnt = fcnt+1
                    

        prev_time = this_time

        if fcnt < nframes:
            if atime == None: ## Zero-fill
                data[fcnt*fsize : (fcnt+1)*fsize] = 0.0
            else: ## Fill with data
                data[fcnt*fsize : (fcnt+1)*fsize] = oneframe
        else:
            print('Frame overflows the end of the array (due to zero-fills before it)')
        
        fcnt = fcnt+1
        
    pbar.close()
    return atime


def read_frame_set(fh=None, nframes=1,fsize=4000,fix_drops=True,vb=True,frame_time=None):
    data = np.zeros(nframes*fsize);### don't keep allocating each time. reuse.
    atime = None
    fcnt = 0
    
    prev_time=None
    this_time = None
    while fcnt < nframes:
        atime,oneframe = read_frame(fh)

        this_time = atime.mjd

        if prev_time != None:
            if np.abs( this_time - prev_time ) > 1.5*frame_time/(24*60*60):
                nframes_dropped = int( ((this_time - prev_time)*(24*60*60)/(frame_time)) ) - 1
                if vb==True:
                    print("Frame %d --   Prev time : %3.7f  This time : %3.7f  --- Diff : %d frames"%(fcnt, prev_time, this_time,nframes_dropped) )

                if fix_drops==True:
                    for i in range(nframes_dropped):
                        data[fcnt*fsize : (fcnt+1)*fsize] = 0.0
                        fcnt = fcnt+1
                        if fcnt >= nframes:
                            break
                    

        prev_time = this_time

        if fcnt < nframes:
            if atime == None: ## Zero-fill
                data[fcnt*fsize : (fcnt+1)*fsize] = 0.0
            else: ## Fill with data
                data[fcnt*fsize : (fcnt+1)*fsize] = oneframe
        else:
            print('Dropped frames cross PRP boundary')
        
        fcnt = fcnt+1
    return atime,data




def get_loc(nowt, srate, time0):
    """
    Calculate the index into times, where nowt is.  Units are seconds, referenced to ref_mjd 
    """

    #dt = times[2]*1e6 - times[1]*1e6

    dt = (1.0/srate)*1e6

    return int(  )


## Apply time varying dopper shift, based on "fint", and interpolation function. 
def doppler_shift_frame_set_2(data=None, tim=None, fint=None, vb=True, ref_mjd=None,offset=0.0):
    """
    Apply Doppler corrections to the time series, prior to the FFTs
    ref_amp*np.exp((0+1j)*(2*np.pi*doppler*tim)
    
    data : 1D array of time series, before the FFT
    end_mjd : MJD of the end of the frame set
    fint : Doppler shift interp1d function (from scipy.interpolate)

    """
    if fint == None:
        print('Error. Empty Doppler model')
        return
    
    d2s = 24*60*60    
    ref_dop = fint( ref_mjd ) + offset

#    if vb==True:
#    Ttim1 = Time(tim[0]/d2s + ref_mjd ,format='mjd')
#    Ttim2 = Time(tim[len(tim)-1]/d2s + ref_mjd, format='mjd')
#    print("MJDrange : %s - %s  --> Doppler offsets : %3.6f -- %3.6f Hz"%(Ttim1.isot,Ttim2.isot,dop[0],dop[len(dop)-1]))

    psize = 4000 * 32 ### Pick some nsamples to break up the long list.

    nsamples = len(data)
    
    for pbeg in tqdm(range(0,nsamples,psize)):
        pend = pbeg + psize
        if pend > nsamples:
            pend = nsamples
    
        dop = -0.5 * ( fint( (tim[pbeg:pend]/d2s + ref_mjd) ) - ref_dop  )

        data[pbeg:pend] = data[pbeg:pend] * np.exp((0+1j)*(2*np.pi*dop*tim[pbeg:pend]))

    return


## Apply time varying delay correction, based on "tint", and interpolation function. 
def delay_shift_frame_set_2(data=None, tim=None, tint=None, vb=True,ref_mjd=None, offset=0.0):
    """
    Apply Delay corrections to the time series, prior to the FFTs
     
    data : 1D array of time series, before the FFT
    end_mjd : MJD of the end of the frame set
    tint : Delay shift interp1d function (from scipy.interpolate)

    """
    if tint == None:
        print('Error. Empty Delay model')
        return
    
    d2s = 24*60*60

    ## delr is the list of 'time delay', interpolated onto the data timesteps. 
    ref_del = tint( ref_mjd ) + offset

#    if vb==True:
#        Ttim1 = Time(tim[0]/d2s + ref_mjd ,format='mjd')
#        Ttim2 = Time(tim[len(tim)-1]/d2s + ref_mjd, format='mjd')
#        print("MJDrange : %s - %s  --> Delay offsets :  %3.6f -- %3.6f microsec"%(Ttim1.isot,Ttim2.isot,delr[0]*1e+6,delr[len(delr)-1]*1e+6))

    psize = 4000 * 32  ### Pick some nsamples to break up the long list.
    pbuf = 4000 * 16   ### Amount to add to each side, for the dat/tim dint calculation

    nsamples = len(data)
    
    for pbeg in tqdm(range(0,nsamples,psize)):
        pend = pbeg + psize
        if pend > nsamples:
            pend = nsamples

        delr = +1.0 * ( tint( (tim[pbeg:pend]/d2s + ref_mjd) ) - ref_del  )      

        ## delayed time is tim + delr
        del_tim = tim[pbeg:pend] + delr
        
        ## Construct interpolation function for data at original timesteps. Use a larger span than pbeg:pend to cover delr
        ibeg = pbeg - pbuf
        if ibeg < 0:
            ibeg=0
        iend = pend + pbuf
        if iend > nsamples:
            iend = nsamples
        dint = interpol.interp1d(tim[ibeg:iend],data[ibeg:iend], fill_value=0.0,bounds_error=False)

        ##dint = interpol.interp1d(tim,data,fill_value='extrapolate',bounds_error=False)
        ##dint = interpol.CubicSpline(tim, data, extrapolate=True)
        
        ## Interpolate data onto delayed timesteps.
        data[pbeg:pend] = dint(del_tim)
        #print('Max del_data is %3.5f at %d'% ( np.max(np.abs(del_data)),  np.argmax(np.abs(del_data)) ) )

    return 


## Apply time varying dopper shift, based on "fint", and interpolation function. 
def doppler_shift_frame_set(data=None, tim=None, fint=None, vb=True, ref_mjd=None):
    """
    Apply Doppler corrections to the time series, prior to the FFTs
    ref_amp*np.exp((0+1j)*(2*np.pi*doppler*tim)
    
    data : 1D array of time series, before the FFT
    end_mjd : MJD of the end of the frame set
    fint : Doppler shift interp1d function (from scipy.interpolate)

    """
    if fint == None:
        print('Error. Empty Doppler model')
        return
    
    d2s = 24*60*60
    
    ref_dop = fint( ref_mjd )
    dop = -0.5 * ( fint( (tim/d2s + ref_mjd) ) - ref_dop  )

    if vb==True:
        Ttim1 = Time(tim[0]/d2s + ref_mjd ,format='mjd')
        Ttim2 = Time(tim[len(tim)-1]/d2s + ref_mjd, format='mjd')
        print("MJDrange : %s - %s  --> Doppler offsets : %3.6f -- %3.6f Hz"%(Ttim1.isot,Ttim2.isot,dop[0],dop[len(dop)-1]))

    data[:] = data[:] * np.exp((0+1j)*(2*np.pi*dop*tim))

    return


## Apply time varying delay correction, based on "tint", and interpolation function. 
def delay_shift_frame_set(data=None, tim=None, tint=None, vb=True,ref_mjd=None):
    """
    Apply Delay corrections to the time series, prior to the FFTs
     
    data : 1D array of time series, before the FFT
    end_mjd : MJD of the end of the frame set
    tint : Delay shift interp1d function (from scipy.interpolate)

    """
    if tint == None:
        print('Error. Empty Delay model')
        return
    
    d2s = 24*60*60

    ## delr is the list of 'time delay', interpolated onto the data timesteps. 
    ref_del = tint( ref_mjd )
    delr = +1.0 * ( tint( (tim/d2s + ref_mjd) ) - ref_del  )      
        
    if vb==True:
        Ttim1 = Time(tim[0]/d2s + ref_mjd ,format='mjd')
        Ttim2 = Time(tim[len(tim)-1]/d2s + ref_mjd, format='mjd')
        print("MJDrange : %s - %s  --> Delay offsets :  %3.6f -- %3.6f microsec"%(Ttim1.isot,Ttim2.isot,delr[0]*1e+6,delr[len(delr)-1]*1e+6))


    ## delayed time is tim + delr
    del_tim = tim + delr
    ## Construct interpolation function for data at original timesteps.
    if len(tim) != len(data):
        print("HEY !!", len(tim), len(data))
    dint = interpol.interp1d(tim,data, fill_value=0.0,bounds_error=False)
    ##dint = interpol.interp1d(tim,data,fill_value='extrapolate',bounds_error=False)
    ##dint = interpol.CubicSpline(tim, data, extrapolate=True)

    ## Interpolate data onto delayed timesteps.
    data[:] = dint(del_tim)
    #print('Max del_data is %3.5f at %d'% ( np.max(np.abs(del_data)),  np.argmax(np.abs(del_data)) ) )

    return 
    
## Other helper methods. 
    
def cut_freqs(freqs, frange=None):
    """
    Get the start and stop freq chan indices
    """
    if frange == None:
        return 0,len(freqs)

    if len(frange) != 2:
        print("Cannot sub-select a spectrum. Need freq list of len 2")
        return 0,len(freqs)

    start = np.argmin( np.abs(freqs - frange[0]))
    stop = np.argmin( np.abs(freqs - frange[1]))

    return start,stop


def read_doppler_and_delay(fname='',dat=False):
    fp = open(fname)
    alines = fp.readlines()
    fp.close()

    tstr = []
    dops = []
    dels = []
    
    for aline in alines:
        if aline[0]=='#':
            continue
        afields = aline.split()
        tstr.append( afields[0]+' ' +afields[1] )
        dops.append( float(afields[7]) )
        dels.append( float(afields[6]) )

    tmjd = Time(tstr).mjd

    fint = interpol.interp1d(tmjd,dops)
    tint = interpol.interp1d(tmjd,dels)

    if dat==True:
        return tmjd, dops, fint, dels, tint
    else:
        return fint, tint


def read_waveform(wname=''):
    matrixVar = loadmat( wname )
    return matrixVar

#In [12]: fft,freqs = setup_fft(int(128e+6),32e+3)
#    ...: aa = fft(qq['Y'][0,:])
#    ...: pl.clf();pl.plot(freqs,np.abs(aa))




    
### Add smarts in this to do the 'mean' in only one dimension, if the input/output dims match
def bin_2d(arr,np0, np1, fstartfrac=0.0): 
    step0 = int(arr.shape[0]/np0)
    step1 = int(arr.shape[1]/np1)
    binarr = np.zeros((np0,np1))
    for ii in range(0,np0):
        for jj in range(0,np1):
            binarr[ii,jj] = (np.mean(np.abs(arr[ii*step0:(ii*step0)+step0, jj*step1:(jj*step1)+step1])))

##            binarr[ii,jj] = np.abs(np.mean(arr[ii*step0:(ii*step0)+step0, jj*step1:(jj*step1)+step1]))

    binarr = binarr[:,int(fstartfrac*np1):int((1.0-fstartfrac)*np1)]
    return binarr


def low_pass_filter(data, cavg_factor):
    """
    datastack = [ ndelays, ndopps ]
    Convolve with a smoothing function, along the Doppler axis
    Decimate by the cavg_factor - return an array indexed by strides. 
    """
    ndopps = data.shape[1]
    ndelays = data.shape[0]

    #win = windows.hann(cavg_factor*10)
    win = windows.kaiser(M=cavg_factor*4, beta=100)
    norm = np.sum(win)
    
    print('Convolve along the slow-time axis with a smoothing function (area = %3.2f)'%(norm))
    for ii in tqdm(range(0,ndelays)):
        data[ii,:] = convolve(data[ii,:], win, mode='same')/norm

    return data[:,0:ndopps:cavg_factor].transpose()


def resample_2d_avg(array, new_shape, fstartfrac=0.0,coherent=False):
    """Resamples a 2D array to a new shape by averaging.

    Args:
        array: The 2D numpy array to resample.
        new_shape: A tuple representing the desired shape of the resampled array (rows, cols).

    Returns:
        A new 2D numpy array with the specified shape, resampled by averaging.
    """

    old_rows, old_cols = array.shape
    new_rows, new_cols = new_shape

    row_ratio = old_rows // new_rows
    col_ratio = old_cols // new_cols

    if coherent==False: ## incoherent averaging
        # Reshape the array to have extra dimensions for averaging
        reshaped_array = array.reshape(new_rows, row_ratio, new_cols, col_ratio)
        # Average along the extra dimensions -- this allocates memory too.... 
        resampled_array = np.mean(np.abs(reshaped_array),axis=(1, 3))
    else:  ## coherent averaging
        # Reshape the array to have extra dimensions for averaging
        reshaped_array = array.reshape(new_rows, row_ratio, new_cols, col_ratio)
        # Average along the extra dimensions
        resampled_array = np.mean(reshaped_array,axis=(1, 3))

        
    return resampled_array[:,int(fstartfrac*new_cols):int((1.0-fstartfrac)*new_cols)]

## Example usage
#original_array = np.arange(24).reshape(4, 6)
#new_shape = (2, 3)
#resampled_array = resample_2d_avg(original_array, new_shape)
#
#print("Original array:\n", original_array)
#print("Resampled array:\n", resampled_array)
#



def disp_binned_ddm(pfile,np0,np1,frac):
    print('Unpickling..')
    with open(pfile,'rb') as f:
        arr = pickle.load(f)

    print('Binning for plotting : ', arr.shape)
#    bin2plot = bin_2d(arr,np0,np1,frac)

    bin2plot = resample_2d_avg(arr,(np0,np1))
    
    pl.figure(2)
    pl.ion()
    pl.clf()

    im1 = pl.imshow(np.abs(bin2plot), cmap='viridis',
                    aspect='auto', origin='lower') 
                  #  vmin = 100, vmax=3e+3)#,
                    #extent=[0,new_nfreqs,0,nsamples])
    #                extent=[0,new_nfreqs,sample_times[0],sample_times[-1]])
    #im1.axes.set_yticks(range(len(dtimes))[0:-1:int(len(dtimes)/10)])
    #im1.axes.set_yticklabels(dtimes[0:-1:int(len(dtimes)/10)])
    pl.xlabel('Frequency (channel id)')
    pl.ylabel('Time/Delay')
    pl.colorbar()
    pl.title(pfile)
    pl.show()
    return bin2plot



def disp_raster(data=None, pnum=1, title='',pname='',frange=None,xaxis='',
                nplot_delay=None,nplot_doppler=None,
                slow_time_inc=0.0,slow_time_span=0.0):
    """
        datastack = [ ndelays, ndopps ]
        xaxis='dopp' -- Frequency on the x axis. 
        xaxis= 'time' -- Slow Time on the x axis.
    """
    print('Binning %s for plotting'%(title))
    shp = data.shape

    if nplot_delay==None or nplot_doppler==None:
        nplot_delay = shp[1] ## This needs to be an integer divisor of 128000
        nplot_doppler = shp[1]

    #bin2plot = bin_2d(data,nplot_delay,nplot_doppler)

    if nplot_delay < shp[0] or nplot_doppler < shp[1]:
        bin2plot = resample_2d_avg(data,(nplot_delay,nplot_doppler))
    else:
        bin2plot = data
    
    pl.figure(pnum,figsize=(8,6))
    pl.clf()
 
    im1 = pl.imshow(np.abs(bin2plot), cmap='viridis',
                    aspect='auto', origin='lower')
#                        vmin = 100, vmax=3e+3)#,
                    #extent=[0,new_nfreqs,0,nsamples])
    #                extent=[0,new_nfreqs,sample_times[0],sample_times[-1]])
    #im1.axes.set_yticks(range(len(dtimes))[0:-1:int(len(dtimes)/10)])
    #im1.axes.set_yticklabels(dtimes[0:-1:int(len(dtimes)/10)])
    if xaxis=='time':
        pl.xlabel('Slow Time -> 0 to %3.2f sec'%(slow_time_span) )
    if xaxis=='dopp':
        pl.xlabel('Doppler Frequency => 0 to %3.2f Hz'%(1.0/(slow_time_inc)) )
    pl.ylabel('Time/Delay')
    pl.colorbar()
    pl.title(title)
    pl.ion()
    pl.show()

    if frange!=None:
        x1 = frange[0] * nplot_doppler
        x2 = frange[1] * nplot_doppler
        pl.xlim([x1,x2])

    if pname != '':
        pl.savefig(pname+'.png')
        print('Pickling DDM array of shape ', bin2plot.shape)
        with open(pname+'.pkl','wb') as f:
            pickle.dump(bin2plot,f)


    return



def average_into_steps2(arr, step_size):
    # Check if the array can be evenly divided into steps
    if len(arr) % step_size != 0:
        raise ValueError("Array length must be divisible by step size.")

    npts = int(len(arr)/step_size)
    avg = np.zeros(npts,dtype='complex')
    for ii in range(0,npts):
        avg[ii] = np.mean((arr[ii*step_size : (ii*step_size)+step_size]))

    return avg


def make_waveform(nsamples,sample_times,start_freq,end_freq):
    sample_freqs = np.arange(start_freq, end_freq, (end_freq - start_freq)/nsamples)

    if len(sample_freqs) != len(sample_times):
        print('Check array lengths in make_waveform()')
        return None
    
    wform = np.exp((0+1j)*(np.pi*sample_freqs*sample_times))
    
    print('\nWaveform Time range : %3.5f - %3.5f seconds \nWaveform Freq range : %3.3f - %3.3f MHz '%(sample_times[0],sample_times[-1],sample_freqs[0]/1e+6,sample_freqs[-1]/1e+6) )

    return wform


def check_waveform(waveform,bw):
    fft1,freqs1 = setup_fft( len(waveform), bw ) ## BW in MHz
    f_wf = do_fft(fft1, waveform)
    nf = len(freqs1)
    pl.clf()
    pl.ion()
    pl.plot(freqs1[0:int(nf/2)], np.abs(f_wf[0:int(nf/2)]))
    pl.show()




### FFT math

def setup_fft(ndata,bw):
    """
    bw is in MHz. 
    """
    fft_maker.set('numpy')
    fft = fft_maker(shape=(ndata,),dtype='complex128', #'float64',
                    direction= 'forward',
                    ortho=True,
                    sample_rate=2*bw*u.MHz)
    print(fft)
    freqs = fft.frequency.value
    return fft,freqs

def do_fft(fft,data):
    return fft(data)


def take_fft(datastack,fft):
    '''
    Do FFT along the slow-time axis and return
    datastack = [ ndelays, ndopps ]
    '''
    shp = datastack.shape
    print('Take FFTs along the slow-time axis (%d points) for %d Delay bins'%(shp[1],shp[0]))

    fftlen = shp[1] ## ndopps
    for i in tqdm(range(0, shp[0])):
        #ms = np.mean(datastack[i,:])
        datastack[i,:] = np.abs(do_fft(fft,datastack[i,:]))

def remove_dc(datastack):
    '''
    Remove DC along the slow-time axis and return
    datastack = [ ndelays, ndopps ]
    ### TODO -- make this with a zero sum filter, to remove time varying DC
    '''
    shp = datastack.shape
    print('Remove DC along the slow-time axis (%d points) for %d Delay bins'%(shp[1],shp[0]))

    for i in tqdm(range(0, shp[0])):
        ms = np.mean(datastack[i,:])
        datastack[i,:] = datastack[i,:]-ms



def take_fft_zpad(datastack,fft,zpad):
    '''
    Create FFT maker
    Do FFT along the long-time axis and return
    Array size to increase by factor of zpad for zero padding
    '''
    shp = datastack.shape
    nsamples = shp[0]
    arr = np.zeros(shp[1]*zpad,dtype='complex')
    fftlen = len(arr)
    for i in range(0, nsamples):
#        datastack[i,:] = np.abs(do_fft(fft,datastack[i,:]))
        arr[int(fftlen/2-shp[1]/2):int(fftlen/2+shp[1]/2)] = datastack[i,:]
#        farr = np.abs(do_fft(fft,datastack[i,:]))
        farr = (do_fft(fft,arr))
        datastack[i,:] = farr[int(fftlen/2-shp[1]/2):int(fftlen/2+shp[1]/2)]

        if np.mod(i, int(nsamples/10)) == 0:
            print('FFT at delay %d/%d'%(i,nsamples))

    return datastack




def run_matched_filter(datastack,wform):
    """
    datastack = [ ndopps, ndelays ]
    """
    shp = datastack.shape
    print('Run a matched filter along the fast-time axis (%d points) for %d PRIs or Doppler bins'%(shp[1],shp[0]))

    
    for pri in tqdm(range(0,shp[0])):
        datastack[pri,:] = correlate(datastack[pri,:], wform, mode='same')


