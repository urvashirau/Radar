import baseband
import pylab as pl
from baseband_tasks.fourier import fft_maker
from baseband import vdif
import numpy as np
import astropy.units as u
from scipy.fft import fft,fftfreq
from scipy.signal import correlate
from astropy.time import Time
import scipy.interpolate as interpol
from   scipy.io   import  loadmat
import pickle

ref_mjd = None

################################################
### Helper functions
################################################

### Basic Info

def fhead(fname='',mode='rb',seekto=0):
    fh = open_file(fname,mode,seekto)
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
    global ref_mjd

    
    fh = vdif.open(fname,mode)
    fin = fh.info()
    print(fin)

    n4k = int(fin['samples_per_frame']/4000)
    fbuf = int(n4k*1000 + 32)
    ###fbuf = 1032  ## 1032 for rtvlba files.  5032 for disk-copied files

    print('Samples per frame : %d'%(fin['samples_per_frame']) )
    print('Frame size : %d '%(fbuf))
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
    atime = fh.read_frame().header.time
    fh.seek(seek_loc) ## Go back 1 frame
    print('Location tell() %d and time %s (MJD %3.8f)'%(fh.tell(),atime.isot,atime.mjd) )

    ref_mjd = atime.mjd
    
    return fh


def read_frame(fh=None):
    global ref_mjd
    try:
        frame = fh.read_frameset([0])
        samples = frame.data[:,0,0]  ## N samples per frame
        atime = frame.header0.time ## time of first sample
        if ref_mjd == None:
            ref_mjd = atime.mjd   ### Time of the first sample in the file.
        return atime, samples.squeeze()
    except Exception as qq:
        print('Exception in read_frame : Caught!'+str(qq))
        return None, None



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



## Apply time varying dopper shift, based on "fint", and interpolation function. 
def doppler_shift_frame_set(data, tim, fint=None, vb=True):
    """
    Apply Doppler corrections to the time series, prior to the FFTs
    ref_amp*np.exp((0+1j)*(2*np.pi*doppler*tim)
    
    data : 1D array of time series, before the FFT
    end_mjd : MJD of the end of the frame set
    fint : Doppler shift interp1d function (from scipy.interpolate)

    """
    global ref_mjd
    
    d2s = 24*60*60
    
    if fint != None:
        ref_dop = fint( ref_mjd )
        dop = -0.5 * ( fint( (tim/d2s + ref_mjd) ) - ref_dop  )
    else:
        ref_dop = 0.0
        dop = np.zeros( len(tim) )

    if vb==True:
        Ttim1 = Time(tim[0]/d2s + ref_mjd ,format='mjd')
        Ttim2 = Time(tim[len(tim)-1]/d2s + ref_mjd, format='mjd')
        print("MJDrange : %s - %s  --> Start and End dop is %3.6f -- %3.6f Hz"%(Ttim1.isot,Ttim2.isot,dop[0],dop[len(dop)-1]))

    if fint != None:
        sdata = data * np.exp((0+1j)*(2*np.pi*dop*tim))
        return sdata
    else:
        return data


## Apply time varying delay correction, based on "tint", and interpolation function. 
def delay_shift_frame_set(data, tim, tint=None, vb=True):
    """
    Apply Delay corrections to the time series, prior to the FFTs
     
    data : 1D array of time series, before the FFT
    end_mjd : MJD of the end of the frame set
    tint : Delay shift interp1d function (from scipy.interpolate)

    """
    global ref_mjd
    
    d2s = 24*60*60

    ## delr is the list of 'time delay', interpolated onto the data timesteps. 
    if tint != None:
        ref_del = tint( ref_mjd )
        delr = +1.0 * ( tint( (tim/d2s + ref_mjd) ) - ref_del  )
    else:
        ref_del = 0.0
        delr = np.zeros( len(tim) )       
        
    if vb==True:
        Ttim1 = Time(tim[0]/d2s + ref_mjd ,format='mjd')
        Ttim2 = Time(tim[len(tim)-1]/d2s + ref_mjd, format='mjd')
        print("MJDrange : %s - %s  --> Start and End Delay is %3.6f -- %3.6f microsec"%(Ttim1.isot,Ttim2.isot,delr[0]*1e+6,delr[len(delr)-1]*1e+6))

    if tint != None:
        ## delayed time is tim + delr
        del_tim = tim + delr
        ## Construct interpolation function for data at original timesteps.
        if len(tim) != len(data):
            print("HEY !!", len(tim), len(data))
        dint = interpol.interp1d(tim,data, fill_value=0.0,bounds_error=False)
        ##dint = interpol.interp1d(tim,data,fill_value='extrapolate',bounds_error=False)
        ##dint = interpol.CubicSpline(tim, data, extrapolate=True)
        ## Interpolate data onto delayed timesteps.
        del_data = dint(del_tim)
        #print('Max del_data is %3.5f at %d'% ( np.max(np.abs(del_data)),  np.argmax(np.abs(del_data)) ) )
        return del_data
    else:
        return data


    
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

    fint = interpol.interp1d(tmjd,dops, fill_value=0.0,bounds_error=False)
    tint = interpol.interp1d(tmjd,dels, fill_value=0.0,bounds_error=False)

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
def bin_2d(arr,np0, np1, fstartfrac=0.1): 
    step0 = int(arr.shape[0]/np0)
    step1 = int(arr.shape[1]/np1)
    binarr = np.zeros((np0,np1))
    for ii in range(0,np0):
        for jj in range(0,np1):
            binarr[ii,jj] = (np.mean(np.abs(arr[ii*step0:(ii*step0)+step0, jj*step1:(jj*step1)+step1])))

    binarr = binarr[:,int(fstartfrac*np1):int((1.0-fstartfrac)*np1)]
    return binarr

def disp_binned_ddm(pfile,np0,np1,frac):
    print('Unpickling..')
    with open(pfile,'rb') as f:
        arr = pickle.load(f)

    print('Binning for plotting : ', arr.shape)
    bin2plot = bin_2d(arr,np0,np1,frac)

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


def disp_raster(data=None, pnum=1, title=''):
    """
        datastack = [ ndelays, ndopps ]
    """
    print('Binning %s for plotting'%(title))
    shp = data.shape
    nplot_delay = shp[1] ## This needs to be an integer divisor of 128000
    nplot_doppler = shp[1]
    bin2plot = bin_2d(data,nplot_delay,nplot_doppler) 
    
    pl.figure(pnum,figsize=(8,6))
    pl.clf()
 
    im1 = pl.imshow(np.abs(bin2plot), cmap='viridis',
                    aspect='auto', origin='lower')
#                        vmin = 100, vmax=3e+3)#,
                    #extent=[0,new_nfreqs,0,nsamples])
    #                extent=[0,new_nfreqs,sample_times[0],sample_times[-1]])
    #im1.axes.set_yticks(range(len(dtimes))[0:-1:int(len(dtimes)/10)])
    #im1.axes.set_yticklabels(dtimes[0:-1:int(len(dtimes)/10)])
    pl.xlabel('Frequency (channel id)')
    pl.ylabel('Time/Delay')
    pl.colorbar()
    pl.title(title)
    pl.ion()
    pl.show()

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

    print(len(sample_freqs))
    print(len(sample_times))
    
    wform = np.exp((0+1j)*(np.pi*sample_freqs*sample_times))
    
    print('Time range : %3.5f - %3.5f    Freq range : %3.5f - %3.5f MHz '%(sample_times[0],sample_times[-1],sample_freqs[0]/1e+6,sample_freqs[-1]/1e+6) )

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


### Without zero padding....
def take_fft(datastack,fft,zpad):
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



def make_ddm(fname='',mode='rb',
             nframes=1, npri=1,
             #bw=32.0,
             seekto=1032*0,
             frange=[15.0,16.0], dodop='',
             pname='ddm_out',vb=True,
             focus_dop=True, focus_del=True,fix_drops=True,
             zpad=1):
    """
    frange : [start,end] in units of MHz. This is used only for the plot range to display. 
                  None is the full range. 
    dodop : File name from osod.  Empty string means 'no doppler correction'.
    """
    global ref_mjd
    fh = open_file(fname,mode,seekto)
    fsize = fh.info()['samples_per_frame']
    srate = fh.info()['sample_rate'].value

    nsamples = nframes*fsize
    sample_times = np.arange(0,nsamples/srate,1/srate) ## seconds ( later convert to distance ? )
    if len(sample_times)>nsamples:
        sample_times = sample_times[0:nsamples]

    start_freq = 1e+6 ##1e+6 #Hz
    end_freq = 31e+6 ##31e+6 #Hz
    wform = make_waveform(nsamples,sample_times,start_freq,end_freq)  

    #check_waveform(wform, bw=30.0) ## (end_freq-start_freq)/1e+6)
    #return
    
    nchans = npri
    ### t_pri is the effective 'sample rate' which will then translate to bandwidth for the FFT.
    ### For now, call it 'bw'... automate this later.
    bw = ( 0.5/(nsamples/srate) ) / 1e+6  ## MHz

    
    fft,freqs1 = setup_fft(zpad*nchans,zpad*bw)
    freqs1 = freqs1 * 1e+6 ## to Hz.
    fstart=0 #int(0.0*len(freqs1))
    fstop=int(len(freqs1)/2)-1
    new_nfreqs = fstop - fstart
    chanres = np.abs(freqs1[1]-freqs1[0])
    
    print('\nPRI length is %3.7f sec (%d samples)\n'%(nsamples/srate, nsamples) )
    print('\nNchans for fft = %d (zero padding factor %3.2f)\nChan res = %3.7f Hz'%(len(freqs1),zpad, chanres))
    print('\nFreqs : %3.6f - %3.6f Hz'%(freqs1[fstart],freqs1[fstop]) )

    #print(freqs1)
    
    ## Setup Doppler correction
    a,b = read_frame(fh)

    fint = None
    if dodop != '':
        fint, tint = read_doppler_and_delay(dodop)
        ref_dop = fint(ref_mjd)
        print("\nReference Doppler Shift : %3.6f Hz  ( %3.6f MHz )\n"%(ref_dop, ref_dop/1e+6) )
        ref_del = tint(ref_mjd)
        print("\nReference Delay Shift : %3.6f sec  ( %3.6f microsec )\n"%(ref_del, ref_del*1e+6) )


    dstack1, dtimes, dataraw = stack_pri_and_match_waveform(fh,nframes,fsize,npri,fint,tint,vb,wform, focus_dop, focus_del,fix_drops)
    print(dstack1.shape)

    disp_raster(dataraw, pnum=1,title='Raw Data')
    
    disp_raster(dstack1, pnum=2,title='After Waveform matched filter')
    
    dstack2 = take_fft(dstack1,fft,zpad)

    print(dstack2.shape)

    disp_raster(dstack2, pnum=3,title='After Doppler FFT') 
    
    print('Pickling')
    with open(pname+'.pkl','wb') as f:
        pickle.dump(dstack2,f)


    return dstack1, dstack2





def stack_pri_and_match_waveform(fh,nframes,fsize,npri,fint, tint,vb, waveform, focus_dop, focus_del,fix_drops):
    global ref_mjd
    stepsize = 1
    
    datastack = np.zeros( (int(nframes*fsize/stepsize), npri) ,dtype='complex' )
    dataraw = np.zeros( (int(nframes*fsize/stepsize), npri) ,dtype='float' )
    datatimes = ['' for _ in range(npri)]

    fin = fh.info()
    frame_time = fin['samples_per_frame']/fin['sample_rate'].value  ## time range per frame. 

    atime = None
    prev_time = None
    for pri in range(0,npri):
        atime, data = read_frame_set(fh, nframes,fsize,fix_drops,vb=False,frame_time=frame_time)
        
        if atime is None:
            datastack[:,pri].fill(0.0)
            datatimes[pri] = '-------'

        else:
            ### Calculate a time array
            d2s = 24*60*60
            diff_mjd = (atime.mjd - ref_mjd)*d2s  ## seconds. 
            tim = np.arange( diff_mjd   , diff_mjd + (len(data)*(1/64e+06)) , 1/64e+06 )  ##### HARD CODED for 64 MHz sampling.
            #tim = tim + ref_mjd*d2s.

            if (len(tim) > len(data)):
                dd = len(tim) - len(data)
                tim = tim[dd:len(tim)]

#            print("------", len(tim), len(data))


            this_time = atime.mjd
            
            if prev_time != None :
                if np.abs(this_time - prev_time) > (0.002 + 0.5*(frame_time))/(24*60*60) :
                    nframe_offset = int( ((this_time - prev_time)*(24*60*60)/(frame_time))-32 )
                    print('Offset at PRI : %d  Prev time : %3.8f mjd  This time : %3.8f mjd   Diff : %3.8f frames'%(pri,prev_time, this_time,nframe_offset))

                    if fix_drops==True:
                        ## Roll the data array to match this starting time, and pre-fill with zeros.
                        data = np.roll( data, fsize * nframe_offset)
                        data[: fsize * nframe_offset ] = 0.0
                        ## Re-calc the time array to start from the new (correct) PRP start time.
                        diff_mjd = diff_mjd - nframe_offset * frame_time
                        tim = np.arange( diff_mjd   , diff_mjd + (len(data)*(1/64e+06)) , 1/64e+06 )
                        if (len(tim) > len(data)):
                            dd = len(tim) - len(data)
                            tim = tim[dd:len(tim)]

                        
            prev_time = this_time


            ### Apply the Delay and Doppler Tracking    
            if focus_dop==True and focus_del==True:
                data0 = delay_shift_frame_set(data, tim, tint, vb)
                data1 = doppler_shift_frame_set(data0, tim, fint, vb)
            if focus_dop==True and focus_del==False:
                data1 = doppler_shift_frame_set(data, tim, fint, vb)
            if focus_dop==False and focus_del==True:
                data1 = delay_shift_frame_set(data, tim, tint, vb)
            if focus_dop==False and focus_del==False:
                data1 = data
           
            datastack[:,pri] = correlate(data1, waveform, mode='same')
            dataraw[:,pri] = data
        
            datatimes[pri] = atime.value[11:24]


        if np.mod(pri, int(npri/10)) == 0:
            print('For pri %d/%d (%s) the max is %3.5f at index %d'%(pri,npri,datatimes[pri],np.max(np.abs(datastack[:,pri])),np.argmax(np.abs(datastack[:,pri]))))

    return datastack, datatimes, dataraw



### Time averaged spectrum
def make_spec(fh,mode,fft,nframes, navg=1,fsize=4000,nchans=1000,
              frange=None,fint=None,vb=True,fix_drops=True,frame_time=None):
    """
    """
    global ref_mjd
    fstart,fstop = cut_freqs(fft.frequency.value, frange)

    
#    avg = np.zeros(fft.frequency_shape[0])
    avg = np.zeros(fstop - fstart)
    atime = None
    avgcnt=0
    for j in range(0,navg):
        atime1, data1 = read_frame_set(fh, nframes,fsize,fix_drops,vb=False,frame_time=frame_time)       
        if atime1 is None:
            continue
        atime = atime1

        ## generate time array
        d2s = 24*60*60
        diff_mjd = (atime.mjd - ref_mjd)*d2s  ## seconds. 
        tim = np.arange( diff_mjd   , diff_mjd + (len(data1)*(1/64e+06)) , 1/64e+06 )      
        if (len(tim) > len(data1)):
            dd = len(tim) - len(data1)
            tim = tim[dd:len(tim)]
                
        data = doppler_shift_frame_set(data1, tim, fint, vb)
             
        for pdata in range(0,int(len(data)/nchans)):
            fdata = do_fft(fft,data[pdata*nchans : (pdata+1)*nchans])
            fdata[int(len(fdata)/2)]=fdata[int(len(fdata)/2)-1]   #### Clip the middle channel
            avg = avg + np.abs(fdata[fstart:fstop])
            avgcnt = avgcnt+1

    if avgcnt>0:
        avg = avg/avgcnt

#    print("%d %d-frame ffts averaged at %s"%(avgcnt,nframes,atime.value if atime != None else '---'))  
    return atime, fft.frequency.value[fstart:fstop],avg  ## time is from the last frame read




def plot_specs(fname='',mode='rb',
               nframes=1,navg=1,nsteps=1,
               nchans=1000,bw=32.0,
               seekto=1032*0,
               frange=[15.853, 15.857], dodop='',
               ptype='spectrum',pname='fig.png',vb=True,
               fix_drops=True):
    """
    frange : [start,end] in units of MHz. This is used only for the plot range to display. 
                  None is the full range. 
    dodop : File name from osod.  Empty string means 'no doppler correction'.
    """
    global ref_mjd
    fh = open_file(fname,mode,seekto)
    fsize = fh.info()['samples_per_frame']

    fin = fh.info()
    frame_time = fin['samples_per_frame']/fin['sample_rate'].value  ## time range per frame. 
    
    
    fft,freqs1 = setup_fft(nchans,bw)
    fstart,fstop = cut_freqs(fft.frequency.value, frange)
    new_nfreq = fstop - fstart

    nsec_per_fft = (len(freqs1)) / (2*bw*1e+6)
    chanres = np.abs(freqs1[1]-freqs1[0])
    
    print('\n-- Nchans for fft = %d\n-- Time range per fft = %3.5f sec\n-- Avg over %d ffts = %3.5f sec \n-- Chan res = %3.7f MHz \n-- Framesize = %d'%(len(freqs1),nsec_per_fft, navg, navg*nsec_per_fft, chanres, fsize))
    #print('Freqs : %3.5f to %3.5f'%(freqs1[0],freqs1[int(len(freqs1)/2)]))

    print("Start and Stop chans : %d %d  (for freq range %s MHz)\n"%(fstart,fstop,str(frange)))

    a,b = read_frame(fh)

    fint = None
    if dodop != '':
        fint,tint = read_doppler_and_delay(dodop)
        ref_dop = fint(ref_mjd)
        print("\nReference Doppler Shift : %3.6f Hz  ( %3.6f MHz )\n"%(ref_dop, ref_dop/1e+6) )

    
    #pl.ion()
    pl.figure(1)
    pl.clf()

    if ptype == 'waterfall':
        waterfall_data = np.zeros( (nsteps, new_nfreq-1 ) )
        waterfall_times = ['' for _ in range(nsteps)]
        waterfall_freqs = freqs1[fstart:fstop]
        
    else:
        waterfall_data = None
        waterfall_times = None
        waterfall_freqs = None
    
    
    for i in range(0,nsteps):
        tloc = fh.tell()
        atime, freqs, avg = make_spec(fh,mode,fft,nframes,navg,fsize,nchans,frange,fint,vb,fix_drops,frame_time)
        lmax = np.argmax(np.abs(avg[1:]))
        print("Step %d/%d \t (from %d) \t Max at freq %2.8f MHz is %3.4f \t Time: %s"%(i,nsteps, tloc,freqs[lmax], max(np.abs(avg[1:])), atime.value if atime != None else '---' ))

        if ptype=='spectrum':
            pl.plot(freqs[1:], np.abs(avg[1:]))
            if atime is None:
                pl.title(fname + "-----")
            else:
                pl.title(fname + "\n" + atime.value)
            pl.xlabel('Frequency (MHz)')

        if ptype=='waterfall':
            waterfall_data[i,:] = np.abs(avg[1:])
            if atime is None:
                waterfall_times[i] = "--------"
            else:
                waterfall_times[i] = atime.value[11:19]
            
    if ptype == 'waterfall':
        im1 = pl.imshow(waterfall_data, cmap='viridis',
                     aspect='auto', origin='lower',
                     extent=[waterfall_freqs[1],waterfall_freqs[-1],0,nsteps])
        im1.axes.set_yticks(range(len(waterfall_times))[0:-1:int(nsteps/10)])
        im1.axes.set_yticklabels(waterfall_times[0:-1:int(nsteps/10)])
        pl.xlabel('Frequency (MHz)')
        pl.ylabel('Time')
        pl.title(fname)
        
    fh.close()
    pl.savefig(pname)
    pl.ion()
    pl.show()

