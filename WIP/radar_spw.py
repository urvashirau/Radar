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

from radar_lib import *

################################################
### Helper functions
################################################

#
#def read_frame(fh=None):
#    global ref_mjd
#    try:
#        frame = fh.read_frameset([0])
#        samples = frame.data[:,0,0]  ## N samples per frame
#        atime = frame.header0.time ## time of first sample
#        if ref_mjd == None:
#            ref_mjd = atime.mjd   ### Time of the first sample in the file.
#        return atime, samples.squeeze()
#  except Exception as qq:
#        print('Exception in read_frame : Caught!'+str(qq))
#        return None, None
#

def read_frame_set(fh=None, nframes=1,fsize=4000,fix_drops=True,vb=True):
    data = np.zeros(nframes*fsize);### don't keep allocating each time. reuse.
    atime = None
    fcnt = 0
    prev_time=None
    this_time = None
    while fcnt < nframes:
        atime,oneframe = read_frame(fh)

        this_time = atime.mjd

        if prev_time != None:
            if np.abs( this_time - prev_time ) > 1.5*6.25e-05/(24*60*60):
                nframes_dropped = int( ((this_time - prev_time)*(24*60*60)/(6.25e-05)) ) - 1
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
    fh,ref_mjd = open_file(fname,mode,seekto)
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


    dstack1, dtimes, dataraw = stack_pri_and_match_waveform(fh,nframes,fsize,npri,fint,tint,vb,wform, focus_dop, focus_del,fix_drops, ref_mjd)
    print(dstack1.shape)

    disp_raster(dataraw, pnum=1,title='Raw Data')
    
    disp_raster(dstack1, pnum=2,title='After Waveform matched filter')
    
    dstack2 = take_fft_zpad(dstack1,fft,zpad)

    print(dstack2.shape)

    disp_raster(dstack2, pnum=3,title='After Doppler FFT') 
    
    print('Pickling')
    with open(pname+'.pkl','wb') as f:
        pickle.dump(dstack2,f)

    return dstack1, dstack2




def stack_pri_and_match_waveform(fh,nframes,fsize,npri,fint, tint,vb, waveform, focus_dop, focus_del,fix_drops,ref_mjd):
    stepsize = 1
    
    datastack = np.zeros( (int(nframes*fsize/stepsize), npri) ,dtype='complex' )
    dataraw = np.zeros( (int(nframes*fsize/stepsize), npri) ,dtype='float' )
    datatimes = ['' for _ in range(npri)]

    atime = None
    prev_time = None
    for pri in range(0,npri):
        atime, data = read_frame_set(fh, nframes,fsize,fix_drops)
        
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
                if np.abs(this_time - prev_time) > (0.002 + 0.5*(6.25e-05))/(24*60*60) :
                    nframe_offset = int( ((this_time - prev_time)*(24*60*60)/(6.25e-05))-32 )
                    print('Offset at PRI : %d  Prev time : %3.8f mjd  This time : %3.8f mjd   Diff : %3.8f frames'%(pri,prev_time, this_time,nframe_offset))

                    if fix_drops==True:
                        ## Roll the data array to match this starting time, and pre-fill with zeros.
                        data = np.roll( data, fsize * nframe_offset)
                        data[: fsize * nframe_offset ] = 0.0
                        ## Re-calc the time array to start from the new (correct) PRP start time.
                        diff_mjd = diff_mjd - nframe_offset * 6.25e-05
                        tim = np.arange( diff_mjd   , diff_mjd + (len(data)*(1/64e+06)) , 1/64e+06 )
                        if (len(tim) > len(data)):
                            dd = len(tim) - len(data)
                            tim = tim[dd:len(tim)]

                        
            prev_time = this_time


            ### Apply the Delay and Doppler Tracking    
            if focus_dop==True and focus_del==True:
                data0 = delay_shift_frame_set(data, tim, tint, vb,ref_mjd)
                data1 = doppler_shift_frame_set(data0, tim, fint, vb, ref_mjd)
            if focus_dop==True and focus_del==False:
                data1 = doppler_shift_frame_set(data, tim, fint, vb,ref_mjd)
            if focus_dop==False and focus_del==True:
                data1 = delay_shift_frame_set(data, tim, tint, vb,ref_mjd)
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
              frange=None,fint=None,vb=True,fix_drops=True,ref_mjd=None):
    """
    """
    fstart,fstop = cut_freqs(fft.frequency.value, frange)
#    avg = np.zeros(fft.frequency_shape[0])
    avg = np.zeros(fstop - fstart)
    atime = None
    avgcnt=0

    nsamples = nframes*fsize
    
    sample_data = np.zeros(nsamples, dtype='complex')
    sample_times = np.arange(0,nsamples/srate,1/srate) ## seconds
    print("\nMemory allocated for data : %3.2f GB  and times : %3.2f GB"%(sample_data.nbytes*1e-9, sample_times.nbytes*1e-9) )
    if len(sample_times)>nsamples:
        sample_times = sample_times[0:nsamples]

    
    for j in range(0,navg):
        ### Get the current time in the file. Add to sample_times.
        
        start_time = Time(sample_times[0]/d2s + ref_mjd, format='mjd')
        end_time = Time(sample_times[-1]/d2s + ref_mjd, format='mjd')
        print("\nTime span of spec : %s to %s"%(start_time.isot, end_time.isot))


        atime1, data1 = read_frame_set(fh, nframes,fsize,fix_drops,vb=False)       
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
    fh,ref_mjd = open_file(fname,mode,seekto)
    fsize = fh.info()['samples_per_frame']

    fft,freqs1 = setup_fft(nchans,bw)
    fstart,fstop = cut_freqs(fft.frequency.value, frange)
    new_nfreq = fstop - fstart

    nsec_per_fft = (len(freqs1)) / (2*bw*1e+6)
    chanres = np.abs(freqs1[1]-freqs1[0])
    
    print('\n-- Nchans for fft = %d\n-- Time range per fft = %3.5f sec\n-- Avg over %d ffts = %3.5f sec \n-- Chan res = %3.7f MHz \n-- Framesize = %d'%(len(freqs1),nsec_per_fft, navg, navg*nsec_per_fft, chanres, fsize))
    #print('Freqs : %3.5f to %3.5f'%(freqs1[0],freqs1[int(len(freqs1)/2)]))

    print("Start and Stop chans : %d %d  (for freq range %s MHz)\n"%(fstart,fstop,str(frange)))

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
        atime, freqs, avg = make_spec(fh,mode,fft,nframes,navg,fsize,nchans,frange,fint,vb,fix_drops)
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

