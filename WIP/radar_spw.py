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
################################################

### Time averaged spectrum
def make_spec(fh=None, fft=None,
              nframes=1, navg=1,fsize=4000, srate=64e6,
              nchans=1000,  frange=None,
              fint=None,tint=None,vb=True,
              fix_drops=True, focus_dop=True, focus_del=True,
              ref_mjd=None, frame_timespan=None):
    """
    """
    fstart,fstop = cut_freqs(fft.frequency.value, frange)
#    avg = np.zeros(fft.frequency_shape[0])
    avg = np.zeros(fstop - fstart)
    atime = None
    avgcnt=0

    nsamples = nframes*fsize

    ## Empty arrays for the data and times.  Time array starts from zero. Need to add 'current location' to this as needed. 
    sample_data = np.zeros(nsamples, dtype='complex')
    sample_times = np.arange(0,nsamples/srate,1/srate)[0:nsamples] ## seconds
    print("\nMemory allocated for data : %3.2f GB  and times : %3.2f GB"%(sample_data.nbytes*1e-9, sample_times.nbytes*1e-9) )
    ## There is also fdata, and avg to keep track of memory.
   
    for j in range(0,navg):

        ## Read the current start time for this spectrum
        current_time = read_time(fh)

        d2s = 24*60*60
        diff_mjd = (current_time.mjd - ref_mjd)*d2s  ## seconds.
        sample_times = np.arange( diff_mjd   , diff_mjd + (nsamples*(1/srate)) , 1/srate )[0:nsamples] 

        ### Print the timerange for this spectrum
        start_time = Time(sample_times[0]/d2s + ref_mjd, format='mjd')
        end_time = Time(sample_times[-1]/d2s + ref_mjd, format='mjd')
        print("\nTime span of spec : %s to %s"%(start_time.isot, end_time.isot))

        ## Read a data stream to fill the data at the desired timesteps (referenced to ref_mjd)
        read_stream(fh, sample_data, sample_times, ref_mjd)

        ### Apply Delay and Doppler correction to the 1D array
        if focus_dop==True and focus_del==True:
            print("Applying Delay and Doppler corrections from OSOD predictions")
            sample_data = delay_shift_frame_set(sample_data, sample_times, tint, vb, ref_mjd)
            sample_data = doppler_shift_frame_set(sample_data, sample_times, fint, vb, ref_mjd)
        if focus_dop==True and focus_del==False:
            print("Applying only Doppler corrections from OSOD predictions")
            sample_data = doppler_shift_frame_set(sample_data, sample_times, fint, vb, ref_mjd)
        if focus_dop==False and focus_del==True:
            print("Applying only Delay corrections from OSOD predictions")
            sample_data = delay_shift_frame_set(sample_data, sample_times, tint, vb,ref_mjd)
        if focus_dop==False and focus_del==False:
            print("Applying NO Delay or Doppler corrections")

            
        for pdata in range(0,int(len(sample_data)/nchans)):
            fdata = do_fft(fft,sample_data[pdata*nchans : (pdata+1)*nchans])
            fdata[int(len(fdata)/2)]=fdata[int(len(fdata)/2)-1]   #### Clip the middle channel
            avg = avg + np.abs(fdata[fstart:fstop])
            avgcnt = avgcnt+1

        print('Data len', len(sample_data))
        print('fft len', len(fdata))

#        fdata = do_fft(fft,sample_data)
#        fdata[int(len(fdata)/2)]=fdata[int(len(fdata)/2)-1]   #### Clip the middle channel
        
#        avg = avg + np.abs(fdata[fstart:fstop])
#        avgcnt = avgcnt+1

    if avgcnt>0:
        avg = avg/avgcnt

#    print("%d %d-frame ffts averaged at %s"%(avgcnt,nframes,atime.value if atime != None else '---'))  
    return current_time, fft.frequency.value[fstart:fstop],avg  ## time is from the last frame read




def plot_specs(fname='',mode='rb',
               nframes=1,navg=1,nsteps=1,
               nchans=1000,bw=32.0,
               seekto=1032*0,
               frange=[15.853, 15.857], dodop='',
               ptype='spectrum',pname='fig.png',vb=True,
               fix_drops=True, focus_dop=True, focus_del=True,):
    """
    frange : [start,end] in units of MHz. This is used only for the plot range to display. 
                  None is the full range. 
    dodop : File name from osod.  Empty string means 'no doppler correction'.
    """
    fh,ref_mjd = open_file(fname,mode,seekto)
    fsize = fh.info()['samples_per_frame']

    fin = fh.info()
    frame_timespan = fin['samples_per_frame']/fin['sample_rate'].value  ## time range per frame. 
    srate = fin['sample_rate'].value

    
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
        ref_del = tint(ref_mjd)
        print("\nReference Delay : %3.6f s  ( %3.6f micro-sec )\n"%(ref_del, ref_del*1e+6) )
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
        atime, freqs, avg = make_spec(fh=fh,fft=fft,
                                      nframes=nframes,navg=navg,fsize=fsize,srate=srate,
                                      nchans=nchans,frange=frange,
                                      fint=fint,tint=tint,vb=vb,
                                      fix_drops=fix_drops, focus_dop=focus_dop, focus_del=focus_del,
                                      ref_mjd=ref_mjd,frame_timespan=frame_timespan)
        
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

