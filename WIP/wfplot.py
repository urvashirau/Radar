#fname = '/home/datacopy-03/HN/BT161A5_HN_No0004'
#fname = '../Data/BT161A1_PT_No0008
#fname = '../Data/BT161A3_NL_No0006'

import baseband
import pylab as pl
from baseband_tasks.fourier import fft_maker
import numpy as np
import astropy.units as u


def pspecfile(sname=''):
    fp = open(sname)
    dat = fp.readlines()
    fp.close()
    freq = np.zeros(len(dat))
    volt = np.zeros(len(dat))
    for i in range(0,len(dat)):
        freq[i] = float(dat[i].split()[0])
        volt[i] = float(dat[i].split()[1])

    pl.clf()
    pl.plot(freq,volt)
    pl.show()



def open_file(fname='',mode='rb'):
    """
    mode='rb' is in units of frames
    mode='rs' is in units of samples
    https://baseband.readthedocs.io/en/stable/vdif/index.html
    """
    fh = baseband.open(fname,mode)
    print(fh.info())
    fh.seek(0)
    return fh


def read_frame(fh=None):
    try:
        frame = fh.read_frame()
        samples = frame.data  ## 4000 samples per frame
        atime = frame.header.time
        return atime, samples.squeeze()
    except Exception as qq:
        #print('Caught!'+str(qq))
        return None, None

def read_block(fh=None,nsamples=4000): ## 4000 samples per frame
    try:
        samples = fh.read(nsamples)
        return None, samples
    except Exception as qq:
        print('Caught!'+str(qq))
        return None, None
        

def setup_fft(ndata):
    fft_maker.set('numpy')
    fft = fft_maker((ndata,), 'float64',
                    direction='forward',
                    ortho=True,
                    sample_rate=64.0*u.MHz)
    freqs = fft.frequency.value
    return fft,freqs

def do_fft(fft,data):
    fdata = fft(data)
    return fdata

def plot_spec(atime,freqs,fdata):
    pl.plot(freqs[1:], np.abs(fdata[1:]))
    if atime is None:
        pl.title("-----")
    else:
        pl.title(atime.value)


def make_spec(fh,mode,fft, nframes=1, navg=1):
    """
    """
    avg = np.zeros(fft.frequency_shape[0])
    atime = None
    avgcnt=0
    for j in range(0,navg):
        if mode=='rs':
            atime, data = read_block(fh,4000*nframes)
        else: ## mode='rb'
            atime, data = read_frame(fh)
        if data is None:
            continue
        fdata = do_fft(fft,data)
        avg = avg + np.abs(fdata)
        avgcnt = avgcnt+1

    if avgcnt>0:
        avg = avg/avgcnt

    print("%d frames averaged at %s"%(avgcnt,atime.value if atime != None else '---'))  
    return atime, avg  ## time is from the last frame read


def plot_specs(fname='',mode='rb',nframes=1,navg=1,nsteps=1):
    if mode == "rb":
        if nframes>1:
            print('forcing nframes=1')
        nframes=1
    fh = open_file(fname,mode)
    fft,freqs = setup_fft(4000*nframes)
    #print(len(freqs))
    pl.ion()
    pl.clf()
    for i in range(0,nsteps):
        avg = atime, make_spec(fh,mode,fft,nframes,navg)
        plot_spec(atime,freqs,avg)

    fh.close()
    pl.savefig('tst.png')


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

waterfall_data = None
waterfall_times = None
ani = None

def plot_waterfall(fname='',mode='rb',nframes=1, navg=1,nsteps=100):
    global waterfall_data
    global waterfall_times
    global ani

    if mode == "rb":
        if nframes>1:
            print('forcing nframes=1')
        nframes=1
        
    #### Open File and set up FFT
    fh = open_file(fname,mode)
    print(fh.info())
    
    fft,freqs = setup_fft(4000*nframes)
    print(len(freqs))


    #### Initialize Plots
                             
    # Create figure and axes
    fig, (ax2,ax1) = plt.subplots(2,sharex=True)

    spectrum_data = np.zeros(2000*nframes+1)
    im2 = ax2.plot(freqs[1:], spectrum_data[1:])
    #ax2.set_xlabel('Frequency (MHz)')
    ax2.set_ylabel('Amplitude')
    ax2.set_title('------------------------------')

    
    waterfall_data = np.zeros((nsteps, 2000*nframes))
    waterfall_times =  ['--------' for _ in range(nsteps)]
    im1 = ax1.imshow(waterfall_data, cmap='viridis',
                     aspect='auto', origin='lower',
                     extent=[freqs[1],freqs[-1],0,nsteps])
    im1.axes.set_yticks(range(len(waterfall_times))[0:-1:int(nsteps/10)])
    im1.axes.set_yticklabels(waterfall_times[0:-1:int(nsteps/10)])
    
#    plt.colorbar(im1)
#    plt.xlabel('Frequency')
#    plt.ylabel('Time')
    ax1.set_xlabel('Frequency (MHz)')
    ax1.set_ylabel('Time')

    
    #### Start animation
    ani = FuncAnimation(fig,
                        update_plot,
                        fargs=(im1,ax2,
                               fh, mode, freqs, nsteps, 
                               fft,nframes,navg),
                        interval=10,
                        cache_frame_data=False)
    plt.show()


def update_plot(frame,im1,ax2,  fh,mode,freqs,nsteps, fft, nframes, navg):
    global waterfall_data
    global waterfall_times
    ## Get new data
    new_time, new_spec = make_spec(fh,mode, fft,nframes,navg)

    if new_time is None:
        print("No more data to plot")
        ani.event_source.stop()
        return
    
    if new_time != None:
        new_time_str = new_time.value
    else:
        new_time_str = '-----'
    
    ## Update the spectrum plot
    ax2.lines[0].set_data(freqs[1:], new_spec[1:])
    ax2.set_ylim(ymin=np.min(new_spec[1:]),
                 ymax=1e-04+np.max(new_spec[1:])*1.1)
    ax2.set_title( new_time_str )
    

    ## Add new data to waterfall plot
    waterfall_data = np.roll(waterfall_data, -1, axis=0)
    waterfall_data[-1, :] = new_spec[1:]
    waterfall_times = np.roll(waterfall_times, -1)
    waterfall_times[-1] = new_time_str[11:19]

    # Update the image
    im1.set_data(waterfall_data)
    im1.set_clim(vmin=np.min(waterfall_data),
                 vmax=np.max(waterfall_data))
    im1.axes.set_yticks(range(len(waterfall_times))[0:-1:int(nsteps/10)])
    im1.axes.set_yticklabels(waterfall_times[0:-1:int(nsteps/10)])

    return
    

