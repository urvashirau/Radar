from spec_math import *

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
        #ani.event_source.stop()
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
    

