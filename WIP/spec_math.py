import baseband
import pylab as pl
from baseband_tasks.fourier import fft_maker
from baseband import vdif
import numpy as np
import astropy.units as u


### Basic Plots

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


def plot_specs(fname='',mode='rs',nframes=1,navg=1,nsteps=1):
    if mode == "rb":
        if nframes>1:
            print('forcing nframes=1')
        nframes=1
    fh = open_file(fname,mode)
    fft,freqs = setup_fft(4000*nframes)
    #print(len(freqs))
    #pl.ion()
    pl.clf()
    for i in range(0,nsteps):
        atime, avg = make_spec(fh,mode,fft,nframes,navg)
        pl.plot(freqs[1:], np.abs(avg[1:]))
        if atime is None:
            pl.title("-----")
        else:
            pl.title(atime.value)
        
    fh.close()
    pl.savefig('tst.png')
    pl.show()

    
### File I/O

def open_file(fname='',mode='rb'):
    """
    mode='rb' is in units of frames
    mode='rs' is in units of samples
    https://baseband.readthedocs.io/en/stable/vdif/index.html
    """
    fh = vdif.open(fname,mode)
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
        atime = fh.time
        return atime, samples
    except Exception as qq:
        #print('Caught!'+str(qq))
        fh.seek( ( int(fh.tell()/4000)+1 )*4000 )
        return None, None
    

### Do the Math

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


def make_spec(fh,mode,fft, nframes=1, navg=1):
    """
    """
    avg = np.zeros(fft.frequency_shape[0])
    atime = None
    avgcnt=0
    for j in range(0,navg):
        if mode=='rs':
            atime1, data = read_block(fh,4000*nframes)
        else: ## mode='rb'
            atime1, data = read_frame(fh)
        if data is None:
            continue
        atime = atime1
        fdata = do_fft(fft,data)
        avg = avg + np.abs(fdata)
        avgcnt = avgcnt+1

    if avgcnt>0:
        avg = avg/avgcnt

    print("%d frames averaged at %s"%(avgcnt,atime.value if atime != None else '---'))  
    return atime, avg  ## time is from the last frame read


