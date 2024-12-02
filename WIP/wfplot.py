#fname = '/home/datacopy-03/HN/BT161A5_HN_No0004'
#fname = '../Data/BT161A1_PT_No0008
#fname = '../Data/BT161A3_NL_No0006'

import baseband
import pylab as pl
from baseband_tasks.fourier import fft_maker
import numpy as np
import astropy.units as u


def open_file(fname=''):
    fh = baseband.open(fname)
    print(fh.info())
    fh.seek(0)
    return fh

global cnt
cnt=0

def read_block(fh=None,nsamples=4000): ## 4000 samples per frame
    global cnt
    try:
        here = fh.tell()
        samples = fh.read(nsamples)
    except Exception as qq:
        print('Caught!'+str(qq))
        cnt = cnt+1
        if cnt<5:
            print("loc : "+str(here))
            fh.seek(here+4000)
            samples = read_block(fh,nsamples)
        
    return samples 

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

def plot_spec(freqs,fdata):
    pl.plot(freqs[1:], np.abs(fdata[1:]))

def plot_waterfall(fname='',nframes=1, navg=1, nsteps=1):
    cnt=0
    fh = open_file(fname)
    fft,freqs = setup_fft(4000*nframes)
    pl.ion()
    pl.clf()

    avg = np.zeros(freqs.shape)
    for i in range(0,nsteps):
        for j in range(0,navg):
            data = read_block(fh,4000*nframes)
            fdata = do_fft(fft,data)
            #plot_spec(freqs,fdata)
            avg = avg + np.abs(fdata)

        avg = avg/navg
        plot_spec(freqs,avg)
        avg.fill(0.0)

    fh.close()
    pl.savefig('tst.png')
    

def plot_specs(fname='',nframes=1,navg=1,nsteps=1):
    cnt=0
    fh = open_file(fname)
    fft,freqs = setup_fft(4000*nframes)
    pl.ion()
    pl.clf()
    avg = np.zeros(freqs.shape)
    for i in range(0,nsteps):
        for j in range(0,navg):
            data = read_block(fh,4000*nframes)
            fdata = do_fft(fft,data)
            #plot_spec(freqs,fdata)
            avg = avg + np.abs(fdata)

        avg = avg/navg
        plot_spec(freqs,avg)
        avg.fill(0.0)

    fh.close()
    pl.savefig('tst.png')


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

