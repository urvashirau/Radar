
#myloc = 'laptop'
myloc = 'desktop'

import sys
if myloc == 'desktop':
    sys.path.insert(1, '/home/vega/rurvashi/RADAR/Code/Radar/WIP')
else:
    sys.path.insert(1, '/Users/rurvashi/SCIwork/RADAR/Code/Radar/WIP')

from radar_spw import *
from radar_ddm import *

### DATA
vfile_moon_tycho_desktop ="/lustre/rurvashi/RADAR/DataTest/Tycho/1pol/BT161D2_FD_No0004"
ddop_moon_tycho_desktop = '../DATA/Doppler/Tycho-2024-10-30-1sec/PRDX.09v-93-93.2024-10-30.OUT'

vfile_moon_tycho_laptop =  '../Data/BT161D2_FD_No0004'
ddop_moon_tycho_laptop = '../Data/Doppler/Tycho-2024-10-30-1sec/PRDX.09v-93-93.2024-10-30.OUT'

vfile_sat_152_desktop = "/lustre/rurvashi/RADAR/DataTest/Molniya152/1pol/BT161C4_HN_No0010"
ddop_sat_152_desktop = '../DATA/Doppler/Molniya1-52-20241029-1sec/PRDX.09v-91-91.2024-10-29.OUT'

vfile_sat_152_laptop ='../Data/BT161C4_HN_No0010' 
ddop_sat_152_laptop = '../Data/Doppler/Molniya1-52-20241029-1sec/PRDX.09v-91-91.2024-10-29.OUT'


### Tests


def test_spec_moon_band(myloc='desktop'):
    '''
    SPECTRUM : Moon -- Plot full bandpass to see the LFM with Gibbs ringing
    '''
    if myloc == 'desktop':
        vfile = vfile_moon_tycho_desktop
        dodop= ddop_moon_tycho_desktop
    else:
        vfile = vfile_moon_tycho_laptop
        dodop= ddop_moon_tycho_laptop


    plot_specs(vfile,'rb',nframes=32,navg=200,nsteps=1,nchans=4000*32,bw=32.0,seekto=int(50e+5),frange=[0.0,32.0],vb=False,ptype='spectrum',pname='tt.png',dodop=dodop,focus_dop=False,focus_del=False)
    pl.ylim([0.75,2.5])
    pl.savefig('tfig_spec_moon_band.png')


def test_spec_sat_line(myloc='desktop'):
    '''
    SPECTRUM : Molniya -- High resolution spectrum to check Doppler correction.
    '''
    if myloc == 'desktop':
        vfile = vfile_sat_152_desktop
        dodop = ddop_sat_152_desktop
    else:
        vfile = vfile_sat_152_laptop
        dodop = ddop_sat_152_laptop

    plot_specs(vfile,'rb',nframes=128*20,navg=100,nchans=128*4000*20,bw=32.0,seekto=20,frange=[15.853,15.858],dodop=dodop,ptype='spectrum',nsteps=3,vb=False,fix_drops=True,focus_dop=True,focus_del=True)
    pl.savefig('tfig_spec_sat_line.png')


def test_wfall_sat_line(myloc='desktop'):
    '''
    WATERFALL : Molniya
    '''
    if myloc == 'desktop':
        vfile = vfile_sat_152_desktop
        dodop = ddop_sat_152_desktop
    else:
        vfile = vfile_sat_152_laptop
        dodop = ddop_sat_152_laptop

    plot_specs(vfile,'rb',nframes=128*10,navg=10,nchans=128*4000*10,bw=32.0,seekto=100,frange=[15.8542,15.8550],dodop=dodop,ptype='waterfall',nsteps=50,vb=False)
    pl.savefig('tfig_wfall_sat_line.png')



def test_ddm_tycho_2sec(myloc='desktop',npri=1000):
    '''
    Make a DDM from 2sec of Tycho data
    '''
    if myloc == 'desktop':
        vfile = vfile_moon_tycho_desktop
        dodop= ddop_moon_tycho_desktop
    else:
        vfile = vfile_moon_tycho_laptop
        dodop= ddop_moon_tycho_laptop


    make_ddm(fname=vfile,nframes=32,npri=npri,dodop=dodop,seekto=int(5.5e+6 + 14 + 2*32000) ,focus_dop=True,focus_del=True,vb=False,fix_drops=True,frange=[0.08,0.20])

    pl.savefig('tfig_ddm_'+str(npri)+'.png')

