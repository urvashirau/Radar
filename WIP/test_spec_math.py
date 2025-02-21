
#myloc = 'laptop'
myloc = 'desktop'

import sys
if myloc == 'desktop':
    sys.path.insert(1, '/home/vega/rurvashi/RADAR/Code/Radar/WIP')
else:
    sys.path.insert(1, '/Users/rurvashi/SCIwork/RADAR/Code/Radar/WIP')

from spec_math import *

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

def test_oldspec_moon_band(myloc='desktop'):
    '''
    #### Test with spec_math
    ### Moon -- Plot full bandpass to see the LFM with Gibbs ringing
    '''

    if myloc == 'desktop':
        vfile = vfile_moon_tycho_desktop
        dodop= ddop_moon_tycho_desktop
    else:
        vfile = vfile_moon_tycho_laptop
        dodop= ddop_moon_tycho_laptop

    plot_specs(vfile,'rb',nframes=32,navg=200,nsteps=1,nchans=4000*32,bw=32.0,seekto=int(50e+5),frange=[0.0,32.0],vb=False,ptype='spectrum',pname='tt.png',dodop=dodop)
    pl.ylim([0.75,2.5])


def test_oldspec_moon_ddm(myloc='desktop'):
    '''
    Test with spec_math
    Moon : DDM
    '''

    if myloc == 'desktop':
        vfile = vfile_moon_tycho_desktop
        dodop= ddop_moon_tycho_desktop
    else:
        vfile = vfile_moon_tycho_laptop
        dodop= ddop_moon_tycho_laptop
   
    make_ddm(fname=vfile,nframes=32,npri=1000,dodop=dodop,seekto=int(5.5e+6 + 12 + 2*32000) ,focus_dop=True,focus_del=True,vb=False,fix_drops=True)

    pl.xlim([80,200])

