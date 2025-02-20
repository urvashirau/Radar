
import sys
sys.path.insert(1, '/home/vega/rurvashi/RADAR/Code/Radar/WIP')
from radar_spw import *

### SPECTRUM

### Moon -- Plot full bandpass to see the LFM with Gibbs ringing
vfile = "/lustre/rurvashi/RADAR/DataTest/Tycho/1pol/BT161D2_FD_No0004"
dodop='../DATA/Doppler/Tycho-2024-10-30-1sec/PRDX.09v-93-93.2024-10-30.OUT'
plot_specs(vfile,'rb',nframes=32,navg=200,nsteps=1,nchans=4000*32,bw=32.0,seekto=int(50e+5),frange=[0.0,32.0],vb=False,ptype='spectrum',pname='tt.png',dodop=dodop,focus_dop=False,focus_del=False)
#pl.ylim([0.75,2.5])



vfile = "/lustre/rurvashi/RADAR/DataTest/Molniya152/1pol/BT161C4_HN_No0010"
dodop = '../DATA/Doppler/Molniya1-52-20241029-1sec/PRDX.09v-91-91.2024-10-29.OUT'

fss = [15.853,15.858]
fss = [14.0,17.0]

plot_specs(vfile,'rb',nframes=1,nchans=1000,navg=1,nsteps=1,bw=32.0,seekto=20,frange=fss,dodop=dodop,ptype='spectrum',vb=False,fix_drops=True,focus_dop=True, focus_del=True)



### WATERFALL
vfile = "/lustre/rurvashi/RADAR/DataTest/Molniya152/1pol/BT161C4_HN_No0010"
dodop = '../DATA/Doppler/Molniya1-52-20241029-1sec/PRDX.09v-91-91.2024-10-29.OUT'

plot_specs(vfile,'rb',nframes=128*10,navg=1,nchans=128*4000*10,bw=32.0,seekto=100,frange=[15.8542,15.8550],dodop=dodop,ptype='waterfall',nsteps=50,vb=False)

## change navg = 10


#### Test with spec_math

import sys
sys.path.insert(1, '/home/vega/rurvashi/RADAR/Code/Radar/WIP')
from spec_math import *

### Moon -- Plot full bandpass to see the LFM with Gibbs ringing
vfile = "/lustre/rurvashi/RADAR/DataTest/Tycho/1pol/BT161D2_FD_No0004"
dodop='../DATA/Doppler/Tycho-2024-10-30-1sec/PRDX.09v-93-93.2024-10-30.OUT'
plot_specs(vfile,'rb',nframes=32,navg=200,nsteps=1,nchans=4000*32,bw=32.0,seekto=int(50e+5),frange=[0.0,32.0],vb=False,ptype='spectrum',pname='tt.png',dodop=dodop)
pl.ylim([0.75,2.5])

### Molniya -- Plot narrow freq range to resolve the line to check Doppler shifts and correction.

