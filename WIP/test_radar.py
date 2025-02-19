
import sys
sys.path.insert(1, '/home/vega/rurvashi/RADAR/Code/Radar/WIP')
from radar_spw import *


vfile = "/lustre/rurvashi/RADAR/DataTest/Molniya152/1pol/BT161C4_HN_No0010"
dodop = '../DATA/Doppler/Molniya1-52-20241029-1sec/PRDX.09v-91-91.2024-10-29.OUT'

fss = [15.853,15.858]
fss = [14.0,17.0]

plot_specs(vfile,'rb',nframes=1,navg=1,nchans=4000,nsteps=1,bw=32.0,seekto=20,frange=fss,dodop=dodop,ptype='spectrum',vb=False,fix_drops=True,focus_dop=True, focus_del=True)
