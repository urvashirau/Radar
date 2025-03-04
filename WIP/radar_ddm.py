from radar_lib import *

def make_ddm(fname='',mode='rb',
             nframes=1, npri=1, ### nframes per PRP  x  n PRP
             #bw=32.0,
             ref_time='',
             seekto=1032*0,
             frange=[0.08,0.20], dodop='',
             pname='',vb=True,
             fix_drops=True,
             focus_dop=True, focus_del=True,
             debug=False, dop_offset=0.0,del_offset=0.0,
             cavg_factor=1.0,
             in_pnum=None):
    """
    frange : [start,end] : Fraction of the frequency range to display :  Tycho-Oct30 : 0.12 - 0.223
    dodop : File name from osod.  Empty string means 'no doppler correction'.
    dop_offset : In Hz, apply an extra Doppler offset to move the DDM signal to lower or higher frequencies.
    """

    t1 = time.time()
    
    ### Open the file and seek to desired location
    d2s = 24*60*60
    fh, ref_mjd = open_file(fname,mode,seekto)
    fin = fh.info()
    fsize = fin['samples_per_frame']
    srate = fin['sample_rate'].value
    frame_timespan = fin['samples_per_frame']/fin['sample_rate'].value  ## time range per frame. 


    if ref_time != '':
        ref_mjd = Time(ref_time).mjd
    
    ## Total number of samples 
    nsamples = nframes*fsize * npri

    ## Number of delay and doppler bins
    ndelays = nframes*fsize
    ndopps = npri

    ### t_pri is the effective 'sample rate' which will then translate to bandwidth for the FFT.
    ### For now, call it 'bw'... automate this later.
    bw = ( 0.5/(ndelays/srate) )/ 1e+6  ## MHz
    fft,freqs1 = setup_fft(ndopps,bw)
    freqs1 = freqs1 * 1e+6 ## to Hz.
    fstart=0 
    fstop=int(len(freqs1)/2)-1
    new_nfreqs = fstop - fstart
    chanres = np.abs(freqs1[1]-freqs1[0])
    
    print('\nPRI length is %3.7f sec (%d delay samples)'%(ndelays/srate, ndelays) )
    print('N_PRI or nchans for fft = %d \nChan res = %3.7f Hz'%(len(freqs1), chanres))
    print('Freqs : %3.6f - %3.6f Hz'%(freqs1[fstart],freqs1[fstop]) )
    
    ## Setup Doppler correction
    fint = None
    if dodop != '':
        fint, tint = read_doppler_and_delay(dodop)
        ref_dop = fint(ref_mjd)
        print("\nReference Doppler Shift : %3.6f Hz  ( %3.6f MHz ) with offset : %3.6f Hz"%(ref_dop, ref_dop/1e+6, dop_offset) )
        ref_del = tint(ref_mjd)
        print("Reference Delay Shift : %3.6f sec  ( %3.6f microsec ) with offset : %3.6f microsec"%(ref_del, ref_del*1e+6, del_offset*1e+6) )
    else: ## dodop=''
        focus_del=False
        focus_dop=False
        print('No Del-Dop tracking model')


    ## Make the waveform for matched filtering
    start_freq = 1e+6 ##1e+6 #Hz
    end_freq = 31e+6 ##31e+6 #Hz
    wf_times = np.arange(0,ndelays/srate,1/srate)[0:ndelays]
    wform = make_waveform(ndelays, wf_times,start_freq,end_freq)
    #print("Waveform from %3.5f MHz to %3.5f MHz"%(start_freq/1e6, end_freq/1e6))

    #check_waveform(wform, bw=30.0) ## (end_freq-start_freq)/1e+6)
    #return

        
    ### Allocate the data and time 1D arrays
    sample_data = np.zeros(nsamples, dtype='complex')
    sample_times = np.arange(0, nsamples/srate,1/srate)[0:nsamples] ## seconds
    print("\nMEMORY allocated for data : %3.5f GB  and times : %3.5f GB"%(sample_data.nbytes*1e-9, sample_times.nbytes*1e-9) )

    start_time = Time(sample_times[0]/d2s + ref_mjd, format='mjd')
    end_time = Time(sample_times[-1]/d2s + ref_mjd, format='mjd')
    print("\nTime span of entire DDM : %s to %s"%(start_time.isot, end_time.isot))

    slow_time_inc = ndelays/srate  ### Multiply by coherent averaging timescale. 
    slow_time_span = slow_time_inc * npri

    print('Doppler Frequency span : 0 to %3.2f Hz'%(1.0/slow_time_inc) )
    print("Number of Delay bins : %d    Number of Doppler bins : %d\n"%(ndelays, ndopps) )

    print("\nSTART PROCESSING\n")
    
    ### Read data into a 1D array
    ## NEW 
    ##read_stream(fh, sample_data, sample_times, ref_mjd)

    #### OLD_2
    print('Reading %d M samples'%(len(sample_data)/1e+6) )
    beg_time = read_frame_set_2(fh,sample_data,nframes*npri, fsize,fix_drops,vb=vb,frame_time=frame_timespan,ref_mjd=ref_mjd)
    fh.close()
    if beg_time is None:
        print('No valid time. Exiting.')
        return;

    #### Remove DC - along the slow-time axis (!!) 
    data_matrix_dc = sample_data.reshape([ndopps,ndelays]).transpose()
    print('DC : Reshape to 2D : Memory shared between 1D and 2D array views : '+str( np.shares_memory(sample_data,data_matrix_dc) ))
    remove_dc(data_matrix_dc)
    del data_matrix_dc 

#    ### Low pass filter.
#    if cavg_factor > 1:
#        print('Apply coherent averaging along the slow-time axis x %d'%(cavg_factor))
#        data_matrix_1 = resample_2d_avg( data_matrix_dc, (ndelays, int(ndopps/cavg_factor)) ,coherent=True)
#        print('DATA SHAPE : ',data_matrix_1.shape)
#        sample_data = data_matrix_1.transpose().reshape(data_matrix_1.shape[0]*data_matrix_1.shape[1])
#        del data_matrix_dc
#
#        time_matrix_dc = sample_times.reshape([ndopps,ndelays]).transpose()
#        time_matrix_1 = resample_2d_avg( time_matrix_dc, (ndelays, int(ndopps/cavg_factor)) ,coherent=False)
#        print('TIME SHAPE : ',time_matrix_1.shape)
#        sample_times = time_matrix_1.transpose().reshape(time_matrix_1.shape[0]*time_matrix_1.shape[1])
#        del time_matrix_dc
#
#        gc.collect()
#
#        slow_time_inc = slow_time_inc * cavg_factor
#        ndopps = int(ndopps/cavg_factor)
#        
    
    ### Apply Delay and Doppler correction to the 1D array

    if focus_dop==True and focus_del==True:
        print("Applying Delay and Doppler corrections from OSOD predictions")
        delay_shift_frame_set_2(sample_data, sample_times, tint, vb, ref_mjd,del_offset)
        doppler_shift_frame_set_2(sample_data, sample_times, fint, vb, ref_mjd,dop_offset)
    if focus_dop==True and focus_del==False:
        print("Applying only Doppler corrections from OSOD predictions")
        doppler_shift_frame_set_2(sample_data, sample_times, fint, vb, ref_mjd,dop_offset)
    if focus_dop==False and focus_del==True:
        print("Applying only Delay corrections from OSOD predictions")
        delay_shift_frame_set_2(sample_data, sample_times, tint, vb,ref_mjd,del_offset)
    if focus_dop==False and focus_del==False:
        print("Applying NO Delay or Doppler corrections")


    ### Delete sample_times and free up the mem.
    del sample_times
        
    ### Reshape to 2D with delay as last axis
    data_matrix_1 = sample_data.reshape([ndopps,ndelays])  ## Make the delay axis the 'fast' one.
    print('Reshape to 2D : Memory shared between 1D and 2D array views : '+str( np.shares_memory(sample_data,data_matrix_1) ))

    if cavg_factor > 1:
        print('Apply coherent averaging along the slow-time axis x %d'%(cavg_factor))
        data_matrix_1 = resample_2d_avg( data_matrix_1, (int(ndopps/cavg_factor), ndelays) ,coherent=True)
        #data_matrix_1 = low_pass_filter( data_matrix_1.transpose(), cavg_factor )
        slow_time_inc = slow_time_inc * cavg_factor
        ## Release memory from original array. 
        del sample_data
        gc.collect()

    print('\nMEMORY needed for averaged matrix : %3.4f GB'%(data_matrix_1.nbytes*1e-9))
    print('DATA SHAPE : ',data_matrix_1.shape)

    ### Display
    if debug==True:
        disp_raster(data_matrix_1.transpose(),pnum=1,title='Data matrix',xaxis='time',slow_time_inc=slow_time_inc,slow_time_span=slow_time_span)
    
    ### Run correlation on fast time axis
    run_matched_filter(data_matrix_1, wform)

    ## Reshape to 2D with doppler as last axis
    data_matrix_2 = data_matrix_1.transpose() ##reshape([ndelays,ndopps]) ## Make doppler axis the 'fast' one.
    print('Reshape to 2D : Memory shared between 1D and 2D array views : '+str( np.shares_memory(data_matrix_1,data_matrix_2) ))

    ### Display
    if debug==True:
        disp_raster(data_matrix_2,pnum=2,title='After Matched filter',xaxis='time',slow_time_inc=slow_time_inc,slow_time_span=slow_time_span)

    
    ### Run FFT on slow time axis
    take_fft(data_matrix_2, fft)

    
#    if pname != '':
#        print('Pickling DDM array of shape ', data_matrix_2.shape)
#        with open(pname+'_fullres.pkl','wb') as f:
#            pickle.dump(data_matrix_2,f)

    ### Display
    if debug==True:
        pnum=3
    else:
        pnum=in_pnum
    disp_raster(data_matrix_2,pnum=pnum,title='After Doppler FFT',pname=pname,frange=frange,xaxis='dopp',
                nplot_delay=int(ndopps), nplot_doppler=int(ndopps/cavg_factor),
#                nplot_delay=int(ndopps*cavg_factor), nplot_doppler=int(ndopps),
                slow_time_inc=slow_time_inc,slow_time_span=slow_time_span)

    ### Release processed arrays.
    del data_matrix_1,data_matrix_2
    
    t2 = time.time()

    print('Runtime : %3.4f min'%( (t2-t1)/60.0 ))

    gc.collect()
    
    return




def make_sequence_ddm(fname='',dodop='',
                      tstart='2024-10-30T15:21:00.000', tinc='0.5s', nsteps=5,
                      pname='ex',npri=1000):
    """
    Construct DDMs at regular intervals and save the plot arrays.
    tstart : start time as a string.
    tinc : time increment between steps ( 30s, or 2min...  )
    nsteps : number of steps to make DDMs at
    """

    d2s = 24*60*60.0
    
    t0 = Time(tstart)
    td = TimeDelta(tinc)
    
    for tnow in range(0,nsteps):
        seekto = t0.isot
        pfname = pname+'_'+seekto
        
        make_ddm(fname=fname,nframes=32,npri=npri,
                 dodop=dodop,focus_dop=True,focus_del=True,
                 vb=False,fix_drops=True,frange=[0.0,1.0],dop_offset=0.0,del_offset=0.0,
                 cavg_factor=1,debug=False,ref_time='',
                 seekto=seekto,pname=pfname)
            
            #pl.clf();
            #pl.imshow(np.abs(arr[:,wmin:wmax]),aspect='auto',origin='lower',
            #          cmap='gray',norm=mcolors.PowerNorm(gamma=0.8,vmin=0.0,vmax=2500));
            ##pl.colorbar();
            #pl.title(pfname);pl.xlabel('Doppler');pl.ylabel('Range');
            #input()
            
        t0 = t0 + td

    return


ani_t0=None
ani_tcnt=0
ani=None


def plot_sequence_ddm(fname='',dodop='',
                     reftime='2024-10-30T15:21:00.000',
                     tstart=int(5.5e6 + 10 + 32*30000 ),
                     tinc=int(32*1000), nsteps=5,
                     pname='ex',npri=1000):

    """
    Plot  DDMs at regular intervals and save the plot arrays.
    tstart : start time as a string.
    tinc : time increment between steps ( 30s, or 2min...  )
    nsteps : number of steps to make DDMs at
    """
    global ani

    d2s = 24*60*60.0

    seekto = tstart

    pfname = pname+'_'+str(seekto)
    ref_pfname = pfname

    parr = read_arr_match(ref_pfname,pfname,npri)
    
    #parr = read_arr(pfname,npri)
    
    fig, ax = plt.subplots(1)
    im = ax.imshow(parr,aspect='auto',origin='lower',
                   cmap='gray',norm=mcolors.PowerNorm(gamma=0.8,vmin=0.0,vmax=3000))
    ax.set_xlabel('Doppler')
    ax.set_ylabel('Delay')
    
    ani = FuncAnimation(fig, update_plot, fargs=(im,ax,pname,tstart,tinc,nsteps,npri,ref_pfname),
                        interval=1000,
                        cache_frame_data=False)
    plt.show()




def update_plot(frame,im,ax,pname,tstart,tinc,nsteps,npri,ref_pfname):
    global ani_t0
    global ani_tcnt
    global ani

    if ani_t0 == None:
        ani_t0 = tstart + tinc

    if ani_tcnt==nsteps:
        print('Done!')
        ani.event_source.stop()
        plt.close(frame)
        return
        
    seekto = ani_t0
    pfname = pname+'_'+str(seekto)
    
    parr = read_arr_match(ref_pfname,pfname,npri)

    im.set_data(parr)
    ax.set_title(pfname)

    ani_tcnt += 1
    
    ani_t0 = ani_t0 + tinc


    return


def try_sequence_ddm(fname='',dodop='',
                     reftime='2024-10-30T15:21:00.000',
                     tstart=int(5.5e6 + 10 + 32*30000 ),
                     tinc=int(32*1000), nsteps=5,
                     pname='ex',npri=50,compute=False,cropf=1.0):
    """
    Construct DDMs at regular intervals and save the plot arrays.
    tstart : start time as a string.
    tinc : time increment between steps ( 30s, or 2min...  )
    nsteps : number of steps to make DDMs at
    """

    d2s = 24*60*60.0

    seekto = tstart

    refpfname=''
    
    for tnow in range(0,nsteps):
        pfname = pname+'_'+str(seekto)
        print('\n------------------')
        print('\nFrame %d : %s'%(tnow+1,pfname))

        if tnow==0:
            ref_pfname = pfname

        pkl_name = pfname


        if compute==True:

            tnpri = 50

            for tnpri in [100]: ##50,1000]:
            
                make_ddm(fname=fname,nframes=32,npri=tnpri,
                         dodop=dodop,focus_dop=True,focus_del=True,
                         vb=False,fix_drops=True,frange=[0.0,1.0],
                         dop_offset=0.0,del_offset=0.0,
                         cavg_factor=1,debug=False,ref_time=reftime,
                         seekto=seekto,pname=pfname,in_pnum=1)


                print('Reading ',pkl_name)
                with open(pkl_name+'.pkl','rb') as f:
                    arr = pickle.load(f)

                # find max.
                x,y = np.unravel_index( np.argmax(np.abs(arr)), arr.shape )
                print('\n---------------------')
                print('\nMax of %3.2f at ( %d , %d )'%(arr[x,y], x, y) )
                
                x_off = int(tnpri/2) - x
                y_off = int(tnpri/2) - y
                
                dop_off = y_off/tnpri * 500.0 * 2 # Hz
                del_off = x_off/tnpri * 0.002  # sec
                ###del_off = int(x_off/tnpri * 32/2)
                
                print('Offsets :   Del = %3.4f frames  Dop = %3.4f Hz \n'%(del_off,dop_off))
                print('\n---------------------')

            make_ddm(fname=fname,nframes=32,npri=npri,
                     dodop=dodop,focus_dop=True,focus_del=True,
                     vb=False,fix_drops=True,frange=[0.45,0.55],
                     dop_offset=dop_off,del_offset=del_off,
                     cavg_factor=1,debug=False,ref_time=reftime,
                     seekto=seekto, pname=pfname,in_pnum=2)

        #parr = read_arr(pfname,npri)
        parr = read_arr_match(ref_pfname,pfname,npri,cropf=cropf)

        pl.ioff()
        pl.figure(1,figsize=(8,8))
        pl.clf()
        pl.imshow(parr,aspect='auto',origin='lower',
                  cmap='gray',norm=mcolors.PowerNorm(gamma=0.8,vmin=0.0,vmax=2500))

        fn = tnow+1
        if fn<10:
            fns = '0'+str(fn)
        else:
            fns = str(fn)
        pl.savefig(pname+'_frame_'+ fns+'.png')
        pl.ion()
        
        seekto = seekto + tinc

    return
