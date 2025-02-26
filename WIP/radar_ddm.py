from radar_lib import *

def make_ddm(fname='',mode='rb',
             nframes=1, npri=1, ### nframes per PRP  x  n PRP
             #bw=32.0,
             seekto=1032*0,
             frange=[0.08,0.20], dodop='',
             pname='',vb=True,
             fix_drops=True,
             focus_dop=True, focus_del=True,
             debug=False, dop_offset=0.0,
             cavg_factor=1.0):
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
        print("Reference Delay Shift : %3.6f sec  ( %3.6f microsec )"%(ref_del, ref_del*1e+6) )
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
        delay_shift_frame_set_2(sample_data, sample_times, tint, vb, ref_mjd)
        doppler_shift_frame_set_2(sample_data, sample_times, fint, vb, ref_mjd,dop_offset)
    if focus_dop==True and focus_del==False:
        print("Applying only Doppler corrections from OSOD predictions")
        doppler_shift_frame_set_2(sample_data, sample_times, fint, vb, ref_mjd,dop_offset)
    if focus_dop==False and focus_del==True:
        print("Applying only Delay corrections from OSOD predictions")
        delay_shift_frame_set_2(sample_data, sample_times, tint, vb,ref_mjd)
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

    
    if pname != '':
        print('Pickling DDM array of shape ', data_matrix_2.shape)
        with open(pname+'_fullres.pkl','wb') as f:
            pickle.dump(data_matrix_2,f)

    ### Display
    disp_raster(data_matrix_2,pnum=3,title='After Doppler FFT',pname=pname,frange=frange,xaxis='dopp',
                nplot_delay=int(ndopps), nplot_doppler=int(ndopps/cavg_factor),
#                nplot_delay=int(ndopps*cavg_factor), nplot_doppler=int(ndopps),
                slow_time_inc=slow_time_inc,slow_time_span=slow_time_span)

    ### Release processed arrays.
    del data_matrix_1,data_matrix_2
    
    t2 = time.time()

    print('Runtime : %3.4f min'%( (t2-t1)/60.0 ))

    gc.collect()
    
    return

