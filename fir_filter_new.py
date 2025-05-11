    # Unified transition width calculation
    if isinstance(cutoff, (list, tuple, np.ndarray)):
        if len(cutoff) != 2:
            raise ValueError("Bandpass/bandstop requires exactly 2 cutoff frequencies")
        
        # Calculate bandwidth and transition
        bandwidth = abs(cutoff[1] - cutoff[0])
        transition = min(500, 0.1 * bandwidth)  # 10% of bandwidth or max 500Hz
        
        # Ensure cutoffs stay within Nyquist
        f1 = max(20, cutoff[0])
        f2 = min(nyq-20, cutoff[1])
    else:
        # For lowpass/highpass
        transition = min(500, 0.1 * cutoff)
        f1 = f2 = cutoff
    
    # Band definitions for all filter types
    if filter_type == 'bandstop':
        bands = [0, 
                f1-transition, f1,
                f2, f2+transition,
                nyq]
        desired = [1, 1, 0, 1]  # Stop between f1 and f2
        weight = [1, 10, 1]     # Strong attenuation in stopband
        firwin_pass_zero = True
    elif filter_type == 'bandpass':
        bands = [0,
                f1-transition, f1,
                f2, f2+transition,
                nyq]
        desired = [0, 0, 1, 0]  # Pass between f1 and f2
        weight = [10, 1, 10]    # Strong attenuation in stopbands
        firwin_pass_zero = False
    elif filter_type == 'lowpass':
        bands = [0, f1, f1+transition, nyq]
        desired = [1, 0]
        weight = [1, 10]
        firwin_pass_zero = True
    elif filter_type == 'highpass':
        bands = [0, f1-transition, f1, nyq]
        desired = [0, 1]
        weight = [10, 1]
        firwin_pass_zero = False
    else:
        raise ValueError(f"Invalid filter_type '{filter_type}'")
    
    if method == 'remez':
        try:
            coeffs = signal.remez(
                numtaps,
                bands,
                desired,
                weight=weight,
                fs=samplerate,
                maxiter=100
            )
            
            return coeffs.astype(np.float32)
        except Exception as e:
            print(f"remez failed ({str(e)}), falling back to firwin")
            method = 'firwin'
    # Filter design
    if method == 'firwin':
        if filter_type in ['bandpass', 'bandstop']:
            return signal.firwin(
                numtaps,
                [f1, f2],
                window=window_type,
                pass_zero=firwin_pass_zero,
                fs=samplerate
            ).astype(np.float32)
        else:
            return signal.firwin(
                numtaps,
                f1,
                width=transition,
                window=window_type,
                pass_zero=firwin_pass_zero,
                fs=samplerate
            ).astype(np.float32)
