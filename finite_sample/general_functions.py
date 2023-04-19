# -*- coding: utf-8 -*-
import numpy as np

def get_signal_power(signal_spk, avg_across_recs = False):
    """ Compute the signal power
    
    The function is used to compute the signal power. You can do it by averaging
    the signal power across all the receivers or in a way that the signal power
    of each receiver is independent.
    
    Parameters
    ----------
    signal_spk : numpy ndArray
        Signal spectrum. It can be a MxNf matrix, with M receivers and Nf frequencies
    snr : float
        The signal to noise ratio you want to emulate
    avg_across_recs : bool
        If signal power is one for each receiver or computed as a mean power
        across the array. If avg_across_recs is False, the the noise power is 
        different for each receiver and frequency. If avg_across_recs is True,
        the noise power is calculated from the average signal magnitude of all 
        receivers (for each frequency). The default value is False
    Returns
        ----------
        signalPower_lin : numpy ndArray
            Signal power in linear scale
        signalPower_dB : numpy ndArray
            Signal power in dB
    """
    if avg_across_recs:
        signalPower_lin = np.mean((np.abs(signal_spk)**2)/np.sqrt(2), axis=0)
    else:
        signalPower_lin = (np.abs(signal_spk)/np.sqrt(2))**2
    signalPower_dB = 10 * np.log10(signalPower_lin)
    return signalPower_lin, signalPower_dB

def get_noise_power(signalPower_dB, snr):
    """ Compute the noise power
    
    Parameters
    ----------
        signalPower_dB : numpy ndArray
            Signal power in dB
        snr : float
            The signal to noise ratio you want to emulate
    Returns
        ----------
        noisePower_lin : numpy ndArray
            Signal power in linear scale
        noisePower_dB : numpy ndArray
            Signal power in dB
    """
    noisePower_dB = signalPower_dB - snr
    noisePower_lin = 10 ** (noisePower_dB/10)
    return noisePower_lin, noisePower_dB

def add_noise(signal_spk, snr = 30, avg_across_recs = False):
        """ Add gaussian noise to the simulated data.

        The function is used to add noise to the spectrum of a signal.
        It reads the clean signal and estimate its power. Then, it estimates the power
        of the noise that would lead to the target SNR. Then, it draws random numbers
        from a Normal distribution with standard deviation =  noise power

        Parameters
        ----------
        signal_spk : numpy ndArray
            Signal spectrum. It can be a MxNf matrix, with M receivers and Nf frequencies
        snr : float
            The signal to noise ratio you want to emulate
        uncorr : bool
            If added noise to each receiver is uncorrelated or not.
            If uncorr is True the the noise power is different for each receiver
            and frequency. If uncorr is False the noise power is calculated from
            the average signal magnitude of all receivers (for each frequency).
            The default value is False
        
        Returns
        ----------
        signal_spk_noisy : numpy ndArray
            Noisy signal spectrum. Same shape as signal_spk
        """
        if uncorr:
            signalPower_lin = (np.abs(signal_spk)/np.sqrt(2))**2
            signalPower_dB = 10 * np.log10(signalPower_lin)
            noisePower_dB = signalPower_dB - snr
            noisePower_lin = 10 ** (noisePower_dB/10)
        else:
            signalPower_lin = np.mean((np.abs(signal_spk)**2)/np.sqrt(2), axis=0)
            signalPower_dB = 10 * np.log10(signalPower_lin)
            noisePower_dB = signalPower_dB - snr
            noisePower_lin = 10 ** (noisePower_dB/10)
        np.random.seed(0)
        noise = np.random.normal(0, np.sqrt(noisePower_lin), size = signal_spk.shape) +\
                1j*np.random.normal(0, np.sqrt(noisePower_lin), size = signal_spk.shape)
        signal_spk_noisy = signal_spk + noise
        return signal_spk_noisy
    
def add_noise2(signal_spk, snr = 30, uncorr = False):
        """ Add gaussian noise to the simulated data.

        The function is used to add noise to the spectrum of a signal.
        It reads the clean signal and estimate its power. Then, it estimates the power
        of the noise that would lead to the target SNR. Then, it draws random numbers
        from a Normal distribution with standard deviation =  noise power

        Parameters
        ----------
        signal_spk : numpy ndArray
            Signal spectrum. It can be a MxNf matrix, with M receivers and Nf frequencies
        snr : float
            The signal to noise ratio you want to emulate
        uncorr : bool
            If added noise to each receiver is uncorrelated or not.
            If uncorr is True the the noise power is different for each receiver
            and frequency. If uncorr is False the noise power is calculated from
            the average signal magnitude of all receivers (for each frequency).
            The default value is False
        
        Returns
        ----------
        signal_spk_noisy : numpy ndArray
            Noisy signal spectrum. Same shape as signal_spk
        """
        if uncorr:
            signalAmp_lin = np.abs(signal_spk)
            signalAmp_dB = 20 * np.log10(signalAmp_lin)
            noiseAmp_dB = signalAmp_dB - snr
            noiseAmp_lin = 20 ** (noiseAmp_dB/10)
        else:
            signalAmp_lin = np.mean(np.abs(signal_spk), axis=0)
            signalAmp_dB = 10 * np.log10(signalAmp_lin)
            noiseAmp_dB = signalAmp_dB - snr
            noiseAmp_lin = 20 ** (noiseAmp_dB/10)
        #np.random.seed(0)
        noise = np.random.normal(0, np.sqrt(noiseAmp_lin/2), size = signal_spk.shape) +\
                1j*np.random.normal(0, np.sqrt(noiseAmp_lin/2), size = signal_spk.shape)
        signal_spk_noisy = signal_spk + noise
        meas_snr = np.mean(20 * np.log10(np.abs(signal_spk)) - 20 * np.log10(np.abs(noise)))
        print('mean snr {}'.format(meas_snr))
        return signal_spk_noisy