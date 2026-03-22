
import numpy as np
import pywt
from scipy.signal import wiener

def wavelet_denoise(signal, wavelet='sym4', level=1):
    """
    Decomposes signal and thresholds high-frequency details to isolate 1/f trend.
    
    Args:
        signal (array): Input signal.
        wavelet (str): Wavelet family (e.g. 'sym4', 'db4').
        level (int): Decomposition level.
        
    Returns:
        array: Denoised signal (reconstructed from approximation coeffs).
    """
    # Decompose
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    
    # Thresholding: We want to KEEP the trend (approx) and reduce noise (details)
    # Strategy: Zero out the detail coefficients (high frequency noise)
    # This effectively acts as a low-pass filter but adaptable with wavelets
    
    # cA scales as 1/sqrt(2)^j, so it contains the low-freq trend (1/f)
    # cD contains the high-freq fluctuations (white noise)
    
    # Hard thresholding of details
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    threshold = sigma * np.sqrt(2 * np.log(len(signal)))
    
    new_coeffs = [coeffs[0]] # Keep approximation
    for i in range(1, len(coeffs)):
        # Soft thresholding
        new_coeffs.append(pywt.threshold(coeffs[i], threshold, mode='soft'))
        
    return pywt.waverec(new_coeffs, wavelet)

def spectral_whitening(signal):
    """
    Flattens the 1/f spectrum to reveal hidden peaks.
    Effectively differentiation, but in frequency domain or via AR model.
    Here we use a simple differentiation difference filter approach.
    """
    return np.diff(signal, prepend=signal[0])

def sax_encoding(signal, alphabet_size=5, behavior='quantile'):
    """
    Symbolic Aggregate approXimation (SAX).
    Maps signal values to letters.
    
    Args:
        signal (array): Input time series.
        alphabet_size (int): Size of alphabet.
        behavior (str): 'gaussian' (default Z-norm assumption) or 'quantile' (empirical equiprobable).
    
    Returns:
        str: Symbol string.
    """
    from scipy.stats import norm
    
    # 1. Normalize? 
    # If partial gaussian, Z-norm is good.
    # If quantile, rank matters.
    sig_norm = (signal - np.mean(signal)) / (np.std(signal) + 1e-10)
    
    if behavior == 'gaussian':
        # Define breakpoints based on Normal dist
        breakpoints = norm.ppf(np.linspace(0, 1, alphabet_size + 1)[1:-1])
    else:
        # Empirical Quantiles (Force equiprobable symbols)
        breakpoints = np.percentile(sig_norm, np.linspace(0, 100, alphabet_size + 1)[1:-1])
    
    # Map values to symbols
    symbols = []
    # Vectorize this? For clarity loop is fine for these lengths
    # But for speed, numpy digitize is better
    indices = np.digitize(sig_norm, breakpoints)
    # indices will be 0..alphabet_size-1 (if < bp[0] -> 0)
    # digitize returns 0 if < bp[0], 1 if bp[0]<=x<bp[1], ... len(bp) if > last
    # We have alphabet_size-1 breakpoints.
    # So indices: 0..alphabet_size-1.
    
    for idx in indices:
        # clamp just in case
        idx = min(max(0, idx), alphabet_size - 1)
        symbols.append(chr(97 + idx))
        
    return "".join(symbols)
