"""
Signal Processing Utility for rPPG
Filtering, detrending, and frequency analysis
"""

import numpy as np
from scipy import signal
from scipy.fft import fft, fftfreq
from scipy.signal import butter, filtfilt, detrend, welch

class SignalProcessor:
    def __init__(self, fps=30):
        """
        Initialize signal processor
        Args:
            fps: Frames per second of the video
        """
        self.fps = fps
        self.nyquist = fps / 2
        
        # Frequency ranges for different vitals (in Hz)
        self.hr_range = (0.7, 4.0)  # 42-240 bpm
        self.resp_range = (0.1, 0.5)  # 6-30 rpm
        
    def filter_signal(self, signal_data, method='bandpass'):
        """
        Apply filtering to rPPG signal
        Args:
            signal_data: Raw rPPG signal
            method: 'bandpass', 'lowpass', or 'highpass'
        Returns:
            Filtered signal
        """
        if len(signal_data) < 10:
            return signal_data
        
        # Remove DC component (detrend)
        detrended = detrend(signal_data, type='linear')
        
        # Apply bandpass filter for heart rate
        if method == 'bandpass':
            filtered = self._bandpass_filter(
                detrended, 
                self.hr_range[0], 
                self.hr_range[1]
            )
        elif method == 'lowpass':
            filtered = self._lowpass_filter(detrended, 5.0)
        else:
            filtered = self._highpass_filter(detrended, 0.5)
        
        # Normalize
        if np.std(filtered) > 0:
            filtered = (filtered - np.mean(filtered)) / np.std(filtered)
        
        return filtered
    
    def _bandpass_filter(self, data, low_freq, high_freq, order=4):
        """Apply bandpass Butterworth filter"""
        try:
            low = low_freq / self.nyquist
            high = high_freq / self.nyquist
            
            # Ensure frequencies are valid
            low = max(0.01, min(low, 0.99))
            high = max(0.01, min(high, 0.99))
            
            if low >= high:
                return data
            
            b, a = butter(order, [low, high], btype='band')
            filtered = filtfilt(b, a, data)
            return filtered
        except Exception as e:
            print(f"Bandpass filter error: {e}")
            return data
    
    def _lowpass_filter(self, data, cutoff_freq, order=4):
        """Apply lowpass Butterworth filter"""
        try:
            cutoff = cutoff_freq / self.nyquist
            cutoff = max(0.01, min(cutoff, 0.99))
            
            b, a = butter(order, cutoff, btype='low')
            filtered = filtfilt(b, a, data)
            return filtered
        except Exception as e:
            print(f"Lowpass filter error: {e}")
            return data
    
    def _highpass_filter(self, data, cutoff_freq, order=4):
        """Apply highpass Butterworth filter"""
        try:
            cutoff = cutoff_freq / self.nyquist
            cutoff = max(0.01, min(cutoff, 0.99))
            
            b, a = butter(order, cutoff, btype='high')
            filtered = filtfilt(b, a, data)
            return filtered
        except Exception as e:
            print(f"Highpass filter error: {e}")
            return data
    
    def compute_fft(self, signal_data):
        """
        Compute FFT of signal
        Returns:
            frequencies, power spectrum
        """
        n = len(signal_data)
        
        # Apply window to reduce spectral leakage
        window = np.hanning(n)
        windowed_signal = signal_data * window
        
        # Compute FFT
        fft_values = fft(windowed_signal)
        fft_freqs = fftfreq(n, 1.0 / self.fps)
        
        # Take positive frequencies only
        positive_freqs = fft_freqs[:n // 2]
        power_spectrum = np.abs(fft_values[:n // 2]) ** 2
        
        return positive_freqs, power_spectrum
    
    def compute_psd(self, signal_data):
        """
        Compute Power Spectral Density using Welch's method
        Returns:
            frequencies, psd
        """
        try:
            freqs, psd = welch(
                signal_data,
                fs=self.fps,
                nperseg=min(len(signal_data), 256),
                noverlap=min(len(signal_data) // 2, 128),
                window='hann'
            )
            return freqs, psd
        except Exception as e:
            print(f"PSD computation error: {e}")
            return self.compute_fft(signal_data)
    
    def find_peak_frequency(self, signal_data, freq_range=None):
        """
        Find dominant frequency in signal
        Args:
            signal_data: Input signal
            freq_range: (min_freq, max_freq) tuple to search within
        Returns:
            Peak frequency in Hz
        """
        if freq_range is None:
            freq_range = self.hr_range
        
        # Compute PSD
        freqs, psd = self.compute_psd(signal_data)
        
        # Find indices within frequency range
        mask = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
        
        if not np.any(mask):
            return 0.0
        
        # Find peak
        valid_freqs = freqs[mask]
        valid_psd = psd[mask]
        
        if len(valid_psd) == 0:
            return 0.0
        
        peak_idx = np.argmax(valid_psd)
        peak_freq = valid_freqs[peak_idx]
        
        return peak_freq
    
    def extract_respiratory_signal(self, rppg_signal):
        """
        Extract respiratory signal from rPPG
        Respiratory rate modulates heart rate
        """
        # Lowpass filter for respiratory range
        resp_signal = self._bandpass_filter(
            rppg_signal,
            self.resp_range[0],
            self.resp_range[1]
        )
        
        return resp_signal
    
    def calculate_snr(self, signal_data, peak_freq):
        """
        Calculate Signal-to-Noise Ratio
        Args:
            signal_data: Input signal
            peak_freq: Expected peak frequency
        Returns:
            SNR in dB
        """
        freqs, psd = self.compute_psd(signal_data)
        
        # Find power at peak frequency (±0.1 Hz)
        peak_mask = (freqs >= peak_freq - 0.1) & (freqs <= peak_freq + 0.1)
        signal_power = np.sum(psd[peak_mask])
        
        # Find noise power (excluding peak region)
        noise_mask = ~peak_mask & (freqs >= self.hr_range[0]) & (freqs <= self.hr_range[1])
        noise_power = np.mean(psd[noise_mask]) if np.any(noise_mask) else 1e-10
        
        # Calculate SNR
        snr = 10 * np.log10(signal_power / (noise_power + 1e-10))
        
        return max(0, snr)
    
    def adaptive_filter(self, signal_data, reference_signal=None):
        """
        Adaptive filtering using LMS algorithm
        Removes motion artifacts
        """
        if reference_signal is None or len(reference_signal) != len(signal_data):
            return signal_data
        
        # Simple LMS adaptive filter
        n = len(signal_data)
        mu = 0.01  # Step size
        order = 10
        
        w = np.zeros(order)  # Filter weights
        output = np.zeros(n)
        
        for i in range(order, n):
            x = reference_signal[i-order:i][::-1]
            y = np.dot(w, x)
            e = signal_data[i] - y
            w = w + mu * e * x
            output[i] = e
        
        return output[order:]
    
    def detect_motion_artifacts(self, signal_data, threshold=3.0):
        """
        Detect motion artifacts in signal
        Returns boolean array indicating artifact locations
        """
        # Compute moving standard deviation
        window_size = min(30, len(signal_data) // 10)
        
        if window_size < 3:
            return np.zeros(len(signal_data), dtype=bool)
        
        moving_std = np.zeros(len(signal_data))
        
        for i in range(len(signal_data)):
            start = max(0, i - window_size // 2)
            end = min(len(signal_data), i + window_size // 2)
            moving_std[i] = np.std(signal_data[start:end])
        
        # Detect outliers
        median_std = np.median(moving_std)
        artifacts = moving_std > threshold * median_std
        
        return artifacts
    
    def interpolate_artifacts(self, signal_data, artifacts):
        """
        Interpolate over detected artifacts
        Args:
            signal_data: Input signal
            artifacts: Boolean array of artifact locations
        Returns:
            Cleaned signal
        """
        if not np.any(artifacts):
            return signal_data
        
        cleaned = signal_data.copy()
        indices = np.arange(len(signal_data))
        
        # Good data points
        good_mask = ~artifacts
        good_indices = indices[good_mask]
        good_values = signal_data[good_mask]
        
        if len(good_indices) < 2:
            return signal_data
        
        # Interpolate bad points
        bad_indices = indices[artifacts]
        interpolated = np.interp(bad_indices, good_indices, good_values)
        cleaned[artifacts] = interpolated
        
        return cleaned
    
    def quality_assessment(self, signal_data):
        """
        Assess signal quality
        Returns quality score (0-1)
        """
        if len(signal_data) < 30:
            return 0.0
        
        # Calculate multiple quality metrics
        
        # 1. SNR
        peak_freq = self.find_peak_frequency(signal_data)
        snr = self.calculate_snr(signal_data, peak_freq)
        snr_score = min(1.0, snr / 20.0)  # Normalize
        
        # 2. Kurtosis (should be close to 3 for normal distribution)
        kurtosis = np.abs(3 - np.mean((signal_data - np.mean(signal_data)) ** 4) / (np.std(signal_data) ** 4))
        kurtosis_score = max(0, 1 - kurtosis / 10)
        
        # 3. Peak prominence
        freqs, psd = self.compute_psd(signal_data)
        mask = (freqs >= self.hr_range[0]) & (freqs <= self.hr_range[1])
        
        if np.any(mask):
            peak_value = np.max(psd[mask])
            mean_value = np.mean(psd[mask])
            prominence = peak_value / (mean_value + 1e-10)
            prominence_score = min(1.0, prominence / 5.0)
        else:
            prominence_score = 0.0
        
        # Combined quality score
        quality = (snr_score * 0.4 + kurtosis_score * 0.3 + prominence_score * 0.3)
        
        return quality