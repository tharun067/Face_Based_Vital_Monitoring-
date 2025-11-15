"""
Vitals Calculator
Calculates heart rate, HRV, SpO2, stress index, and respiratory rate
"""

import numpy as np
from scipy import signal
from collections import deque

class VitalsCalculator:
    def __init__(self):
        """Initialize vitals calculator"""
        self.hr_history = deque(maxlen=30)
        self.hrv_history = deque(maxlen=30)
        self.spo2_history = deque(maxlen=30)
        self.rr_history = deque(maxlen=30)
        
        self.previous_peaks = []
        self.last_calculation_time = 0
    
    def calculate_vitals(self, rppg_signal, fps=30):
        """
        Calculate all vital signs from rPPG signal
        Args:
            rppg_signal: Filtered rPPG signal
            fps: Frame rate
        Returns:
            Dictionary with all vitals
        """
        vitals = {
            'heart_rate': 0.0,
            'hrv': 0.0,
            'spo2': 0.0,
            'stress_index': 0.0,
            'respiratory_rate': 0.0,
            'signal_quality': 0.0
        }
        
        if len(rppg_signal) < 30:
            return vitals
        
        try:
            # Calculate heart rate
            hr, peak_indices = self._calculate_heart_rate(rppg_signal, fps)
            vitals['heart_rate'] = hr
            self.hr_history.append(hr)
            
            # Calculate HRV
            if len(peak_indices) > 2:
                hrv = self._calculate_hrv(peak_indices, fps)
                vitals['hrv'] = hrv
                self.hrv_history.append(hrv)
            
            # Calculate SpO2 (estimation from signal characteristics)
            spo2 = self._estimate_spo2(rppg_signal, hr)
            vitals['spo2'] = spo2
            self.spo2_history.append(spo2)
            
            # Calculate stress index
            if len(self.hr_history) > 10 and len(self.hrv_history) > 10:
                stress = self._calculate_stress_index()
                vitals['stress_index'] = stress
            
            # Calculate respiratory rate
            rr = self._calculate_respiratory_rate(rppg_signal, fps)
            vitals['respiratory_rate'] = rr
            self.rr_history.append(rr)
            
            # Apply smoothing to vitals
            vitals = self._smooth_vitals(vitals)
            
        except Exception as e:
            print(f"Error calculating vitals: {e}")
        
        return vitals
    
    def _calculate_heart_rate(self, rppg_signal, fps):
        """
        Calculate heart rate from rPPG signal
        Returns: HR in bpm and peak indices
        """
        # Find peaks in signal
        # Use adaptive threshold based on signal statistics
        threshold = np.mean(rppg_signal) + 0.3 * np.std(rppg_signal)
        
        peaks, properties = signal.find_peaks(
            rppg_signal,
            height=threshold,
            distance=int(fps * 0.4),  # Minimum 0.4s between peaks (150 bpm max)
            prominence=0.1
        )
        
        if len(peaks) < 2:
            # Try FFT method as fallback
            return self._calculate_hr_fft(rppg_signal, fps), peaks
        
        # Calculate heart rate from peak intervals
        peak_intervals = np.diff(peaks) / fps  # Convert to seconds
        
        if len(peak_intervals) == 0:
            return 0.0, peaks
        
        # Remove outliers
        median_interval = np.median(peak_intervals)
        valid_intervals = peak_intervals[
            (peak_intervals > median_interval * 0.7) & 
            (peak_intervals < median_interval * 1.3)
        ]
        
        if len(valid_intervals) == 0:
            valid_intervals = peak_intervals
        
        # Calculate BPM
        avg_interval = np.mean(valid_intervals)
        hr = 60.0 / avg_interval if avg_interval > 0 else 0.0
        
        # Constrain to physiological range
        hr = np.clip(hr, 40, 200)
        
        return hr, peaks
    
    def _calculate_hr_fft(self, rppg_signal, fps):
        """Calculate heart rate using FFT"""
        n = len(rppg_signal)
        
        # Apply window
        window = np.hanning(n)
        windowed = rppg_signal * window
        
        # Compute FFT
        fft_vals = np.fft.fft(windowed)
        fft_freqs = np.fft.fftfreq(n, 1.0 / fps)
        
        # Positive frequencies only
        positive_freqs = fft_freqs[:n // 2]
        power = np.abs(fft_vals[:n // 2]) ** 2
        
        # Heart rate range: 0.7-4.0 Hz (42-240 bpm)
        hr_mask = (positive_freqs >= 0.7) & (positive_freqs <= 4.0)
        
        if not np.any(hr_mask):
            return 0.0
        
        # Find peak frequency
        hr_freqs = positive_freqs[hr_mask]
        hr_power = power[hr_mask]
        
        peak_freq = hr_freqs[np.argmax(hr_power)]
        hr = peak_freq * 60.0  # Convert to BPM
        
        return hr
    
    def _calculate_hrv(self, peak_indices, fps):
        """
        Calculate Heart Rate Variability (SDNN)
        Args:
            peak_indices: Indices of detected peaks
            fps: Frame rate
        Returns:
            HRV in milliseconds
        """
        if len(peak_indices) < 3:
            return 0.0
        
        # Calculate RR intervals (in milliseconds)
        rr_intervals = np.diff(peak_indices) / fps * 1000
        
        # Remove outliers
        median_rr = np.median(rr_intervals)
        valid_rr = rr_intervals[
            (rr_intervals > median_rr * 0.7) & 
            (rr_intervals < median_rr * 1.3)
        ]
        
        if len(valid_rr) < 2:
            return 0.0
        
        # Calculate SDNN (Standard Deviation of NN intervals)
        hrv_sdnn = np.std(valid_rr)
        
        # Constrain to reasonable range
        hrv_sdnn = np.clip(hrv_sdnn, 0, 200)
        
        return hrv_sdnn
    
    def _estimate_spo2(self, rppg_signal, heart_rate):
        """
        Estimate SpO2 from rPPG signal characteristics
        Note: This is a simplified estimation. Real SpO2 requires
        dual-wavelength measurements (red and infrared).
        """
        # Use signal characteristics as proxy
        # Higher signal quality and amplitude typically correlates with better oxygenation
        
        if len(rppg_signal) < 30 or heart_rate < 40 or heart_rate > 200:
            # Return normal value with uncertainty
            return 97.0
        
        # Calculate signal characteristics
        signal_std = np.std(rppg_signal)
        signal_range = np.max(rppg_signal) - np.min(rppg_signal)
        
        # Normalize features
        quality_score = min(1.0, signal_range / 2.0)
        regularity_score = 1.0 / (1.0 + signal_std)
        
        # Estimate SpO2 (baseline 95%, adjust based on signal quality)
        base_spo2 = 95.0
        quality_adjustment = quality_score * 3.0
        regularity_adjustment = regularity_score * 2.0
        
        spo2 = base_spo2 + quality_adjustment + regularity_adjustment
        
        # Constrain to physiological range
        spo2 = np.clip(spo2, 90, 100)
        
        return spo2
    
    def _calculate_stress_index(self):
        """
        Calculate stress index from HR and HRV
        Lower HRV and higher HR = higher stress
        Returns: Stress index (0-100)
        """
        if len(self.hr_history) < 5 or len(self.hrv_history) < 5:
            return 50.0
        
        # Get recent averages
        avg_hr = np.mean(list(self.hr_history)[-10:])
        avg_hrv = np.mean(list(self.hrv_history)[-10:])
        
        # Normalize HR (assume 60-80 is normal, >90 is stressed)
        hr_stress = (avg_hr - 60) / 30.0
        hr_stress = np.clip(hr_stress, 0, 1)
        
        # Normalize HRV (higher is better, assume 50ms is normal)
        hrv_stress = 1.0 - min(1.0, avg_hrv / 100.0)
        
        # Calculate HR variability (inconsistent HR = stress)
        hr_variance = np.std(list(self.hr_history)[-10:])
        variance_stress = min(1.0, hr_variance / 20.0)
        
        # Combined stress index
        stress_index = (
            hr_stress * 0.4 + 
            hrv_stress * 0.4 + 
            variance_stress * 0.2
        ) * 100
        
        stress_index = np.clip(stress_index, 0, 100)
        
        return stress_index
    
    def _calculate_respiratory_rate(self, rppg_signal, fps):
        """
        Calculate respiratory rate from rPPG signal
        Breathing modulates heart rate (respiratory sinus arrhythmia)
        """
        if len(rppg_signal) < fps * 10:  # Need at least 10 seconds
            return 15.0  # Return normal value
        
        # Apply lowpass filter for respiratory frequency range (0.1-0.5 Hz)
        nyquist = fps / 2
        low_cutoff = 0.1 / nyquist
        high_cutoff = 0.5 / nyquist
        
        try:
            sos = signal.butter(4, [low_cutoff, high_cutoff], btype='band', output='sos')
            resp_signal = signal.sosfiltfilt(sos, rppg_signal)
            
            # Find peaks in respiratory signal
            peaks, _ = signal.find_peaks(
                resp_signal,
                distance=int(fps * 2),  # Minimum 2 seconds between breaths
                prominence=0.05
            )
            
            if len(peaks) < 2:
                return 15.0
            
            # Calculate respiratory rate
            breath_intervals = np.diff(peaks) / fps
            avg_interval = np.mean(breath_intervals)
            
            if avg_interval > 0:
                rr = 60.0 / avg_interval  # Breaths per minute
                rr = np.clip(rr, 8, 30)  # Physiological range
                return rr
            
        except Exception as e:
            print(f"Respiratory rate calculation error: {e}")
        
        return 15.0
    
    def _smooth_vitals(self, vitals):
        """Apply temporal smoothing to vitals"""
        smoothed = vitals.copy()
        
        # Smooth heart rate
        if len(self.hr_history) > 0:
            recent_hrs = list(self.hr_history)[-5:]
            if len(recent_hrs) > 0:
                smoothed['heart_rate'] = np.mean([vitals['heart_rate']] + recent_hrs)
        
        # Smooth HRV
        if len(self.hrv_history) > 0:
            recent_hrv = list(self.hrv_history)[-5:]
            if len(recent_hrv) > 0:
                smoothed['hrv'] = np.mean([vitals['hrv']] + recent_hrv)
        
        # Smooth SpO2
        if len(self.spo2_history) > 0:
            recent_spo2 = list(self.spo2_history)[-5:]
            if len(recent_spo2) > 0:
                smoothed['spo2'] = np.mean([vitals['spo2']] + recent_spo2)
        
        # Smooth respiratory rate
        if len(self.rr_history) > 0:
            recent_rr = list(self.rr_history)[-5:]
            if len(recent_rr) > 0:
                smoothed['respiratory_rate'] = np.mean([vitals['respiratory_rate']] + recent_rr)
        
        return smoothed
    
    def get_vital_trends(self):
        """
        Get trends in vitals over time
        Returns dictionary with trend information
        """
        trends = {
            'hr_trend': 'stable',
            'hrv_trend': 'stable',
            'spo2_trend': 'stable',
            'rr_trend': 'stable'
        }
        
        if len(self.hr_history) > 10:
            hr_list = list(self.hr_history)
            early_hr = np.mean(hr_list[:5])
            recent_hr = np.mean(hr_list[-5:])
            
            if recent_hr > early_hr + 5:
                trends['hr_trend'] = 'increasing'
            elif recent_hr < early_hr - 5:
                trends['hr_trend'] = 'decreasing'
        
        # Similar analysis for other vitals
        if len(self.hrv_history) > 10:
            hrv_list = list(self.hrv_history)
            if np.mean(hrv_list[-5:]) > np.mean(hrv_list[:5]) + 5:
                trends['hrv_trend'] = 'increasing'
            elif np.mean(hrv_list[-5:]) < np.mean(hrv_list[:5]) - 5:
                trends['hrv_trend'] = 'decreasing'
        
        return trends