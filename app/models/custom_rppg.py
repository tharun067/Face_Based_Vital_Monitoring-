"""
Custom rPPG Model - Trained from scratch
Uses 3D CNN + Attention mechanism for vital signs extraction
"""

import tensorflow as tf
import numpy as np
import cv2
from keras import layers, Model
import os

class CustomRPPGModel:
    def __init__(self, model_path='weights/custom_rppg.h5'):
        """Initialize custom trained model"""
        self.model_path = model_path
        self.input_shape = (64, 64, 3)
        self.temporal_depth = 64  # Number of frames
        
        if os.path.exists(model_path):
            self.model = tf.keras.models.load_model(
                model_path, 
                custom_objects={'attention_block': self.attention_block},
                compile=False
            )
            print(f"Loaded custom model from {model_path}")
        else:
            self.model = self._build_custom_model()
            print("Built new custom model (no weights found)")
    
    def attention_block(self, x, name='attention'):
        """Attention mechanism for focusing on relevant features"""
        # Channel attention
        gap = layers.GlobalAveragePooling2D()(x)
        dense1 = layers.Dense(x.shape[-1] // 4, activation='relu')(gap)
        dense2 = layers.Dense(x.shape[-1], activation='sigmoid')(dense1)
        channel_attention = layers.Reshape((1, 1, x.shape[-1]))(dense2)
        
        # Apply attention
        x = layers.Multiply()([x, channel_attention])
        return x
    
    def _build_custom_model(self):
        """
        Build custom 3D CNN model with attention
        Optimized for real-time vitals monitoring
        """
        input_layer = layers.Input(shape=(self.temporal_depth, 64, 64, 3))
        
        # First 3D Conv block
        x = layers.Conv3D(16, (3, 3, 3), padding='same', activation='relu')(input_layer)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling3D((2, 2, 2))(x)
        
        # Second 3D Conv block
        x = layers.Conv3D(32, (3, 3, 3), padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling3D((2, 2, 2))(x)
        
        # Third 3D Conv block
        x = layers.Conv3D(64, (3, 3, 3), padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling3D((2, 1, 1))(x)
        
        # Reshape for 2D processing with attention
        time_steps = x.shape[1]
        x = layers.Reshape((time_steps, -1))(x)
        
        # Bidirectional LSTM with attention
        x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
        x = layers.Dropout(0.4)(x)
        x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)
        x = layers.Dropout(0.3)(x)
        
        # Self-attention mechanism
        attention = layers.Dense(1, activation='tanh')(x)
        attention = layers.Flatten()(attention)
        attention = layers.Activation('softmax')(attention)
        attention = layers.RepeatVector(128)(attention)
        attention = layers.Permute([2, 1])(attention)
        
        attended = layers.Multiply()([x, attention])
        
        # Multiple outputs for different vitals
        # rPPG signal
        rppg_output = layers.TimeDistributed(
            layers.Dense(1, activation='linear', name='rppg')
        )(attended)
        
        # Heart rate estimation
        hr_features = layers.GlobalAveragePooling1D()(attended)
        hr_output = layers.Dense(64, activation='relu')(hr_features)
        hr_output = layers.Dropout(0.2)(hr_output)
        hr_output = layers.Dense(1, activation='relu', name='heart_rate')(hr_output)
        
        # SpO2 estimation
        spo2_output = layers.Dense(64, activation='relu')(hr_features)
        spo2_output = layers.Dropout(0.2)(spo2_output)
        spo2_output = layers.Dense(1, activation='sigmoid', name='spo2')(spo2_output)
        
        model = Model(
            inputs=input_layer,
            outputs={
                'rppg': rppg_output,
                'heart_rate': hr_output,
                'spo2': spo2_output
            }
        )
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
            loss={
                'rppg': 'mse',
                'heart_rate': 'mae',
                'spo2': 'mae'
            },
            loss_weights={
                'rppg': 1.0,
                'heart_rate': 0.5,
                'spo2': 0.5
            },
            metrics={
                'rppg': ['mae'],
                'heart_rate': ['mae'],
                'spo2': ['mae']
            }
        )
        
        return model
    
    def preprocess_frames(self, frames):
        """Preprocess frames for the custom model"""
        processed = []
        
        for frame in frames:
            # Resize
            resized = cv2.resize(frame, (64, 64))
            
            # Convert to YCrCb (better for skin detection)
            ycrcb = cv2.cvtColor(resized, cv2.COLOR_BGR2YCrCb)
            
            # Normalize
            normalized = ycrcb.astype(np.float32) / 255.0
            
            # Standardize
            mean = np.mean(normalized, axis=(0, 1), keepdims=True)
            std = np.std(normalized, axis=(0, 1), keepdims=True)
            standardized = (normalized - mean) / (std + 1e-7)
            
            processed.append(standardized)
        
        return np.array(processed)
    
    def extract_rppg(self, frames):
        """
        Extract rPPG signal using custom model
        Args:
            frames: List of face ROI frames
        Returns:
            rPPG signal as 1D array
        """
        if len(frames) < self.temporal_depth:
            # Pad with replication
            while len(frames) < self.temporal_depth:
                frames.append(frames[-1].copy())
        
        # Process in sliding windows
        signals = []
        step = max(1, len(frames) // 8)
        
        for i in range(0, len(frames) - self.temporal_depth + 1, step):
            window = frames[i:i + self.temporal_depth]
            
            # Preprocess
            processed = self.preprocess_frames(window)
            
            # Add batch dimension
            batch = np.expand_dims(processed, axis=0)
            
            # Predict
            try:
                predictions = self.model.predict(batch, verbose=0)
                
                # Extract rPPG signal
                rppg = predictions['rppg'][0, :, 0]
                signals.append(rppg)
                
            except Exception as e:
                print(f"Prediction error: {e}")
                # Fallback to spatial averaging
                fallback = self._extract_chrom(window)
                signals.append(fallback)
        
        # Combine signals
        if len(signals) > 0:
            full_signal = np.concatenate(signals)
        else:
            full_signal = np.zeros(len(frames))
        
        return full_signal
    
    def _extract_chrom(self, frames):
        """
        CHROM algorithm as fallback
        Chrominance-based rPPG extraction
        """
        rgb_signals = []
        
        for frame in frames:
            resized = cv2.resize(frame, (64, 64))
            roi = resized[16:48, 16:48, :]
            mean_rgb = np.mean(roi, axis=(0, 1))
            rgb_signals.append(mean_rgb)
        
        rgb_signals = np.array(rgb_signals)
        
        # CHROM computation
        R = rgb_signals[:, 2]
        G = rgb_signals[:, 1]
        B = rgb_signals[:, 0]
        
        # Normalize
        R_norm = (R - np.mean(R)) / np.std(R)
        G_norm = (G - np.mean(G)) / np.std(G)
        B_norm = (B - np.mean(B)) / np.std(B)
        
        # CHROM signal
        X_s = 3 * R_norm - 2 * G_norm
        Y_s = 1.5 * R_norm + G_norm - 1.5 * B_norm
        
        # Combine
        alpha = np.std(X_s) / np.std(Y_s)
        S = X_s - alpha * Y_s
        
        return S
    
    def predict_vitals(self, frames):
        """
        Directly predict vitals from frames
        Returns dictionary with all vital signs
        """
        if len(frames) < self.temporal_depth:
            while len(frames) < self.temporal_depth:
                frames.append(frames[-1].copy())
        
        # Take last temporal_depth frames
        window = frames[-self.temporal_depth:]
        
        # Preprocess
        processed = self.preprocess_frames(window)
        batch = np.expand_dims(processed, axis=0)
        
        # Predict
        try:
            predictions = self.model.predict(batch, verbose=0)
            
            vitals = {
                'heart_rate': float(predictions['heart_rate'][0, 0] * 120),  # Scale to bpm
                'spo2': float(predictions['spo2'][0, 0] * 100),  # Scale to percentage
                'rppg_signal': predictions['rppg'][0, :, 0]
            }
            
            return vitals
            
        except Exception as e:
            print(f"Vitals prediction error: {e}")
            return None
    
    def save_model(self, path='weights/custom_rppg.h5'):
        """Save model weights"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.model.save(path)
        print(f"Custom model saved to {path}")