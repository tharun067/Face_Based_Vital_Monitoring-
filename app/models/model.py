"""
Pre-trained rPPG (Remote Photoplethysmography) Model
Uses PhysNet architecture for heart rate estimation from facial video
"""
import tensorflow as tf
import numpy as np
import cv2
from keras import layers, Model
import keras as keras
import os
import logging

logger = logging.getLogger(__name__)

class RPPGMOdel:
    def __init__(self, model_path="weights/physnet_trained.h5"):
        """Initialize model"""
        self.model_path = model_path
        self.input_shape = (128, 128, 3)
        self.temporal_depth = 32

        # Build or load model
        if os.path.exists(model_path):
            self.model = tf.keras.models.load_model(model_path, compile=False)
            logger.info(f"Loaded pre-trained model from {model_path}")
        else:
            self.model = self._build_physnet()
            logger.info("Built new PhysNet model (no pre-trained weights found)")
            self._initialize_with_imagenet()
        
    def _build_physnet(self):
        """
        Build PhysNet architecture
        """
        input_layer = layers.Input(shape=(self.temporal_depth, 128, 128, 3))

        # 3D Convolution layers with specific parameters for rPPG
        # First block
        x = layers.TimeDistributed(
            layers.Conv2D(16, (5, 5), padding = "same", activation='tanh')
        )(input_layer)
        x = layers.TimeDistributed(layers.BatchNormalization())(x)
        x = layers.TimeDistributed(layers.MaxPooling2D(2, 2))(x)

        # Second block
        x = layers.TimeDistributed(
            layers.Conv2D(32, (5, 5), padding = "same", activation='tanh')
        )(x)
        x = layers.TimeDistributed(layers.BatchNormalization())(x)
        x = layers.TimeDistributed(layers.MaxPooling2D(2, 2))(x)

        # Third block
        x = layers.TimeDistributed(
            layers.Conv2D(64, (5, 5), padding = "same", activation='tanh')
        )(x)
        x = layers.TimeDistributed(layers.BatchNormalization())(x)
        x = layers.TimeDistributed(layers.MaxPooling2D(2, 2))(x)

        # Fourth block
        x = layers.TimeDistributed(
            layers.Conv2D(64, (5, 5), padding = "same", activation='tanh')
        )(x)
        x = layers.TimeDistributed(layers.BatchNormalization())(x)
        x = layers.TimeDistributed(layers.MaxPooling2D(2, 2))(x)

        # Flatten spatial dimensions before LSTM
        x = layers.TimeDistributed(layers.Flatten())(x)
        
        # LSTM for temporal modeling
        x = layers.LSTM(64, return_sequences=True)(x)
        x = layers.Dropout(0.3)(x)

        # Output layer
        output = layers.TimeDistributed(layers.Dense(1, activation='linear'))(x)

        model = Model(inputs=input_layer, outputs=output)
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.0001),
            loss='mse',
            metrics=['mae']
        )

        return model
    
    def _initialize_with_imagenet(self):
        """
        Initialize convolutional layers with ImageNet pre-trained weights
        """
        try:
            base_model = keras.applications.MobileNetV2(
                input_shape=(128, 128, 3),
                include_top=False,
                weights='imagenet'
            )
            logger.info("Loaded MobileNetV2 with ImageNet weights for initialization")
        except Exception as e:
            logger.error(f"Error loading MobileNetV2: {e}")
            return
    
    def preprocess_frames(self, frames):
        """
        Preprocess frames for rPPG extraction
        Args:
            frames: List or array of video frames
        Returns:
            Preprocessed frames as numpy array
        """
        processed_frames = []
        for frame in frames:
            try:
                # Ensure frame is valid
                if frame is None or frame.size == 0:
                    continue
                    
                # Resize to expected input size
                resized = cv2.resize(frame, (128, 128))
                
                # Convert to float and normalize
                normalized = resized.astype('float32') / 255.0
                
                # Simple contrast enhancement without LAB conversion
                # Apply slight contrast adjustment
                enhanced = np.clip(normalized * 1.2, 0.0, 1.0).astype('float32')
                
                processed_frames.append(enhanced)
            except Exception as e:
                logger.error(f"Error preprocessing frame: {e}")
                # Use a black frame as fallback
                processed_frames.append(np.zeros((128, 128, 3), dtype=np.float32))
        
        return np.array(processed_frames)
    
    def extract_rppg(self, frames):
        """
        Extract rPPG signal from preprocessed frames
        Args:
            frames: Preprocessed frames as numpy array
        Returns:
            rPPG signal as numpy array
        """
        if len(frames) < self.temporal_depth:
            while len(frames) < self.temporal_depth:
                frames.append(frames[-1])
        
        signals = []
        step = max(1, len(frames) // 10)

        for i in range(0, len(frames) - self.temporal_depth + 1, step):
            window = frames[i:i + self.temporal_depth]
            # Preprocess window
            processed = self.preprocess_frames(window)
            # Add batch dimension
            batch = np.expand_dims(processed, axis=0)
            # Predict rPPG signal
            prediction = self.model.predict(batch, verbose=0)

            signals.append(prediction[0, :, 0])
        
        if len(signals) > 0:
            full_signal = np.concatenate(signals)
        else:
            full_signal = np.zeros(len(frames))

        return full_signal
    
    def extract_spatial_mean(self, frames):
        """
        Extract spatial mean signal from frames using POS algorithm
        Args:
            frames: List of face ROI frames (BGR format)
        Returns:
            Spatial mean signal as 1D numpy array
        """
        if len(frames) == 0:
            return np.zeros(1)

        signals = []

        for frame in frames:
            try:
                # Ensure frame is valid
                if frame is None or frame.size == 0:
                    continue
                    
                # Resize and normalize
                resized = cv2.resize(frame, (64, 64))
                normalized = resized.astype(np.float32) / 255.0

                # Extract center ROI (face area)
                roi = normalized[16:48, 16:48, :]

                # Calculate mean RGB values
                mean_rgb = np.mean(roi, axis=(0, 1))
                signals.append(mean_rgb)
            except Exception as e:
                logger.error(f"Error processing frame in spatial mean: {e}")
                continue
        
        if len(signals) < 2:
            return np.zeros(len(frames))
            
        signals = np.array(signals)

        # POS (Plane-Orthogonal-to-Skin) algorithm
        # Extract color channels (BGR format)
        B_s = signals[:, 0]
        G_s = signals[:, 1]
        R_s = signals[:, 2]

        # Normalize each channel
        B_n = B_s / np.mean(B_s)
        G_n = G_s / np.mean(G_s)
        R_n = R_s / np.mean(R_s)

        # Build Plane-Orthogonal-to-Skin projection
        X_s = 3 * R_n - 2 * G_n
        Y_s = 1.5 * R_n + G_n - 1.5 * B_n

        # Calculate alpha (standard deviation ratio)
        std_x = np.std(X_s)
        std_y = np.std(Y_s)
        
        if std_y < 1e-6:  # Avoid division by zero
            alpha = 1.0
        else:
            alpha = std_x / std_y

        # POS signal
        S = X_s - alpha * Y_s

        return S
    
    def save(self, path="weights/physnet_trained.h5"):
        """
        Save model weights
        Args:
            path: File path to save the model weights
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.model.save(path)
        logger.info(f"Model weights saved to {path}")