"""
Face detection utility functions.
"""

import cv2
import numpy as np
import mediapipe as mp
from collections import deque
import logging

logger = logging.getLogger(__name__)

class FaceDetector:
    def __init__(self, use_mediapipe=True):
        """Initialize the FaceDetector with the chosen detection method."""
        self.use_mediapipe = use_mediapipe

        # OpenCV Cascade Classifier 
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )

        # MediaPipe Face Detection
        if use_mediapipe:
            try:
                self.mp_face_detection = mp.solutions.face_detection
                self.mp_face_mesh = mp.solutions.face_mesh
                self.face_detection = self.mp_face_detection.FaceDetection(
                    model_selection=1,
                    min_detection_confidence=0.5
                )
                self.face_mesh = self.mp_face_mesh.FaceMesh(
                    max_num_faces=1,
                    refine_landmarks=True,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5
                )
                logger.info("MediaPipe Face Detection initialized successfully.")
            except Exception as e:
                logger.error(f"Failed to initialize MediaPipe Face Detection: {e}")
                self.use_mediapipe = False
        
        # Tracking variables
        self.last_face_roi = None
        self.last_face_coords = None
        self.tracking_history = deque(maxlen=10)
        self.detection_confidence = 0.0
    def detect_faces(self, frame):
        """Detect face in frame
        Returns:
            face_roi: Cropped face region
            face_coords: Coordinates of the detected face (x, y, w, h)
        """

        if self.use_mediapipe:
            return self._detect_faces_mediapipe(frame)
        else:
            return self._detect_faces_opencv(frame)
    
    def _detect_faces_mediapipe(self, frame):
        """Detect faces using MediaPipe."""
        try:
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w = frame.shape[:2]
            
            # Detect faces
            results = self.face_detection.process(rgb_frame)
            
            if results.detections:
                detection = results.detections[0]
                bboxC = detection.location_data.relative_bounding_box
                
                # Convert to pixel coordinates
                x = int(bboxC.xmin * w)
                y = int(bboxC.ymin * h)
                width = int(bboxC.width * w)
                height = int(bboxC.height * h)
                
                # Expand ROI slightly for better coverage
                expand = 0.2
                x = max(0, int(x - width * expand / 2))
                y = max(0, int(y - height * expand / 2))
                width = min(w - x, int(width * (1 + expand)))
                height = min(h - y, int(height * (1 + expand)))
                
                # Extract ROI
                face_roi = frame[y:y+height, x:x+width]
                
                # Smooth tracking
                coords = self._smooth_tracking((x, y, width, height))
                
                self.last_face_roi = face_roi
                self.last_face_coords = coords
                self.detection_confidence = detection.score[0]
                
                return face_roi, coords
            
            # Use last known face if available
            elif self.last_face_roi is not None and self.detection_confidence > 0.7:
                return self.last_face_roi, self.last_face_coords
            
            return None, None
        except Exception as e:
            logger.error(f"Error in MediaPipe face detection: {e}")
            return None, None
    def _detect_faces_opencv(self, frame):
        """Detect face using OpenCV Cascade Classifier"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(100, 100)
        )
        
        if len(faces) > 0:
            # Take largest face
            x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
            
            # Expand ROI
            expand = 0.1
            x = max(0, int(x - w * expand / 2))
            y = max(0, int(y - h * expand / 2))
            w = int(w * (1 + expand))
            h = int(h * (1 + expand))
            
            # Ensure within bounds
            h_img, w_img = frame.shape[:2]
            w = min(w, w_img - x)
            h = min(h, h_img - y)
            
            face_roi = frame[y:y+h, x:x+w]
            coords = self._smooth_tracking((x, y, w, h))
            
            self.last_face_roi = face_roi
            self.last_face_coords = coords
            
            return face_roi, coords
        
        # Use last known face
        elif self.last_face_roi is not None:
            return self.last_face_roi, self.last_face_coords
        
        return None, None
    
    def _smooth_tracking(self, coords):
        """Apply temporal smoothing to face coordinates"""
        self.tracking_history.append(coords)
        
        if len(self.tracking_history) < 3:
            return coords
        
        # Average last few detections
        x_vals = [c[0] for c in self.tracking_history]
        y_vals = [c[1] for c in self.tracking_history]
        w_vals = [c[2] for c in self.tracking_history]
        h_vals = [c[3] for c in self.tracking_history]
        
        smoothed = (
            int(np.mean(x_vals)),
            int(np.mean(y_vals)),
            int(np.mean(w_vals)),
            int(np.mean(h_vals))
        )
        
        return smoothed
    
    def get_facial_landmarks(self, frame):
        """
        Get facial landmarks using MediaPipe Face Mesh
        Returns list of (x, y) coordinates
        """
        if not self.use_mediapipe:
            return None
        
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb_frame)
            
            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0]
                h, w = frame.shape[:2]
                
                points = []
                for landmark in landmarks.landmark:
                    x = int(landmark.x * w)
                    y = int(landmark.y * h)
                    points.append((x, y))
                
                return points
            
            return None
        except Exception as e:
            logger.error(f"Error in MediaPipe face mesh: {e}")
            return None
    def extract_skin_region(self, frame):
        """
        Extract skin-dominant regions for better rPPG signal
        Returns masked frame
        """
        # Convert to YCrCb color space
        ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        
        # Define skin color range in YCrCb
        lower = np.array([0, 133, 77], dtype=np.uint8)
        upper = np.array([255, 173, 127], dtype=np.uint8)
        
        # Create mask
        skin_mask = cv2.inRange(ycrcb, lower, upper)
        
        # Morphological operations to clean up mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)
        
        # Apply mask
        masked_frame = cv2.bitwise_and(frame, frame, mask=skin_mask)
        
        return masked_frame, skin_mask
    
    def get_roi_regions(self, frame):
        """
        Get multiple ROI regions (forehead, cheeks) for robust rPPG
        Returns dictionary of ROI frames
        """
        landmarks = self.get_facial_landmarks(frame)
        
        if landmarks is None:
            return {'full_face': frame}
        
        h, w = frame.shape[:2]
        rois = {}
        
        # Forehead region (landmarks 10, 151, 9, 107, 66, 109, 69)
        try:
            forehead_points = [landmarks[i] for i in [10, 151, 9, 107, 66, 109, 69]]
            x_coords = [p[0] for p in forehead_points]
            y_coords = [p[1] for p in forehead_points]
            
            x_min, x_max = max(0, min(x_coords)), min(w, max(x_coords))
            y_min, y_max = max(0, min(y_coords)), min(h, max(y_coords))
            
            rois['forehead'] = frame[y_min:y_max, x_min:x_max]
        except:
            pass
        
        # Left cheek
        try:
            left_cheek = [landmarks[i] for i in [234, 93, 132, 58, 172]]
            x_coords = [p[0] for p in left_cheek]
            y_coords = [p[1] for p in left_cheek]
            
            x_min, x_max = max(0, min(x_coords)), min(w, max(x_coords))
            y_min, y_max = max(0, min(y_coords)), min(h, max(y_coords))
            
            rois['left_cheek'] = frame[y_min:y_max, x_min:x_max]
        except:
            pass
        
        # Right cheek
        try:
            right_cheek = [landmarks[i] for i in [454, 323, 361, 288, 397]]
            x_coords = [p[0] for p in right_cheek]
            y_coords = [p[1] for p in right_cheek]
            
            x_min, x_max = max(0, min(x_coords)), min(w, max(x_coords))
            y_min, y_max = max(0, min(y_coords)), min(h, max(y_coords))
            
            rois['right_cheek'] = frame[y_min:y_max, x_min:x_max]
        except:
            pass
        
        if not rois:
            rois['full_face'] = frame
        
        return rois
    