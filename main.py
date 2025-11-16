"""
Live Face-Based Vitals Monitoring System - Streamlit Version
Supports both pre-trained models and custom-trained models
"""

import cv2
import numpy as np
import streamlit as st
import time
from datetime import datetime
import queue
from threading import Lock

from app.models.model import RPPGMOdel
from app.utils.face_detector import FaceDetector
from app.utils.signal_processor import SignalProcessor
from app.utils.vital_calculator import VitalsCalculator

# Page configuration
st.set_page_config(
    page_title="Face-Based Vitals Monitor",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    /* Vitals metrics styling */
    [data-testid="stMetricValue"] {
        font-size: 2rem !important;
        font-weight: bold !important;
        color: #1f77b4 !important;
    }
    
    [data-testid="stMetricLabel"] {
        font-size: 1rem !important;
        font-weight: 600 !important;
        color: #262730 !important;
    }
    
    /* Individual metric containers with colored backgrounds */
    div[data-testid="column"]:has([data-testid="stMetric"]) {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    /* Override metric colors for visibility */
    div[data-testid="stMetric"] {
        background-color: rgba(255, 255, 255, 0.95) !important;
        padding: 1rem !important;
        border-radius: 8px !important;
        border-left: 4px solid #667eea !important;
    }
    
    /* Heart Rate - Red theme */
    div[data-testid="column"]:nth-child(1) div[data-testid="stMetric"] {
        border-left-color: #ef4444 !important;
    }
    
    div[data-testid="column"]:nth-child(1) [data-testid="stMetricValue"] {
        color: #ef4444 !important;
    }
    
    /* HRV - Green theme */
    div[data-testid="column"]:nth-child(2) div[data-testid="stMetric"] {
        border-left-color: #10b981 !important;
    }
    
    div[data-testid="column"]:nth-child(2) [data-testid="stMetricValue"] {
        color: #10b981 !important;
    }
    
    /* SpO2 - Blue theme */
    div[data-testid="column"]:nth-child(3) div[data-testid="stMetric"] {
        border-left-color: #3b82f6 !important;
    }
    
    div[data-testid="column"]:nth-child(3) [data-testid="stMetricValue"] {
        color: #3b82f6 !important;
    }
    
    /* Stress - Orange theme */
    div[data-testid="column"]:nth-child(4) div[data-testid="stMetric"] {
        border-left-color: #f59e0b !important;
    }
    
    div[data-testid="column"]:nth-child(4) [data-testid="stMetricValue"] {
        color: #f59e0b !important;
    }
    
    /* Respiratory Rate - Purple theme */
    div[data-testid="stMetric"]:has([aria-label*="Respiratory"]) {
        border-left-color: #8b5cf6 !important;
    }
    
    .stMetric [data-testid="stMetricValue"]:has(+ [aria-label*="Respiratory"]) {
        color: #8b5cf6 !important;
    }
    
    /* Caption/timestamp styling */
    .stCaptionContainer {
        color: #6b7280 !important;
        font-style: italic;
        margin-top: 0.5rem;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'models_initialized' not in st.session_state:
    st.session_state.models_initialized = False
    st.session_state.pretrained_model = None
    st.session_state.face_detector = None
    st.session_state.signal_processor = None
    st.session_state.vitals_calculator = None
    st.session_state.frame_buffer = queue.Queue(maxsize=300)
    st.session_state.current_vitals = {
        'heart_rate': 0.0,
        'hrv': 0.0,
        'spo2': 0.0,
        'stress_index': 0.0,
        'respiratory_rate': 0.0,
        'timestamp': datetime.now().isoformat()
    }
    st.session_state.processing_lock = Lock()

@st.cache_resource
def initialize_models():
    """Initialize all models and processors"""
    print("Initializing models...")
    
    models = {
        'face_detector': None,
        'signal_processor': None,
        'vitals_calculator': None,
        'pretrained_model': None
    }
    
    try:
        # Initialize face detector
        models['face_detector'] = FaceDetector()
        print("✓ Face detector loaded")
        
        # Initialize signal processor
        models['signal_processor'] = SignalProcessor(fps=30)
        print("✓ Signal processor loaded")
        
        # Initialize vitals calculator
        models['vitals_calculator'] = VitalsCalculator()
        print("✓ Vitals calculator loaded")
        
        # Initialize pre-trained model
        try:
            models['pretrained_model'] = RPPGMOdel()
            print("✓ Pre-trained model loaded")
        except Exception as e:
            print(f"✗ Error loading pre-trained model: {e}")
            
    except Exception as e:
        print(f"Error during initialization: {e}")
    
    return models

def process_frame(frame, models):
    """Process a single frame and extract vitals"""
    face_detector = models['face_detector']
    signal_processor = models['signal_processor']
    vitals_calculator = models['vitals_calculator']
    pretrained_model = models['pretrained_model']
    
    # Detect face
    face_roi, face_coords = face_detector.detect_faces(frame)
    
    if face_roi is None:
        return frame, None, False
    
    # Draw face rectangle
    x, y, w, h = face_coords
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    # Add frame to buffer
    if st.session_state.frame_buffer.full():
        st.session_state.frame_buffer.get()
    st.session_state.frame_buffer.put(face_roi)
    
    vitals_updated = False
    buffer_size = st.session_state.frame_buffer.qsize()
    
    # Process if we have enough frames (reduced to 90 frames = 3 seconds)
    if buffer_size >= 90:
        frames = list(st.session_state.frame_buffer.queue)
        
        if pretrained_model is not None:
            try:
                # Extract rPPG signal using spatial mean method
                rppg_signal = pretrained_model.extract_spatial_mean(frames)
                
                print(f"DEBUG: Signal length: {len(rppg_signal)}, Min: {np.min(rppg_signal):.3f}, Max: {np.max(rppg_signal):.3f}, Mean: {np.mean(rppg_signal):.3f}")
                
                # Check if signal is valid
                if len(rppg_signal) < 30 or np.std(rppg_signal) < 0.001:
                    print("WARNING: Signal too weak or too short")
                else:
                    # Process signal
                    filtered_signal = signal_processor.filter_signal(rppg_signal)
                    print(f"DEBUG: Filtered signal - Mean: {np.mean(filtered_signal):.3f}, Std: {np.std(filtered_signal):.3f}")
                    
                    # Calculate vitals
                    vitals = vitals_calculator.calculate_vitals(
                        filtered_signal, 
                        signal_processor.fps
                    )
                    
                    print(f"DEBUG: Calculated HR: {vitals['heart_rate']:.1f}, HRV: {vitals['hrv']:.1f}, SpO2: {vitals['spo2']:.1f}")
                    
                    with st.session_state.processing_lock:
                        st.session_state.current_vitals.update(vitals)
                        st.session_state.current_vitals['timestamp'] = datetime.now().isoformat()
                    
                    vitals_updated = True
                
            except Exception as e:
                import traceback
                print(f"Error processing vitals: {e}")
                traceback.print_exc()
    
    # Overlay vitals on frame
    frame = overlay_vitals(frame, st.session_state.current_vitals)
    
    return frame, st.session_state.current_vitals, vitals_updated

def overlay_vitals(frame, vitals):
    """Overlay vital signs on the frame"""
    h, w = frame.shape[:2]
    
    # Create semi-transparent overlay
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (400, 280), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
    
    # Add buffer status
    buffer_size = st.session_state.frame_buffer.qsize()
    buffer_text = f"Buffer: {buffer_size}/90 frames"
    cv2.putText(frame, buffer_text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    
    # Add text
    y_offset = 60
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    color = (0, 255, 0)
    thickness = 2
    
    texts = [
        f"Heart Rate: {vitals['heart_rate']:.1f} bpm",
        f"HRV (SDNN): {vitals['hrv']:.1f} ms",
        f"SpO2: {vitals['spo2']:.1f} %",
        f"Stress Index: {vitals['stress_index']:.1f}",
        f"Resp Rate: {vitals['respiratory_rate']:.1f} rpm"
    ]
    
    for text in texts:
        cv2.putText(frame, text, (20, y_offset), font, font_scale, color, thickness)
        y_offset += 35
    
    return frame

def main():
    # Title and description
    st.title("❤️ Live Face-Based Vitals Monitoring System")
    st.markdown("Real-time health monitoring using facial video analysis")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Camera settings
        st.subheader("📹 Camera Settings")
        camera_index = st.number_input("Camera Index", min_value=0, max_value=10, value=0)
        fps_target = st.slider("Target FPS", min_value=15, max_value=60, value=30)
        
        st.divider()
        
        # Status indicators
        st.subheader("🔌 System Status")
        
        # Initialize models if not done
        if not st.session_state.models_initialized:
            with st.spinner("Loading models..."):
                models = initialize_models()
                st.session_state.face_detector = models['face_detector']
                st.session_state.signal_processor = models['signal_processor']
                st.session_state.vitals_calculator = models['vitals_calculator']
                st.session_state.pretrained_model = models['pretrained_model']
                st.session_state.models_initialized = True
        
        # Display status
        st.metric("Pre-trained Model", "✓" if st.session_state.pretrained_model else "✗")
        st.metric("Face Detector", "✓" if st.session_state.face_detector else "✗")
        
        st.divider()
        
        # Info
        st.info("💡 Position your face in front of the camera and stay still for best results.")
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📹 Live Video Feed")
        video_placeholder = st.empty()
        status_placeholder = st.empty()
    
    with col2:
        st.markdown("### 📊 Current Vitals")
        
        # Create vitals container with custom styling
        vitals_container = st.container()
        
        with vitals_container:
            # Vitals display metrics with spacing
            metric1, metric2 = st.columns(2, gap="small")
            metric3, metric4 = st.columns(2, gap="small")
            
            # Store placeholders in session state for dynamic updates
            if 'vitals_placeholders' not in st.session_state:
                st.session_state.vitals_placeholders = {
                    'hr': metric1.empty(),
                    'hrv': metric2.empty(),
                    'spo2': metric3.empty(),
                    'stress': metric4.empty(),
                    'rr': st.empty(),
                    'timestamp': st.empty()
                }
            
            # Initialize with default values and emojis
            st.session_state.vitals_placeholders['hr'].metric(
                "❤️ Heart Rate", 
                "-- bpm", 
                help="Beats per minute"
            )
            st.session_state.vitals_placeholders['hrv'].metric(
                "💚 HRV (SDNN)", 
                "-- ms", 
                help="Heart rate variability"
            )
            st.session_state.vitals_placeholders['spo2'].metric(
                "🩸 SpO2", 
                "-- %", 
                help="Blood oxygen saturation"
            )
            st.session_state.vitals_placeholders['stress'].metric(
                "😰 Stress Index", 
                "--", 
                help="Stress level (0-100)"
            )
            st.session_state.vitals_placeholders['rr'].metric(
                "🫁 Respiratory Rate", 
                "-- rpm", 
                help="Breaths per minute"
            )
            st.session_state.vitals_placeholders['timestamp'].caption("⏱️ Waiting for data...")
        
        st.divider()
        
        # Chart placeholder for future implementation
        st.subheader("📈 Trend Graph")
        chart_placeholder = st.empty()
        chart_placeholder.info("Real-time trend visualization coming soon...")
    
    # Control buttons
    col_btn1, col_btn2, col_btn3 = st.columns(3)
    
    with col_btn1:
        start_button = st.button("▶️ Start Monitoring", width="stretch")
    with col_btn2:
        stop_button = st.button("⏸️ Stop Monitoring", width="stretch")
    with col_btn3:
        reset_button = st.button("🔄 Reset Buffer", width="stretch")
    
    # Handle reset
    if reset_button:
        st.session_state.frame_buffer = queue.Queue(maxsize=300)
        st.success("Frame buffer reset!")
    
    # Main monitoring loop
    if start_button and not stop_button:
        cap = cv2.VideoCapture(camera_index)
        cap.set(cv2.CAP_PROP_FPS, fps_target)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        if not cap.isOpened():
            st.error("❌ Failed to open camera. Please check camera index.")
            return
        
        status_placeholder.success("✅ Camera active - Monitoring in progress...")
        
        models = {
            'face_detector': st.session_state.face_detector,
            'signal_processor': st.session_state.signal_processor,
            'vitals_calculator': st.session_state.vitals_calculator,
            'pretrained_model': st.session_state.pretrained_model
        }
        
        frame_count = 0
        start_time = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                st.error("❌ Failed to read frame from camera")
                break
            
            # Process frame
            processed_frame, vitals, vitals_updated = process_frame(frame, models)
            
            # Convert BGR to RGB for Streamlit
            processed_frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            
            # Display video with output_format to avoid caching issues
            video_placeholder.image(processed_frame_rgb, channels="RGB", width="stretch", output_format="JPEG")
            
            # Update vitals display with colors
            if vitals and vitals['heart_rate'] > 0:
                st.session_state.vitals_placeholders['hr'].metric(
                    "❤️ Heart Rate",
                    f"{vitals['heart_rate']:.1f} bpm",
                    delta=f"{vitals['heart_rate'] - 70:.0f}" if vitals['heart_rate'] > 0 else None,
                    help="Beats per minute"
                )
                
                st.session_state.vitals_placeholders['hrv'].metric(
                    "💚 HRV (SDNN)",
                    f"{vitals['hrv']:.1f} ms",
                    help="Heart rate variability"
                )
                
                st.session_state.vitals_placeholders['spo2'].metric(
                    "🩸 SpO2",
                    f"{vitals['spo2']:.1f}%",
                    delta=f"{vitals['spo2'] - 98:.0f}%" if vitals['spo2'] > 0 else None,
                    help="Blood oxygen saturation"
                )
                
                # Color code stress level
                stress_emoji = "😌" if vitals['stress_index'] < 30 else "😰" if vitals['stress_index'] < 60 else "😫"
                st.session_state.vitals_placeholders['stress'].metric(
                    f"{stress_emoji} Stress Index",
                    f"{vitals['stress_index']:.1f}",
                    help="Stress level (0-100)"
                )
                
                st.session_state.vitals_placeholders['rr'].metric(
                    "🫁 Respiratory Rate",
                    f"{vitals['respiratory_rate']:.1f} rpm",
                    help="Breaths per minute"
                )
                
                st.session_state.vitals_placeholders['timestamp'].caption(f"⏱️ Last updated: {vitals['timestamp'][:19]}")
            
            # Calculate FPS
            frame_count += 1
            if frame_count % 30 == 0:
                elapsed = time.time() - start_time
                fps = frame_count / elapsed
                status_placeholder.success(f"✅ Camera active - FPS: {fps:.1f}")
            
            # Check for stop
            if stop_button:
                break
            
            # Small delay to prevent overwhelming the UI and caching issues
            time.sleep(0.033)  # ~30 FPS max
        
        cap.release()
        status_placeholder.info("⏸️ Monitoring stopped")

if __name__ == "__main__":
    main()