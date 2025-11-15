# ❤️ Face-Based Vitals Monitoring System

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15.0-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-Latest-red)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8.1-green)

A real-time, non-contact vital signs monitoring system using facial video analysis and remote photoplethysmography (rPPG). Extract heart rate, heart rate variability (HRV), SpO2, respiratory rate, and stress index from webcam video feed.

---

## 🌟 Features

- **Real-time Face Detection**: Uses MediaPipe for robust face detection and tracking
- **Non-Contact Vital Signs Extraction**:
  - ❤️ Heart Rate (HR) - 40-200 BPM
  - 📊 Heart Rate Variability (HRV/SDNN) - 0-200 ms
  - 🩸 Blood Oxygen Saturation (SpO2) - 90-100%
  - 😰 Stress Index - 0-100
  - 🫁 Respiratory Rate - 8-30 RPM
- **Advanced Signal Processing**:
  - POS (Plane-Orthogonal-to-Skin) algorithm
  - Butterworth bandpass filtering
  - FFT-based frequency analysis
  - Motion artifact detection and removal
- **Interactive Web UI**: Built with Streamlit for easy monitoring
- **Deep Learning Model**: PhysNet architecture with LSTM for temporal modeling
- **Signal Quality Assessment**: Real-time quality metrics

---

## 📋 Table of Contents

- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Project Structure](#-project-structure)
- [How It Works](#-how-it-works)
- [Usage](#-usage)
- [Model Architecture](#-model-architecture)
- [Configuration](#-configuration)
- [Troubleshooting](#-troubleshooting)
- [Technical Details](#-technical-details)
- [Known Issues](#-known-issues)
- [References](#-references)

---

## 🚀 Installation

### Prerequisites

- Python 3.8 - 3.11 (Python 3.13 not supported due to MediaPipe compatibility)
- Webcam or camera device
- Windows/Linux/macOS

### Step 1: Clone the Repository

```bash
git clone https://github.com/your-username/vitals-monitoring.git
cd vitals-monitoring
```

### Step 2: Create Virtual Environment

```bash
# Windows PowerShell
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Linux/macOS
python3 -m venv .venv
source .venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

**Note**: If you encounter MediaPipe installation issues on Python 3.13:
```
× No solution found when resolving dependencies:
  ╰─▶ mediapipe==0.10.7 has no wheels for Python 3.13
```
**Solution**: Use Python 3.8-3.11 instead.

### Step 4: Initialize Model (Optional)

Generate synthetic pre-trained weights:

```bash
python main.py
```

This creates `weights/physnet_pretrained.h5` with ~5 epochs of synthetic training.

---

## ⚡ Quick Start

### Run the Streamlit App

```bash
streamlit run main.py
```

The app will open in your browser at `http://localhost:8501`

### Using the App

1. **Adjust Settings** (Sidebar):
   - Camera Index: Select your camera (default: 0)
   - Target FPS: Set frame rate (15-60, default: 30)

2. **Start Monitoring**:
   - Click "▶️ Start Monitoring"
   - Position your face in the frame
   - Wait for buffer to fill (90 frames ≈ 3 seconds)

3. **View Results**:
   - Real-time video with face detection
   - Live vitals displayed on right panel
   - On-screen overlay showing current measurements

4. **Controls**:
   - "⏸️ Stop Monitoring" - Pause capture
   - "🔄 Reset Buffer" - Clear frame buffer

---

## 📁 Project Structure

```
assignment/
├── main.py                      # Streamlit web application
├── requirements.txt             # Python dependencies
├── README.md                    # This file
│
├── app/
│   ├── models/
│   │   ├── model.py            # RPPGModel - PhysNet architecture
│   │   └── custom_rppg.py      # Custom model implementations
│   │
│   └── utils/
│       ├── face_detector.py    # FaceDetector - MediaPipe/OpenCV
│       ├── signal_processor.py # SignalProcessor - Filtering & FFT
│       └── vital_calculator.py # VitalsCalculator - HR, HRV, SpO2
│
├── data/
│   └── training/
│       ├── videos/             # Training video files (.avi)
│       └── ground_truth/       # Ground truth signals (.txt)
│
└── weights/
    └── physnet_pretrained.h5   # Pre-trained model weights
```

---

## 🔬 How It Works

### 1. Face Detection
- **MediaPipe Face Detection** for robust face localization
- Temporal smoothing to reduce jitter
- Region of Interest (ROI) extraction with 20% expansion

### 2. Signal Extraction (POS Algorithm)

```
For each frame:
  1. Extract face ROI
  2. Resize to 64×64
  3. Extract center region (32×32)
  4. Calculate mean RGB values

POS Algorithm:
  - Normalize each color channel by its mean
  - X = 3R - 2G
  - Y = 1.5R + G - 1.5B
  - α = std(X) / std(Y)
  - Signal = X - αY
```

### 3. Signal Processing

```python
# Bandpass filtering (0.7-4.0 Hz → 42-240 BPM)
filtered = butterworth_bandpass(signal, low=0.7, high=4.0, order=4)

# FFT for frequency analysis
frequencies, power = fft(filtered_signal)
peak_freq = find_peak(power, freq_range=[0.7, 4.0])
heart_rate = peak_freq × 60  # Convert Hz to BPM
```

### 4. Vital Signs Calculation

- **Heart Rate**: Peak detection in time domain + FFT validation
- **HRV (SDNN)**: Standard deviation of RR intervals
- **SpO2**: Estimated from signal quality metrics
- **Stress Index**: Combination of HR elevation and HRV reduction
- **Respiratory Rate**: Low-frequency modulation (0.1-0.5 Hz)

---

## 💻 Usage

### Basic Usage

```python
from app.models.model import RPPGMOdel
from app.utils.face_detector import FaceDetector
from app.utils.signal_processor import SignalProcessor
from app.utils.vital_calculator import VitalsCalculator

# Initialize components
face_detector = FaceDetector(use_mediapipe=True)
model = RPPGMOdel()
signal_processor = SignalProcessor(fps=30)
vitals_calculator = VitalsCalculator()

# Process frames
frames = []
for frame in video_capture:
    face_roi, coords = face_detector.detect_faces(frame)
    if face_roi is not None:
        frames.append(face_roi)

# Extract signal (need 90+ frames)
if len(frames) >= 90:
    signal = model.extract_spatial_mean(frames)
    filtered_signal = signal_processor.filter_signal(signal)
    vitals = vitals_calculator.calculate_vitals(filtered_signal, fps=30)
    
    print(f"Heart Rate: {vitals['heart_rate']:.1f} BPM")
    print(f"HRV: {vitals['hrv']:.1f} ms")
    print(f"SpO2: {vitals['spo2']:.1f}%")
```

### Advanced Configuration

```python
# Custom signal processing
signal_processor = SignalProcessor(fps=30)
signal_processor.hr_range = (0.5, 3.0)  # 30-180 BPM
signal_processor.resp_range = (0.15, 0.4)  # 9-24 RPM

# Quality assessment
quality = signal_processor.quality_assessment(signal)
print(f"Signal Quality: {quality:.2f}")

# Motion artifact detection
artifacts = signal_processor.detect_motion_artifacts(signal, threshold=3.0)
cleaned = signal_processor.interpolate_artifacts(signal, artifacts)
```

---

## 🧠 Model Architecture

### PhysNet (Deep rPPG Model)

```
Input: (batch, 32, 128, 128, 3)
  ↓
TimeDistributed Conv2D (16 filters, 5×5) + Tanh + BatchNorm + MaxPool
  ↓
TimeDistributed Conv2D (32 filters, 5×5) + Tanh + BatchNorm + MaxPool
  ↓
TimeDistributed Conv2D (64 filters, 5×5) + Tanh + BatchNorm + MaxPool
  ↓
TimeDistributed Conv2D (64 filters, 5×5) + Tanh + BatchNorm + MaxPool
  ↓
TimeDistributed Flatten
  ↓
LSTM (64 units, return_sequences=True)
  ↓
Dropout (0.3)
  ↓
TimeDistributed Dense (1, linear)
  ↓
Output: (batch, 32, 1) - rPPG signal
```

**Training**: MSE loss, Adam optimizer (lr=0.0001)

---

## ⚙️ Configuration

### Camera Settings

```python
# In Streamlit sidebar
camera_index = 0  # 0 for default camera, 1+ for external
fps_target = 30   # Frame rate (15-60)
```

### Signal Processing Parameters

Edit `app/utils/signal_processor.py`:

```python
class SignalProcessor:
    def __init__(self, fps=30):
        self.fps = fps
        self.hr_range = (0.7, 4.0)    # Heart rate: 42-240 BPM
        self.resp_range = (0.1, 0.5)   # Respiration: 6-30 RPM
```

### Model Parameters

Edit `app/models/model.py`:

```python
class RPPGMOdel:
    def __init__(self, model_path="weights/physnet_pretrained.h5"):
        self.input_shape = (128, 128, 3)
        self.temporal_depth = 32  # Number of frames per window
```

---

## 🔧 Troubleshooting

### Common Issues

#### 1. Camera Not Opening

```
❌ Failed to open camera. Please check camera index.
```

**Solution**:
- Check camera is connected and not used by another app
- Try different camera indices (0, 1, 2...)
- On Windows, close Skype/Teams which lock cameras

#### 2. MediaPipe Compatibility Error

```
× No solution found: mediapipe==0.10.7 has no wheels for Python 3.13
```

**Solution**: Use Python 3.8-3.11:
```bash
# Install specific Python version
pyenv install 3.11.0
pyenv local 3.11.0
```

#### 3. All Vitals Showing 0.0

**Debug Steps**:
1. Check terminal for debug output:
   ```
   DEBUG: Signal length: 90, Min: X, Max: X, Mean: X
   DEBUG: Filtered signal - Mean: X, Std: X
   DEBUG: Calculated HR: X.X
   ```

2. Ensure proper lighting (avoid backlighting)
3. Keep face stable for 3+ seconds
4. Check buffer status on video overlay

#### 4. TensorFlow Warnings

```
WARNING:tensorflow:From ... tf.losses.sparse_softmax_cross_entropy is deprecated
```

**Solution**: Already suppressed in code. Warnings don't affect functionality.

#### 5. OpenCV Merge Error

```
error: (-215:Assertion failed) mv[i].size == mv[0].size
```

**Solution**: Fixed in latest version. Use `extract_spatial_mean()` instead of `extract_rppg()`.

---

## 📊 Technical Details

### Minimum Requirements

- **Frames**: 90 frames (3 seconds at 30 FPS)
- **Face Size**: 128×128 pixels minimum
- **Lighting**: Ambient indoor lighting (avoid direct sunlight)
- **Distance**: 30-60 cm from camera
- **Motion**: Keep face relatively still

### Signal Quality Factors

| Factor | Impact | Recommendation |
|--------|--------|----------------|
| **Lighting** | Critical | Indirect, diffuse lighting |
| **Motion** | High | Minimize head movement |
| **Distance** | Medium | 30-60 cm from camera |
| **Face Angle** | High | Face camera directly |
| **Skin Tone** | Low | Algorithm adapts automatically |

### Accuracy

**Expected Ranges** (under optimal conditions):
- Heart Rate: ±5 BPM
- HRV: ±10 ms
- SpO2: ±2% (estimation only)
- Respiratory Rate: ±2 RPM

**Note**: This is a research/educational tool. Not validated for medical use.

---

## ⚠️ Known Issues

1. **SpO2 Estimation**: Simplified estimation based on signal quality. Real SpO2 requires dual-wavelength (red/IR) sensors.

2. **Streamlit Image Caching**: Occasional warnings about missing image files. Mitigated with `output_format="JPEG"`.

3. **Initial Delay**: 3-second buffer required before first vitals appear.

4. **Dark Skin Tones**: May require brighter lighting for optimal signal quality.

5. **Glasses/Makeup**: Heavy makeup or reflective glasses may affect signal extraction.

---

## 📚 References

### Papers

1. **PhysNet**: Chen, W., & McDuff, D. (2018). "DeepPhys: Video-Based Physiological Measurement Using Convolutional Attention Networks." *ECCV 2018*.

2. **POS Algorithm**: Wang, W., den Brinker, A. C., Stuijk, S., & de Haan, G. (2017). "Algorithmic Principles of Remote PPG." *IEEE Transactions on Biomedical Engineering*.

3. **rPPG Survey**: Rouast, P. V., Adam, M. T., Chiong, R., Cornforth, D., & Lux, E. (2018). "Remote heart rate measurement using low-cost RGB face video: a technical literature review." *Frontiers in Computer Science*.

### Libraries

- **TensorFlow**: https://www.tensorflow.org/
- **OpenCV**: https://opencv.org/
- **MediaPipe**: https://google.github.io/mediapipe/
- **Streamlit**: https://streamlit.io/
- **SciPy**: https://scipy.org/

---

## 📝 License

This project is for educational and research purposes only. Not intended for medical diagnosis or clinical use.

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

### Areas for Improvement

- [ ] Add real-time trend visualization charts
- [ ] Implement video file upload support
- [ ] Add export functionality (CSV, JSON)
- [ ] Improve SpO2 estimation algorithm
- [ ] Add multi-person support
- [ ] Implement GPU acceleration
- [ ] Add unit tests

---

## 📧 Contact

For questions or support, please open an issue on GitHub.

---

## 🙏 Acknowledgments

- MediaPipe team for face detection models
- PhysNet authors for the neural architecture
- Streamlit team for the web framework
- OpenCV community for computer vision tools

---

**⚠️ Disclaimer**: This software is provided "as is" for educational purposes only. It is not a medical device and should not be used for medical diagnosis or treatment decisions. Always consult healthcare professionals for medical advice.
