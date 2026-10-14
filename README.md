# Face Mask Detection 
##
semester project

Face Mask Detection is a small computer vision app. It can check a still image or a live camera feed, find faces, and estimate whether each face is wearing a mask. The app shows the result with a box and a label around each detected face.

The app uses a Streamlit web interface. The models run in the Python environment where the app is started. If you deploy the app on a server, images and camera frames are processed by that server.

## What the app uses

- **Python** runs the application code.
- **Streamlit** builds the web interface and accepts image uploads.
- **OpenCV** loads the face detector, prepares camera frames, and draws result boxes.
- **TensorFlow and Keras** load the mask classification model and make predictions.
- **MobileNetV2 image preprocessing** prepares face crops in the format expected by the classifier.
- **Pillow and NumPy** help read and prepare images.
- **streamlit-webrtc** connects a browser camera to the live detection mode.

The project includes the trained model files, so you do not need to train a model before running the app.

## Requirements

- Python 3.8–3.11
- A webcam and browser camera permission for live detection
- Internet access to install the Python packages

## Installation and running

Open a terminal in the project folder.

### 1. Create and activate a virtual environment

Linux or macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Windows PowerShell:

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
```

### 2. Install the packages

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Start the app

```bash
streamlit run app.py
```

Streamlit prints a local address in the terminal, usually `http://localhost:8501`. Open that address in a browser.

Choose **Image** to upload a JPG, JPEG, or PNG file and select **Analyse image**. Choose **Live camera** to allow camera access and start the live feed. Use the camera control to stop the feed.

## How live camera detection works

1. Select **Live camera** in the app. The browser asks for permission to use the camera; allow it to start the feed.
2. `streamlit-webrtc` sends video frames from the browser to the running Python app. Audio is not requested.
3. For each frame, the app finds faces and predicts mask or no mask using the same steps and models described above.
4. The app draws a box and label on each frame and sends the annotated frame back to the browser to display.
5. Use the stream's **Stop** control to end the camera feed. The app stops receiving and processing frames.

The live camera mode needs a working webcam and browser permission. When running locally, video frames are processed by the app on your computer. If the app is hosted on another machine, frames are sent to and processed by that machine.

## How detection works

The app has two detection modes, but both use the same face and mask models:

1. **Get an image frame.** In image mode, the frame comes from the uploaded image. In live mode, it comes from the camera.
2. **Find faces.** OpenCV's face detector uses the included model configuration and weights in `models/face_detector/`. It keeps detections with a confidence above `0.5`.
3. **Prepare each face.** The app crops each face from the frame, converts its color format, resizes it to `224 × 224`, and applies MobileNetV2 preprocessing.
4. **Predict mask status.** The TensorFlow/Keras model in `models/mask_classifier/` predicts two scores for each face: mask and no mask.
5. **Show the result.** The app draws a box around each face and labels it **Mask** or **No Mask**, with the higher score shown as a percentage. In image mode, it also reports how many faces were found.

If no faces are detected, the app returns no face predictions for that frame.

## Project architecture

```text
.
├── app.py                         # Starts the Streamlit app
├── requirements.txt               # Python packages and versions
├── assets/
│   └── images/                    # Images used in the page design
├── detection/
│   └── mask_detector.py           # Face detection and mask prediction steps
├── models/
│   ├── face_detector/
│   │   ├── deploy.prototxt        # Face detector network definition
│   │   └── res10_300x300_ssd_iter_140000.caffemodel
│   │                               # Face detector trained weights
│   └── mask_classifier/
│       └── mask_detector.model    # Trained mask/no-mask classifier
└── ui/
    ├── app.py                     # Page layout, upload, camera, and results
    └── styles.css                 # Page styling
```

`app.py` calls `run_app()` in `ui/app.py`. The UI loads the included models and calls the shared detection function in `detection/mask_detector.py` for both uploaded images and camera frames.

## Notes

- Keep the files in `models/` in place; the app needs them to detect faces and classify masks.
- A clear, well-lit image with visible faces usually gives better results.
- The model's output is an estimate and may be wrong. It should not be used as the only basis for important decisions.
