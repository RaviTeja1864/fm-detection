from pathlib import Path

import cv2
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array


def load_face_detector(face_detector_dir=None):
    detector_dir = (
        Path(face_detector_dir)
        if face_detector_dir is not None
        else Path(__file__).resolve().parent.parent / "models" / "face_detector"
    )
    prototxt_path = detector_dir / "deploy.prototxt"
    weights_path = detector_dir / "res10_300x300_ssd_iter_140000.caffemodel"
    return cv2.dnn.readNet(str(prototxt_path), str(weights_path))


def detect_and_predict_mask(frame, face_net, mask_net, confidence=0.5):
    height, width = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(
        frame, 1.0, (300, 300), (104.0, 177.0, 123.0)
    )
    face_net.setInput(blob)
    detections = face_net.forward()

    faces = []
    locations = []
    for index in range(detections.shape[2]):
        if detections[0, 0, index, 2] <= confidence:
            continue

        box = detections[0, 0, index, 3:7] * np.array(
            [width, height, width, height]
        )
        start_x, start_y, end_x, end_y = box.astype("int")
        start_x = max(0, start_x)
        start_y = max(0, start_y)
        end_x = min(width - 1, end_x)
        end_y = min(height - 1, end_y)

        if end_x <= start_x or end_y <= start_y:
            continue

        face = frame[start_y:end_y, start_x:end_x]
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face = cv2.resize(face, (224, 224))
        face = img_to_array(face)
        faces.append(preprocess_input(face))
        locations.append((start_x, start_y, end_x, end_y))

    if not faces:
        return locations, []

    predictions = mask_net.predict(np.array(faces, dtype="float32"), verbose=0)
    return locations, predictions
