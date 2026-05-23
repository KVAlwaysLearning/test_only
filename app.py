import streamlit as st
import os
import gdown
import pandas as pd
import numpy as np
import cv2
from PIL import Image
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import time

st.set_page_config(layout="wide", page_title="Video Analysis Browser")

# --- INITIALIZATION ---
BASE_MODEL_DIR = os.path.join(os.getcwd(), "all_models")
SECRET_FOLDER_ID = st.secrets["drive_folder_id"] if "drive_folder_id" in st.secrets else None

@st.cache_resource
def setup_environment(drive_folder_id):
    import signal
    def dummy_signal_handler(signum, frame): pass
    original_signal = signal.signal
    def patched_signal(signalnum, handler):
        try: return original_signal(signalnum, handler)
        except ValueError: return dummy_signal_handler
    signal.signal = patched_signal

    os.environ["ULTRALYTICS_HUB_DISABLED"] = "true"
    from ultralytics import YOLO
    from transformers import pipeline
    from tensorflow import keras
    
    if not os.path.exists(BASE_MODEL_DIR):
        gdown.download_folder(id=drive_folder_id, output=BASE_MODEL_DIR, quiet=True)

    yolo = YOLO(os.path.join(BASE_MODEL_DIR, "yolo/yolov8n.pt"))
    emotion_pipe = pipeline("image-classification", model=os.path.join(BASE_MODEL_DIR, "emotion"))
    gender_pipe = pipeline("image-classification", model=os.path.join(BASE_MODEL_DIR, "gender"))
    age_model = keras.models.load_model(os.path.join(BASE_MODEL_DIR, "age/best_model.h5"), compile=False)
    return yolo, emotion_pipe, gender_pipe, age_model

class FaceAnalyzer(VideoTransformerBase):
    def __init__(self, models):
        self.last_process_time = time.time()
        self.yolo, self.emo, self.gen, self.age = models

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        # ... (YOLO logic as before) ...
        
        # CAPTURE (Every 5 seconds)
        if time.time() - self.last_process_time >= 5:
            self.last_process_time = time.time()
            # ... (Prediction logic as before) ...
            
            # DIRECT PUSH to session state (Bypass ctx entirely)
            timestamp = time.strftime("%H:%M:%S")
            key = f"Capture at {timestamp}"
            if 'webcam_frames' not in st.session_state:
                st.session_state['webcam_frames'] = {}
            st.session_state['webcam_frames'][key] = (img.copy(), frame_results)
            
        return img

# --- MAIN APP ---
st.title("🎥 Video Face Analysis Browser")
models = setup_environment(SECRET_FOLDER_ID)

if models:
    yolo, emotion_pipe, gender_pipe, age_model = models
    mode = st.radio("Choose Input Method:", ["File Upload", "Live Webcam"])

    if mode == "File Upload":
        uploaded_video = st.file_uploader("Upload a video", type=["mp4", "mov", "avi"])
        if uploaded_video:
            current_file_key = f"{uploaded_video.name}_{uploaded_video.size}"
            if st.session_state.get('last_uploaded_file') != current_file_key:
                st.session_state['processed_frames'] = {}
                st.session_state['last_uploaded_file'] = current_file_key
                st.rerun()
            
            with open("temp_vid.mp4", "wb") as f: f.write(uploaded_video.read())
            
            if 'processed_frames' not in st.session_state or not st.session_state['processed_frames']:
                with st.spinner("Processing video..."):
                    cap = cv2.VideoCapture("temp_vid.mp4")
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    fps = cap.get(cv2.CAP_PROP_FPS) or 30
                    target_indices = np.linspace(0, max(0, total_frames - 1), 5, dtype=int)
                    frames_data = {}
                    for frame_idx in range(total_frames):
                        ret, frame = cap.read()
                        if not ret: break
                        if frame_idx in target_indices:
                            results = yolo(frame, classes=[0], verbose=False)
                            coords = [list(map(int, b.xyxy[0])) for b in results[0].boxes]
                            pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                            frame_results = []
                            for i, (x1, y1, x2, y2) in enumerate(coords):
                                crop = pil_img.crop((x1, y1, x2, y2))
                                age = int(age_model.predict(np.expand_dims(np.array(crop.resize((224,224)), dtype=np.float32)/255.0, axis=0), verbose=0)[0][0])
                                emo = max(emotion_pipe(crop), key=lambda x: x['score'])['label']
                                gen = max(gender_pipe(crop), key=lambda x: x['score'])['label']
                                frame_results.append({'ID': i+1, 'Age': age, 'Emotion': emo.capitalize(), 'Gender': gen.capitalize()})
                                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 165, 0), 2)
                            frames_data[f"Frame at {frame_idx/fps:.1f}s"] = (frame, frame_results)
                    st.session_state['processed_frames'] = frames_data
                    cap.release()

            if st.session_state.get('processed_frames'):
                selection = st.selectbox("Select a frame to inspect:", list(st.session_state['processed_frames'].keys()))
                frame_img, frame_data = st.session_state['processed_frames'][selection]
                col1, col2 = st.columns([2, 1])
                with col1: st.image(cv2.cvtColor(frame_img, cv2.COLOR_BGR2RGB), use_container_width=True)
                with col2: st.dataframe(pd.DataFrame(frame_data).set_index('ID'), use_container_width=True)

    elif mode == "Live Webcam":
        # We track if we are currently recording or in review mode
        if 'is_reviewing' not in st.session_state:
            st.session_state['is_reviewing'] = False

        if not st.session_state['is_reviewing']:
            st.write("Recording... Snapshots are being saved in the background.")
            webrtc_streamer(key="face-analysis", video_transformer_factory=lambda: FaceAnalyzer(models))
            
            if st.button("Stop & Review Results"):
                st.session_state['is_reviewing'] = True
                st.rerun()
        
        else:
            # --- REVIEW MODE ---
            st.success("Analysis Complete. Review your snapshots below:")
            if st.button("Start New Session"):
                st.session_state['webcam_frames'] = {}
                st.session_state['is_reviewing'] = False
                st.rerun()

            if 'webcam_frames' in st.session_state and st.session_state['webcam_frames']:
                keys = sorted(list(st.session_state['webcam_frames'].keys()))
                selected = st.selectbox("Select Snapshot to Review:", keys)
                
                img, data = st.session_state['webcam_frames'][selected]
                st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), use_container_width=True)
                st.dataframe(pd.DataFrame(data).set_index('ID'), use_container_width=True)
            else:
                st.warning("No frames were captured during the session.")
