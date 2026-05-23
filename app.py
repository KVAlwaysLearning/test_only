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
        if 'webcam_frames' not in st.session_state:
            st.session_state['webcam_frames'] = {}

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # 1. Real-time Detection
        results = self.yolo(img, classes=[0], verbose=False)
        coords = [list(map(int, b.xyxy[0])) for b in results[0].boxes]
        pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        
        # 2. Capture and Analyze every 5 seconds
        if time.time() - self.last_process_time >= 5:
            self.last_process_time = time.time()
            frame_results = []
            
            for i, (x1, y1, x2, y2) in enumerate(coords):
                crop = pil_img.crop((x1, y1, x2, y2))
                age = int(self.age.predict(np.expand_dims(np.array(crop.resize((224,224)), dtype=np.float32)/255.0, axis=0), verbose=0)[0][0])
                emo = max(self.emo(crop), key=lambda x: x['score'])['label']
                gen = max(self.gen(crop), key=lambda x: x['score'])['label']
                frame_results.append({'ID': i+1, 'Age': age, 'Emotion': emo.capitalize(), 'Gender': gen.capitalize()})
            
            # Save snapshot
            timestamp = time.strftime("%H:%M:%S")
            st.session_state['webcam_frames'][f"Capture at {timestamp}"] = (img.copy(), frame_results)
            
        # Draw boxes on live feed
        for i, (x1, y1, x2, y2) in enumerate(coords):
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 165, 0), 2)
            cv2.putText(img, f"ID: {i+1}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
        return img

# --- MAIN APP ---
st.title("🎥 Video Face Analysis Browser")
models = setup_environment(SECRET_FOLDER_ID)

if models:
    mode = st.radio("Choose Input Method:", ["File Upload", "Live Webcam"])

    if mode == "Live Webcam":
        st.write("Snapshots saved automatically every 5 seconds.")
        # Only one streamer allowed
        webrtc_streamer(key="face-analysis", video_transformer_factory=lambda: FaceAnalyzer(models))
        
        if st.button("Refresh Results"): st.rerun()

        if st.session_state.get('webcam_frames'):
            snapshot_keys = list(st.session_state['webcam_frames'].keys())
            selected_key = st.selectbox("Select a snapshot to review:", snapshot_keys)
            cap_img, cap_data = st.session_state['webcam_frames'][selected_key]
            
            col1, col2 = st.columns(2)
            with col1: st.image(cv2.cvtColor(cap_img, cv2.COLOR_BGR2RGB), use_container_width=True)
            with col2:
                if cap_data: st.dataframe(pd.DataFrame(cap_data).set_index('ID'), use_container_width=True)
