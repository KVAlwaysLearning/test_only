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
        self.latest_capture = None

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        results = self.yolo(img, classes=[0], verbose=False)
        coords = [list(map(int, b.xyxy[0])) for b in results[0].boxes]
        
        for i, (x1, y1, x2, y2) in enumerate(coords):
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 165, 0), 2)
        
        if time.time() - self.last_process_time >= 5:
            self.last_process_time = time.time()
            frame_results = []
            pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            for i, (x1, y1, x2, y2) in enumerate(coords):
                crop = pil_img.crop((x1, y1, x2, y2))
                age = int(self.age.predict(np.expand_dims(np.array(crop.resize((224,224)), dtype=np.float32)/255.0, axis=0), verbose=0)[0][0])
                emo = max(self.emo(crop), key=lambda x: x['score'])['label']
                gen = max(self.gen(crop), key=lambda x: x['score'])['label']
                frame_results.append({'ID': i+1, 'Age': age, 'Emotion': emo.capitalize(), 'Gender': gen.capitalize()})
            
            timestamp = time.strftime("%H:%M:%S")
            self.latest_capture = (img.copy(), frame_results, timestamp)
            
        return img

# --- MAIN APP ---
st.title("🎥 Video Face Analysis Browser")
models = setup_environment(SECRET_FOLDER_ID)

if models:
    mode = st.radio("Choose Input Method:", ["File Upload", "Live Webcam"])

    if mode == "File Upload":
        uploaded_video = st.file_uploader("Upload a video", type=["mp4", "mov", "avi"])
        if uploaded_video:
            # (Keeping your existing logic exactly as it was)
            current_file_key = f"{uploaded_video.name}_{uploaded_video.size}"
            if st.session_state.get('last_uploaded_file') != current_file_key:
                st.session_state['processed_frames'] = {} 
                st.session_state['last_uploaded_file'] = current_file_key
                st.rerun() 
            with open("temp_vid.mp4", "wb") as f: f.write(uploaded_video.read())
            if 'processed_frames' not in st.session_state or not st.session_state['processed_frames']:
                with st.spinner("Processing video..."):
                    cap = cv2.VideoCapture("temp_vid.mp4")
                    # ... [Your full existing processing loop logic here] ...
                    cap.release()

    elif mode == "Live Webcam":
        ctx = webrtc_streamer(key="face-analysis", video_transformer_factory=lambda: FaceAnalyzer(models))
        
        if ctx.video_transformer and ctx.video_transformer.latest_capture:
            cap_img, cap_data, ts = ctx.video_transformer.latest_capture
            key = f"Capture at {ts}"
            if 'webcam_frames' not in st.session_state: st.session_state['webcam_frames'] = {}
            if key not in st.session_state['webcam_frames']:
                st.session_state['webcam_frames'][key] = (cap_img, cap_data)
        
        if st.session_state.get('webcam_frames'):
            selected = st.selectbox("Select Snapshot:", list(st.session_state['webcam_frames'].keys()))
            img, data = st.session_state['webcam_frames'][selected]
            st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), use_container_width=True)
            st.dataframe(pd.DataFrame(data))
