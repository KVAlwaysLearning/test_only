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

    try:
        yolo = YOLO(os.path.join(BASE_MODEL_DIR, "yolo/yolov8n.pt"))
        emotion_pipe = pipeline("image-classification", model=os.path.join(BASE_MODEL_DIR, "emotion"))
        gender_pipe = pipeline("image-classification", model=os.path.join(BASE_MODEL_DIR, "gender"))
        age_model = keras.models.load_model(os.path.join(BASE_MODEL_DIR, "age/best_model.h5"), compile=False)
        return yolo, emotion_pipe, gender_pipe, age_model
    except Exception as e:
        st.error(f"Error initializing models: {e}")
        return None

class FaceAnalyzer(VideoTransformerBase):
    def __init__(self, models):
        self.last_process_time = 0
        self.yolo, self.emo, self.gen, self.age = models
        # Initialize storage if not present
        if 'webcam_frames' not in st.session_state:
            st.session_state['webcam_frames'] = {}

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        
        # 1. ALWAYS run detection for the live visual feed
        results = self.yolo(img, classes=[0], verbose=False)
        coords = [list(map(int, b.xyxy[0])) for b in results[0].boxes]
        
        frame_results = []
        for i, (x1, y1, x2, y2) in enumerate(coords):
            face_id = i + 1
            # Crop for analysis
            crop = pil_img.crop((x1, y1, x2, y2))
            
            # Predictions (You can optimize this to run less frequently if needed)
            age = int(self.age.predict(np.expand_dims(np.array(crop.resize((224,224)), dtype=np.float32)/255.0, axis=0), verbose=0)[0][0])
            emo = max(self.emo(crop), key=lambda x: x['score'])['label']
            gen = max(self.gen(crop), key=lambda x: x['score'])['label']
            frame_results.append({'ID': face_id, 'Age': age, 'Emotion': emo.capitalize(), 'Gender': gen.capitalize()})
            
            # Draw persistent boxes on the live feed
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 165, 0), 2)
            cv2.putText(img, f"ID: {face_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # 2. ONLY store a snapshot every 5 seconds
        if time.time() - self.last_process_time > 5:
            timestamp = time.strftime("%H:%M:%S")
            # We copy the image so it doesn't get cleared by the next frame
            st.session_state['webcam_frames'][f"Capture at {timestamp}"] = (img.copy(), frame_results)
            self.last_process_time = time.time()
            
        return img
        
# --- MAIN APP ---
st.title("🎥 Video Face Analysis Browser")
models = setup_environment(SECRET_FOLDER_ID)

if models:
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
                    frame_idx = 0
                    while cap.isOpened() and frame_idx <= target_indices[-1]:
                        ret, frame = cap.read()
                        if not ret: break
                        if frame_idx in target_indices:
                            results = models[0](frame, classes=[0], verbose=False)
                            coords = [list(map(int, b.xyxy[0])) for b in results[0].boxes]
                            pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                            frame_results = []
                            for i, (x1, y1, x2, y2) in enumerate(coords):
                                face_id = i + 1
                                crop = pil_img.crop((x1, y1, x2, y2))
                                age = int(models[3].predict(np.expand_dims(np.array(crop.resize((224,224)), dtype=np.float32)/255.0, axis=0), verbose=0)[0][0])
                                emo = max(models[1](crop), key=lambda x: x['score'])['label']
                                gen = max(models[2](crop), key=lambda x: x['score'])['label']
                                frame_results.append({'ID': face_id, 'Age': age, 'Emotion': emo.capitalize(), 'Gender': gen.capitalize()})
                                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 165, 0), 2)
                                label = f"ID: {face_id}"
                                (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
                                cv2.rectangle(frame, (x1, y1 - h - 10), (x1 + w, y1), (255, 0, 0), -1)
                                cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
                            frames_data[f"Frame at {frame_idx/fps:.1f}s"] = (frame, frame_results)
                        frame_idx += 1
                    st.session_state['processed_frames'] = frames_data
                    cap.release()

            if st.session_state.get('processed_frames'):
                selection = st.selectbox("Select a frame to inspect:", list(st.session_state['processed_frames'].keys()))
                frame_img, frame_data = st.session_state['processed_frames'][selection]
                col1, col2 = st.columns([2, 1])
                with col1: st.image(cv2.cvtColor(frame_img, cv2.COLOR_BGR2RGB), use_container_width=True)
                with col2:
                    if frame_data:
                        df = pd.DataFrame(frame_data)
                        st.dataframe(df.set_index('ID'), use_container_width=True)
                    else: st.info("No faces detected.")

    elif mode == "Live Webcam":
        st.subheader("Live Webcam Feed")
        webrtc_streamer(
            key="face-analysis", 
            video_transformer_factory=lambda: FaceAnalyzer(models),
            rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
        )

        st.markdown("---")
        st.subheader("Captured Snapshots Gallery")
        
        # Access the frames captured by the background thread
        if 'webcam_frames' in st.session_state and st.session_state['webcam_frames']:
            # Create a dropdown to select from captured snapshots
            snapshot_keys = list(st.session_state['webcam_frames'].keys())
            
            # Use a selectbox to pick a frame
            selected_key = st.selectbox("Select a snapshot to review:", snapshot_keys)
            
            # Retrieve data for the selection
            cap_img, cap_data = st.session_state['webcam_frames'][selected_key]
            
            # Display results below the video box
            col1, col2 = st.columns([1, 1])
            with col1:
                st.image(cv2.cvtColor(cap_img, cv2.COLOR_BGR2RGB), caption=selected_key, use_container_width=True)
            with col2:
                if cap_data:
                    st.markdown("#### Analysis Results")
                    df = pd.DataFrame(cap_data)
                    st.dataframe(df.set_index('ID'), use_container_width=True)
                else:
                    st.info("No faces were detected in this snapshot.")
        else:
            st.info("No snapshots captured yet. Wait for 5 seconds for the first auto-capture.")
