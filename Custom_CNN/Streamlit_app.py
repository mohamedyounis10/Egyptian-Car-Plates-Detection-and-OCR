import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
import itertools
import streamlit as st
from datetime import datetime
from PIL import Image
from tensorflow.keras import layers, Model, backend as K

# 1. CONFIGURATION AND PATHS
DET_MODEL_PATH = r'C:\Users\moham\Desktop\Computer Vision\Project\best_plate_model.keras'
OCR_WEIGHTS_PATH = r'C:\Users\moham\Desktop\Computer Vision\Project\plate_ocr_weights.weights.h5'

IMG_W_OCR, IMG_H_OCR = 256, 128
MAX_TEXT_LEN = 7
CONF_THRESHOLD = 0.3

SAVE_DIR = r"C:\Users\moham\Desktop\Computer Vision\Project\Radar_Detections"
CROPS_DIR = os.path.join(SAVE_DIR, "Plates_Crops")
FULL_DIR = os.path.join(SAVE_DIR, "Full_Vehicle")
LOG_FILE = os.path.join(SAVE_DIR, "radar_log.csv")

for folder in [SAVE_DIR, CROPS_DIR, FULL_DIR]:
    if not os.path.exists(folder):
        os.makedirs(folder)

unique_letters = sorted(set([
    '0','1','2','3','4','5','6','7','8','9',
    'A','B','C','D','E','F','G','H','I','J','K','L','M',
    'N','O','P','Q','R','S','T','U','V','W','Y','Z','a','b','n'
]))
if 'X' not in unique_letters: unique_letters.append('X')

CHAR_VECTOR = "".join(unique_letters)
NUM_CLASSES = len(unique_letters) + 1

en_to_ar = {
    "A": "ا", "B": "ب", "C": "ت", "D": "ث", "E": "ج",
    "F": "ح", "G": "خ", "H": "د", "I": "ذ", "J": "ر",
    "K": "ز", "L": "س", "M": "ش", "N": "ص", "O": "ض",
    "P": "ط", "Q": "ظ", "R": "ع", "S": "غ", "T": "ف",
    "U": "ق", "V": "ك", "W": "ل", "n": "م", "Y": "ن",
    "Z": "ه", "a": "و", "b": "ي",
    "0": "٠", "1": "١", "2": "٢", "3": "٣", "4": "٤",
    "5": "٥", "6": "٦", "7": "٧", "8": "٨", "9": "٩"
}

# 2. MODEL BUILDING
@st.cache_resource
def build_ocr_model():
    inputs = layers.Input(name="the_input", shape=(IMG_W_OCR, IMG_H_OCR, 1))
    
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.15)(x)

    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.15)(x)

    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.15)(x)

    conv_shape = x.shape
    x = layers.Reshape(target_shape=(int(conv_shape[1]), int(conv_shape[2] * conv_shape[3])))(x)
    
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Bidirectional(layers.LSTM(256, return_sequences=True, dropout=0.1))(x)
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True, dropout=0.1))(x)
    
    outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    model.load_weights(OCR_WEIGHTS_PATH)
    return model

@st.cache_resource
def load_detection_model():
    return tf.keras.models.load_model(DET_MODEL_PATH, compile=False)

# 3. HELPER FUNCTIONS
def decode_det(pred, w, h):
    grid = 26
    conf_map = pred[..., 0]
    i, j = np.unravel_index(np.argmax(conf_map), conf_map.shape)
    conf = float(pred[i, j, 0])
    x_g = (j + pred[i, j, 1]) / grid
    y_g = (i + pred[i, j, 2]) / grid
    wd, hd = pred[i, j, 3] * 1.15, pred[i, j, 4] * 1.15
    x1, y1 = max(0, int((x_g - wd/2)*w)), max(0, int((y_g - hd/2)*h))
    x2, y2 = min(w, int((x_g + wd/2)*w)), min(h, int((y_g + hd/2)*h))
    return x1, y1, x2, y2, conf

def process_ocr(plate_crop, ocr_model):
    gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
    
    kernel = np.array([[0, -1, 0], 
                       [-1, 5, -1], 
                       [0, -1, 0]])
    sharpened = cv2.filter2D(gray, -1, kernel)
    
    resized = cv2.resize(sharpened, (IMG_W_OCR, IMG_H_OCR))
    img_in = resized.T.reshape(1, IMG_W_OCR, IMG_H_OCR, 1).astype(np.float32) / 255.0
    
    preds = ocr_model.predict(img_in, verbose=0)
    best_path = np.argmax(preds[0], axis=1)
    out = [k for k, g in itertools.groupby(best_path)]

    raw = "".join([CHAR_VECTOR[i] for i in out if i < len(unique_letters) and CHAR_VECTOR[i] != 'X'])

    letters_en = [c for c in raw if c.isalpha()]
    digits_en  = [c for c in raw if c.isdigit()]
    
    letters_correct = letters_en[::-1]
    
    text_en = ''.join(digits_en) + ' ' + ''.join(letters_correct)
    text_ar = ' '.join([en_to_ar[c] for c in letters_correct if c in en_to_ar]) + \
              ' ' + \
              ''.join([en_to_ar[c] for c in digits_en if c in en_to_ar])
    
    return text_ar, text_en

def save_radar_data(frame, plate_crop, text_en, text_ar):
    clean_text = text_en.replace(" ", "")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}_{clean_text}.jpg"
    cv2.imwrite(os.path.join(CROPS_DIR, filename), plate_crop)
    cv2.imwrite(os.path.join(FULL_DIR, filename), frame)
    log_entry = {"Date": datetime.now().strftime("%Y-%m-%d"), "Time": datetime.now().strftime("%H:%M:%S"), "Plate_English": text_en, "Plate_Arabic": text_ar}
    pd.DataFrame([log_entry]).to_csv(LOG_FILE, mode='a', header=not os.path.exists(LOG_FILE), index=False)
    return filename

# 4. STREAMLIT APP
st.set_page_config(page_title="Radar System", layout="wide")

st.markdown("""
    <style>
    [data-testid="stSidebar"] h2 { font-size: 35px !important; font-weight: bold; }
    [data-testid="stSidebar"] label, [data-testid="stSidebar"] p { font-size: 24px !important; color: #ffffff !important; }
    div[role="radiogroup"] label { font-size: 22px !important; }
    h1 { font-size: 50px !important; }
    h3 { font-size: 30px !important; }
    </style>
    """, unsafe_allow_html=True)

st.title("🛡️ Vehicle Radar OCR System")

det_model = load_detection_model()
ocr_model = build_ocr_model()

st.sidebar.header("Input Settings")
input_mode = st.sidebar.radio("Source Select:", ("Upload Image", "Use Camera"))

input_image = None
if input_mode == "Upload Image":
    uploaded_file = st.sidebar.file_uploader("Choose File:", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        input_image = cv2.imdecode(np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8), 1)
else:
    camera_file = st.camera_input("Snapshot")
    if camera_file:
        input_image = cv2.imdecode(np.asarray(bytearray(camera_file.read()), dtype=np.uint8), 1)

if input_image is not None:
    h, w = input_image.shape[:2]
    blob = cv2.resize(cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB), (416, 416))
    blob = np.expand_dims(blob.astype(np.float32) / 255.0, axis=0)
    
    with st.spinner('Scanning...'):
        pred = det_model.predict(blob, verbose=0)[0]
        x1, y1, x2, y2, conf = decode_det(pred, w, h)

    if conf > CONF_THRESHOLD:
        plate_crop = input_image[y1:y2, x1:x2]
        if plate_crop.size > 0:
            text_ar, text_en = process_ocr(plate_crop, ocr_model)
            save_radar_data(input_image, plate_crop, text_en, text_ar)
            
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Detection View")
                rgb_display = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
                cv2.rectangle(rgb_display, (x1, y1), (x2, y2), (0, 255, 0), 3)
                st.image(rgb_display, use_container_width=True)
                
            with col2:
                st.subheader("Recognition Results")
                st.image(cv2.cvtColor(plate_crop, cv2.COLOR_BGR2RGB), caption="Plate Crop")
                
                st.markdown(f"""
                    <div style="background-color:#1e1e1e; padding:20px; border-radius:10px; border-left: 10px solid #2ecc71;">
                        <p style="color:#ffffff; font-size:22px; margin:0;">Detected Plate:</p>
                        <h1 style="color:#2ecc71; font-size:75px; margin:0; direction:rtl; text-align:right;">{text_ar}</h1>
                        <hr style="border: 1px solid #333;">
                        <p style="color:#3498db; font-size:30px; margin:0;">{text_en} | Conf: {conf:.2f}</p>
                    </div>
                """, unsafe_allow_html=True)

                if os.path.exists(LOG_FILE):
                    st.write("---")
                    st.write("📝 Recent Radar Logs")
                    try:
                        st.dataframe(pd.read_csv(LOG_FILE).tail(5))
                    except:
                        st.warning("Log file error.")
    else:
        st.warning("No license plate detected.")