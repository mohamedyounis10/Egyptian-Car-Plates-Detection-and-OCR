import os
import cv2
import numpy as np
import pandas as pd
import streamlit as st
import textwrap
from datetime import datetime
from ultralytics import YOLO

# 1. CONFIGURATION AND PATHS
DET_MODEL_PATH = r'C:\Users\moham\Desktop\Computer Vision\Project\YOLO_Models\Detection Model\best.pt'
OCR_MODEL_PATH = r'C:\Users\moham\Desktop\Computer Vision\Project\YOLO_Models\OCR Model\best.pt'

CONF_THRESHOLD = 0.3
IOU_THRESHOLD = 0.4

SAVE_DIR  = r"C:\Users\moham\Desktop\Computer Vision\Project\Radar_Detections"
CROPS_DIR = os.path.join(SAVE_DIR, "Plates_Crops")
FULL_DIR  = os.path.join(SAVE_DIR, "Full_Vehicle")
LOG_FILE  = os.path.join(SAVE_DIR, "radar_log.csv")

for folder in [SAVE_DIR, CROPS_DIR, FULL_DIR]:
    os.makedirs(folder, exist_ok=True)

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

en_to_clean_en = {
    "A": "A", "B": "B", "C": "T", "D": "TH", "E": "G", "F": "H", "G": "KH", "H": "D",
    "I": "Z", "J": "R", "K": "Z", "L": "S", "M": "SH", "N": "C", "O": "D", "P": "T",
    "Q": "Z", "R": "E", "S": "G", "T": "F", "U": "K", "V": "K", "W": "L", "n": "M",
    "Y": "N", "Z": "H", "a": "W", "b": "Y",
    "0": "0", "1": "1", "2": "2", "3": "3", "4": "4", "5": "5", "6": "6", "7": "7", "8": "8", "9": "9"
}

ar_to_en = {v: k for k, v in en_to_ar.items() if k != 'n'}
ar_to_en['م'] = 'M'
for i in range(10):
    ar_to_en[str(i)] = str(i)
    ar_to_en[list(en_to_ar.values())[list(en_to_ar.keys()).index(str(i))]] = str(i)

# 2. MODEL LOADING
@st.cache_resource
def load_detection_model():
    return YOLO(DET_MODEL_PATH)

@st.cache_resource
def load_ocr_model():
    return YOLO(OCR_MODEL_PATH)

# 3. HELPER FUNCTIONS (Integrated from your Notebook functions)
def process_ocr(plate_crop, ocr_model, conf=0.3, iou=0.4):
    res = ocr_model.predict(
        source  = plate_crop,
        conf    = conf,
        iou     = iou,
        verbose = False
    )[0]

    if len(res.boxes) == 0:
        return "لم يتم رصد حروف", "لم يتم رصد حروف", "Unknown"

    boxes  = res.boxes.xyxy.cpu().numpy()   # x1 y1 x2 y2
    clsids = res.boxes.cls.cpu().numpy().astype(int)

    order = np.argsort(boxes[:, 0])
    boxes, clsids = boxes[order], clsids[order]

    chars = [ocr_model.names[i] for i in clsids]

    ordered_digits = [c for c in chars if c.isdigit()]
    ordered_letters = [c for c in chars if c.isalpha()]

    ar_letters_list = [en_to_ar.get(c, c) for c in reversed(ordered_letters)]
    ar_digits_list = [en_to_ar.get(c, c) for c in reversed(ordered_digits)]
    
    ar_letters_disp = " &nbsp; ".join(ar_letters_list)
    ar_digits_disp = " &nbsp; ".join(ar_digits_list)
    
    text_ar_display = f"{ar_letters_disp} &nbsp; &nbsp; &nbsp; {ar_digits_disp}"

    text_ar_log = f"{' '.join(ar_letters_list)} {' '.join(ar_digits_list)}"

    en_digits = [ar_to_en.get(c, c) for c in ordered_digits]
    en_letters = [ar_to_en.get(c, c) for c in ordered_letters]
    
    clean_digits = [en_to_clean_en.get(c, c) for c in en_digits]
    clean_letters = [en_to_clean_en.get(c, c) for c in en_letters]
    text_en = "".join(clean_digits) + " " + "".join(clean_letters)

    return text_ar_display, text_ar_log, text_en


def save_radar_data(frame, plate_crop, text_en, text_ar_log):
    clean_text = text_en.replace(" ", "")
    timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename   = f"{timestamp}_{clean_text}.jpg"

    cv2.imwrite(os.path.join(CROPS_DIR, filename), plate_crop)
    cv2.imwrite(os.path.join(FULL_DIR,  filename), frame)

    log_entry = {
        "Date":          datetime.now().strftime("%Y-%m-%d"),
        "Time":          datetime.now().strftime("%H:%M:%S"),
        "Plate_English": text_en,
        "Plate_Arabic":  text_ar_log
    }
    pd.DataFrame([log_entry]).to_csv(
        LOG_FILE, mode='a',
        header=not os.path.exists(LOG_FILE),
        index=False,
        encoding='utf-8-sig'
    )
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

st.title("🛡️ Vehicle Radar OCR System Enhanced (YOLOv8)")

det_model = load_detection_model()
ocr_model = load_ocr_model()

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
    with st.spinner('Scanning...'):
        results = det_model(input_image, conf=CONF_THRESHOLD, verbose=False)[0]

    if len(results.boxes) > 0:
        best_idx       = int(results.boxes.conf.argmax())
        best_box       = results.boxes[best_idx]
        conf           = float(best_box.conf[0])
        x1, y1, x2, y2 = map(int, best_box.xyxy[0].tolist())

        plate_crop = input_image[y1:y2, x1:x2]

        if plate_crop.size > 0:
            text_ar_display, text_ar_log, text_en = process_ocr(plate_crop, ocr_model, conf=CONF_THRESHOLD, iou=IOU_THRESHOLD)
            save_radar_data(input_image, plate_crop, text_en, text_ar_log)

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Detection View")
                rgb_display = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
                cv2.rectangle(rgb_display, (x1, y1), (x2, y2), (0, 255, 0), 3)
                st.image(rgb_display, use_container_width=True)

            with col2:
                st.subheader("Recognition Results")
                st.image(cv2.cvtColor(plate_crop, cv2.COLOR_BGR2RGB), caption="Plate Crop")

                plate_html = f"""<div style="background-color:#1e1e1e; padding:15px; border-radius:10px; border-left: 10px solid #2ecc71;">
<p style="color:#ffffff; font-size:20px; margin:0; text-align:left;">Detected Plate:</p>
<div style="background-color:#111111; padding:15px; border-radius:8px; margin-top:10px; border: 1px solid #333; text-align:center; direction:rtl;">
<span style="color:#2ecc71; font-size:48px; font-weight:bold; font-family:sans-serif; letter-spacing: 1px;">
{text_ar_display}
</span>
</div>
<hr style="border: 1px solid #333; margin-top:15px;">
<div style="direction: ltr; text-align: left; color:#3498db; font-size:22px; font-weight:bold; margin:0;">
<span>Plate: {text_en}</span> &nbsp; | &nbsp; <span>Conf: {conf:.2f}</span>
</div>
</div>"""
                st.markdown(plate_html, unsafe_allow_html=True)

                if os.path.exists(LOG_FILE):
                    st.write("---")
                    st.write("📝 Recent Radar Logs")
                    try:
                        st.dataframe(pd.read_csv(LOG_FILE, encoding='utf-8-sig').tail(5))
                    except:
                        st.warning("Log file error.")
    else:
        st.warning("No license plate detected.")