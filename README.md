<div align="center">
  <!-- 🖼️ PLACEHOLDER: Insert your project header image here -->
  <!-- <img src="URL_TO_YOUR_HEADER_IMAGE" alt="Project Banner" width="100%"> -->
  
  <h1>Egyptian Car Plates Detection & OCR 🚗🇪🇬📸</h1>
  <p>End-to-end computer vision pipeline for detecting and reading Egyptian car license plates using YOLO, Custom CNNs, and an interactive Streamlit dashboard.</p>

  <p>
    <a href="#overview">Overview 🧾</a> •
    <a href="#academic-context">Academic Context 🎓</a> •
    <a href="#project-structure">Project Structure 🗂️</a> •
    <a href="#pipeline">Pipeline & Models ⚙️</a> •
    <a href="#streamlit-preview">Streamlit Preview 🖼️</a> •
    <a href="#how-to-run">How to Run ▶️</a>
  </p>

  <p>
    <img alt="Python" src="https://img.shields.io/badge/Python-3.x-blue" />
    <img alt="OpenCV" src="https://img.shields.io/badge/OpenCV-Computer_Vision-5C3EE8" />
    <img alt="YOLO" src="https://img.shields.io/badge/YOLO-Detection-00FFFF" />
    <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-Deep_Learning-EE4C2C" />
    <img alt="Streamlit" src="https://img.shields.io/badge/Streamlit-Web_App-FF4B4B" />
  </p>
</div>

---

## Table of Contents 🧭

- [Overview 🧾](#overview)
- [Academic Context 🎓](#academic-context)
- [Project Structure 🗂️](#project-structure)
- [Pipeline & Models ⚙️](#pipeline)
- [Streamlit Preview 🖼️](#streamlit-preview)
- [How to Run ▶️](#how-to-run)
- [Author ✍️](#author)

---

<a id="overview"></a>
## Overview 🧾

This repository focuses on **Egyptian License Plate Detection and Character Recognition**, providing a robust, business-ready computer vision workflow. 

- **Object Detection 🔍**: Localizes vehicles and license plates within dynamic street environments.
- **Optical Character Recognition (OCR) 🔠**: Extracts Arabic letters and numbers from the detected plates.
- **Custom CNN Architectures 🧠**: Explores custom deep learning approaches tailored for regional plate characteristics.
- **Interactive UI 🖥️**: Provides a sleek web interface via Streamlit to upload images, process them, and log detections (e.g., radar speed logs).

---

<a id="academic-context"></a>
## Academic Context 🎓

This project was developed as part of an academic curriculum focusing on applied Computer Vision and Deep Learning. It demonstrates the ability to:
- Integrate state-of-the-art models (YOLO) with custom-built neural networks (CNNs).
- Handle real-world, noisy data (varying angles, lighting, and occlusions).
- Deploy machine learning pipelines into user-friendly, interactive web applications.
- Document and present technical architectures clearly.

---

<a id="project-structure"></a>
## Project Structure 🗂️

- [`Car images/`](#) — Sample datasets and testing images
- [`Custom_CNN/`](#) — Custom Convolutional Neural Network notebooks and models
- [`YOLO_Models/`](#) — YOLO-based detection and OCR models, plus the web application
  - `Detection Model/`
  - `OCR Model/`
  - `Streamlit_app.py`
- [`Radar_Detections/`](#) — Logs and outputs from the detection pipeline
- `license Egypt Presentation.pdf` — Technical presentation slides

Tree 🌳:

```text
Project/
├─ Car images/
├─ Custom_CNN/
├─ Radar_Detections/
├─ YOLO_Models/
│  ├─ Detection Model/
│  ├─ OCR Model/
│  └─ Streamlit_app.py
├─ README.md
└─ license Egypt Presentation.pdf
```

---

<a id="pipeline"></a>
## Pipeline & Models ⚙️

The system utilizes a multi-stage approach for maximum accuracy:

1. **Plate Detection**: Uses YOLO to quickly and accurately draw bounding boxes around license plates in an image.
2. **Character Segmentation & OCR**: 
   - Crops the detected plate.
   - Passes the crop to a specialized OCR model (or Custom CNN) to read the specific Arabic characters and numbers.
3. **Actionable Logging**: Integrates with simulated "radar" logic to record the detected plate string alongside timestamp and vehicle data into a localized log.

---

<a id="streamlit-preview"></a>
## Streamlit Preview 🖼️

<!-- 🖼️ PLACEHOLDER: Insert your Streamlit app screenshots here -->
<div align="center">
  <p><i>Replace this text with a screenshot of your Streamlit application</i></p>
  <!-- <img alt="Streamlit App Screenshot" src="URL_TO_YOUR_STREAMLIT_IMAGE" width="100%" /> -->
</div>

To run the Streamlit app locally:

```bash
cd YOLO_Models
streamlit run Streamlit_app.py
```

---

<a id="how-to-run"></a>
## How to Run ▶️

### 1) Setup Environment 🧪
```bash
# It is recommended to use a virtual environment
python -m venv .venv
.\.venv\Scripts\activate
```

### 2) Install Dependencies 📦
*(Ensure you have your specific `requirements.txt` installed, example below)*
```bash
pip install streamlit opencv-python ultralytics torch torchvision pandas numpy
```

### 3) Launch the App 🚀
```bash
cd YOLO_Models
streamlit run Streamlit_app.py
```

Upload a car image through the web interface to see the detection and OCR results in real-time!

---

<a id="author"></a>
## Author ✍️

- **Name**: Mohamed Younis
- **Program**: MSC KFS - Data Science Phase 2 💙
