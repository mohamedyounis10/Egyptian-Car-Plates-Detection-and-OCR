<div align="center">
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

<img width="1920" height="1080" alt="License Egypt Presentation" src="https://github.com/user-attachments/assets/ba07a6a2-7e94-4716-847c-e0a1be21b744" />

---

## Table of Contents 🧭

- [Overview 🧾](#overview)
- [Academic Context 🎓](#academic-context)
- [Project Structure 🗂️](#project-structure)
- [Approaches & Models ⚙️](#pipeline)
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
- [`Radar_Detections/`](#) — Logs and outputs from the detection pipeline
- `license Egypt Presentation.pdf` — Technical presentation slides

Tree 🌳:

```text
Project/
├─ Car images/
├─ Custom_CNN/
│  ├─ Detection Model/
│  ├─ OCR Model/
│  └─ Streamlit_app.py
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
## Approaches & Models ⚙️

This project explores two distinct, powerful approaches to solve the Egyptian license plate detection and OCR problem, each with its own strengths and implementations.

### 1. Custom CNN Architecture 🧠
The first approach is a highly specialized **Custom Convolutional Neural Network (CNN)** built entirely from scratch. 
- **Uniqueness & Power**: This model is structurally tailored to exclusively handle the unique aspect ratios, distinctive fonts, and specific layouts of Egyptian license plates. It does not rely on pre-trained generic weights, making its learned features highly domain-specific.
- **Custom Loss Function**: It introduces a robust, custom-designed loss function to severely penalize background noise (e.g., bumpers, text on cars) and force the network to focus purely on the plate's characters.
- **Advanced Data Augmentation**: Extensive, aggressive augmentation techniques were applied to ensure the model remains resilient against harsh lighting conditions, varied camera angles, and partial occlusions common in Egyptian streets.

#### Custom CNN Streamlit Preview 🖼️

<img width="1919" height="765" alt="image" src="https://github.com/user-attachments/assets/61d598ca-4d98-44b9-8ecd-0e8f8eccd543" />

---

### 2. YOLO-Based Real-Time Pipeline 🚀
The second approach focuses on speed and production-readiness by utilizing the **YOLO (You Only Look Once)** framework, designed for real-time tracking and logging.
- **Lightning-Fast Detection**: YOLO rapidly and accurately localizes bounding boxes around vehicles and their plates in a fraction of a second, making it ideal for video streams or live cameras.
- **OCR Integration**: The detected plate regions are cropped and passed to an integrated OCR module to cleanly extract the specific Arabic letters and numbers.
- **Radar Logic Integration**: This approach goes beyond just detection; it hooks the YOLO outputs into a simulated "radar" system that logs the detected plate characters, calculates speed data, and records timestamps for actionable insights.

#### YOLO Streamlit Preview 🖼️

<img width="1902" height="885" alt="Screenshot 2026-05-16 222047" src="https://github.com/user-attachments/assets/eb02744b-edae-4452-9f89-f7af0ef0ce17" />

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

### 3) Launch the Applications 🚀

Since this project features two distinct approaches, there are two separate Streamlit interfaces you can explore:

**A. Run the Custom CNN Application:**
```bash
cd Custom_CNN
streamlit run Streamlit_app.py
```

**B. Run the YOLO & Radar Application:**
```bash
cd YOLO_Models
streamlit run Streamlit_app.py
```

Upload a car image through either web interface to see the detection and OCR results in real-time!

---

<a id="author"></a>
## Author ✍️

- **Name**: Mohamed Younis
