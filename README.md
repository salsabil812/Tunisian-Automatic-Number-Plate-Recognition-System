# 🇹🇳 Automatic License Plate Recognition – Tunisia (ALPR Tunisie)

This project implements a full **Tunisian License Plate Recognition pipeline**, including:

- 🚗 **Vehicle detection** (YOLOv8)
- 🔳 **Plate detection** (YOLOv8)
- 🟦 **Super-resolution on plates** (Real-ESRGAN)
- 🔤 **OCR extraction** (Microsoft TrOCR)
- 🇹🇳 **Normalization of Tunisian plate format**
- 🎨 **Streamlit web interface**

---

## 🧠 Project Structure

📂 Tunisian-Automatic-Number-Plate-Recognition-System/

│── app.py # Streamlit web app

│── best_Vehicules.pt # YOLOv8 model for vehicle detection

│── best_plaques_final.pt # YOLOv8 model for plate detection

│── requirements.txt # Python dependencies

│── README.md # Documentation

│── notebook.ipynb # Colab notebook (full pipeline training + ESRGAN generation)


## ⚠️ Real-ESRGAN Model Not Included Here :The file "**RealESRGAN_x4plus.pth**" is **NOT uploaded to GitHub**, because:

- GitHub forbids files **> 100 MB**
- Real-ESRGAN weights are **111 MB**
 

➡️ **You must run the notebook `notebook.ipynb` once**, which automatically downloads: RealESRGAN_x4plus.pth and saves it in the project directory.


📌 The Streamlit app will then work normally.
---
## 🚀 Running the Application Locally
### 1. Install dependencies
pip install -r requirements.txt

### 2. Run Streamlit app
streamlit run app.py


The interface includes:
*Upload of custom image
*Test images included in repo
*Vehicle detection results
*Plate crops (raw / GAN / OCR)
*OCR text + Tunisian normalization


⭐ Technologies Used :

- Task	Model / Library

- Vehicle Detection	YOLOv8

- Plate Detection	YOLOv8

- Super Resolution	Real-ESRGAN

- OCR	Microsoft TrOCR

- UI	Streamlit

👩‍🎓 Author :

Salsabil Ben Halima
Étudiante en terminale Ingénierie des Données
Faculté des Sciences — Université de Sfax
Département d’Informatique et des Communications
