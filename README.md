# 🌍 Image Classification App using Deep Learning

## 📌 Project Overview

This project implements a complete **image classification pipeline** using Deep Learning techniques.  
The goal is to classify natural scene images into six categories:

- Buildings 🏢
- Forest 🌳
- Glacier 🧊
- Mountain ⛰️
- Sea 🌊
- Street 🛣️

Two Convolutional Neural Network (CNN) models were developed and compared:
- PyTorch model
- TensorFlow model

---

## 🧠 Project Features

✔️ Data preprocessing and augmentation  
✔️ CNN model design (from scratch)  
✔️ Training and evaluation  
✔️ Model comparison (PyTorch vs TensorFlow)  
✔️ Command-line prediction script  
✔️ Web application (Flask / Streamlit)  
✔️ Model deployment  

---

## 📂 Project Structure

```
project/
│
├── data/
│   ├── train/
│   └── test/
│
├── models/
│   ├── model_pytorch.py
│   └── model_tensorflow.py
│
├── saved_models/         
│   ├── ahmed_model.pth
│   └── ahmed_model.keras
│
├── utils/                
│   ├── data_loader.py
│   └── data_loader_tf.py
│
├── static/
│   └── style.css
│
├── templates/
│   └── index.html
│
├── app.py                
├── train.py               
├── predict.py             
├── requirements.txt     
└── README.md             
```

---
