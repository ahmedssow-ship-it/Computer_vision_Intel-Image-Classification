# 🌍 Image Classification using Deep Learning (PyTorch & TensorFlow)

This repository implements a complete **image classification pipeline** for natural scenes using Deep Learning.

The goal is to classify images into 6 categories:

- Buildings 🏢  
- Forest 🌳  
- Glacier 🧊  
- Mountain ⛰️  
- Sea 🌊  
- Street 🛣️  

The project includes:
- CNN model built from scratch in **PyTorch**
- CNN model built using **TensorFlow**
- Training, evaluation, and prediction pipeline
- Web application (Streamlit)

---

# 📊 Dataset

The dataset used is the **Intel Image Classification Dataset**.

It contains approximately:
- 14,000 training images
- 3,000 test images

Images are 150x150 pixels and distributed into 6 classes:

```bash
{'buildings': 0,
'forest': 1,
'glacier': 2,
'mountain': 3,
'sea': 4,
'street': 5}
```

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


---

# ⚙️ Installation

1. **Create environment (optional):**

```bash
conda create --name cv_project python=3.11
```

2. **Activate environment:**

```bash
conda activate cv_project
```

3. **Install dependencies:**

```bash
pip install -r requirements.txt
```


