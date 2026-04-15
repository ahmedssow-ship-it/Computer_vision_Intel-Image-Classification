
import streamlit as st
import numpy as np
from PIL import Image
import random 

random.seed(42)
print(random.random())

# CONFIG
st.set_page_config(page_title="Image Classifier", layout="wide")

CLASSES = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']

# STYLE CSS
st.markdown("""
    <style>
    .main-title {
        font-size: 40px;
        font-weight: bold;
        text-align: center;
        color: #2c3e50;
    }
    .subtitle {
        text-align: center;
        font-size: 18px;
        color: gray;
        margin-bottom: 30px;
    }
    .prediction-box {
        padding: 20px;
        border-radius: 15px;
        background-color: #f0f2f6;
        text-align: center;
        font-size: 22px;
        font-weight: bold;
        color: #2c3e50;
    }
    .info-box {
    padding: 15px;
    border-radius: 12px;
    background: linear-gradient(to right, #36d1dc, #5b86e5);
    color: white;
    margin: 8px;
    text-align: center;
    font-weight: bold;
    font-size: 16px;
}
</style>
""", unsafe_allow_html=True)


# HEADER
st.markdown('<div class="main-title">🌍 Image Classification App</div>', unsafe_allow_html=True)
st.markdown(
    """
    <div style='
        text-align: center;
        font-size:17px;
        color:#6c757d;
        max-width:750px;
        margin:auto;
        line-height:1.6;
        background-color:#f8f9fa;
        padding:15px;
        border-radius:12px;
    '>
    Classify natural scenes using <b style="color:#4facfe;">Deep Learning</b><br><br>
    This application uses deep learning models to classify natural scene images into categories such as
    <b>buildings, forest, glacier, mountain, sea,</b> and <b>street</b>.  
    Upload an image and select a model to get an instant prediction.
    </div>
    """,
    unsafe_allow_html=True
)

# SIDEBAR
st.sidebar.header("⚙️ Settings")

model_type = st.sidebar.selectbox("Choose Model", ["PyTorch", "TensorFlow"])

uploaded_file = st.sidebar.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

# CLASSES DANS MAIN
st.markdown(
    """
    <h3 style='text-align: center; color:#1f4e79; font-size:28px;'>
        Predictable classes
    </h3>
    """,
    unsafe_allow_html=True
)



cols = st.columns(3)
for i, cls in enumerate(CLASSES):
    cols[i % 3].markdown(f"""
    <div class="info-box">
        <b>{cls}</b>
    </div>
    """, unsafe_allow_html=True)


# PREPROCESSING
def preprocess_pytorch(image):
    import torchvision.transforms as transforms

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    return transform(image).unsqueeze(0)


def preprocess_tf(image):
    image = image.resize((150, 150))
    image = np.array(image) / 255.0
    return np.expand_dims(image, axis=0)


# MAIN DISPLAY
if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📸 Uploaded Image")
        st.image(image, use_container_width=True)

    with col2:
        st.subheader("🔍 Prediction")

        if st.button("🚀 Predict"):

            if model_type == "PyTorch":
                import torch
                import sys, os
                sys.path.append(os.path.abspath("."))

                from models.model_pytorch import IntelCNNPyTorch

                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

                model = IntelCNNPyTorch()
                model.load_state_dict(torch.load("saved_models/ahmed_model.pth", map_location=device))
                model.to(device)
                model.eval()

                input_tensor = preprocess_pytorch(image).to(device)

                with torch.no_grad():
                    outputs = model(input_tensor)
                    probs = torch.softmax(outputs, dim=1)
                    _, predicted = torch.max(outputs, 1)

                prediction = CLASSES[predicted.item()]
                confidence = probs.max().item()

            else:
                import tensorflow as tf

                model = tf.keras.models.load_model("saved_models/ahmed_model.keras")

                input_tensor = preprocess_tf(image)

                preds = model.predict(input_tensor)
                prediction = CLASSES[np.argmax(preds)]
                confidence = np.max(preds)

            st.markdown(f'<div class="prediction-box">Prediction: {prediction}</div>', unsafe_allow_html=True)
            st.write(f"Confidence: {confidence:.2f}")

else:
    st.info("👈 Upload an image from the sidebar to start")
    
