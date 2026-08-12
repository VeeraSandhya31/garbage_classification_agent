
import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image

st.set_page_config(page_title="Smart Garbage Classifier", page_icon="♻️", layout="centered")

@st.cache_resource
def load_my_model():
    MODEL_DIR = "/content/drive/MyDrive/garbage_classification/models"
    return load_model(f"{MODEL_DIR}/garbage_classifier_efficientnetb3.keras")

model = load_my_model()

class_names = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]

recycling_advice = {
    "cardboard": "Recyclable — flatten and place in paper/cardboard recycling.",
    "glass": "Recyclable — rinse and place in glass recycling bin.",
    "metal": "Recyclable — rinse cans, place in metal recycling.",
    "paper": "Recyclable — keep dry, place in paper recycling.",
    "plastic": "Check resin code; most PET/HDPE plastics are recyclable.",
    "trash": "Not recyclable — dispose in general waste.",
}

st.title("♻️ Smart Garbage Classification & Recycling Recommendation Agent")
st.write("Upload a photo of garbage/trash to get instant classification and recycling advice.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_container_width=True)

    img_resized = img.resize((300, 300))
    arr = img_to_array(img_resized)
    arr = efficientnet_preprocess(arr)
    arr = np.expand_dims(arr, axis=0)

    pred = model.predict(arr, verbose=0)[0]
    idx = np.argmax(pred)
    label = class_names[idx]
    confidence = pred[idx] * 100

    st.subheader(f"Prediction: {label.upper()}")
    st.write(f"Confidence: {confidence:.1f}%")
    st.success(recycling_advice.get(label, "No advice available."))

    st.subheader("Confidence per class")
    probs_dict = {class_names[i]: float(pred[i]) for i in range(len(class_names))}
    st.bar_chart(probs_dict)
