import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

model = tf.keras.models.load_model('anchal_model.h5')

def preprocess_image(img_path):
    img = Image.open(img_path).convert('RGB')
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

def predict_image(model, img_path):
    img_array = preprocess_image(img_path)
    predictions = model.predict(img_array)
    decoded_predictions = decode_predictions(predictions, top=5)[0]
    return decoded_predictions

st.title("Model Minde Bike Classifier")

st.markdown(
    """
    <style>
    .reportview-container, .main {
        background-color: #f5f5f5;
    }
    </style>
    """,
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "webp"])
if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")
    
    with open("temp_image", "wb") as f:
        f.write(uploaded_file.getbuffer())

    predictions = predict_image(model, "temp_image")
    
    labels = [label for _, label, _ in predictions]
    scores = [score * 100 for _, _, score in predictions]

    st.write("Top-5 Predictions:")
    for i, (label, score) in enumerate(zip(labels, scores)):
        st.write(f"{i + 1}: {label} ({score:.2f}%)")

    fig, ax = plt.subplots()
    ax.barh(labels, scores, color='skyblue')
    ax.set_xlabel('Confidence (%)')
    ax.set_title('Top-5 Predictions')
    plt.gca().invert_yaxis()
    st.pyplot(fig)
    
    top_prediction = predictions[0]
    st.write(f"**Top Prediction:** {top_prediction[1]} ({top_prediction[2] * 100:.2f}%)")


    st.download_button(
        label="Download Image",
        data=uploaded_file,
        file_name=uploaded_file.name,
        mime="image/jpeg"
    )
