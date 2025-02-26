import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter
import os
import subprocess
import sys

# Charger le modèle entraîné
MODEL_PATH = "./model/classification_vetements_model.h5"
model = load_model(MODEL_PATH)

# Classes du modèle
class_labels = ["dress", "hat", "longsleeve", "outwear", "pants", "shirts", "shoes", "shorts", "skirt", "t-shirt"]

# 📌 Fonction de compression JPEG pour suppression du bruit
def jpeg_compression(image, quality=50):
    from io import BytesIO
    buffer = BytesIO()
    image.save(buffer, format='JPEG', quality=quality)
    return Image.open(buffer)

# 📌 Fonction de prétraitement avec compression JPEG
def preprocess_image(image, use_filters=True):
    if use_filters:
        image = jpeg_compression(image, quality=50)  # Appliquer compression JPEG
    image = image.resize((128, 128))  # Redimensionner pour MobileNetV2
    image = image.convert("RGB")  # Assurer 3 canaux (RVB)
    img_array = np.array(image) / 255.0  # Normalisation [0,1]
    img_array = img_array.reshape(1, 128, 128, 3)  # Adapter aux dimensions attendues
    return img_array

# 📌 Fonction de prédiction
def predict_image(image, use_filters=True):
    img_array = preprocess_image(image, use_filters)  # Prétraitement avec ou sans filtres
    pred = model.predict(img_array)  # Prédiction
    predicted_class = class_labels[np.argmax(pred)]  # Classe prédite

    # Résultats sous forme de DataFrame
    results_df = pd.DataFrame({'Classe': class_labels, 'Probabilité': pred.flatten()}).sort_values(by='Probabilité', ascending=False)

    return predicted_class, results_df

# Interface Streamlit
st.set_page_config(page_title="👕🧢 Classificateur de Vêtements", layout="centered")

st.title("👕🧢 Classificateur de vêtements par Jean Frédéric Sobel GOMIS & Malayni SAMBOU")
st.write("""
Téléchargez une image pour la classer.

**Catégories disponibles :**
- 👗 Dress -> Robe
- 🧢 Hat -> Casquette
- 👕 Longsleeve -> Manches longues
- 🧥 Outwear -> Vêtements d'extérieur
- 👖 Pant -> Pantalon
- 👔 Shirt -> Chemise
- 👟 Shoes -> Chaussures
- 🩳 Short -> Short
- 👚 Skirt -> Jupe
- 👕 T-shirt -> T-shirt
""")

uploaded_file = st.file_uploader("Choisissez une image...", type=["jpg", "png", "jpeg"])

use_filters = st.checkbox("Utiliser la compression JPEG pour réduire le bruit", value=True)

if uploaded_file is not None:
    # Charger l'image
    image = Image.open(uploaded_file)
    st.image(image, caption="Image téléchargée", use_container_width=True)

    # Prédiction avec ou sans compression JPEG
    predicted_class, results_df = predict_image(image, use_filters)

    # Affichage du résultat
    st.subheader(f"🛍️ Classe prédite : **{predicted_class}**")

    # Afficher les probabilités sous forme de tableau
    st.write("### 🔍 Résultat détaillé de la classification :")
    st.dataframe(results_df.style.format({"Probabilité": "{:.2f}"}))
