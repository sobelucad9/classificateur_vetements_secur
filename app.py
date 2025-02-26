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
import pywt

# Charger le modèle entraîné
MODEL_PATH = "./model/classification_vetements_model.h5"
model = load_model(MODEL_PATH)

# Classes du modèle
class_labels = ["dress", "hat", "longsleeve", "outwear", "pants", "shirts", "shoes", "shorts", "skirt", "t-shirt"]

# 📌 Fonction de filtrage par ondelettes (atténuation plutôt que suppression)
def wavelet_denoise(image, attenuation_factor=0.5):
    img_gray = image.convert('L')  # Convertir en niveaux de gris
    img_array = np.array(img_gray, dtype=np.float32) / 255.0
    coeffs = pywt.wavedec2(img_array, 'haar', level=1)
    coeffs = list(coeffs)
    coeffs[0] *= attenuation_factor  # Réduction des hautes fréquences au lieu de suppression
    img_denoised = pywt.waverec2(coeffs, 'haar')
    img_denoised = np.clip(img_denoised * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(img_denoised)

# 📌 Fonction de prétraitement avec filtrage par ondelettes
def preprocess_image(image, use_filters=True):
    if use_filters:
        image = wavelet_denoise(image, attenuation_factor=0.5)  # Appliquer la transformée en ondelettes avec atténuation
    image = image.resize((128, 128))  # Redimensionner pour MobileNetV2
    image = image.convert("RGB")  # Assurer 3 canaux (RVB)
    img_array = img_to_array(image)
    img_array = preprocess_input(img_array)  # Appliquer le prétraitement MobileNetV2
    img_array = img_array.reshape(1, 128, 128, 3)  # Adapter aux dimensions attendues
    return img_array

# 📌 Fonction de prédiction
def predict_image(image, use_filters=True):
    img_array = preprocess_image(image, use_filters)  # Prétraitement avec ou sans filtrage par ondelettes
    pred = model.predict(img_array)  # Prédiction
    predicted_class = class_labels[np.argmax(pred)]  # Classe prédite

    # Résultats sous forme de DataFrame
    results_df = pd.DataFrame({'Classe': class_labels, 'Probabilité': pred.flatten()}).sort_values(by='Probabilité', ascending=False)

    return predicted_class, results_df

# Interface Streamlit
st.set_page_config(page_title="👕🧢 Classificateur de Vêtements", layout="centered")

st.title("👕🧢 Classificateur de vêtements sécurisé avec filtrage par ondelettes pour les images. *jfsg*")
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

use_filters = st.checkbox("Utiliser le filtrage par ondelettes pour réduire le bruit", value=True)

if uploaded_file is not None:
    # Charger l'image
    image = Image.open(uploaded_file)
    st.image(image, caption="Image téléchargée", use_container_width=True)

    # Prédiction avec ou sans filtrage par ondelettes
    predicted_class, results_df = predict_image(image, use_filters)

    # Affichage du résultat
    st.subheader(f"🛍️ Classe prédite : **{predicted_class}**")

    # Afficher les probabilités sous forme de tableau
    st.write("### 🔍 Résultat détaillé de la classification :")
    st.dataframe(results_df.style.format({"Probabilité": "{:.2f}"}))
