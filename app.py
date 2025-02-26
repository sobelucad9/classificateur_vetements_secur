import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter
import pywt

# Charger le modèle entraîné
MODEL_PATH = "./model/classification_vetements_model.h5"
model = load_model(MODEL_PATH)

# Classes du modèle
class_labels = ["dress", "hat", "longsleeve", "outwear", "pants", "shirts", "shoes", "shorts", "skirt", "t-shirt"]

# 📌 Fonction de filtrage par ondelettes
def wavelet_denoise(image):
    img_gray = image.convert('L')  # Convertir en niveaux de gris
    img_array = np.array(img_gray, dtype=np.float32) / 255.0
    coeffs = pywt.wavedec2(img_array, 'haar', level=1)
    coeffs = list(coeffs)
    coeffs[0] *= 0  # Supprimer le bruit basse fréquence
    img_denoised = pywt.waverec2(coeffs, 'haar')
    img_denoised = np.clip(img_denoised * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(img_denoised)

# 📌 Fonction de compression JPEG pour suppression du bruit
def jpeg_compression(image, quality=50):
    from io import BytesIO
    buffer = BytesIO()
    image.save(buffer, format='JPEG', quality=quality)
    return Image.open(buffer)

# 📌 Fonction de prétraitement avec filtrage
def preprocess_image(image):
    image = jpeg_compression(image, quality=50)  # Appliquer compression JPEG
    image = wavelet_denoise(image)  # Appliquer le filtrage par ondelettes
    image = image.resize((128, 128))  # Redimensionner pour MobileNetV2
    image = image.convert("RGB")  # Assurer 3 canaux (RVB)
    img_array = np.array(image) / 255.0  # Normalisation [0,1]
    img_array = img_array.reshape(1, 128, 128, 3)  # Adapter aux dimensions attendues
    return img_array

# 📌 Fonction de prédiction
def predict_image(image):
    img_array = preprocess_image(image)  # Prétraitement
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

if uploaded_file is not None:
    # Charger l'image
    image = Image.open(uploaded_file)
    st.image(image, caption="Image téléchargée", use_container_width=True)

    # Prédiction
    predicted_class, results_df = predict_image(image)

    # Affichage du résultat
    st.subheader(f"🛍️ Classe prédite : **{predicted_class}**")

    # Afficher les probabilités sous forme de tableau
    st.write("### 🔍 Résultat détaillé de la classification :")
    st.dataframe(results_df.style.format({"Probabilité": "{:.2f}"}))
