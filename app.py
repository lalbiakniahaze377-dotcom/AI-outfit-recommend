import streamlit as st
import numpy as np
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from sklearn.metrics.pairwise import cosine_similarity
import os
from PIL import Image
import random

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    return ResNet50(weights="imagenet", include_top=False, pooling="avg")

model = load_model()

# ---------------- LOAD EMBEDDINGS ----------------
def load_embeddings(category):
    features_path = f"data/embeddings/{category}_features.npy"
    paths_path = f"data/embeddings/{category}_paths.npy"

    if not os.path.exists(features_path) or not os.path.exists(paths_path):
        st.error(f"❌ {category} embeddings missing")
        st.stop()

    features = np.load(features_path)
    paths = np.load(paths_path, allow_pickle=True)
    return features, paths

shirts_features, shirts_paths = load_embeddings("shirts")
shorts_features, shorts_paths = load_embeddings("shorts")
shoes_features, shoes_paths = load_embeddings("shoes")

# ---------------- FEATURE EXTRACTION ----------------
def extract_feature(img):
    img = img.resize((224, 224))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    feature = model.predict(img_array, verbose=0)
    return feature.flatten()

# ---------------- RECOMMEND SIMILAR ----------------
def recommend_similar(query_feature, features, paths, top_n=5):
    similarity = cosine_similarity([query_feature], features)[0]
    indices = similarity.argsort()[-top_n:][::-1]
    return [paths[i] for i in indices]

# ---------------- RANDOM PICK ----------------
def random_items(paths, n=5):
    return random.sample(list(paths), min(n, len(paths)))

# ---------------- UI ----------------
st.title("👕 AI Outfit Recommendation System")

uploaded_file = st.file_uploader("Upload a shirt image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", width=300)

    if st.button("Generate Outfit"):
        with st.spinner("Analyzing and styling..."):

            query_feature = extract_feature(img)

            # 👕 Similar shirts
            similar_shirts = recommend_similar(query_feature, shirts_features, shirts_paths)

            # 🩳 Random shorts
            shorts = random_items(shorts_paths)

            # 👟 Random shoes
            shoes = random_items(shoes_paths)

        # ---------------- DISPLAY ----------------

        st.subheader("👕 Recommended Shirts")
        cols = st.columns(5)
        for col, img_path in zip(cols, similar_shirts):
            try:
                rec_img = Image.open(img_path)
                col.image(rec_img, width=150)
            except:
                col.write("Error")

        st.subheader("🩳 Recommended Shorts")
        cols = st.columns(5)
        for col, img_path in zip(cols, shorts):
            try:
                rec_img = Image.open(img_path)
                col.image(rec_img, width=150)
            except:
                col.write("Error")

        st.subheader("👟 Recommended Shoes")
        cols = st.columns(5)
        for col, img_path in zip(cols, shoes):
            try:
                rec_img = Image.open(img_path)
                col.image(rec_img, width=150)
            except:
                col.write("Error")