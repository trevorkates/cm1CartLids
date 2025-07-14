# app.py
import streamlit as st
from fastai.vision.all import *
from zipfile import ZipFile
import shutil
import os
from PIL import Image

# Setup paths
MODEL_PATH = 'trash_classifier.pkl'
TRAIN_DIR = Path('train_data')
TEST_DIR = Path('test_data')

# Load model if exists
model = None
if os.path.exists(MODEL_PATH):
    model = load_learner(MODEL_PATH)

# Function to train model from uploaded zip
def train_model_from_zip(zip_file):
    shutil.rmtree(TRAIN_DIR, ignore_errors=True)
    with ZipFile(zip_file, 'r') as zf:
        zf.extractall(TRAIN_DIR)
    data_root = next(TRAIN_DIR.iterdir())  # get inside folder
    dblock = DataBlock(
        blocks=(ImageBlock, CategoryBlock),
        get_items=get_image_files,
        get_y=parent_label,
        splitter=RandomSplitter(seed=42),
        item_tfms=Resize(224),
        batch_tfms=aug_transforms()
    )
    dls = dblock.dataloaders(data_root, bs=4)
    learn = vision_learner(dls, resnet18, metrics=accuracy)
    learn.fine_tune(3)
    learn.export(MODEL_PATH)
    return learn

# Function to predict from uploaded test zip
def predict_from_zip(zip_file):
    shutil.rmtree(TEST_DIR, ignore_errors=True)
    with ZipFile(zip_file, 'r') as zf:
        zf.extractall(TEST_DIR)
    test_root = next(TEST_DIR.iterdir())
    test_images = get_image_files(test_root)
    learn = load_learner(MODEL_PATH)
    results = []
    for img in test_images:
        pred, _, probs = learn.predict(img)
        results.append((img, pred, probs.max().item()))
    return results

# Streamlit UI
st.set_page_config(page_title="Trash Classifier", layout="wide")
st.title("🧠 Trash Classifier")

st.sidebar.header("📥 Upload Files")

# Training upload
train_zip = st.sidebar.file_uploader("Upload Training ZIP", type=['zip'])
if train_zip and st.sidebar.button("Train Model"):
    with st.spinner("Training... This may take a minute."):
        model = train_model_from_zip(train_zip)
    st.success("✅ Model trained and saved as 'trash_classifier.pkl'")

# Testing upload
test_zip = st.sidebar.file_uploader("Upload Test ZIP", type=['zip'])
if test_zip and st.sidebar.button("Run Predictions"):
    if not os.path.exists(MODEL_PATH):
        st.error("❌ No trained model found. Please train a model first.")
    else:
        with st.spinner("Running predictions..."):
            results = predict_from_zip(test_zip)

        st.subheader("📊 Predictions")
        for img_path, pred, conf in results:
            cols = st.columns([1, 2])
            with cols[0]:
                st.image(Image.open(img_path), width=150)
            with cols[1]:
                st.markdown(f"**Prediction:** `{pred}`")
                st.markdown(f"**Confidence:** `{conf:.2%}`")
                correction = st.radio(
                    f"Is this correct?",
                    ["Yes", "Should be good", "Should be bad"],
                    key=str(img_path.name)
                )
                # Save correction logic could go here
