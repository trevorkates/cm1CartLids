import streamlit as st
from fastai.vision.all import *
from zipfile import ZipFile
import os, shutil
from pathlib import Path

st.set_page_config(page_title="Trash Classifier Trainer", layout="wide")
st.title("🗑️ Trash Lid Image Classifier - Train & Test")

# Set folders
TRAIN_DIR = Path("train_data")
TEST_DIR = Path("test_data")
MODEL_NAME = "trash_model.pkl"

# Clean folders
def reset_folder(path):
    if path.exists(): shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)

# Extract zip
def extract_zip(uploaded_file, dest):
    reset_folder(dest)
    with ZipFile(uploaded_file, 'r') as zip_ref:
        zip_ref.extractall(dest)

# Load data
@st.cache_resource
def load_learner_from_file():
    if os.path.exists(MODEL_NAME):
        return load_learner(MODEL_NAME)
    return None

# Create DataLoaders
def get_dls(path, bs=4):
    dblock = DataBlock(
        blocks=(ImageBlock, CategoryBlock),
        get_items=get_image_files,
        get_y=parent_label,
        splitter=RandomSplitter(valid_pct=0.2, seed=42),
        item_tfms=Resize(224)
    )
    return dblock.dataloaders(path, bs=bs)

# Sidebar
with st.sidebar:
    st.header("1️⃣ Upload Training Data (.zip)")
    train_zip = st.file_uploader("Upload ZIP (containing good/ and bad/ folders inside a folder)", type=["zip"])
    if st.button("Train Model") and train_zip:
        extract_zip(train_zip, TRAIN_DIR)
        subfolder = next(TRAIN_DIR.iterdir()) if any(TRAIN_DIR.iterdir()) else TRAIN_DIR
        dls = get_dls(subfolder)
        learn = vision_learner(dls, resnet18, metrics=accuracy)
        with st.spinner("Training..."):
            learn.fine_tune(3)
        learn.export(MODEL_NAME)
        st.success("Model trained and saved!")

    st.header("2️⃣ Upload Test Data (.zip)")
    test_zip = st.file_uploader("Upload ZIP with test images (in good/ and bad/ folders)", type=["zip"], key="test")
    run_predictions = st.button("Run Predictions")

# Main App
learn = load_learner_from_file()
if run_predictions and test_zip:
    if not learn:
        st.error("Train a model first!")
    else:
        extract_zip(test_zip, TEST_DIR)
        test_folder = next(TEST_DIR.iterdir()) if any(TEST_DIR.iterdir()) else TEST_DIR
        test_images = get_image_files(test_folder)

        preds = []
        for img in test_images:
            pred, _, probs = learn.predict(img)
            preds.append((img, pred, probs))

        st.subheader("🔍 Results - Review Predictions")
        correct_labels = []
        for img, pred, probs in preds:
            col1, col2 = st.columns([1, 2])
            with col1:
                st.image(img, width=224, caption=f"Predicted: {pred}")
            with col2:
                true_label = st.radio(
                    f"What is the correct label for `{img.name}`?", ["good", "bad"],
                    index=["good", "bad"].index(pred), key=str(img)
                )
                correct_labels.append((img, true_label))

        if st.button("✅ Retrain with Corrections"):
            # Move corrected images into TRAIN_DIR structure
            for img, label in correct_labels:
                dest_folder = TRAIN_DIR / "corrected" / label
                dest_folder.mkdir(parents=True, exist_ok=True)
                shutil.copy(img, dest_folder / img.name)

            all_train_data = TRAIN_DIR / "corrected"
            dls = get_dls(all_train_data)
            learn = vision_learner(dls, resnet18, metrics=accuracy)
            with st.spinner("Retraining with feedback..."):
                learn.fine_tune(2)
            learn.export(MODEL_NAME)
            st.success("Model retrained with corrected labels!")
