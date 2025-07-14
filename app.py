import streamlit as st
from fastai.vision.all import *
import zipfile, os, shutil
from pathlib import Path
import tempfile

st.set_page_config(layout="wide")
st.title("🔍 Trash Lid Quality Classifier")

# Globals
model_path = Path("model.pkl")
learner = None
dls = None
class_labels = []

# Helper: Extract zip
@st.cache_resource(show_spinner=False)
def extract_zip(uploaded_file):
    temp_dir = tempfile.mkdtemp()
    with zipfile.ZipFile(uploaded_file, 'r') as zip_ref:
        zip_ref.extractall(temp_dir)
    return Path(temp_dir)

# Helper: Load images from test folder
def get_images_from_folder(path):
    return get_image_files(path)

# Helper: Visual feedback
def display_predictions(imgs, preds):
    cols = st.columns(3)
    for i, (img, pred) in enumerate(zip(imgs, preds)):
        with cols[i % 3]:
            st.image(img, width=300, caption=f"Predicted: {pred}")
            correct = st.radio(
                f"Was this correct?",
                ["Yes", "No - Should be good", "No - Should be bad"],
                key=str(img),
                horizontal=True
            )
            if correct != "Yes":
                corrected_label = "good" if "good" in correct else "bad"
                shutil.copy(img, Path("./feedback")/corrected_label)

# Train model
st.header("📦 Upload Training Dataset (.zip)")
train_zip = st.file_uploader("Upload a zip with 'good' and 'bad' folders", type="zip", key="train")

if train_zip:
    train_dir = extract_zip(train_zip)
    folder = next(train_dir.iterdir()) if len(list(train_dir.iterdir())) == 1 else train_dir
    st.success(f"✅ Found training images in: {folder.name}")
    
    def label_func(fp): return fp.parent.name
    dblock = DataBlock(
        blocks=(ImageBlock, CategoryBlock),
        get_items=get_image_files,
        splitter=RandomSplitter(seed=42),
        get_y=label_func,
        item_tfms=Resize(224)
    )
    dls = dblock.dataloaders(folder, bs=4)
    class_labels = dls.vocab

    learner = vision_learner(dls, resnet18, metrics=accuracy)

    with st.spinner("Training the model..."):
        learner.fine_tune(3)
        learner.export(model_path)
    st.success("✅ Model trained and saved!")

# Predict from test zip
st.header("🧪 Upload Test Dataset (.zip)")
test_zip = st.file_uploader("Upload a test zip with images", type="zip", key="test")

if test_zip:
    if not model_path.exists():
        st.warning("Please train a model first.")
    else:
        test_dir = extract_zip(test_zip)
        test_folder = next(test_dir.iterdir()) if len(list(test_dir.iterdir())) == 1 else test_dir

        if learner is None:
            learner = load_learner(model_path)

        test_imgs = get_images_from_folder(test_folder)
        preds = [learner.predict(img)[0] for img in test_imgs]

        st.header("🖼️ Predictions")
        display_predictions(test_imgs, preds)

# Retrain from feedback
if Path("./feedback").exists():
    st.header("🔁 Improve with Feedback")
    if st.button("Retrain with Feedback"):
        corrected_path = Path("./feedback")
        if len(list(corrected_path.rglob("*.jpg"))) > 1:
            dblock_feedback = DataBlock(
                blocks=(ImageBlock, CategoryBlock),
                get_items=get_image_files,
                splitter=RandomSplitter(seed=24),
                get_y=label_func,
                item_tfms=Resize(224)
            )
            dls_feedback = dblock_feedback.dataloaders(corrected_path, bs=4)
            learner = vision_learner(dls_feedback, resnet18, metrics=accuracy)
            learner.fine_tune(2)
            learner.export(model_path)
            shutil.rmtree(corrected_path)
            st.success("✅ Model updated with feedback!")
        else:
            st.warning("Not enough feedback examples to retrain.")

# Generate visual summary
if learner and st.button("📸 Show Visual Summary"):
    from fastai.vision.utils import plot_top_losses
    interp = ClassificationInterpretation.from_learner(learner)
    interp.plot_confusion_matrix(figsize=(4,4))
    st.pyplot()
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.info("This summary shows where the model struggles the most.")
