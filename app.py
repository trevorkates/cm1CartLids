import streamlit as st
import openai
import base64
import os
from PIL import Image

# === Config ===
IMAGE_FOLDER = "images"
client = openai.OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

st.set_page_config(page_title="Trash Lid Classifier", layout="centered")

st.title("♻️ Trash Can Lid Inspector")
st.write("Select a lid image and send it to GPT-4o to determine if it's ACCEPTED or REJECTED.")

# === File selection ===
images = [f for f in os.listdir(IMAGE_FOLDER) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
if not images:
    st.warning("No images found in the 'images' folder.")
    st.stop()

selected_file = st.selectbox("Choose an image to evaluate", images)
image_path = os.path.join(IMAGE_FOLDER, selected_file)

image = Image.open(image_path)
st.image(image, caption=selected_file, use_container_width=True)

# === Tolerance Control (moved here)
tolerance = st.slider("🔧 Defect Tolerance", min_value=0, max_value=10, value=5,
                      help="0 = strict (small issues cause REJECT), 10 = lenient (minor defects allowed)")

# === Few-shot examples from Google Drive ===
example_images = {
    "MissingBrand.jpeg": "REJECT – Branding is missing or faded.",
    "GoodIML.jpeg": "ACCEPT – IML is printed clearly and aligned.",
    "GoodBrandButBadWhiteStreaksNearTop.jpeg": "REJECT – White streaks near the top make it defective.",
    "GoodBrand.jpeg": "ACCEPT – Brand is correct and clearly placed.",
    "BadStreaks.jpeg": "REJECT – White streaks make the lid unacceptable."
}

drive_ids = {
    "MissingBrand.jpeg": "1ebxDmqwDkVVBen7DZh4M5eb0IWyZ2dLO",
    "GoodIML.jpeg": "1TW4-kUvd8C_qzPsk_XMIK8r6iRVSmqeR",
    "GoodBrandButBadWhiteStreaksNearTop.jpeg": "1vsTsq_haEtx8pFYAeo_DZTl4WdxD_hQT",
    "GoodBrand.jpeg": "1GCsMHYybyLNT_HVDXz4w17NFPTKmrNVt",
    "BadStreaks.jpeg": "1TV8vH_igEnZ-4027AbEqbpIkCwg5hCzE"
}

examples = []
for filename, explanation in example_images.items():
    file_id = drive_ids.get(filename)
    if file_id:
        examples.append({"type": "image_url", "image_url": {"url": f"https://drive.google.com/uc?id={file_id}"}})
        examples.append({"type": "text", "text": explanation})

# === GPT Prompt ===
prompt = f"""
You are an expert inspector of plastic trash can lids.
Classify each lid as either ACCEPT or REJECT.

Use this tolerance level (0–10): {tolerance}
- 0 = strict (even small defects cause REJECT)
- 10 = lenient (only major issues = REJECT)

Rules:
- ACCEPT if branding and IML print are clear, centered, and legible
- Minor defects (scuffs, small dirt, slight misalignment) are OK
- REJECT if branding is missing, warped, unreadable, or severely off-center

Respond ONLY with ACCEPT or REJECT and a short reason.
"""

# === Send to GPT ===
if st.button("Send to GPT-4o"):
    with open(image_path, "rb") as img_file:
        b64_img = base64.b64encode(img_file.read()).decode()

    with st.spinner("Sending image to GPT-4o..."):
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": prompt},
                    *examples,
                    {"role": "user", "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}}
                    ]}
                ]
            )
            content = response.choices[0].message.content.strip()
            st.markdown("### 🧠 GPT-4o Decision")
            if "REJECT" in content.upper():
                st.error(content)
            elif "ACCEPT" in content.upper():
                st.success(content)
            else:
                st.info(content)

        except Exception as e:
            st.error(f"❌ Error: {e}")
