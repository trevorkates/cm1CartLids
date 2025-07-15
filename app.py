import streamlit as st
import openai
import base64
import os
from PIL import Image

# === Config ===
IMAGE_FOLDER = "images"  # Make sure this folder exists in your repo
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

if st.button("Send to GPT-4o"):
    with open(image_path, "rb") as img_file:
        b64_img = base64.b64encode(img_file.read()).decode()

    with st.spinner("Sending image to GPT-4o..."):
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "user", "content": [
                        {"type": "text", "text": "This is a photo of a trash can lid. Decide if the lid should be ACCEPTED or REJECTED based on part correctness, print quality, and branding. Respond with a decision and a short explanation."},
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
