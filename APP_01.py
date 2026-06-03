import streamlit as st
from ultralytics import YOLO
import numpy as np
from PIL import Image
import tempfile

# LOAD MODEL
model = YOLO("best.pt")

# GRAM PER PIXEL
g_per_px = {"Rice": 0.0087, "chicken": 0.0039, "Tempe": 0.0036, "Tofu": 0.0045}

# KCAL PER GRAM
kcal_per_g = {"Rice": 1.739, "chicken": 2.721, "Tempe": 1.223, "Tofu": 1.253}

# CLASS NAMES
class_names = {0: "plate", 1: "Rice", 2: "Tofu", 3: "Tempe", 4: "chicken"}

st.title("Food Calorie Estimation")

uploaded_file = st.file_uploader("Upload Food Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # SHOW IMAGE
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # YOLO PREDICT — use temp file path
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
        temp_file.write(uploaded_file.getbuffer())
        temp_path = temp_file.name

    results = model.predict(
        source=temp_path,
        save=False,
        conf=0.25,
        imgsz=640
    )

    st.write("Prediction Finished!")

    total_calorie = 0
    for r in results:
        if r.masks is not None:
            masks = r.masks.data
            classes = r.boxes.cls.cpu().numpy()
            pixel_dict = {}
            for i, mask in enumerate(masks):
                mask_np = mask.cpu().numpy()
                pixel_area = np.sum(mask_np)
                class_name = class_names[int(classes[i])]
                if class_name != "plate":
                    pixel_dict[class_name] = pixel_dict.get(class_name, 0) + pixel_area

            for food, total_pixel in pixel_dict.items():
                predicted_gram = total_pixel * g_per_px[food]
                predicted_calorie = predicted_gram * kcal_per_g[food]
                total_calorie += predicted_calorie
                st.subheader(food)
                st.write(f"Pixel Area: {total_pixel:.0f} px")
                st.write(f"Predicted Gram: {predicted_gram:.2f} g")
                st.write(f"Predicted Calorie: {predicted_calorie:.2f} kcal")

    st.header(f"Total Calorie: {total_calorie:.2f} kcal")
