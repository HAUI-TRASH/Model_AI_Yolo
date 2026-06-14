import cv2
import numpy as np
import streamlit as st

from trashsmart.pipeline.main import classify_image


st.set_page_config(page_title="TrashSmart ResNet50", layout="centered")

st.title("TrashSmart - phan loai chat lieu")
st.caption("ResNet50 phan loai mot vat the: nhua, kim loai, thuy tinh hoac giay.")

uploaded = st.file_uploader("Chon anh vat the", type=["jpg", "jpeg", "png", "webp"])

if uploaded is not None:
    image_bytes = uploaded.read()
    np_arr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if image is None:
        st.error("Khong doc duoc anh.")
    else:
        st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="Anh dau vao", use_container_width=True)

        with st.spinner("Dang phan loai bang ResNet50..."):
            result = classify_image(image)

        st.success("Hoan tat")
        st.metric("Label ResNet", result["label"])
        st.json(result)
