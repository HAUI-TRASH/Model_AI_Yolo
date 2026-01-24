import streamlit as st
import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
import joblib
from datetime import datetime
import time
import plotly.express as px
import plotly.graph_objects as go

# =====================================================
# 1. CẤU HÌNH CHUNG & TẢI MÔ HÌNH
# =====================================================
st.set_page_config(page_title="Traffic Congestion Demo", layout="wide")

st.title("🚦 HỆ THỐNG NHẬN DIỆN VÀ DỰ BÁO ÙN TẮC GIAO THÔNG")
st.markdown("**YOLOv11 + DeepSORT + Logistic Regression** – Demo minh họa theo mật độ phương tiện")

# Load mô hình
yolo_model = YOLO("models/best0.pt")  # YOLOv11 đã fine-tune
logistic_model = joblib.load("models/logistic_congestion_model_refined.pkl")  # Logistic Regression (theo mật độ)

# =====================================================
# 2. GIAO DIỆN CHỌN VIDEO & THAM SỐ MÔI TRƯỜNG
# =====================================================
video_file = st.file_uploader(" Chọn video giao thông", type=["mp4", "avi"])
if video_file is not None:
    tfile = open("temp_video.mp4", "wb")
    tfile.write(video_file.read())
    video_path = "temp_video.mp4"
else:
    video_path = "video_samples/hanoi.mp4"

with st.sidebar:
    st.header("⚙ Thiết lập môi trường")
    selected_weather = st.selectbox("Thời tiết", ["Clear", "Cloudy", "Rainy", "Foggy"], index=0)
    selected_temp = st.slider("Nhiệt độ (°C)", 10, 45, 20)


# =====================================================
# 3. HÀM DỰ ĐOÁN
# =====================================================
def predict_state_from_features(model, vehicle_density, avg_speed, weather, temperature):
    """Tạo DataFrame đầu vào đúng cấu trúc Logistic Regression."""
    now = datetime.now()
    input_df = pd.DataFrame([{
        "hour": now.hour,
        "vehicle_density": vehicle_density,
        "avg_speed": avg_speed,
        "weather": weather,
        "temperature": temperature,
        "day_of_week": now.strftime("%a")  # Mon, Tue, ...
    }])
    state = int(model.predict(input_df)[0])
    return state


label_map = {0: "Thong thoang", 1: "Dang dong tuc", 2: "Co kha nang un tac"}
color_map = {"Thong thoang": (0, 255, 0), "Dang dong tuc": (0, 255, 255), "Co kha nang un tac": (0, 0, 255)}

# =====================================================
# 4. XỬ LÝ VIDEO
# =====================================================
cap = cv2.VideoCapture(video_path)
frame_count = 0
timestamps, vehicle_counts, densities, speeds, states = [], [], [], [], []

progress = st.progress(0)
frame_display = st.empty()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame_count += 1

    # Phát hiện phương tiện
    results = yolo_model(frame, verbose=False)
    boxes = results[0].boxes
    vehicle_count = len(boxes)
    vehicle_density = 0
    for box in boxes:
        cls_id = int(box.cls[0])  # ví dụ 2
        cls_name = yolo_model.names[cls_id]  # ví dụ 'car'
        if cls_name == "Motorcycle":
            vehicle_density += 2.5
        if cls_name == "Car":
            vehicle_density += 5
        if cls_name == "Truck":
            vehicle_density += 8
        if cls_name == "Bus":
            vehicle_density += 10
        else:
            vehicle_density += 1

    # Giả lập mật độ (density) và vận tốc
    # vehicle_density = min(vehicle_count * 2.8, 100)  # mỗi xe tương đương ~2.5 điểm mật độ
    avg_speed = np.random.randint(25, 35)

    # Dự đoán Logistic
    state = predict_state_from_features(
        logistic_model,
        vehicle_density=vehicle_density,
        avg_speed=avg_speed,
        weather=selected_weather,
        temperature=selected_temp
    )

    label = label_map[state]
    color = color_map[label]

    # Hiển thị kết quả trên khung hình
    nowTime = datetime.now().strftime("%H:%M:%S");
    annotated = results[0].plot()
    cv2.putText(annotated, f"STATE: {label}", (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
    cv2.putText(annotated,
        f"Time: {nowTime}",(20, 100),
        cv2.FONT_HERSHEY_SIMPLEX,0.8,(255, 255, 255),2
    )
    cv2.putText(annotated,
                f"Vehicles: {vehicle_count}", (20, 140),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2
                )
    cv2.putText(annotated,
                f"Avg_speed: {avg_speed} Km/h", (20, 180),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2
                )
    frame_display.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), channels="RGB")

    # Ghi log
    timestamps.append(datetime.now().strftime("%H:%M:%S"))
    vehicle_counts.append(vehicle_count)
    densities.append(vehicle_density)
    speeds.append(avg_speed)
    states.append(label)

    progress.progress(min(frame_count / 200, 1.0))
    time.sleep(0.05)

cap.release()

# =====================================================
# 5. TẠO DATAFRAME
# =====================================================
df = pd.DataFrame({
    "Thời gian": timestamps,
    "Số lượng xe": vehicle_counts,
    "Mật độ (%)": densities,
    "Vận tốc (km/h)": speeds,
    "Trạng thái": states
})

# =====================================================
# 6. BIỂU ĐỒ & THỐNG KÊ
# =====================================================
st.subheader(" PHÂN TÍCH VÀ THỐNG KÊ GIAO THÔNG")

col1, col2 = st.columns(2)

# --- Biểu đồ 1: Mật độ theo thời gian ---
fig1 = px.line(df, x="Thời gian", y="Mật độ (%)", color="Trạng thái",
               title="Mật độ phương tiện theo thời gian",
               markers=True,
               color_discrete_map={"THÔNG": "green", "ĐÔNG": "orange", "TẮC": "red"})
col1.plotly_chart(fig1, use_container_width=True)

# --- Biểu đồ 2: Tỷ lệ trạng thái ---
state_counts = df["Trạng thái"].value_counts()
fig2 = px.pie(values=state_counts.values, names=state_counts.index,
              title="Tỷ lệ thời gian các trạng thái giao thông",
              color=state_counts.index,
              color_discrete_map={"THÔNG": "green", "ĐÔNG": "orange", "TẮC": "red"})
col2.plotly_chart(fig2, use_container_width=True)

# --- Biểu đồ 3: Mối quan hệ mật độ - vận tốc ---
st.subheader(" Mối quan hệ giữa mật độ và vận tốc trung bình")
fig3 = go.Figure()
fig3.add_trace(go.Scatter(x=df["Thời gian"], y=df["Mật độ (%)"],
                          name="Mật độ (%)", mode="lines+markers", line=dict(color='blue')))
fig3.add_trace(go.Scatter(x=df["Thời gian"], y=df["Vận tốc (km/h)"],
                          name="Vận tốc (km/h)", mode="lines", yaxis="y2", line=dict(color='orange')))
fig3.update_layout(
    xaxis=dict(title="Thời gian"),
    yaxis=dict(title="Mật độ (%)"),
    yaxis2=dict(title="Vận tốc (km/h)", overlaying="y", side="right"),
    legend=dict(x=0.01, y=1.1, orientation="h")
)
st.plotly_chart(fig3, use_container_width=True)

# =====================================================
# 7. THỐNG KÊ & CẢNH BÁO
# =====================================================
st.subheader("Thống kê tổng hợp")

colA, colB, colC, colD = st.columns(4)
colA.metric("Tổng khung hình xử lý", len(df))
colB.metric("Xe trung bình/frame", round(np.mean(df["Số lượng xe"]), 2))
colC.metric("Mật độ TB (%)", round(np.mean(df["Mật độ (%)"]), 1))
colD.metric("Trạng thái cuối", df["Trạng thái"].iloc[-1] if not df.empty else "N/A")

# Chỉ số ùn tắc tổng hợp
df["Congestion Index"] = df["Mật độ (%)"] * (100 / (df["Vận tốc (km/h)"] + 1))
avg_CI = np.mean(df["Congestion Index"])
st.info(f"**Chỉ số ùn tắc trung bình (CI):** {avg_CI:.2f} — "
        + ("Giao thông thông thoáng" if avg_CI < 100 else
           "Giao thông đông đúc" if avg_CI < 200 else " Có khả năng ùn tắc nghiêm trọng!"))

# =====================================================
# 8. LƯU DỮ LIỆU
# =====================================================
st.download_button(
    label=" Tải xuống dữ liệu thống kê (CSV)",
    data=df.to_csv(index=False).encode("utf-8"),
    file_name=f"traffic_analysis_{datetime.now().strftime('%H%M%S')}.csv",
    mime="text/csv"
)

st.success(" Phân tích hoàn tất! Bạn có thể xem biểu đồ hoặc tải kết quả về.")
