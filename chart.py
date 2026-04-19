import pandas as pd
import matplotlib.pyplot as plt

# Đọc file
df = pd.read_csv("runs/detect/yolo_fast_accurate/results.csv")

# Vẽ 2 đường
plt.figure(figsize=(10, 6))

plt.plot(df["epoch"], df["metrics/mAP50(B)"], label="mAP@50")
plt.plot(df["epoch"], df["metrics/mAP50-95(B)"], label="mAP@50-95")

# Label
plt.xlabel("Epoch")
plt.ylabel("mAP")
plt.title("Diễn biến mAP theo Epoch")

# Hiển thị legend
plt.legend()

# Grid cho đẹp
plt.grid(True)

# Show
plt.show()