# main.py
import cv2
import os
import uuid
import json
import base64
from models import load_yolo_model, load_resnet_model
from utils import preprocess_for_resnet, get_class_label, enhance_crop
import torch
from config import (
    CLASS_NAMES, NUM_CLASSES,
    YOLO_MODEL_PATH, RESNET_MODEL_PATH,
    CONF_THRESHOLD, YOLO_IMG_SIZE, MIN_CROP_SIZE,
    OUTPUT_DIR, CROPS_DIR,
    BOX_COLOR, BOX_THICKNESS, FONT_SCALE, FONT_THICKNESS,
    TEXT_OFFSET_Y, MIN_TEXT_Y, INPUT_IMAGE
)


def process_image(image_path, yolo_model, resnet_model):
    """
    Xử lý một ảnh: detect bằng YOLO -> crop -> classify bằng ResNet.

    Returns:
        dict: API response format:
        {
            "request_id": str,
            "detections": [
                {
                    "crop_image": str (đường dẫn ảnh crop),
                    "bounding_box": {"x1": int, "y1": int, "x2": int, "y2": int},
                    "label1": str (label YOLO),
                    "label2": str (label ResNet),
                    "confident1": float (confident YOLO),
                    "confident2": float (confident ResNet)
                },
                ...
            ]
        }
    """
    # Tạo request_id duy nhất
    request_id = str(uuid.uuid4())

    response = {
        "request_id": request_id,
        "detections": []
    }

    try:
        image = cv2.imread(image_path)
        if image is None:
            print("Không đọc được ảnh!")
            return response

        # Detection bằng YOLO
        results = yolo_model(image, imgsz=YOLO_IMG_SIZE)

        # Tạo thư mục lưu crops
        os.makedirs(CROPS_DIR, exist_ok=True)

        detection_count = 0

        for result in results:
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                print("Không phát hiện đối tượng nào")
                continue

            print(f"YOLO phát hiện {len(boxes)} đối tượng")

            # Lấy tên classes của YOLO model
            yolo_class_names = result.names  # dict {id: name}

            for idx, box in enumerate(boxes):
                try:
                    # === YOLO info ===
                    yolo_conf = box.conf.item()
                    if yolo_conf < CONF_THRESHOLD:
                        continue

                    yolo_cls_id = int(box.cls.item())
                    yolo_label = yolo_class_names.get(yolo_cls_id, "unknown")

                    detection_count += 1
                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    # Kiểm tra tọa độ hợp lệ
                    if x1 >= x2 or y1 >= y2 or x1 < 0 or y1 < 0 or x2 > image.shape[1] or y2 > image.shape[0]:
                        continue

                    # Crop vùng phát hiện
                    crop = image[y1:y2, x1:x2]
                    if crop.size == 0 or min(crop.shape[:2]) < MIN_CROP_SIZE:
                        continue

                    # Enhance crop quality before classification
                    crop = enhance_crop(crop)

                    # === ResNet info ===
                    input_tensor = preprocess_for_resnet(crop)
                    with torch.no_grad():
                        output = resnet_model(input_tensor)
                    resnet_label, resnet_conf = get_class_label(output, CLASS_NAMES)

                    # === Vẽ label lên ảnh crop ===
                    crop_labeled = crop.copy()
                    label_text = f"YOLO: {yolo_label} | ResNet: {resnet_label}"
                    (tw, th), baseline = cv2.getTextSize(
                        label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    cv2.rectangle(crop_labeled, (0, 0), (tw + 10, th + baseline + 10),
                                  (0, 0, 0), cv2.FILLED)
                    cv2.putText(crop_labeled, label_text, (5, th + 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                    # Lưu ảnh crop
                    crop_filename = f"{resnet_label}_{detection_count}.jpg"
                    crop_path = os.path.join(CROPS_DIR, crop_filename)
                    cv2.imwrite(crop_path, crop_labeled)

                    # === Thêm vào response ===
                    detection_info = {
                        "crop_image": crop_path,
                        "bounding_box": {
                            "x1": x1, "y1": y1,
                            "x2": x2, "y2": y2
                        },
                        "label1": yolo_label,
                        "label2": resnet_label,
                        "confident1": round(yolo_conf, 4),
                        "confident2": round(resnet_conf, 4)
                    }
                    response["detections"].append(detection_info)

                    print(f"  Object {detection_count}: "
                          f"YOLO={yolo_label}({yolo_conf:.2f}) | "
                          f"ResNet={resnet_label}({resnet_conf:.2f})")

                except Exception as e:
                    continue

        print(f"\nTổng kết: {len(response['detections'])} đối tượng phân loại thành công")
        return response

    except Exception as e:
        print(f"Lỗi: {str(e)}")
        return response


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    try:
        yolo_model = load_yolo_model(YOLO_MODEL_PATH)
        resnet_model = load_resnet_model(RESNET_MODEL_PATH, NUM_CLASSES)

        response = process_image(INPUT_IMAGE, yolo_model, resnet_model)

        # Lưu response JSON
        json_path = os.path.join(OUTPUT_DIR, "response.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(response, f, ensure_ascii=False, indent=2)

        # In kết quả API
        print(f"\n{'='*50}")
        print(f"API Response:")
        print(f"{'='*50}")
        print(json.dumps(response, ensure_ascii=False, indent=2))
        print(f"\nResponse JSON lưu tại: {json_path}")
        print(f"Ảnh crop lưu tại: {CROPS_DIR}")

    except Exception as e:
        print(f"Lỗi: {str(e)}")


if __name__ == "__main__":
    main()