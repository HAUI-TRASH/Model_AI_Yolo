# TrashSmart

Package layout:

- `shared/`: shared ResNet architecture, ImageNet constants, and transforms.
- `training/`: ResNet image classification training/evaluation.
- `yolo_training/`: YOLO11 training and inference test scripts.
- `pipeline/`: YOLO + ResNet + optional Real-ESRGAN inference.

Run examples:

```powershell
python -m trashsmart.training.main
python -m trashsmart.yolo_training.train_yolo
python -m trashsmart.yolo_training.test_yolo --model models\best.pt --image sample.jpg --save
python -m trashsmart.pipeline.main
python -m trashsmart.pipeline.roi
python -m trashsmart.pipeline.image_roi --image sample.jpg
```

Configure paths through environment variables or a local `.env` loader if you use one. Keep `.env`, model weights, datasets, and outputs out of git.
pip install basicsr
pip install git+https://github.com/xinntao/Real-ESRGAN.git

