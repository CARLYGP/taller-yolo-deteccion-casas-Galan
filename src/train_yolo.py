from ultralytics import YOLO
from utils import summarize_dataset

data_path = 'data.yaml'

# Mostrar resumen del dataset antes de entrenar
summarize_dataset(data_path)

# Cargar modelo base
model = YOLO('yolo11l.pt')

# Entrenamiento
results = model.train(
    data=data_path,
    epochs=80,
    imgsz=640,
    batch=8,
    patience=25
)
