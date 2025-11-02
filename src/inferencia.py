import os
from utils import show_detections, infer_folder, cargar_modelo


def cargar_modelo(model_path):
    from ultralytics import YOLO
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No se encontró el archivo de pesos en: {model_path}")
    model = YOLO(model_path)
    print(f" Modelo cargado correctamente desde: {model_path}")
    return model


if __name__ == "__main__":
    model_path = "models/best.pt"
    image_path = "dataset/test/test3.png"
    folder_path = "dataset/test"

    model = cargar_modelo(model_path)

    # Inferencia en una sola imagen
    show_detections(image_path=image_path, model_path=model_path, conf=0.1, save=True, show=True)
