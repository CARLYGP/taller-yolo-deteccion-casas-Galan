import os
import sys
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from utils import show_detections, infer_folder
from ultralytics import YOLO

def cargar_modelo(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No se encontró el archivo de pesos en: {model_path}")
    model = YOLO(model_path)
    print(f" Modelo cargado correctamente desde: {model_path}")
    return model


if __name__ == "__main__":
    # Si el usuario pasa una ruta de imagen por consola, se usa esa.
    image_path = sys.argv[1] if len(sys.argv) > 1 else "dataset/test/test3.png"

    model_path = "models/best.pt"
    model = cargar_modelo(model_path)

    show_detections(
        image_path=image_path,
        model_path=model_path,
        conf=0.16,
        save=True,
        show=True
    )

    print(f"\n Inferencia completada sobre: {image_path}")
    print(" Resultados guardados en 'runs/detect/'")
