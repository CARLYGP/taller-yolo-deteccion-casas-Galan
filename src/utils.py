import os
import cv2
import yaml
import matplotlib.pyplot as plt
from ultralytics import YOLO


def cargar_modelo(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No se encontró el archivo de pesos en: {model_path}")
    model = YOLO(model_path)
    print(f" Modelo cargado correctamente desde: {model_path}")
    return model


def show_detections(image_path, model_path, conf=0.25, save=True, show=True):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"No se encontró la imagen en: {image_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No se encontró el modelo en: {model_path}")
    model = YOLO(model_path)
    results = model(image_path, conf=conf, save=save)
    if show:
        annotated = results[0].plot()
        plt.figure(figsize=(10, 8))
        plt.imshow(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
        plt.axis("off")
        plt.title(f"Detecciones {os.path.basename(image_path)}")
        plt.show()
    return results


def summarize_dataset(data_yaml_path):
    if not os.path.exists(data_yaml_path):
        raise FileNotFoundError(f"No se encontró el archivo YAML en: {data_yaml_path}")
    with open(data_yaml_path, 'r') as f:
        data_cfg = yaml.safe_load(f)
    for split in ['train', 'val']:
        img_dir = data_cfg.get(split, '')
        if not os.path.exists(img_dir):
            print(f"No se encontró la carpeta: {img_dir}")
            continue
        n_imgs = len([f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png'))])
        print(f"{split.upper()}: {n_imgs} imágenes")
    print("Clases:", data_cfg.get('names', []))


def infer_folder(model_path, folder_path, conf=0.25, save=True):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No se encontró el modelo en: {model_path}")
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"No se encontró la carpeta: {folder_path}")
    model = YOLO(model_path)
    results = model.predict(source=folder_path, conf=conf, save=save)
    print(f"Inferencias guardadas en: {model.predictor.save_dir}")
    return results


def yolo_to_json(yolo_txt_path, img_width, img_height, class_names=None):
    if not os.path.exists(yolo_txt_path):
        raise FileNotFoundError(f"No se encontró el archivo YOLO: {yolo_txt_path}")
    data = []
    with open(yolo_txt_path, "r") as f:
        for line in f:
            cls, x, y, w, h = map(float, line.strip().split())
            cls = int(cls)
            bbox = {
                "class_id": cls,
                "class_name": class_names[cls] if class_names else str(cls),
                "x_center": x * img_width,
                "y_center": y * img_height,
                "width": w * img_width,
                "height": h * img_height
            }
            data.append(bbox)
    return data
