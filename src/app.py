from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from ultralytics import YOLO
import uvicorn
import cv2
import numpy as np
import io
from PIL import Image

app = FastAPI(title="YOLO Detección de Casas API")

model = YOLO("models/best.pt")


@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = np.array(image)

    
    results = model.predict(source=img, conf=0.25, save=False)

    
    detections = []
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(float, box.xyxy[0].tolist())
            score = float(box.conf[0])
            cls_id = int(box.cls[0])
            cls_name = model.names[cls_id]
            detections.append({
                "class": cls_name,
                "score": round(score, 4),
                "bbox": [x1, y1, x2, y2]
            })

    return JSONResponse(content=detections)


@app.get("/")
async def root():
    return {"message": " API YOLO - Detección de Casas activa"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
