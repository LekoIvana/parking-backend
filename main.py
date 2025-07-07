from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from ultralytics import YOLO
import shutil, os
import cv2
import numpy as np
import base64

app = FastAPI()

origins = [
    "http://localhost:5173",
    "http://localhost:8080",
    "http://127.0.0.1:5173",
    "http://127.0.0.1:8080",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = YOLO("models/best_train_parking_aug_finetune_v2.pt")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    temp_file = f"temp_{file.filename}"
    with open(temp_file, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    results = model(temp_file)
    labels = results[0].boxes.cls.cpu().numpy()
    boxes = results[0].boxes.xyxy.cpu().numpy()

    n_free = int((labels == 0).sum())
    
    total_spots = len(labels)
    
    n_occupied = total_spots - n_free

    # Otvori sliku
    img = cv2.imread(temp_file)

    # Crtaj boxeve SAMO za slobodna mjesta (klasa 0)
    for i, label in enumerate(labels):
        if label == 0:
            x1, y1, x2, y2 = map(int, boxes[i])
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)  # ZELENO za slobodno

    annotated_file = f"annotated_{file.filename}"
    cv2.imwrite(annotated_file, img)

    with open(annotated_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')

    # Očisti privremene fileove
    os.remove(temp_file)
    os.remove(annotated_file)

    return JSONResponse(content={
        "n_free": n_free,                
        "n_occupied": n_occupied,        
        "total_spots": total_spots,      
        "image_base64": encoded_string   
    })
