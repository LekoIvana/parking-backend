from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from ultralytics import YOLO
import shutil, os
import cv2
import numpy as np
import base64
import time, uuid


from supabase import create_client, Client


from firebase_admin import credentials, firestore
import firebase_admin

SUPABASE_URL = "https://unbfufmeavxsqhtoeawg.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InVuYmZ1Zm1lYXZ4c3FodG9lYXdnIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTE3MTQwNzcsImV4cCI6MjA2NzI5MDA3N30.JbUgw_EiQD3sXhWszAsqQHjQKKo_dRMN6MvK11RWC3M"
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
BUCKET_NAME = "predictions"

# FIREBASE inicijalizacija
cred = credentials.Certificate('key/serviceAccountKey.json')
if not firebase_admin._apps:
    firebase_admin.initialize_app(cred)

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

def generate_unique_name(prefix, filename):
    ext = filename.split(".")[-1]
    unique = f"{prefix}_{int(time.time())}_{uuid.uuid4().hex[:8]}.{ext}"
    return unique

def upload_to_supabase(file_path, dest_name):
    with open(file_path, "rb") as f:
        supabase.storage.from_(BUCKET_NAME).upload(dest_name, f)
    url = supabase.storage.from_(BUCKET_NAME).get_public_url(dest_name)
    return url

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    user_id: str = Form(...)    
):
    
    temp_file = f"temp_{file.filename}"
    with open(temp_file, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    
    results = model(temp_file)
    labels = results[0].boxes.cls.cpu().numpy()
    boxes = results[0].boxes.xyxy.cpu().numpy()

    n_free = int((labels == 0).sum())
    total_spots = len(labels)
    n_occupied = total_spots - n_free

    
    img = cv2.imread(temp_file)
    for i, label in enumerate(labels):
        if label == 0:
            x1, y1, x2, y2 = map(int, boxes[i])
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)  

    annotated_file = f"annotated_{file.filename}"
    cv2.imwrite(annotated_file, img)

 
    with open(annotated_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')


    orig_dest = generate_unique_name("orig", file.filename)
    annot_dest = generate_unique_name("annotated", file.filename)
    orig_url = upload_to_supabase(temp_file, orig_dest)
    annotated_url = upload_to_supabase(annotated_file, annot_dest)

    #spremamo u firestore
    db = firestore.client()
    doc_data = {
        "user_id": user_id,                      
        "image_url": orig_url,
        "annotated_image_url": annotated_url,
        "n_free": n_free,
        "n_occupied": n_occupied,
        "total_spots": total_spots,
        "created_at": firestore.SERVER_TIMESTAMP,
    }
    db.collection("predictions").add(doc_data)

  
    os.remove(temp_file)
    os.remove(annotated_file)

    
    return JSONResponse(content={
        "n_free": n_free,
        "n_occupied": n_occupied,
        "total_spots": total_spots,
        "image_base64": encoded_string,          
        "image_url": orig_url,                   
        "annotated_image_url": annotated_url     
    })


@app.get("/get_last_global")
async def get_last_prediction_global():
    db = firestore.client()
    docs = (
        db.collection("predictions")
        .order_by("created_at", direction=firestore.Query.DESCENDING)
        .limit(1)
        .stream()
    )
    last_doc = next(docs, None)
    if last_doc:
        data = last_doc.to_dict()
        created_at = data.get("created_at")
        if hasattr(created_at, "isoformat"):
            data["created_at"] = created_at.isoformat()
    
        elif isinstance(created_at, str):
            try:
                import dateutil.parser
                dt = dateutil.parser.parse(created_at)
                data["created_at"] = dt.isoformat()
            except Exception:
                pass  
        return data
    else:
        raise HTTPException(status_code=404, detail="Nema podataka")

