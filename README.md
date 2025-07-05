# SpotCount Backend (FastAPI)

Ovo je backend servis za projekt prepoznavanja slobodnih parkirnih mjesta korištenjem YOLOv8 i FastAPI.

## Pokretanje

1. Instaliraj ovisnosti:
pip install -r requirements.txt


2. **YOLOv8 model (.pt) nije dio repozitorija**  
Potrebno ga je ručno dodati u direktorij `models/` pod imenom `best_train_parking_aug_finetune_v2.pt`.

3. Pokreni server:
uvicorn main:app --reload

## Napomena

Frontend aplikacija nalazi se u drugom repozitoriju.