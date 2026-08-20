from fastapi import FastAPI, UploadFile, File, Form
from ultralytics import YOLO
import cv2
import numpy as np
import os
import uvicorn
from supabase import create_client, Client

# ==========================================
# 1. ตั้งค่า Supabase (เอา URL และ anon public key ของคุณมาใส่ตรงนี้)
# ==========================================
SUPABASE_URL = "https://zkfqqywpluqgvymmvqqr.supabase.co"
SUPABASE_KEY = "sb_publishable_hYmRXX4MvU0B0b9dF77KKQ_BTICQnjO"
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

app = FastAPI(title="Car Damage Assessment API")

# โหลดโมเดล YOLO
model = YOLO("best.pt")
CONF_THRESHOLD = 0.60 

os.makedirs("needs_review", exist_ok=True)

@app.get("/")
def read_root():
    return {"message": "Hello from AI-API! ระบบพร้อมทำงานครับ"}

@app.post("/predict")
async def predict_damage(
    file: UploadFile = File(...), 
    car_location: str = Form(...) # <-- เพิ่มการรับค่าตำแหน่งรถจาก Flutter
):
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        results = model(img)[0]
        
        parts_detected = []
        needs_human_review = False
        total_cost = 0

        for box in results.boxes:
            conf = float(box.conf[0])
            class_id = int(box.cls[0])
            damage_type = model.names[class_id].replace(" ", "_") # เช่น AI ตรวจเจอ "dent"

            if conf < CONF_THRESHOLD:
                needs_human_review = True

            # ==========================================
            # ผสมคำ: ตำแหน่งรถ (จากผู้ใช้) + รอยที่เจอ (จาก AI)
            # เช่น "door" + "_" + "dent" = "door_dent"
            # ==========================================
            search_query = f"{car_location}_{damage_type}" 
            
            subtotal = 0
            try:
                # ค้นหาราคาจากตารางโดยใช้คำที่ผสมแล้ว (door_dent)
                response = supabase.table("parts_pricing").select("price").eq("part_name", search_query).execute()
                
                if len(response.data) > 0:
                    subtotal = response.data[0]["price"]
                else:
                    print(f"⚠️ ยังไม่มีราคาของ: '{search_query}' ในฐานข้อมูล")
            except Exception as e:
                print(f"❌ Database Error: {e}")

            total_cost += subtotal

            parts_detected.append({
                "part": damage_type,
                "confidence": round(conf, 2),
                "subtotal": subtotal
            })

        if needs_human_review:
            save_path = f"needs_review/{file.filename}"
            cv2.imwrite(save_path, img)

        return {
            "status": "success",
            "filename": file.filename,
            "needs_human_review": needs_human_review,
            "total_cost": total_cost,
            "details": parts_detected
        }
        
    except Exception as e:
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)