from fastapi import FastAPI, UploadFile, File, Form, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from ultralytics import YOLO
import cv2
import numpy as np
import os
import uvicorn
import base64
import json
import bcrypt
from supabase import create_client, Client
from dotenv import load_dotenv # 1. นำเข้า load_dotenv

load_dotenv()

# ==========================================
# 1. ตั้งค่า Supabase (ดึงจาก .env)
# ==========================================
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("Missing Supabase URL or Key in .env file")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

app = FastAPI(title="Car Damage Assessment API")

# อนุญาตให้ Frontend ข้ามโดเมนมาเรียก API ได้ (ป้องกัน Error CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# โหลดโมเดล YOLO
model = YOLO("best.pt")
CONF_THRESHOLD = 0.60 

os.makedirs("needs_review", exist_ok=True)

@app.get("/")
def read_root():
    return {"message": "Hello from AI-API! ระบบพร้อมทำงานครับ"}

# ==========================================
# 2. API: วิเคราะห์ความเสียหายและบันทึกประวัติ
# ==========================================
@app.post("/predict")
async def predict_damage(
    file: UploadFile = File(...), 
    car_locations: str = Form(...),
    vehicle_id: str = Form(...), 
    user_id: str = Form(...)     
):
    try:
        # 2.1 แปลงข้อมูลชิ้นส่วนที่ส่งมาจากหน้าเว็บให้เป็น List
        try:
            locations = json.loads(car_locations)
            if not isinstance(locations, list):
                locations = [car_locations]
        except Exception:
            locations = [car_locations]

        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # วิเคราะห์รูปด้วยโมเดล YOLO
        results = model(img)[0]
        
        parts_detected = []
        needs_human_review = False
        total_cost = 0

        for box in results.boxes:
            conf = float(box.conf[0])
            class_id = int(box.cls[0])
            
            # ดึงชื่อความเสียหายจากโมเดล
            damage_type_original = model.names[class_id].strip().lower().replace(" ", "_")
            
            # ปรับมาตรฐานคำศัพท์ให้ตรงกับตาราง (Normalization)
            if damage_type_original in ["lamp_broken", "light_broken", "head_light_broken", "tail_light_broken", "broken_lamp"]:
                normalized_damage = "broken"
            else:
                normalized_damage = damage_type_original

            if conf < CONF_THRESHOLD:
                needs_human_review = True

            subtotal = 0
            matched_query = ""

            # 2.2 Smart Matching: จับคู่ชิ้นส่วนกับความเสียหาย
            for loc in locations:
                if loc == "windshield_glass" and normalized_damage in ["glass_shatter", "shatter"]:
                    query = "windshield_glass_shatter"
                elif loc in ["headlight", "taillight"] and normalized_damage == "broken":
                    query = f"{loc}_broken"
                else:
                    query = f"{loc}_{normalized_damage}"

                try:
                    response = supabase.table("parts_pricing").select("price").eq("part_name", query).execute()
                    if response.data and len(response.data) > 0:
                        subtotal = float(response.data[0]["price"])
                        matched_query = query
                        break 
                except Exception as e:
                    print(f"Database Query Error: {e}")

            # Fallback: ถ้าหาแบบผสมคำไม่เจอ ให้ลองดึงราคาจากชื่อรอยเพียวๆ
            if subtotal == 0:
                try:
                    fallback_res = supabase.table("parts_pricing").select("price").eq("part_name", normalized_damage).execute()
                    if fallback_res.data and len(fallback_res.data) > 0:
                        subtotal = float(fallback_res.data[0]["price"])
                        matched_query = normalized_damage
                except:
                    pass

            total_cost += subtotal
            parts_detected.append({
                "label": damage_type_original,
                "confidence": round(conf, 2),
                "cost": subtotal,
                "matched_part": matched_query
            })

        # 2.3 วาดกรอบบนรูปภาพและเข้ารหัสเป็น Base64
        img_with_boxes = results.plot()
        _, buffer = cv2.imencode('.jpg', img_with_boxes)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        image_data_uri = f"data:image/jpeg;base64,{img_base64}"

        # 2.4 บันทึกข้อมูลลงตาราง inspections แบบแยกตาม user_id
        try:
            new_inspection = {
                "user_id": user_id,
                "vehicle_id": vehicle_id,
                "image_url": image_data_uri,
                "damage_labels": parts_detected,
                "technician_feedback": []
            }
            insert_response = supabase.table("inspections").insert(new_inspection).execute()
            inserted_id = insert_response.data[0]['id'] if insert_response.data else None
        except Exception as e:
            print(f"❌ Failed to save inspection: {e}")
            inserted_id = None

        if needs_human_review:
            save_path = f"needs_review/{file.filename}"
            cv2.imwrite(save_path, img_with_boxes) 

        return {
            "status": "success",
            "inspection_id": inserted_id,
            "filename": file.filename,
            "needs_human_review": needs_human_review,
            "total_cost": total_cost,
            "damage_labels": parts_detected,   
            "image_url": image_data_uri      
        }
        
    except Exception as e:
        return {"status": "error", "message": str(e)}


# ==========================================
# 3. API: ดึงประวัติเฉพาะของผู้ใช้งานคนนั้นๆ
# ==========================================
@app.get("/api/v1/history/{user_id}")
def get_user_history(user_id: str):
    try:
        response = supabase.table("inspections").select("*").eq("user_id", user_id).order("created_at", desc=True).execute()
        return {"status": "success", "data": response.data}
    except Exception as e:
        return {"status": "error", "message": str(e)}
    
# ==========================================
# 4. API: ดึงผลลัพธ์การประเมิน 1 รายการ (สำหรับหน้า Result)
# ==========================================
@app.get("/api/v1/inspection/{inspection_id}")
def get_single_inspection(inspection_id: int):
    try:
        # ค้นหาข้อมูลจาก id ที่ส่งมา
        response = supabase.table("inspections").select("*").eq("id", inspection_id).execute()
        
        if response.data and len(response.data) > 0:
            return {"status": "success", "data": response.data[0]}
        else:
            return {"status": "error", "message": "ไม่พบข้อมูลการประเมินนี้"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/api/v1/feedback")
async def submit_feedback(
    case_id: str = Form(...),
    damage_type: str = Form(...),
    corrected_price: str = Form(...),
    notes: str = Form("")
):
    try:
        # 1. จัดการแปลง case_id (เช่น "CASE-14" -> 14)
        clean_case_id = case_id.replace("CASE-", "").strip()
        if not clean_case_id.isdigit():
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={"status": "error", "message": "รูปแบบ Case ID ไม่ถูกต้อง"}
            )
        inspection_id = int(clean_case_id)

        # 2. แปลงราคาเป็น float
        try:
            price_float = float(corrected_price)
        except ValueError:
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={"status": "error", "message": "ราคาประเมินต้องเป็นตัวเลขเท่านั้น"}
            )

        # 3. จัดเตรียมข้อมูลให้ตรงกับ Column ใน Supabase
        feedback_data = {
            "case_id": inspection_id,
            "damage_label": damage_type,
            "repair_cost": price_float,
            "note": notes
        }
        
        # 4. บันทึกลง Supabase
        supabase.table("technician_feedback").insert(feedback_data).execute()
        
        return JSONResponse(
            status_code=status.HTTP_201_CREATED,
            content={"status": "success", "message": "บันทึก Feedback เรียบร้อยครับ"}
        )
        
    except Exception as e:
        print(f"❌ Feedback Error: {e}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"status": "error", "message": f"เกิดข้อผิดพลาดภายในเซิร์ฟเวอร์: {str(e)}"}
        )

@app.post("/api/v1/register")
async def register_user(
    full_name: str = Form(...),
    email: str = Form(...),
    password: str = Form(...)
):
    email = email.strip().lower()

    # 1. เช็คความยาวรหัสผ่าน (ต้อง 8 ตัวขึ้นไป)
    if len(password) < 8:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"status": "error", "message": "รหัสผ่านต้องมีอย่างน้อย 8 ตัวอักษร"}
        )

    try:
        # 2. เช็คว่ามีอีเมลนี้ในระบบหรือยัง
        res = supabase.table("users").select("id").eq("email", email).execute()
        if res.data and len(res.data) > 0:
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={"status": "error", "message": "อีเมลนี้ถูกใช้งานแล้ว"}
            )

        # 3. เข้ารหัสผ่านให้ปลอดภัยก่อนลง Database
        hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

        # 4. บันทึกข้อมูลลงตาราง users
        new_user = {
            "full_name": full_name,
            "email": email,
            "password_hash": hashed_password,
            "role": "user"
        }
        supabase.table("users").insert(new_user).execute()
        
        return {"status": "success", "message": "สมัครสมาชิกสำเร็จ"}

    except Exception as e:
        print(f"Register Error: {e}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"status": "error", "message": "ไม่สามารถสมัครสมาชิกได้ในขณะนี้"}
        )

@app.post("/api/v1/login")
async def login_user(
    email: str = Form(...),
    password: str = Form(...)
):
    email = email.strip().lower()

    try:
        # 1. ค้นหาผู้ใช้จากอีเมลใน Supabase
        res = supabase.table("users").select("*").eq("email", email).execute()
        
        if not res.data or len(res.data) == 0:
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={"status": "error", "message": "อีเมลหรือรหัสผ่านไม่ถูกต้อง"}
            )

        user = res.data[0]
        stored_password = user.get("password_hash", "") or user.get("password", "")

        # 2. เช็ครหัสผ่าน (รองรับทั้งแบบข้อความธรรมดา หรือเช็คผ่าน bcrypt)
        is_matched = False
        try:
            if stored_password.startswith("$2b$") or stored_password.startswith("$2a$"):
                is_matched = bcrypt.checkpw(password.encode('utf-8'), stored_password.encode('utf-8'))
            else:
                is_matched = (password == stored_password)
        except Exception:
            is_matched = (password == stored_password)

        if not is_matched:
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={"status": "error", "message": "อีเมลหรือรหัสผ่านไม่ถูกต้อง"}
            )

        # 3. ล็อกอินสำเร็จ ส่งข้อมูลกลับไปให้ Flutter
        return {
            "status": "success",
            "user_id": str(user["id"]),
            "user_name": user["full_name"]
        }

    except Exception as e:
        print(f"❌ Login Error Details: {e}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"status": "error", "message": f"Server Error: {str(e)}"}
        )

@app.post("/api/v1/update_profile")
async def update_profile(
    user_id: str = Form(...),
    full_name: str = Form(...)
):
    try:
        supabase.table("users").update({"full_name": full_name}).eq("id", user_id).execute()
        return {"status": "success", "message": "บันทึกข้อมูลส่วนตัวเรียบร้อยแล้ว"}
    except Exception as e:
        print(f"Update Profile Error: {e}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"status": "error", "message": "ไม่สามารถอัปเดตข้อมูลได้"}
        )

@app.post("/api/v1/update_password")
async def update_password(
    user_id: str = Form(...),
    current_password: str = Form(...),
    new_password: str = Form(...)
):
    try:
        res = supabase.table("users").select("password_hash").eq("id", user_id).execute()
        if not res.data:
            return JSONResponse(status_code=status.HTTP_404_NOT_FOUND, content={"status": "error", "message": "ไม่พบผู้ใช้"})

        stored_hash = res.data[0].get("password_hash", "")
        
        # เช็ครหัสผ่านเดิม
        is_matched = False
        try:
            if stored_hash.startswith("$2b$") or stored_hash.startswith("$2a$"):
                is_matched = bcrypt.checkpw(current_password.encode('utf-8'), stored_hash.encode('utf-8'))
            else:
                is_matched = (current_password == stored_hash)
        except:
            is_matched = (current_password == stored_hash)

        if not is_matched:
            return JSONResponse(status_code=status.HTTP_401_UNAUTHORIZED, content={"status": "error", "message": "รหัสผ่านเดิมไม่ถูกต้อง"})

        if len(new_password) < 8:
            return JSONResponse(status_code=status.HTTP_400_BAD_REQUEST, content={"status": "error", "message": "รหัสผ่านใหม่ต้องมี 8 ตัวขึ้นไป"})

        # อัปเดตรหัสผ่านใหม่
        new_hash = bcrypt.hashpw(new_password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
        supabase.table("users").update({"password_hash": new_hash}).eq("id", user_id).execute()

        return {"status": "success", "message": "เปลี่ยนรหัสผ่านสำเร็จ"}
    except Exception as e:
        print(f"Update Password Error: {e}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"status": "error", "message": "ระบบขัดข้อง ไม่สามารถเปลี่ยนรหัสผ่านได้"}
        )
        
@app.get("/api/v1/feedback_history")
def get_feedback_history():
    try:
        response = supabase.table("technician_feedback").select("*").order("submitted_at", desc=True).execute()
        return {"status": "success", "data": response.data}
    except Exception as e:
        print(f"Feedback History Error: {e}")
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)