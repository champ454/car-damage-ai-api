from fastapi import FastAPI, UploadFile, File, Form, HTTPException, status, Depends
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
import uuid
from supabase import create_client, Client
from dotenv import load_dotenv
import logging
import imghdr

# ==========================================
# 0. ตั้งค่าระบบและ Logging
# ==========================================
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000,http://127.0.0.1:5500").split(",")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("Missing Supabase URL or Key in .env file")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

app = FastAPI(title="Car Damage Assessment API")

# 1. จำกัด CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = YOLO("best.pt")
CONF_THRESHOLD = 0.60 

os.makedirs("needs_review", exist_ok=True)

ALLOWED_VEHICLES = {"HONDA-CIVIC", "HONDA-CITY"}
ALLOWED_LOCATIONS = {
    "front_bumper", "rear_bumper", "door", "hood", "trunk",
    "fender", "headlight", "taillight", "windshield_glass"
}

# ==========================================
# Security Dependency
# ==========================================
def verify_admin_or_tech(user_id: str):
    try:
        res = supabase.table("users").select("role").eq("id", user_id).execute()
        if not res.data or res.data[0].get("role") not in ["admin", "technician"]:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN, 
                detail="ไม่มีสิทธิ์เข้าถึงข้อมูลส่วนนี้"
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Auth error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail="เกิดข้อผิดพลาดในการตรวจสอบสิทธิ์"
        )

@app.get("/")
def read_root():
    return {"message": "Hello from AI-API! ระบบพร้อมทำงานครับ"}

# ==========================================
# 2. API: วิเคราะห์ความเสียหายและคำนวณราคา
# ==========================================
@app.post("/predict")
async def predict_damage(
    file: UploadFile = File(...), 
    car_locations: str = Form(...),
    vehicle_id: str = Form(...), 
    user_id: str = Form(...),
    car_model: str = Form(...)
):
    contents = await file.read()
    detected_type = imghdr.what(None, h=contents)  # ตรวจสอบจาก Byte จริง ไม่ใช่ Header

    if detected_type not in ["jpeg", "png"]:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="ไฟล์ไม่ใช่รูปภาพที่ถูกต้องตามที่ระบบรองรับ")

    if vehicle_id not in ALLOWED_VEHICLES:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="รหัสรถยนต์ไม่ถูกต้องในระบบ")

    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="ไม่สามารถอ่านไฟล์รูปภาพได้")

    try:
        try:
            locations = json.loads(car_locations)
            if not isinstance(locations, list):
                locations = [car_locations]
        except Exception:
            locations = [car_locations]

        invalid_locs = [loc for loc in locations if loc not in ALLOWED_LOCATIONS]
        if invalid_locs:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"ชิ้นส่วนไม่ถูกต้อง: {invalid_locs}")

        # 🟢 REMOVED REDUNDANT READS:
        # contents = await file.read()
        # nparr = np.frombuffer(contents, np.uint8)
        # img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        # if img is None:
        #     raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="ไม่สามารถอ่านไฟล์รูปภาพได้")

        results = model(img)[0]
        
        parts_detected = []
        needs_human_review = False
        
        has_masks = results.masks is not None
        masks = results.masks if has_masks else None

        part_summary = {loc: {"accumulated_cost": 0.0, "force_replace": False, "damages_idx": []} for loc in locations}

        # 🟢 Optimization A: Batch Query (In-Memory Pre-fetching)
        try:
            pricing_res = supabase.table("parts_pricing").select("part_name, price").eq("car_model", car_model).execute()
            pricing_dict = {item["part_name"]: float(item["price"]) for item in pricing_res.data} if pricing_res.data else {}
        except Exception as e:
            logger.error(f"Failed to pre-fetch parts pricing: {e}")
            pricing_dict = {}

        for i, box in enumerate(results.boxes):
            conf = float(box.conf[0])
            class_id = int(box.cls[0])
            damage_type_original = model.names[class_id].strip().lower().replace(" ", "_")
            
            if damage_type_original in ["lamp_broken", "light_broken", "head_light_broken", "tail_light_broken", "broken_lamp"]:
                normalized_damage = "broken"
            else:
                normalized_damage = damage_type_original

            if conf < CONF_THRESHOLD:
                needs_human_review = True

            # 🟢 Optimization C: Polygon Coordinates
            damage_ratio = 0.0
            if has_masks and i < len(masks.xy) and len(masks.xy[i]) > 0:
                w = float(box.xywh[0][2])
                h = float(box.xywh[0][3])
                bbox_area = w * h

                mask_polygon = masks.xy[i]
                mask_area = cv2.contourArea(mask_polygon)

                if bbox_area > 0:
                    damage_ratio = float(mask_area / bbox_area)

            subtotal = 0.0
            matched_query = ""
            matched_loc = None

            # 🟢 In-Memory Dictionary
            for loc in locations:
                if loc == "windshield_glass" and normalized_damage in ["glass_shatter", "shatter"]:
                    query = "windshield_glass_shatter"
                elif loc in ["headlight", "taillight"] and normalized_damage == "broken":
                    query = f"{loc}_broken"
                else:
                    query = f"{loc}_{normalized_damage}"

                if query in pricing_dict:
                    subtotal = pricing_dict[query]
                    matched_query = query
                    matched_loc = loc
                    break 

            if subtotal == 0.0 and normalized_damage in pricing_dict:
                subtotal = pricing_dict[normalized_damage]
                matched_query = normalized_damage
                matched_loc = locations[0] if locations else None

            if subtotal == 0.0:
                needs_human_review = True

            damage_info = {
                "label": damage_type_original,
                "confidence": round(conf, 2),
                "cost": subtotal, 
                "matched_part": matched_query,
                "damage_percent": round(damage_ratio * 100, 2),
                "repair_action": "ซ่อมทำสี (Repair)"
            }
            parts_detected.append(damage_info)
            current_idx = len(parts_detected) - 1

            if matched_loc and matched_loc in part_summary:
                part_summary[matched_loc]["accumulated_cost"] += subtotal
                part_summary[matched_loc]["damages_idx"].append(current_idx)
                
                if damage_ratio > 0.30:
                    part_summary[matched_loc]["force_replace"] = True

        total_cost = 0.0
        for loc, summary in part_summary.items():
            if not summary["damages_idx"]:
                continue
            
            replace_price = pricing_dict.get(loc, 0.0)
            is_replace = summary["force_replace"]
            
            if replace_price > 0 and summary["accumulated_cost"] > replace_price:
                is_replace = True

            if is_replace:
                final_part_cost = replace_price if replace_price > 0 else summary["accumulated_cost"] * 1.5
                total_cost += final_part_cost
                
                for i, dmg_idx in enumerate(summary["damages_idx"]):
                    parts_detected[dmg_idx]["repair_action"] = "เปลี่ยนชิ้นส่วน (Replace)"
                    parts_detected[dmg_idx]["cost"] = final_part_cost if i == 0 else 0.0
            else:
                total_cost += summary["accumulated_cost"]

        # 🟢 Optimization B: Supabase Storage
        img_with_boxes = results.plot()
        _, buffer = cv2.imencode('.jpg', img_with_boxes)
        
        image_filename = f"inspection_{user_id}_{uuid.uuid4().hex}.jpg"
        image_url = ""
        
        try:
            supabase.storage.from_("car-images").upload(
                file=buffer.tobytes(),
                path=image_filename,
                file_options={"content-type": "image/jpeg"}
            )
            image_url = supabase.storage.from_("car-images").get_public_url(image_filename)
        except Exception as e:
            logger.error(f"Supabase Storage Upload Error: {e}")
            img_base64 = base64.b64encode(buffer).decode('utf-8')
            image_url = f"data:image/jpeg;base64,{img_base64}"

        try:
            new_inspection = {
                "user_id": user_id,
                "vehicle_id": vehicle_id,
                "image_url": image_url,  
                "damage_labels": parts_detected,
                "total_cost": total_cost, 
                "needs_human_review": needs_human_review,
                "technician_feedback": []
            }
            insert_response = supabase.table("inspections").insert(new_inspection).execute()
            inserted_id = insert_response.data[0]['id'] if insert_response.data else None
        except Exception as e:
            logger.error(f"Failed to save inspection: {e}")
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="ไม่สามารถบันทึกข้อมูลลงฐานข้อมูลได้")

        if needs_human_review:
            save_path = f"needs_review/{image_filename}"
            cv2.imwrite(save_path, img_with_boxes) 

        return {
            "status": "success",
            "inspection_id": inserted_id,
            "filename": file.filename,
            "needs_human_review": needs_human_review,
            "total_cost": total_cost,
            "damage_labels": parts_detected,   
            "image_url": image_url      
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="เกิดข้อผิดพลาดในการวิเคราะห์รูปภาพ")

# ==========================================
# 3. API: ดึงประวัติเฉพาะของผู้ใช้งาน
# ==========================================
@app.get("/api/v1/history/{user_id}")
def get_user_history(user_id: str):
    try:
        response = supabase.table("inspections").select("*").eq("user_id", user_id).order("created_at", desc=True).execute()
        return {"status": "success", "data": response.data}
    except Exception as e:
        logger.error(f"History fetch error: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="ไม่สามารถดึงข้อมูลประวัติได้")
    
# ==========================================
# 3.5 API: คิวงานที่รอการตรวจสอบจากช่าง (Review Queue)
# ==========================================
@app.get("/api/v1/review_queue")
def get_review_queue(user_id: str):
    verify_admin_or_tech(user_id)
    try:
        response = supabase.table("inspections") \
            .select("*") \
            .eq("needs_human_review", True) \
            .order("created_at", desc=True) \
            .execute()
        return {"status": "success", "data": response.data}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Review queue fetch error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail="ไม่สามารถดึงข้อมูลเคสที่ต้องตรวจสอบได้"
        )
    
# ==========================================
# 4. API: ดึงผลลัพธ์การประเมิน 1 รายการ
# ==========================================
@app.get("/api/v1/inspection/{inspection_id}")
def get_single_inspection(inspection_id: int):
    try:
        response = supabase.table("inspections").select("*").eq("id", inspection_id).execute()
        if response.data:
            return {"status": "success", "data": response.data[0]}
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="ไม่พบข้อมูลการประเมินนี้")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Fetch single inspection error: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="เกิดข้อผิดพลาดในการดึงข้อมูล")

# ==========================================
# 5. API: ส่ง Feedback จากช่าง
# ==========================================
@app.post("/api/v1/feedback")
async def submit_feedback(
    case_id: str = Form(...),
    damage_type: str = Form(...),
    corrected_price: str = Form(...),
    notes: str = Form(""),
    user_id: str = Form(...)
):
    verify_admin_or_tech(user_id)

    try:
        clean_case_id = case_id.replace("CASE-", "").strip()
        if not clean_case_id.isdigit():
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="รูปแบบ Case ID ไม่ถูกต้อง")
        
        inspection_id = int(clean_case_id)

        # ตรวจสอบว่ามี Inspection ID นี้ในระบบจริงหรือไม่
        try:
            inspection_res = supabase.table("inspections").select("id").eq("id", inspection_id).execute()
            if not inspection_res.data:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="ไม่พบเคสที่ต้องการส่ง Feedback"
                )
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Check inspection error: {e}")
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="เกิดข้อผิดพลาดในการตรวจสอบเคส")

        try:
            price_float = float(corrected_price)
        except ValueError:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="ราคาประเมินต้องเป็นตัวเลขเท่านั้น")

        # 🟢 1. บันทึก Feedback พร้อมระบุตัวตนช่าง (Audit Trail)
        feedback_data = {
            "case_id": inspection_id,
            "damage_label": damage_type,
            "repair_cost": price_float,
            "note": notes,
            "submitted_by": user_id  # เพิ่มคอลัมน์เก็บผู้ตรวจสอบ
        }
        supabase.table("technician_feedback").insert(feedback_data).execute()
        
        # 🟢 2. ปลดล็อกสถานะให้ออกจากคิวเคสที่ต้องตรวจสอบ (State Update)
        supabase.table("inspections").update({"needs_human_review": False}).eq("id", inspection_id).execute()
        
        return JSONResponse(
            status_code=status.HTTP_201_CREATED,
            content={"status": "success", "message": "บันทึก Feedback เรียบร้อยครับ"}
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Feedback Error: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="เกิดข้อผิดพลาดในการบันทึก Feedback")

# ==========================================
# 6. API: การยืนยันตัวตนและการจัดการบัญชี
# ==========================================
@app.post("/api/v1/register")
async def register_user(full_name: str = Form(...), email: str = Form(...), password: str = Form(...)):
    email = email.strip().lower()
    if len(password) < 8:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="รหัสผ่านต้องมีอย่างน้อย 8 ตัวอักษร")

    try:
        res = supabase.table("users").select("id").eq("email", email).execute()
        if res.data:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="อีเมลนี้ถูกใช้งานแล้ว")

        hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
        new_user = {"full_name": full_name, "email": email, "password_hash": hashed_password, "role": "user"}
        supabase.table("users").insert(new_user).execute()
        return {"status": "success", "message": "สมัครสมาชิกสำเร็จ"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Register Error: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="ไม่สามารถสมัครสมาชิกได้ในขณะนี้")

@app.post("/api/v1/login")
async def login_user(email: str = Form(...), password: str = Form(...)):
    email = email.strip().lower()
    try:
        res = supabase.table("users").select("*").eq("email", email).execute()
        if not res.data:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="อีเมลหรือรหัสผ่านไม่ถูกต้อง")

        user = res.data[0]
        stored_password = user.get("password_hash", "")
        if not stored_password or not (stored_password.startswith("$2b$") or stored_password.startswith("$2a$")):
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="อีเมลหรือรหัสผ่านไม่ถูกต้อง (Legacy Account)")

        if not bcrypt.checkpw(password.encode('utf-8'), stored_password.encode('utf-8')):
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="อีเมลหรือรหัสผ่านไม่ถูกต้อง")

        return {"status": "success", "user_id": str(user["id"]), "user_name": user["full_name"], "role": user.get("role", "user")}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Login Error Details: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="ระบบขัดข้อง ไม่สามารถเข้าสู่ระบบได้")

@app.post("/api/v1/update_profile")
async def update_profile(user_id: str = Form(...), full_name: str = Form(...)):
    try:
        supabase.table("users").update({"full_name": full_name}).eq("id", user_id).execute()
        return {"status": "success", "message": "บันทึกข้อมูลส่วนตัวเรียบร้อยแล้ว"}
    except Exception as e:
        logger.error(f"Update Profile Error: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="ไม่สามารถอัปเดตข้อมูลได้")

@app.post("/api/v1/update_password")
async def update_password(user_id: str = Form(...), current_password: str = Form(...), new_password: str = Form(...)):
    try:
        res = supabase.table("users").select("password_hash").eq("id", user_id).execute()
        if not res.data:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="ไม่พบผู้ใช้")

        stored_hash = res.data[0].get("password_hash", "")
        if not stored_hash or not bcrypt.checkpw(current_password.encode('utf-8'), stored_hash.encode('utf-8')):
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="รหัสผ่านเดิมไม่ถูกต้อง")
        if len(new_password) < 8:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="รหัสผ่านใหม่ต้องมี 8 ตัวขึ้นไป")

        new_hash = bcrypt.hashpw(new_password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
        supabase.table("users").update({"password_hash": new_hash}).eq("id", user_id).execute()
        return {"status": "success", "message": "เปลี่ยนรหัสผ่านสำเร็จ"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Update Password Error: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="ระบบขัดข้อง ไม่สามารถเปลี่ยนรหัสผ่านได้")
        
# ==========================================
# 7. API: ดึงประวัติ Feedback ทั้งหมด
# ==========================================
@app.get("/api/v1/feedback_history")
def get_feedback_history(user_id: str):
    verify_admin_or_tech(user_id)
    try:
        response = supabase.table("technician_feedback").select("*").order("submitted_at", desc=True).execute()
        return {"status": "success", "data": response.data}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Feedback History Error: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="ไม่สามารถดึงประวัติได้")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)