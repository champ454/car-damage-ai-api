FROM python:3.11-slim

# ติดตั้งไลบรารีระบบที่จำเป็นสำหรับ OpenCV
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ตั้งค่าโฟลเดอร์ชั่วคราวสำหรับ Ultralytics และจำกัดการใช้หน่วยความจำ
ENV YOLO_CONFIG_DIR=/tmp/Ultralytics \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p needs_review && chmod -R 777 /app

EXPOSE 8000

# รองรับตัวแปร $PORT อัตโนมัติของ Render (หากไม่มีจะใช้พอร์ต 8000)
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]