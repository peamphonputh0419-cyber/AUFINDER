# ใช้ Python 3.10 ตัวเต็ม (แนะนำสำหรับ PyTorch)
FROM python:3.10-slim

# กำหนด Working Directory
WORKDIR /app

# ติดตั้ง System Dependencies ที่จำเป็นสำหรับ OpenCV และ SQLite
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# คัดลอกและติดตั้ง Python Dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# คัดลอกโค้ดทั้งหมดเข้ามาใน Container
COPY . .

# สร้างโฟลเดอร์สำหรับเก็บรูปและฐานข้อมูลเผื่อไว้
RUN mkdir -p static/images

# เปิด Port 8000
EXPOSE 8000

# รันแอปพลิเคชันด้วย Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
