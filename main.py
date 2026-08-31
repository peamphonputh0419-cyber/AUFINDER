import io
import os
import re
import sqlite3
import warnings
import traceback
import asyncio
from typing import List, Optional, Tuple

from fastapi import FastAPI, File, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import numpy as np
import cv2
import requests
from bs4 import BeautifulSoup
from PIL import Image, ImageOps
from sklearn.metrics.pairwise import cosine_similarity
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger
from tqdm import tqdm

try:
    from transformers import AutoProcessor, CLIPModel
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

warnings.filterwarnings("ignore")

app = FastAPI(title="Audition Fashion Finder Fast API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DB_NAME = "audition_fashion.db"
VECTOR_FILE = "image_features_fast_v1.npz"
CLIP_MODEL_NAME = os.getenv("AUDITION_CLIP_MODEL", "openai/clip-vit-base-patch32")
IMAGE_DIR = "static/images"

os.makedirs(IMAGE_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

CLIP_WEIGHT = 0.80
COLOR_WEIGHT = 0.20
RERANK_CANDIDATES = 40
RESULT_LIMIT = 20
MIN_SCORE_FLOOR = 0.40

ALL_SEGMENTS = ("full", "top", "bottom", "head", "feet")
SEGMENT_WEIGHTS = {
    "full": 0.40,
    "top": 0.25,
    "bottom": 0.20,
    "head": 0.10,
    "feet": 0.05,
}

CATEGORY_TO_COLOR_SEGMENT = {
    "เสื้อ": "top",
    "กางเกง": "bottom",
    "กระโปรง": "bottom",
    "ใบหน้า/หมวก": "head",
    "ทรงผม": "head",
    "รองเท้า": "feet",
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cpu":
    try: torch.set_num_threads(2)
    except Exception: pass

clip_model = None
clip_processor = None

print(f"\n🚀 กำลังโหลดโมเดล AI (Fast Mode)... (device: {device})")
if TRANSFORMERS_AVAILABLE:
    try:
        clip_processor = AutoProcessor.from_pretrained(CLIP_MODEL_NAME)
        clip_model = CLIPModel.from_pretrained(CLIP_MODEL_NAME).eval().to(device)
        print(f"✅ CLIP พร้อมใช้งาน: {CLIP_MODEL_NAME}")
    except Exception as e:
        print(f"⚠️ โหลด CLIP ล้มเหลว: {e}")

vision_model = None
img_transforms = None
try:
    weights = models.MobileNet_V3_Small_Weights.DEFAULT
    vision_model = models.mobilenet_v3_small(weights=weights)
    vision_model.classifier = torch.nn.Identity()
    vision_model.eval().to(device)
    img_transforms = weights.transforms()
except Exception as e:
    print(f"⚠️ โหลด fallback MobileNet ล้มเหลว: {e}")

vector_cache = {}

def init_db():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS items (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            category TEXT DEFAULT 'ชุดแฟชั่น',
            source_type TEXT DEFAULT 'โปรโมชันพิเศษ',
            source_detail TEXT DEFAULT '',
            gender TEXT DEFAULT 'ชาย/หญิง',
            image_url TEXT NOT NULL,
            source_url TEXT DEFAULT '',
            color_tag TEXT DEFAULT 'อื่นๆ',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    try:
        cursor.execute("ALTER TABLE items ADD COLUMN color_tag TEXT DEFAULT 'อื่นๆ'")
        conn.commit()
    except sqlite3.OperationalError:
        pass

    try:
        cursor.execute("ALTER TABLE items ADD COLUMN published_at TEXT DEFAULT NULL")
        conn.commit()
    except sqlite3.OperationalError:
        pass

    try:
        cursor.execute("ALTER TABLE items ADD COLUMN scrape_order INTEGER DEFAULT NULL")
        conn.commit()
    except sqlite3.OperationalError:
        pass

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS vector_failures (
            item_id INTEGER PRIMARY KEY,
            failed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

init_db()
FAILURE_RETRY_DAYS = 3

import datetime

THAI_MONTHS_FULL = {
    "มกราคม": 1, "กุมภาพันธ์": 2, "มีนาคม": 3, "เมษายน": 4,
    "พฤษภาคม": 5, "มิถุนายน": 6, "กรกฎาคม": 7, "สิงหาคม": 8,
    "กันยายน": 9, "ตุลาคม": 10, "พฤศจิกายน": 11, "ธันวาคม": 12,
}
THAI_MONTHS_ABBR = {
    "ม.ค.": 1, "มค": 1, "ก.พ.": 2, "กพ": 2, "มี.ค.": 3, "มีค": 3,
    "เม.ย.": 4, "เมย": 4, "พ.ค.": 5, "พค": 5, "มิ.ย.": 6, "มิย": 6,
    "ก.ค.": 7, "กค": 7, "ส.ค.": 8, "สค": 8, "ก.ย.": 9, "กย": 9,
    "ต.ค.": 10, "ตค": 10, "พ.ย.": 11, "พย": 11, "ธ.ค.": 12, "ธค": 12,
}
ALL_THAI_MONTHS = {**THAI_MONTHS_FULL, **THAI_MONTHS_ABBR}

_MONTH_NAME_PATTERN = "|".join(sorted(ALL_THAI_MONTHS.keys(), key=len, reverse=True))
_DATE_PATTERNS = [
    re.compile(r"(" + _MONTH_NAME_PATTERN + r")\s+(\d{1,2}),?\s*(\d{4})"),
    re.compile(r"(\d{1,2})\s+(" + _MONTH_NAME_PATTERN + r")\s*,?\s*(\d{4})"),
    re.compile(r"(\d{4})[-/](\d{1,2})[-/](\d{1,2})"),
    re.compile(r"(\d{1,2})[-/](\d{1,2})[-/](\d{4})"),
]

def _normalize_year(year: int) -> int:
    return year - 543 if year > 2400 else year

def _try_build_date(y: int, mo: int, d: int) -> Optional[str]:
    try:
        return datetime.datetime(_normalize_year(y), mo, d).strftime("%Y-%m-%d 00:00:00")
    except Exception:
        return None

def parse_thai_date_from_text(text: str) -> Optional[str]:
    if not text:
        return None
    for pattern in _DATE_PATTERNS:
        m = pattern.search(text)
        if m:
            groups = m.groups()
            if len(groups) == 3:
                if groups[0] in ALL_THAI_MONTHS:
                    month_num = ALL_THAI_MONTHS[groups[0]]
                    res = _try_build_date(int(groups[2]), month_num, int(groups[1]))
                    if res: return res
                elif groups[1] in ALL_THAI_MONTHS:
                    month_num = ALL_THAI_MONTHS[groups[1]]
                    res = _try_build_date(int(groups[2]), month_num, int(groups[0]))
                    if res: return res
                else:
                    res = _try_build_date(int(groups[0]), int(groups[1]), int(groups[2]))
                    if res: return res
    return None

def extract_date_from_article_tag(article) -> Optional[str]:
    try:
        time_tag = article.find("time")
        if time_tag:
            dt_attr = time_tag.get("datetime") or time_tag.get("content")
            if dt_attr:
                parsed = parse_thai_date_from_text(dt_attr)
                if parsed: return parsed
            parsed = parse_thai_date_from_text(time_tag.get_text(" ", strip=True))
            if parsed: return parsed
        date_like = article.find(["span", "div", "p"], class_=re.compile(r"date|time|published|วันที่", re.IGNORECASE))
        if date_like:
            parsed = parse_thai_date_from_text(date_like.get_text(" ", strip=True))
            if parsed: return parsed
    except Exception:
        pass
    return None

def process_raw_bytes_to_pil(image_bytes: bytes) -> Image.Image:
    if not image_bytes: return None
    try:
        img = Image.open(io.BytesIO(image_bytes))
        img.load()
        try: img = ImageOps.exif_transpose(img)
        except Exception: pass
        if img.mode != "RGB":
            bg = Image.new("RGB", img.size, (255, 255, 255))
            if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
                alpha = img.convert("RGBA").split()[3]
                bg.paste(img.convert("RGB"), mask=alpha)
                img = bg
            else:
                img = img.convert("RGB")
        return img
    except Exception:
        try:
            nparr = np.frombuffer(image_bytes, np.uint8)
            cv_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if cv_img is not None:
                return Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))
        except Exception: pass
    return None

def resize_max_dim(pil_img: Image.Image, max_dim: int = 224) -> Image.Image:
    if pil_img is None: return pil_img
    w, h = pil_img.size
    longest = max(w, h)
    if longest <= max_dim: return pil_img
    scale = max_dim / float(longest)
    return pil_img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)

def get_dominant_color_name(pil_img: Image.Image) -> str:
    try:
        cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2HSV)
        pixels = cv_img.reshape(-1, 3)
        valid_pixels = pixels[~((pixels[:, 1] < 20) & (pixels[:, 2] > 220))]
        if len(valid_pixels) > 50:
            pixels = valid_pixels

        mean_hsv = np.mean(pixels, axis=0)
        h, s, v = mean_hsv
        
        if v < 40: return "ดำ"
        if v > 210 and s < 25: return "ขาว"
        if s < 35:
            if v < 120: return "เทา"
            else: return "ขาว"
        
        if h < 8 or h > 172: return "แดง"
        elif 8 <= h < 22: return "ส้ม"
        elif 22 <= h < 33: return "เหลือง"
        elif 33 <= h < 78: return "เขียว"
        elif 78 <= h < 125: return "ฟ้า/น้ำเงิน"
        elif 125 <= h < 155: return "ม่วง"
        elif 155 <= h <= 172: return "ชมพู"
        return "อื่นๆ"
    except Exception:
        return "อื่นๆ"

def extract_advanced_color_histogram(pil_img: Image.Image) -> np.ndarray:
    cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(cv_img, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [12, 6, 6], [0, 180, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()

def encode_images_clip(pil_images: List[Image.Image]) -> Optional[np.ndarray]:
    if not pil_images or clip_model is None or clip_processor is None: return None
    try:
        inputs = clip_processor(images=pil_images, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.inference_mode():
            feat = clip_model.get_image_features(**inputs)
            feat = feat / torch.norm(feat, p=2, dim=-1, keepdim=True)
        return feat.detach().cpu().numpy().astype(np.float32)
    except Exception:
        return None

def encode_images_deep(pil_images: List[Image.Image]) -> Optional[np.ndarray]:
    if not pil_images or vision_model is None or img_transforms is None: return None
    try:
        tensors = torch.stack([img_transforms(img) for img in pil_images]).to(device)
        with torch.inference_mode():
            feat = vision_model(tensors)
            feat = feat / torch.norm(feat, p=2, dim=-1, keepdim=True)
            return feat.cpu().numpy()
    except Exception:
        return None

def encode_images(pil_images: List[Image.Image]) -> Optional[np.ndarray]:
    clip_vecs = encode_images_clip(pil_images)
    if clip_vecs is not None: return clip_vecs
    return encode_images_deep(pil_images)

def auto_crop_subject(pil_img: Image.Image, pad_ratio: float = 0.04) -> Image.Image:
    try:
        arr = np.array(pil_img.convert("RGB"))
        h, w = arr.shape[:2]
        if h < 20 or w < 20: return pil_img
        corner = max(4, min(16, h // 10, w // 10))
        corners = np.concatenate([
            arr[:corner, :corner].reshape(-1, 3),
            arr[:corner, -corner:].reshape(-1, 3),
            arr[-corner:, :corner].reshape(-1, 3),
            arr[-corner:, -corner:].reshape(-1, 3),
        ])
        bg_color = np.median(corners, axis=0)
        diff = np.abs(arr.astype(np.int16) - bg_color.astype(np.int16)).sum(axis=2)
        mask = (diff > 28).astype(np.uint8)
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if num_labels <= 1: return pil_img
        areas = stats[1:, cv2.CC_STAT_AREA]
        best_label = 1 + int(np.argmax(areas))
        x, y, bw, bh, area = stats[best_label]
        if area < 0.03 * w * h or (bw >= w * 0.98 and bh >= h * 0.98): return pil_img
        pad_x, pad_y = int(bw * pad_ratio), int(bh * pad_ratio)
        x0, y0 = max(0, x - pad_x), max(0, y - pad_y)
        x1, y1 = min(w, x + bw + pad_x), min(h, y + bh + pad_y)
        return pil_img.crop((x0, y0, x1, y1))
    except Exception:
        return pil_img

def segment_has_signal(pil_img: Image.Image, std_threshold: float = 8.0) -> bool:
    try:
        arr = np.array(pil_img.convert("L"), dtype=np.float32)
        return float(arr.std()) >= std_threshold
    except Exception:
        return True

def prepare_image_segments(pil_img: Image.Image) -> Tuple[dict, bool]:
    pil_img = resize_max_dim(pil_img, 224)
    w, h = pil_img.size
    boxes = {
        "head": (int(w * 0.08), 0, int(w * 0.92), int(h * 0.42)),
        "top": (int(w * 0.04), int(h * 0.25), int(w * 0.96), int(h * 0.78)),
        "bottom": (int(w * 0.04), int(h * 0.58), int(w * 0.96), h),
        "feet": (int(w * 0.08), int(h * 0.74), int(w * 0.92), h),
    }
    segments = {"full": pil_img}
    for seg, box in boxes.items():
        segments[seg] = pil_img.crop(box)
    return segments, True

def extract_features_bulk_tensor(pil_images: List[Image.Image]) -> List[Optional[np.ndarray]]:
    if not pil_images: return []
    deep_vecs = encode_images(pil_images)
    if deep_vecs is None: return [None] * len(pil_images)
    results = []
    for i, pil_img in enumerate(pil_images):
        color_vec = extract_advanced_color_histogram(pil_img)
        color_vec = color_vec / (np.linalg.norm(color_vec) + 1e-8)
        combined = np.hstack((deep_vecs[i] * CLIP_WEIGHT, color_vec * COLOR_WEIGHT))
        norm = np.linalg.norm(combined)
        results.append(combined / norm if norm > 0 else combined)
    return results

def compute_phash(pil_img: Image.Image, hash_size: int = 8) -> np.ndarray:
    try:
        gray = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2GRAY)
        gray = cv2.resize(gray, (hash_size * 4, hash_size * 4), interpolation=cv2.INTER_AREA)
        dct = cv2.dct(np.float32(gray))
        low = dct[:hash_size, :hash_size]
        med = np.median(low[1:, :])
        return (low > med).astype(np.uint8).flatten()
    except Exception:
        return np.zeros(hash_size * hash_size, dtype=np.uint8)

def phash_similarity(a: np.ndarray, b: np.ndarray) -> float:
    if a is None or b is None or a.size != b.size: return 0.0
    return float(1.0 - np.count_nonzero(a != b) / a.size)

def load_local_item_image(item_info: dict) -> Optional[Image.Image]:
    try:
        filename = os.path.basename(item_info.get("img", ""))
        path = os.path.join(IMAGE_DIR, filename)
        if not os.path.exists(path): return None
        with Image.open(path) as im:
            return im.convert("RGB")
    except Exception:
        return None

def sync_and_load_vector_cache():
    global vector_cache
    print("⏳ กำลังตรวจสอบและโหลด Vector Cache (Fast Mode)...")
    existing_ids = []
    existing_features = {s: [] for s in ALL_SEGMENTS}

    if os.path.exists(VECTOR_FILE):
        try:
            data = np.load(VECTOR_FILE, allow_pickle=True)
            existing_ids = data["ids"].tolist()
            for s in ALL_SEGMENTS:
                existing_features[s] = data[f"features_{s}"].tolist()
        except Exception:
            existing_ids = []
            existing_features = {s: [] for s in ALL_SEGMENTS}

    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT id, image_url FROM items")
    db_items = cursor.fetchall()

    cursor.execute(
        "SELECT item_id FROM vector_failures WHERE datetime(failed_at) > datetime('now', ?)",
        (f"-{FAILURE_RETRY_DAYS} days",),
    )
    recently_failed_ids = {r[0] for r in cursor.fetchall()}
    conn.close()

    existing_set = set(existing_ids)
    new_items = [i for i in db_items if i[0] not in existing_set and i[0] not in recently_failed_ids]

    if new_items:
        batch_size = 64
        newly_failed_ids = []
        for i in tqdm(range(0, len(new_items), batch_size), desc="สกัด Vector (Fast)"):
            batch_chunk = new_items[i:i + batch_size]
            valid_chunk_meta = []
            full_batch, top_batch, bottom_batch, head_batch, feet_batch = [], [], [], [], []

            for item_id, image_url in batch_chunk:
                if not image_url:
                    newly_failed_ids.append(item_id)
                    continue
                filename = os.path.basename(image_url)
                img_path = os.path.join(IMAGE_DIR, filename)

                if not os.path.exists(img_path) or os.path.getsize(img_path) == 0:
                    if image_url.startswith("http"):
                        try:
                            res = requests.get(image_url, timeout=3)
                            if res.status_code == 200 and len(res.content) > 0:
                                with open(img_path, "wb") as f: f.write(res.content)
                            else:
                                newly_failed_ids.append(item_id)
                                continue
                        except Exception:
                            newly_failed_ids.append(item_id)
                            continue
                    else:
                        newly_failed_ids.append(item_id)
                        continue

                try:
                    with Image.open(img_path) as img:
                        base_img = auto_crop_subject(img.convert("RGB"))
                        segments, _ = prepare_image_segments(base_img)
                        full_batch.append(segments["full"])
                        top_batch.append(segments["top"])
                        bottom_batch.append(segments["bottom"])
                        head_batch.append(segments["head"])
                        feet_batch.append(segments["feet"])
                        valid_chunk_meta.append(item_id)
                except Exception:
                    newly_failed_ids.append(item_id)

            if valid_chunk_meta:
                try:
                    full_vecs = extract_features_bulk_tensor(full_batch)
                    top_vecs = extract_features_bulk_tensor(top_batch)
                    bottom_vecs = extract_features_bulk_tensor(bottom_batch)
                    head_vecs = extract_features_bulk_tensor(head_batch)
                    feet_vecs = extract_features_bulk_tensor(feet_batch)

                    for idx, item_id in enumerate(valid_chunk_meta):
                        f_v = full_vecs[idx]
                        t_v = top_vecs[idx]
                        b_v = bottom_vecs[idx]
                        h_v = head_vecs[idx]
                        ft_v = feet_vecs[idx]

                        if all(v is not None for v in (f_v, t_v, b_v, h_v, ft_v)):
                            existing_ids.append(item_id)
                            existing_features["full"].append(f_v)
                            existing_features["top"].append(t_v)
                            existing_features["bottom"].append(b_v)
                            existing_features["head"].append(h_v)
                            existing_features["feet"].append(ft_v)
                        else:
                            newly_failed_ids.append(item_id)
                except Exception:
                    pass

        if existing_ids:
            save_kwargs = {"ids": np.array(existing_ids, dtype=np.int64)}
            for s in ALL_SEGMENTS:
                save_kwargs[f"features_{s}"] = np.array(existing_features[s], dtype=np.float32)
            np.savez_compressed(VECTOR_FILE, **save_kwargs)

    if existing_ids:
        vector_cache["ids"] = np.array(existing_ids, dtype=np.int64)
        for s in ALL_SEGMENTS:
            vector_cache[s] = np.array(existing_features[s], dtype=np.float32)

PROMO_TARGET_URLS = [
    "https://audition.playpark.com/th-th/category/news/promotion/",
    "https://audition.playpark.com/th-th/category/news/event/"
]
SCRAPE_HEADERS = {"User-Agent": "Mozilla/5.0"}

def scrape_article_list(target_url: str) -> List[dict]:
    results = []
    try:
        res = requests.get(target_url, headers=SCRAPE_HEADERS, timeout=5)
        if res.status_code != 200: return results
        soup = BeautifulSoup(res.content, "html.parser")
        articles = soup.find_all("article") or soup.find_all("div", class_=re.compile(r"news|post|item"))
        for article in articles:
            a_tag = article.find("a", href=True)
            img_tag = article.find("img")
            if not a_tag or not img_tag: continue
            title = a_tag.get_text(strip=True) or img_tag.get("alt", "ไอเทม")
            img_src = img_tag.get("src") or img_tag.get("data-src")
            promo_link = a_tag["href"]
            if not img_src: continue
            published_at = extract_date_from_article_tag(article) or \
                parse_thai_date_from_text(article.get_text(" ", strip=True))
            results.append({
                "title": title, "img_src": img_src,
                "promo_link": promo_link, "published_at": published_at,
            })
    except Exception:
        pass
    return results

def backfill_published_dates_for_existing_items() -> int:
    updated = 0
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    try:
        scrape_order_counter = 0
        for target_url in PROMO_TARGET_URLS:
            for art in scrape_article_list(target_url):
                current_scrape_order = scrape_order_counter
                scrape_order_counter += 1
                if art["published_at"]:
                    cursor.execute(
                        "UPDATE items SET published_at = ? WHERE source_url = ? AND published_at IS NULL",
                        (art["published_at"], art["promo_link"]),
                    )
                    updated += cursor.rowcount
                cursor.execute(
                    "UPDATE items SET scrape_order = ? WHERE source_url = ? AND scrape_order IS NULL",
                    (current_scrape_order, art["promo_link"]),
                )
        conn.commit()
    finally:
        conn.close()
    return updated

def fetch_latest_audition_promotions_and_events():
    print("\n🔍 [Auto-Scraper] กำลังตรวจสอบโปรโมชันและอีเวนต์ใหม่จากหน้าเว็บ Audition...")
    new_inserted = 0
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    scrape_order_counter = 0

    for target_url in PROMO_TARGET_URLS:
        try:
            articles = scrape_article_list(target_url)
            for art in articles:
                current_scrape_order = scrape_order_counter
                scrape_order_counter += 1
                title = art["title"]
                img_src = art["img_src"]
                promo_link = art["promo_link"]
                published_at = art["published_at"]

                cursor.execute("SELECT id FROM items WHERE source_url = ?", (promo_link,))
                if cursor.fetchone(): continue

                try:
                    print(f"พบไอเทม/โปรโมชันใหม่: \"{title}\" กำลังดาวน์โหลดรูปภาพ...")
                    img_res = requests.get(img_src, headers=SCRAPE_HEADERS, timeout=3)
                    if img_res.status_code == 200:
                        ext = img_src.split(".")[-1].split("?")[0]
                        if ext not in ["jpg", "png", "jpeg", "webp"]: ext = "jpg"
                        safe_filename = f"promo_{new_inserted}_{int(os.urandom(4).hex(), 16)}.{ext}"
                        local_img_path = os.path.join(IMAGE_DIR, safe_filename)
                        with open(local_img_path, "wb") as f: f.write(img_res.content)

                        web_img_url = f"/static/images/{safe_filename}"
                        pil_temp = Image.open(local_img_path).convert("RGB")
                        color_tag = get_dominant_color_name(pil_temp)

                        category = "ชุดแฟชั่น"
                        if "หน้า" in title or "หมวก" in title or "แว่น" in title: category = "ใบหน้า/หมวก"
                        elif "ผม" in title: category = "ทรงผม"
                        elif "เสื้อ" in title: category = "เสื้อ"
                        elif "กางเกง" in title or "กระโปรง" in title: category = "กางเกง"
                        elif "รองเท้า" in title: category = "รองเท้า"

                        title_lower = title.lower()
                        if "golden gacha" in title_lower or "golden" in title_lower: source_type = "Golden Gacha"
                        elif "gacha" in title_lower or "กาชา" in title_lower: source_type = "Gacha"
                        elif "กิจกรรม" in title_lower or "event" in title_lower or "ฟรี" in title_lower: source_type = "กิจกรรมฟรี"
                        elif "พิเศษ" in title_lower or "special" in title_lower: source_type = "โปรโมชันพิเศษ"
                        else: source_type = "โปรโมชันพิเศษ"

                        cursor.execute("""
                            INSERT INTO items (name, category, source_type, source_detail, gender, image_url, source_url, color_tag, published_at, scrape_order)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, (title, category, source_type, source_type, "ชาย/หญิง", web_img_url, promo_link, color_tag, published_at, current_scrape_order))
                        new_inserted += 1
                except Exception as ex: 
                    print(f"⚠️ ดึงข้อมูลไอเทมล้มเหลว: {ex}")
        except Exception as ex:
            print(f"⚠️ เข้าถึงลิงก์ล้มเหลว ({target_url}): {ex}")

    conn.commit()
    conn.close()
    if new_inserted > 0:
        sync_and_load_vector_cache()

scheduler = BackgroundScheduler()

@app.on_event("startup")
async def startup_event():
    loop = asyncio.get_event_loop()
    loop.run_in_executor(None, sync_and_load_vector_cache)
    loop.run_in_executor(None, backfill_published_dates_for_existing_items)
    loop.run_in_executor(None, fetch_latest_audition_promotions_and_events)
    scheduler.add_job(fetch_latest_audition_promotions_and_events, trigger=IntervalTrigger(days=1), id="daily_sync", replace_existing=True)
    scheduler.start()

@app.on_event("shutdown")
def shutdown_event():
    scheduler.shutdown()

@app.get("/")
def read_index():
    return FileResponse("index.html")

@app.get("/api/items")
def get_items(
    search: str = Query(""),
    category: str = Query("ทั้งหมด"),
    source: str = Query("ทั้งหมด"),
    gender: str = Query("ทั้งหมด"),
    sort: str = Query("newest"),
    page: int = Query(1, ge=1),
    limit: int = Query(20, ge=1, le=100),
):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    count_query = "SELECT COUNT(*) FROM items WHERE 1=1"
    query = "SELECT id, name, category, source_type, source_detail, gender, image_url, source_url, color_tag FROM items WHERE 1=1"
    params = []

    if search.strip():
        query += " AND name LIKE ?"
        count_query += " AND name LIKE ?"
        params.append(f"%{search.strip()}%")

    if category != "ทั้งหมด":
        query += " AND category LIKE ?"
        count_query += " AND category LIKE ?"
        params.append(f"%{category}%")

    # [แก้ไขเพิ่มเติม] รองรับการค้นหา source เป็น Topup, เติมเงิน ฯลฯ ได้อย่างยืดหยุ่น
    if source != "ทั้งหมด":
        if source == "Topup":
            query += " AND (source_type = ? OR source_type = ? OR source_detail LIKE ?)"
            count_query += " AND (source_type = ? OR source_type = ? OR source_detail LIKE ?)"
            params.extend(["Topup", "เติมเงิน", "%Topup%"])
        else:
            query += " AND source_type = ?"
            count_query += " AND source_type = ?"
            params.append(source)

    if gender != "ทั้งหมด":
        query += " AND (gender LIKE ? OR gender = 'ทั้งหมด' OR gender = 'ชาย/หญิง')"
        count_query += " AND (gender LIKE ? OR gender = 'ทั้งหมด' OR gender = 'ชาย/หญิง')"
        params.append(f"%{gender}%")

    cursor.execute(count_query, params)
    total_items = cursor.fetchone()[0]

    offset = (page - 1) * limit
    
    if sort == "newest":
        order_clause = (
            " ORDER BY COALESCE(published_at, created_at) DESC, "
            "COALESCE(scrape_order, id) ASC"
        )
    else: 
        order_clause = (
            " ORDER BY COALESCE(published_at, created_at) ASC, "
            "COALESCE(scrape_order, id) DESC"
        )

    query += f"{order_clause} LIMIT ? OFFSET ?"
    params.extend([limit, offset])

    cursor.execute(query, params)
    rows = cursor.fetchall()
    conn.close()

    return {
        "items": [
            {
                "id": r[0], "name": r[1], "category": r[2],
                "sourceType": r[3], "detail": r[4], "gender": r[5],
                "img": r[6], "url": r[7], "colorTag": r[8],
            }
            for r in rows
        ],
        "total": total_items,
        "page": page,
        "totalPages": (total_items + limit - 1) // limit if total_items > 0 else 1,
    }

@app.post("/api/search-by-image")
async def search_by_image(
    file: UploadFile = File(...),
    target_part: str = Query("auto"),
    category: str = Query("ทั้งหมด"),
):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT id, name, category, source_type, source_detail, gender, image_url, source_url, color_tag FROM items")
    rows = cursor.fetchall()
    conn.close()

    db_items = {
        r[0]: {
            "id": r[0], "name": r[1], "category": r[2],
            "sourceType": r[3], "detail": r[4], "gender": r[5],
            "img": r[6], "url": r[7], "colorTag": r[8],
        } for r in rows
    }

    try:
        await file.seek(0)
        image_bytes = await file.read()
        if len(image_bytes) > 15 * 1024 * 1024:
            return {"results": [], "categories": {}, "error": "ไฟล์ใหญ่เกิน 15MB"}

        pil_img = process_raw_bytes_to_pil(image_bytes)
        if pil_img is None:
            return {"results": [], "categories": {}, "error": "ไม่สามารถเปิดไฟล์รูปภาพได้"}

        pil_img = auto_crop_subject(pil_img)
        processed_img = resize_max_dim(pil_img, 224)
        query_phash = compute_phash(processed_img)

        segments, pose_reliable = prepare_image_segments(processed_img)
        imgs = [segments[s] for s in ALL_SEGMENTS]
        vecs = extract_features_bulk_tensor(imgs)
        query_vectors = dict(zip(ALL_SEGMENTS, vecs))

        query_colors = {s: get_dominant_color_name(segments[s]) for s in ALL_SEGMENTS}
        query_color = query_colors["full"]

        if not any(v is not None for v in query_vectors.values()):
            return {"results": [], "categories": {}, "error": "ไม่สามารถสกัดข้อมูลจากภาพได้"}

        if "ids" not in vector_cache or len(vector_cache["ids"]) == 0:
            return {"results": [], "categories": {}, "error": "ระบบกำลังสร้างดัชนีภาพ กรุณาลองใหม่อีกครั้ง"}

        item_ids = vector_cache["ids"]
        n_items = len(item_ids)

        allowed = np.ones(n_items, dtype=bool)
        if category != "ทั้งหมด":
            allowed = np.array([
                db_items.get(int(item_id), {}).get("category") == category
                for item_id in item_ids
            ])

        score_sum = np.zeros(n_items, dtype=np.float64)
        weight_sum = 0.0
        
        if target_part != "auto" and target_part in ALL_SEGMENTS:
            weights = {target_part: 0.80, "full": 0.20}
        else:
            weights = dict(SEGMENT_WEIGHTS)
            for seg in ("head", "top", "bottom", "feet"):
                if seg in weights and not segment_has_signal(segments[seg]):
                    weights[seg] = 0.0

        for s, w in weights.items():
            q_vec = query_vectors.get(s)
            ds = vector_cache.get(s)
            if q_vec is None or ds is None or len(ds) != n_items:
                continue
            sims = cosine_similarity([q_vec], ds)[0]
            score_sum += sims * w
            weight_sum += w

        if weight_sum == 0:
            return {"results": [], "categories": {}, "error": "ไม่สามารถเปรียบเทียบภาพได้"}

        final_scores = score_sum / weight_sum
        final_scores[~allowed] = -1.0

        for idx, matched_id in enumerate(item_ids):
            if final_scores[idx] < 0: continue
            info = db_items.get(int(matched_id))
            if not info: continue
            cat = info.get("category", "")
            seg_key = CATEGORY_TO_COLOR_SEGMENT.get(cat, "full")
            match_color = query_colors.get(seg_key) or query_color
            
            if match_color and match_color == info.get("colorTag", ""):
                final_scores[idx] += COLOR_WEIGHT * 0.80

        candidate_order = np.argsort(final_scores)[::-1]
        candidate_order = [i for i in candidate_order if final_scores[i] >= MIN_SCORE_FLOOR][:RERANK_CANDIDATES]

        for idx in candidate_order:
            matched_id = int(item_ids[idx])
            info = db_items.get(matched_id)
            if not info: continue
            cand_img = load_local_item_image(info)
            if cand_img is None: continue
            ph = compute_phash(cand_img)
            ph_score = phash_similarity(query_phash, ph)
            final_scores[idx] = (final_scores[idx] * 0.80) + (ph_score * 0.20)

        top_order = np.argsort(final_scores)[::-1][:RESULT_LIMIT]

        def confidence_from_score(score: float) -> float:
            x = max(0.0, min(1.0, score))
            return round(max(15.0, min(98.0, x * 100)), 1)

        detected_results = []
        for idx in top_order:
            score = float(final_scores[idx])
            if score < MIN_SCORE_FLOOR: continue
            matched_id = int(item_ids[idx])
            info = db_items.get(matched_id)
            if not info: continue

            conf = confidence_from_score(score)
            item_data = info.copy()
            item_data["matchScore"] = conf
            detected_results.append({
                "confidence": conf,
                "rawScore": round(max(0.0, min(1.0, score)), 4),
                "item": item_data,
            })

        categories = {}
        for r in detected_results:
            cat = r["item"].get("category", "อื่นๆ")
            categories.setdefault(cat, []).append(r)

        for cat in categories:
            categories[cat].sort(key=lambda x: x["rawScore"], reverse=True)

        return {
            "detected_count": len(detected_results),
            "results": detected_results,
            "categories": categories,
            "query": {
                "color": query_color,
                "targetPart": target_part,
                "category": category,
                "engine": "CLIP + Enhanced Color Matching",
                "poseReliable": pose_reliable,
            },
        }
    except Exception as e:
        traceback.print_exc()
        return {"results": [], "categories": {}, "error": "เกิดข้อผิดพลาดในการประมวลผลค้นหา"}

@app.post("/api/backfill-published-dates")
def backfill_published_dates_endpoint():
    try:
        updated = backfill_published_dates_for_existing_items()
        return {"ok": True, "updated": updated, "message": f"เติมวันที่ย้อนหลังให้ {updated} รายการ"}
    except Exception as e:
        traceback.print_exc()
        return {"ok": False, "error": str(e)}

@app.post("/api/rebuild-image-index")
def rebuild_image_index():
    global vector_cache
    vector_cache = {}
    try:
        if os.path.exists(VECTOR_FILE):
            os.remove(VECTOR_FILE)
        sync_and_load_vector_cache()
        return {"ok": True, "message": "สร้างดัชนีภาพใหม่เรียบร้อย", "count": len(vector_cache.get("ids", []))}
    except Exception as e:
        traceback.print_exc()
        return {"ok": False, "error": str(e)}