from flask import Flask, request, jsonify, render_template
import os, base64, datetime, sqlite3, hashlib, time, csv
from pathlib import Path
import cv2
import numpy as np
from typing import Optional, cast, List, Tuple
import onnxruntime as ort

# -------------------------
# Anti-spoof imports
# -------------------------
from models.antispoof.anti_spoof_predict import AntiSpoofPredict
from models.antispoof.utility import parse_model_name
# --- SCRFD Detector (faster, ONNX)
from models.face_detection.scrfd.wrapper import SCRFDWrapper
# SCRFD model initialization moved below after MODEL_DIR is defined


# --- PRNet Alignment ---
from models.face_alignment.alignment import norm_crop



# -------------------------
TARGET_SIZE = (640, 480)
app = Flask(__name__)

# -------------------------
# Paths
# -------------------------
# -------------------------
# Paths
# -------------------------
BASE = Path.cwd()
BASE = Path.cwd()
MODEL_DIR = BASE / "models"
ANTI_SPOOF_MODEL_DIR = MODEL_DIR / "anti_spoof"
ARCFACE_DIR = MODEL_DIR / "arcface"
USER_DB = MODEL_DIR / "users.db"
ATT_CSV = BASE / "Attendance.csv"

# Directory to save suspicious frames (served via Flask's static folder)
STATIC_DIR = BASE / "static"
SUSPICIOUS_DIR = STATIC_DIR / "suspicious_frames"
ENROLL_DIR = STATIC_DIR / "enrollment_images"
SUSPICIOUS_DIR.mkdir(parents=True, exist_ok=True)
ENROLL_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)
ARCFACE_DIR.mkdir(exist_ok=True)

# Initialize SCRFD model (now that MODEL_DIR is defined)
SCRFD_MODEL = MODEL_DIR / "face_detection" / "scrfd" / "weights" / "scrfd_10g_bnkps.onnx"
retina = SCRFDWrapper(model_path=str(SCRFD_MODEL), input_size=(640,480), score_thresh=0.45)

# -------------------------
# Startup Logs
# -------------------------
print("\n🚀 Starting Attendance Verification Service (port 2000)...")
print("[INFO] Model directories:")
print(f"  - AntiSpoof: {ANTI_SPOOF_MODEL_DIR}")
print(f"  - ArcFace:   {ARCFACE_DIR}")
print(f"  - User DB:   {USER_DB}")
print(f"  - Static:    {STATIC_DIR}")



# Suspicious log CSV
SUSP_CSV = BASE / "suspicious_log.csv"
if not SUSP_CSV.exists():
    with open(SUSP_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["user_id", "date", "time", "type", "saved_path", "score"])

# -------------------------
# Config - tweak these
# -------------------------
MAX_ENROLL_IMAGES = 7        # max images accepted during registration
MIN_VALID_IMAGES = 2         # minimum valid images required to register
MATCH_THRESHOLD = 0.55       # verification threshold
SPOOF_CONF_THRESHOLD = 0.9   # required score for anti-spoof label==1
# -------------------------

# -------------------------
# DB Setup
# -------------------------
def init_user_db():
    conn = sqlite3.connect(str(USER_DB))
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            dim INTEGER,
            embedding BLOB,
            variance BLOB,
            reg_quality REAL,
            created_at TEXT
        )
        """
    )
    conn.commit()
    # add missing columns if old DB
    cur.execute("PRAGMA table_info(users)")
    cols = [r[1] for r in cur.fetchall()]
    if "variance" not in cols:
        try:
            cur.execute("ALTER TABLE users ADD COLUMN variance BLOB")
            conn.commit()
        except Exception:
            pass
    if "reg_quality" not in cols:
        try:
            cur.execute("ALTER TABLE users ADD COLUMN reg_quality REAL")
            conn.commit()
        except Exception:
            pass
    if "created_at" not in cols:
        try:
            cur.execute("ALTER TABLE users ADD COLUMN created_at TEXT")
            conn.commit()
        except Exception:
            pass
    conn.close()

def hash_password(password: str):
    return hashlib.sha256(password.encode()).hexdigest()

# -------------------------
# Attendance logging
# -------------------------
def mark_attendance(user_id, event):
    now = datetime.datetime.now()
    date = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S")

    if not ATT_CSV.exists():
        with open(ATT_CSV, "w") as f:
            f.write("user_id,date,time,event\n")

    with open(ATT_CSV, "r") as f:
        lines = f.read().splitlines()[1:]

    today_records = [line.split(",") for line in lines if line.split(",")[1] == date and line.split(",")[0] == user_id]

    if today_records:
        last_event = today_records[-1][3]
        if last_event == event:
            return False, time_str, event

    with open(ATT_CSV, "a") as f:
        f.write(f"{user_id},{date},{time_str},{event}\n")

    return True, time_str, event


# -------------------------
# ArcFace embedding (ONNX Runtime)
# -------------------------
ONNX_ARCFACE_PATH = ARCFACE_DIR / "model.onnx"
ARCFACE_SESSION = None
if ONNX_ARCFACE_PATH.exists():
    ARCFACE_SESSION = ort.InferenceSession(str(ONNX_ARCFACE_PATH), providers=["CPUExecutionProvider"])
    print("[INFO] ArcFace model loaded successfully.")
    print("[INFO] ArcFace model hash:", hashlib.md5(open(ONNX_ARCFACE_PATH, "rb").read()).hexdigest()[:8])
else:
    print("[ERROR] ArcFace model not found at", ONNX_ARCFACE_PATH)


def preprocess_face_for_arcface(face_bgr: np.ndarray, size=(112,112)):
    if face_bgr is None or face_bgr.size == 0:
        return None
    face = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    face = cv2.resize(face, size)
    x = (face.astype(np.float32) - 127.5) / 128.0
    x = np.transpose(x, (2,0,1))
    return np.expand_dims(x, 0).astype(np.float32)

def get_embedding(face_bgr: np.ndarray):
    if ARCFACE_SESSION is None:
        return None
    inp = preprocess_face_for_arcface(face_bgr)
    if inp is None:
        return None
    input_name = ARCFACE_SESSION.get_inputs()[0].name
    out = ARCFACE_SESSION.run(None, {input_name: inp})
    out0 = cast(np.ndarray, out[0])
    emb = np.squeeze(out0).astype(np.float32)
    emb /= (np.linalg.norm(emb) + 1e-6)
    return emb

# -------------------------
# Initialize anti-spoof + DB
# -------------------------
anti_spoof = AntiSpoofPredict(device_id=0)
init_user_db()



# -------------------------
# Suspicious detection tracker
# -------------------------
suspicious_tracker = {}

def log_suspicious(user_id, suspicious_type, saved_path, score=-1.0, date=None, time_str=None):
    if date is None or time_str is None:
        now = datetime.datetime.now()
        date = now.strftime("%Y-%m-%d")
        time_str = now.strftime("%H:%M:%S")
    with open(SUSP_CSV, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([user_id, date, time_str, suspicious_type, saved_path, score])

# -------------------------
# Helper functions for enrollment quality + weighting
# -------------------------
def face_sharpness(gray_face: np.ndarray) -> float:
    if gray_face is None or gray_face.size == 0:
        return 0.0
    return float(cv2.Laplacian(gray_face, cv2.CV_64F).var())

def face_area_ratio(bbox: Tuple[int,int,int,int], frame_shape: Tuple[int,int]) -> float:
    x,y,w,h = bbox
    frame_h, frame_w = frame_shape
    return float((w * h) / (frame_w * frame_h + 1e-9))

def compute_quality_metrics(face_bgr: np.ndarray, bbox: Tuple[int,int,int,int], frame_shape: Tuple[int,int]) -> dict:
    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    sharp = face_sharpness(gray)
    area_ratio = face_area_ratio(bbox, frame_shape)
    return {"sharpness": sharp, "area_ratio": area_ratio, "raw_sum": sharp + (area_ratio * 1e4)}

def normalize_weights(metrics_list: List[dict]) -> List[float]:
    raw = np.array([m["raw_sum"] for m in metrics_list], dtype=np.float32)
    raw = np.clip(raw, a_min=1e-6, a_max=None)
    exp = np.exp(raw / (np.max(raw)+1e-6))
    weights = exp / (np.sum(exp) + 1e-9)
    return weights.tolist()

def weighted_mean_and_variance(embs: List[np.ndarray], weights: List[float]) -> Tuple[np.ndarray, np.ndarray]:
    embs_arr = np.stack(embs, axis=0)
    weights_arr = np.array(weights, dtype=np.float32).reshape((-1,1))
    mean = np.sum(embs_arr * weights_arr, axis=0) / (np.sum(weights_arr)+1e-9)
    diff = embs_arr - mean
    var = np.sum((diff**2) * weights_arr, axis=0) / (np.sum(weights_arr)+1e-9)
    mean /= (np.linalg.norm(mean) + 1e-6)
    return mean.astype(np.float32), var.astype(np.float32)



# -------------------------
# Routes (UI)
# -------------------------
@app.route("/")
def landing():
    return render_template("landing.html")

@app.route("/admin")
def admin():
    return render_template("admin.html")

@app.route("/login")
def login_page():
    return render_template("login.html")

@app.route("/employee")
def employee():
    return render_template("index.html")

# -------------------------
# Admin: Add new user (only best enrollment image saved)
# -------------------------
@app.route("/admin_add_user", methods=["POST"])
def admin_add_user():
    data = request.get_json()
    user_id = data.get("user_id")
    password = data.get("password")
    images = data.get("images")
    print(f"\n========== [ADMIN ADD USER] ==========")
    print(f"[INFO] Starting enrollment for user_id={user_id}, images_received={len(images)}")
    print(f"\n========== [PROCESSING IMAGES: total={len(images)}] ==========")

    if not user_id or not password or not images:
        return jsonify({"status": "error", "message": "Missing fields"}), 400
    if not isinstance(images, list) or len(images) == 0:
        return jsonify({"status": "error", "message": "images must be list"}), 400

    images = images[:MAX_ENROLL_IMAGES]

    valid_embs = []
    metrics = []
    candidates = []
    # ensure variables are defined for use after the loop (avoids "possibly unbound" warnings)
    label = -1
    value = 0.0
    emb = None

    for idx, img_data in enumerate(images):
        print(f"[DEBUG] Processing enrollment image {idx+1}/{len(images)} for user {user_id}")

        try:
            img_bytes = base64.b64decode(img_data.split(",")[1])
            frame = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
            print(f"[INFO] Frame{idx+1}: Base64 bytes={len(img_bytes)}, Decoded shape={frame.shape}")

        except Exception:
            continue
        faces = retina.detect(frame)

        if faces is None or len(faces) == 0:
            print("[WARN] RetinaFace: No face detected")
            continue


        # Extract bbox
        x1, y1, x2, y2 = faces[0]["bbox"]
        w = x2 - x1
        h = y2 - y1
        x, y = x1, y1
        bbox_norm = [x, y, w, h]
        # Extract 5 landmarks
        landmarks = np.array(faces[0]["landmarks"], dtype=np.float32)

        # Align face
        aligned_face = norm_crop(frame, landmarks, image_size=112)
        

        # NEW CHECK
        if aligned_face is None or aligned_face.shape != (112, 112, 3):
            print("[ERROR] Bad aligned face - skipping frame")
            continue

        # Embed using ArcFace
        emb = get_embedding(aligned_face)

             

        print(f"[INFO] Frame{idx+1}: Detected face bbox=({x},{y},{w},{h}), area_ratio={(w*h)/(frame.shape[0]*frame.shape[1]+1e-9):.4f}")
        prediction = np.zeros((1,3))
        for model_name in os.listdir(ANTI_SPOOF_MODEL_DIR):
            model_path = os.path.join(ANTI_SPOOF_MODEL_DIR, model_name)
            try:
                h_input, w_input, _, _ = parse_model_name(model_name)
                img_rs = cv2.resize(frame, (w_input, h_input))
                prediction += anti_spoof.predict(img_rs, model_path)
               
            except Exception:
                continue

        label = int(np.argmax(prediction))
        value = float(prediction[0][label])
        if not (label == 1 and value > SPOOF_CONF_THRESHOLD):
            continue
        print(f"[INFO] Frame{idx+1}: AntiSpoof label={label}, conf={value:.3f}, result={'REAL' if label==1 else 'SPOOF'}")

        
        if emb is None:
            print(f"[ERROR] Frame{idx+1}: Embedding computation failed.")
        else:
            print(f"[INFO] Frame{idx+1}: Embedding norm={np.linalg.norm(emb):.4f}, first5={np.round(emb[:5],3).tolist()}")
            frame_h, frame_w = frame.shape[:2]
            # Quality metrics based on aligned face
            gray = cv2.cvtColor(aligned_face, cv2.COLOR_BGR2GRAY)
            sharp = face_sharpness(gray)

            # area ratio uses bbox w/h
            area_ratio = (w * h) / (frame_h * frame_w + 1e-9)

            m = {
                "sharpness": sharp,
                "area_ratio": area_ratio,
                "raw_sum": sharp + (area_ratio * 1e4)
                }

            valid_embs.append(emb)
            metrics.append(m)
            candidates.append((m["raw_sum"], aligned_face))

            print(f"[INFO] Frame{idx+1}: Sharpness={m['sharpness']:.2f}, AreaRatio={m['area_ratio']:.4f}, RawSum={m['raw_sum']:.2f}")
    if len(valid_embs) < MIN_VALID_IMAGES:
        return jsonify({"status": "error", "message": "Not enough valid captures"}), 400
    print(f"[INFO] Valid images={len(valid_embs)}, total processed={len(images)}")

    weights = normalize_weights(metrics)
    mean_emb, var_emb = weighted_mean_and_variance(valid_embs, weights)
    reg_quality_score = float(np.mean([m["raw_sum"] for m in metrics]))
    print(f"[INFO] Weighted mean embedding computed (dim={len(mean_emb)})")
    print(f"[DEBUG] mean_emb_norm={np.linalg.norm(mean_emb):.4f}, var_mean={np.mean(var_emb):.4f}, var_std={np.std(var_emb):.4f}")
    print(f"[RESULT] Registration complete for user {user_id}")
    print("")
    # save to DB
    try:
        conn = sqlite3.connect(str(USER_DB))
        cur = conn.cursor()
        cur.execute("INSERT OR REPLACE INTO users (user_id, password, dim, embedding, variance, reg_quality, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (user_id,
             hash_password(password),
             int(mean_emb.shape[0]),
             sqlite3.Binary(mean_emb.tobytes()),
             sqlite3.Binary(var_emb.tobytes()),
             reg_quality_score,
             datetime.datetime.now().isoformat()))
        conn.commit()
        conn.close()
    except Exception as e:
        return jsonify({"status": "error", "message": f"DB error: {str(e)}"}), 500

    # Save only best image
    candidates.sort(key=lambda x: x[0], reverse=True)
    best_path = None
    try:
        user_dir = ENROLL_DIR / user_id
        user_dir.mkdir(parents=True, exist_ok=True)
        best_path = str(user_dir / "best.jpg")
        cv2.imwrite(best_path, candidates[0][1])
    except Exception:
        best_path = None


    return jsonify({"status": "ok", 
                    "user_id": user_id, 
                    "emb_count": len(valid_embs),
                    "reg_quality": reg_quality_score, 
                    "saved_best": best_path})

# -------------------------
# Employee login API
# -------------------------
@app.route("/login", methods=["POST"])
def login():
    data = request.get_json()
    user_id = data.get("user_id")
    password = data.get("password")

    if not user_id or not password:
        return jsonify({"status": "error", "message": "Missing credentials"})

    conn = sqlite3.connect(str(USER_DB))
    cur = conn.cursor()
    cur.execute("SELECT password FROM users WHERE user_id=?", (user_id,))
    row = cur.fetchone()
    conn.close()

    if not row:
        return jsonify({"status": "error", "message": "User not found"})
    if row[0] != hash_password(password):
        return jsonify({"status": "error", "message": "Invalid password"})
    # Simple login result log (avoid referencing enrollment-scoped variables)
    print(f"[RESULT] User logged in: {user_id}")

    return jsonify({"status": "ok", "user_id": user_id})

# -------------------------
# Frame check (suspicious saving as before)
# -------------------------
@app.route("/process_frame", methods=["POST"])


def process_frame():
    
    data = request.get_json()
    frame_data = data.get("frame")
    user_id = data.get("user_id")
    print(f"\n========== [PROCESS FRAME] ==========")
    print(f"[INFO] user_id={user_id}, received frame at {datetime.datetime.now().strftime('%H:%M:%S')}")

    if not frame_data or not user_id:
        return jsonify({"error": "Missing frame or user_id"}), 400

    try:
        img_bytes = base64.b64decode(frame_data.split(",")[1])
        frame = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
        print(f"[INFO] Frame shape: {frame.shape}")
    except Exception as e:
        print(f"[ERROR] Failed to process frame: {e}")
        return jsonify({"error": "Invalid frame data"}), 400

    faces = retina.detect(frame)
    if faces is None or len(faces) == 0:
        print("[WARN] RetinaFace: No face detected")
        return jsonify({"bbox": None, "spoof": "none", "match": False, "saved": False})


    x1, y1, x2, y2 = faces[0]["bbox"]
    w = x2 - x1
    h = y2 - y1
    bbox_norm = [x1, y1, w, h]

    landmarks = np.array(faces[0]["landmarks"], dtype=np.float32)
    aligned_face = norm_crop(frame, landmarks, image_size=112)
    # NEW CHECK
    if aligned_face is None or aligned_face.shape != (112, 112, 3):
        print("[ERROR] Bad aligned face - skipping frame")
        return jsonify({
            "bbox": bbox_norm,
            "spoof": "real",
            "match": False,
            "color": [255,255,0],
            "saved": False
        })


        

    prediction = np.zeros((1,3))
    for model_name in os.listdir(ANTI_SPOOF_MODEL_DIR):
        model_path = os.path.join(ANTI_SPOOF_MODEL_DIR, model_name)
        try:
            h_input, w_input, _, _ = parse_model_name(model_name)
            img_rs = cv2.resize(frame, (w_input, h_input))
            prediction += anti_spoof.predict(img_rs, model_path)
  
        except Exception:
            continue

    label = int(np.argmax(prediction))
    value = float(prediction[0][label])
    print(f"[INFO] AntiSpoof Result: label={label}, conf={value:.3f}, verdict={'REAL' if label==1 else 'SPOOF'}")
    if value < SPOOF_CONF_THRESHOLD:
        print(f"[WARN] Low spoof confidence ({value:.3f}) — likely cause for rejection.")

    now_ts = time.time()
    saved = False
    saved_path = None
    suspicious_type = None
    score = None

    if not (label == 1 and value > SPOOF_CONF_THRESHOLD):
        suspicious_type = "spoof"
        score = float(value)
        print(f"[INFO] AntiSpoof label={label}, conf={value:.3f}")

    
    else:
        # --- SKIP MATCHING DURING REGISTRATION ---
        if user_id == "REGISTRATION":
            print("[INFO] Registration mode: anti-spoof only, no matching")


            
            

            # If no mask → allow
            return jsonify({
                "bbox": bbox_norm,
                "spoof": "real",
                "match": False,
                "color": [0,255,0],
                "saved": False
            })

      
               
        emb = get_embedding(aligned_face)
        print(f"[DEBUG] Using aligned face, shape={aligned_face.shape}")


        if emb is None:
            suspicious_type = "unmatched"
            score = -1.0
        else:
            print(f"[INFO] Embedding computed successfully, norm={np.linalg.norm(emb):.4f}")

            conn = sqlite3.connect(str(USER_DB))
            cur = conn.cursor()
            cur.execute("SELECT dim, embedding, variance FROM users WHERE user_id=?", (user_id,))
            row = cur.fetchone()
            conn.close()
            if not row:
                suspicious_type = "unmatched"
                score = -2.0
            else:
                stored_emb = np.frombuffer(row[1], dtype=np.float32, count=row[0])
                stored_var = np.frombuffer(row[2], dtype=np.float32, count=row[0]) if row[2] else None

                # ---- Compute cosine similarity ----
                cos = float(np.dot(emb, stored_emb) /((np.linalg.norm(emb) + 1e-6) * (np.linalg.norm(stored_emb) + 1e-6)))

                # ---- Apply variance penalty ----
                if stored_var is not None:
                    penalty = float(np.mean(stored_var))
                else:
                    penalty = 0.0

                adjusted_score = cos - penalty
                score = adjusted_score

                print(f"[RESULT] Adjusted similarity score={score:.4f}")

                if score < MATCH_THRESHOLD:
                    print(f"[WARN] Similarity below threshold ({score:.3f} < {MATCH_THRESHOLD}) — likely mismatch.")
                    suspicious_type = "unmatched"
                else:
                    print(f"[INFO] Match successful (score={score:.3f} ≥ {MATCH_THRESHOLD})")

    print(f"[INFO] Cosine score={score:.3f}, match={'YES' if not suspicious_type else 'NO'}")

    if suspicious_type:
        key = (user_id, suspicious_type)
        entry = suspicious_tracker.get(key)
        if entry is None:
            suspicious_tracker[key] = {"first_seen": now_ts, "last_saved": 0}
            entry = suspicious_tracker[key]
        first_seen = entry.get("first_seen", now_ts)
        last_saved = entry.get("last_saved", 0)
        if now_ts - first_seen >= 5 and now_ts - last_saved >= 5:
            user_dir = SUSPICIOUS_DIR / user_id
            user_dir.mkdir(parents=True, exist_ok=True)
            now = datetime.datetime.now()
            date = now.strftime("%Y-%m-%d")
            time_str = now.strftime("%H:%M:%S")
            timestamp = now.strftime("%Y%m%d_%H%M%S")
            filename = f"{suspicious_type}_{timestamp}.jpg"
            filepath = user_dir / filename
            cv2.imwrite(str(filepath), frame)
            saved = True
            saved_path = f"/static/suspicious_frames/{user_id}/{filename}"
            suspicious_tracker[key]["last_saved"] = now_ts
            suspicious_tracker[key]["first_seen"] = now_ts
            log_suspicious(user_id, suspicious_type, saved_path, score=score, date=date, time_str=time_str)
    else:
        keys_to_remove = [k for k in suspicious_tracker.keys() if k[0] == user_id]
        for k in keys_to_remove:
            suspicious_tracker.pop(k, None)

    print(f"[WARN] Suspicious ({suspicious_type}) saved at {saved_path} with score={score:.3f}")

    if suspicious_type == "spoof":
        resp = {"bbox": bbox_norm, "spoof": "spoof", "match": False, "color": [255,0,0], "saved": saved, "score": score}
    elif suspicious_type == "unmatched":
        resp = {"bbox": bbox_norm, "spoof": "real", "match": False, "color": [255,255,0], "saved": saved, "score": score}
    else:
        resp = {"bbox": bbox_norm, "spoof": "real", "match": True, "color": [0,255,0], "saved": False, "score": score}

    if saved and saved_path:
        resp["saved_path"] = saved_path
    return jsonify(resp)

# -------------------------
# Attendance logging endpoint
# Delete suspicious frames on checkout
# -------------------------
@app.route("/mark_attendance", methods=["POST"])
def mark_attendance_endpoint():
    data = request.get_json()
    user_id = data.get("user_id")
    event = data.get("event", "").upper()

    if not user_id:
        return jsonify({"status": "error", "message": "No user_id provided"})
    if event not in ("IN", "OUT"):
        return jsonify({"status": "error", "message": "Invalid event"})

    success, time_str, event = mark_attendance(user_id, event)

    # if checkout → clear suspicious frames
    if success and event == "OUT":
        user_dir = SUSPICIOUS_DIR / user_id
        if user_dir.exists():
            for f in user_dir.glob("*"):
                f.unlink()
            try:
                user_dir.rmdir()
            except Exception:
                pass

    if success:
        return jsonify({"status": "marked", "user_id": user_id, "time": time_str, "event": event})
    else:
        return jsonify({"status": "error", "message": f"Duplicate {event} already recorded"})

# -------------------------
# List users
# -------------------------
@app.route("/list_users", methods=["GET"])
def list_users():
    try:
        conn = sqlite3.connect(str(USER_DB))
        cur = conn.cursor()
        cur.execute("SELECT user_id FROM users")
        rows = cur.fetchall()
        conn.close()
        users = [r[0] for r in rows]
        return jsonify({"users": users})
    except Exception as e:
        return jsonify({"users": [], "error": str(e)})

# -------------------------
# Run app
# -------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=2000, debug=True)
