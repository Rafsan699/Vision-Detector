"""Open-source object detection + vehicle speed estimation (FastAPI + YOLOv8)."""
import base64, os, re, shutil, subprocess, tempfile, threading, uuid
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO

MODEL_NAME = os.getenv("MODEL", "yolov8n.pt")      # try yolov8s.pt / yolov8m.pt for better accuracy
MAX_SECONDS = float(os.getenv("MAX_SECONDS", "30"))  # cap video length (keeps free CPU hosts alive)
VEHICLES = {"bicycle", "car", "motorcycle", "bus", "truck"}
OUT = Path(tempfile.gettempdir()) / "vision_out"
OUT.mkdir(exist_ok=True)

app = FastAPI(title="Vision Detector")
lock = threading.Lock()          # one heavy job at a time
IMAGE_MODEL = os.getenv("IMAGE_MODEL", "yoloe-11s-seg-pf.pt")  # prompt-free: ~4,500 object names


def load_image_model():
    try:
        from ultralytics import YOLOE
        return YOLOE(IMAGE_MODEL) if "yoloe" in IMAGE_MODEL else YOLO(IMAGE_MODEL)
    except Exception as e:  # old ultralytics / download failed -> fall back to 80-class COCO
        print("Could not load", IMAGE_MODEL, "->", e, "; using", MODEL_NAME)
        return YOLO(MODEL_NAME)


detector = load_image_model()


@app.post("/api/detect")
def detect(image: UploadFile = File(...), conf: float = Form(0.35)):
    img = cv2.imdecode(np.frombuffer(image.file.read(), np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "This file is not a readable image.")
    with lock:
        r = detector.predict(img, conf=conf, verbose=False)[0]
    found = [
        {"label": r.names[int(c)], "confidence": round(float(p), 3),
         "box": [round(v) for v in b.tolist()]}
        for b, c, p in zip(r.boxes.xyxy, r.boxes.cls, r.boxes.conf)
    ]
    ok, buf = cv2.imencode(".jpg", r.plot(), [cv2.IMWRITE_JPEG_QUALITY, 90])
    data = "data:image/jpeg;base64," + base64.b64encode(buf).decode()
    return {"image": data, "detections": found}


@app.post("/api/speed")
def speed(video: UploadFile = File(...), line_a: float = Form(40), line_b: float = Form(70),
          distance_m: float = Form(10), conf: float = Form(0.3)):
    """Speed = real distance between two lines / time a vehicle takes to cross both."""
    if distance_m <= 0 or abs(line_a - line_b) < 3:
        raise HTTPException(400, "Distance must be > 0 and the two lines must be apart.")
    src = OUT / f"{uuid.uuid4().hex}_in"
    with open(src, "wb") as f:
        shutil.copyfileobj(video.file, f)
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise HTTPException(400, "This file is not a readable video.")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    w, h = int(cap.get(3)), int(cap.get(4))
    ya, yb = int(h * line_a / 100), int(h * line_b / 100)
    vid = uuid.uuid4().hex
    raw = OUT / f"{vid}_raw.mp4"
    writer = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    prev_y, hit_a, hit_b, kmh, kind = {}, {}, {}, {}, {}

    def crossed(y0, y1, line):
        return (y0 - line) * (y1 - line) <= 0 and y0 != y1

    with lock:
        tracker_model = YOLO(MODEL_NAME)   # fresh instance = fresh tracker state
        ids = [i for i, n in tracker_model.names.items() if n in VEHICLES]
        n = 0
        while n < fps * MAX_SECONDS:
            ok, frame = cap.read()
            if not ok:
                break
            t = n / fps
            r = tracker_model.track(frame, persist=True, conf=conf, classes=ids,
                                    tracker="bytetrack.yaml", verbose=False)[0]
            if r.boxes.id is not None:
                for b, tid, c in zip(r.boxes.xyxy.tolist(), r.boxes.id.int().tolist(),
                                     r.boxes.cls.int().tolist()):
                    x1, y1, x2, y2 = map(int, b)
                    kind[tid] = r.names[c]
                    if tid in prev_y:
                        if tid not in hit_a and crossed(prev_y[tid], y2, ya): hit_a[tid] = t
                        if tid not in hit_b and crossed(prev_y[tid], y2, yb): hit_b[tid] = t
                        if tid not in kmh and tid in hit_a and tid in hit_b:
                            dt = abs(hit_a[tid] - hit_b[tid])
                            if dt > 0.04:
                                kmh[tid] = round(distance_m / dt * 3.6, 1)
                    prev_y[tid] = y2
                    text = f"{kind[tid]} #{tid}" + (f"  {kmh[tid]} km/h" if tid in kmh else "")
                    color = (60, 200, 255) if tid in kmh else (200, 160, 40)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, text, (x1, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX,
                                0.6, color, 2, cv2.LINE_AA)
            for y in (ya, yb):
                cv2.line(frame, (0, y), (w, y), (0, 215, 255), 2)
            writer.write(frame)
            n += 1
    cap.release(); writer.release(); src.unlink(missing_ok=True)

    final = OUT / f"{vid}.mp4"
    if shutil.which("ffmpeg"):   # re-encode to H.264 so browsers can play it
        subprocess.run(["ffmpeg", "-y", "-i", str(raw), "-vcodec", "libx264", "-pix_fmt", "yuv420p",
                        "-movflags", "+faststart", str(final)], capture_output=True)
    if not final.exists():
        raw.rename(final)
    raw.unlink(missing_ok=True)
    vehicles = [{"id": i, "type": kind[i], "kmh": s} for i, s in sorted(kmh.items())]
    return {"video": f"/api/files/{vid}.mp4", "vehicles": vehicles,
            "tracked": len(kind), "seconds": round(n / fps, 1)}


@app.get("/api/files/{name}")
def files(name: str):
    if not re.fullmatch(r"[0-9a-f]{32}\.mp4", name) or not (OUT / name).exists():
        raise HTTPException(404, "File not found.")
    return FileResponse(OUT / name, media_type="video/mp4")


app.mount("/", StaticFiles(directory=Path(__file__).parent / "static", html=True), name="ui")
