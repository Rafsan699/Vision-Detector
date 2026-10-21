---
title: Vision Detector
emoji: 🚗
sdk: docker
app_port: 7860
---
# Vision Detector

Open-source object detection (80 COCO classes) and vehicle speed estimation. FastAPI + YOLOv8 backend, single-file HTML frontend.

## Run locally
    pip install -r requirements.txt
    uvicorn app:app --port 8000      # open http://localhost:8000

## Run on Google Colab (public link, no signup)
    !git clone https://github.com/YOUR_NAME/vision-detector && cd vision-detector && pip -q install -r requirements.txt
    !wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -O cf && chmod +x cf
    import subprocess; subprocess.Popen("cd vision-detector && uvicorn app:app --port 8000", shell=True)
    !./cf tunnel --url http://localhost:8000      # prints a https://...trycloudflare.com link

## Deploy permanently (free): Hugging Face Spaces
Create a new Space, choose Docker, push these files. The Dockerfile is ready.

## Notes
- Image tab uses `IMAGE_MODEL` (default `yoloe-11s-seg-pf.pt`, detects thousands of object types; try `yoloe-11l-seg-pf.pt` for accuracy). The speed tab uses `MODEL` (COCO vehicles).
- Set `MODEL=yolov8s.pt` for better accuracy, `MAX_SECONDS` to change the video cap.
- Speed uses two lines and a known real distance; it needs a fixed camera. Estimates only.
- Licence note: Ultralytics YOLO is AGPL-3.0, so publish your project under AGPL-3.0 too.
# Vision-Detector
