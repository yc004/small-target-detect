#!/usr/bin/env python3
"""
Web-based real-time detection demo.

Launches a server that any device (phone, tablet, laptop) can connect to.
Camera feed is streamed via WebSocket to the backend, YOLO inference runs,
and detection results are drawn live on the client.

Usage:
    python scripts/web_demo.py                           # default: stage1, port 8000
    python scripts/web_demo.py --stage stage2_p2         # use specific stage
    python scripts/web_demo.py --weights best.pt --port 8080
    python scripts/web_demo.py --device mps --conf 0.3
    python scripts/web_demo.py --no-ssl                 # localhost dev mode (no HTTPS)

Then open the printed URL on any device with a camera.
NOTE: Camera access requires HTTPS (except on localhost). The server
auto-generates a self-signed certificate. Accept the browser warning.
"""

from __future__ import annotations

import argparse
import asyncio
import datetime
import io
import ipaddress
import json
import logging
import socket
import ssl
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
from PIL import Image

# ── Project path ─────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.model_builder import register_custom_modules

register_custom_modules()

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════════
# HTML Frontend (single-page, embedded)
# ═══════════════════════════════════════════════════════════════════════════════

FRONTEND_HTML = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0, user-scalable=no">
<title>小目标交通标志实时检测</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         background: #0f172a; color: #e2e8f0; overflow: hidden; height: 100dvh; }
  #app { display: flex; flex-direction: column; height: 100%; }

  /* Header */
  #header { padding: 10px 16px; display: flex; align-items: center; gap: 12px;
            background: #1e293b; border-bottom: 1px solid #334155; flex-shrink: 0; }
  #header h1 { font-size: 16px; font-weight: 600; white-space: nowrap; }
  #status { font-size: 12px; padding: 3px 10px; border-radius: 99px; font-weight: 500; }
  .connected { background: #166534; color: #4ade80; }
  .disconnected { background: #7f1d1d; color: #fca5a5; }

  /* Video area */
  #video-container { position: relative; flex: 1; display: flex;
                    align-items: center; justify-content: center;
                    background: #000; overflow: hidden; }
  #canvas { position: absolute; top: 0; left: 0; width: 100%; height: 100%;
            object-fit: contain; }
  #placeholder { color: #475569; font-size: 14px; text-align: center; padding: 24px; }

  /* Controls bar */
  #controls { display: flex; gap: 8px; padding: 10px 16px; flex-wrap: wrap;
              background: #1e293b; border-top: 1px solid #334155; flex-shrink: 0;
              align-items: center; }
  button { padding: 10px 18px; border: none; border-radius: 8px; font-size: 14px;
           font-weight: 600; cursor: pointer; transition: all .15s; }
  button:active { transform: scale(0.97); }
  #btn-start { background: #2563eb; color: #fff; }
  #btn-stop  { background: #dc2626; color: #fff; }
  button:disabled { opacity: 0.4; pointer-events: none; }

  select { padding: 10px 12px; border-radius: 8px; border: 1px solid #334155;
           background: #0f172a; color: #e2e8f0; font-size: 14px; }

  #fps { font-size: 13px; color: #94a3b8; margin-left: auto; white-space: nowrap; }

  /* Stats overlay on video */
  #stats-overlay { position: absolute; top: 8px; left: 8px; font-size: 12px;
                   background: rgba(0,0,0,.6); padding: 6px 10px; border-radius: 6px;
                   color: #f1f5f9; pointer-events: none; }
</style>
</head>
<body>
<div id="app">
  <div id="header">
    <h1>🚦 小目标交通标志实时检测</h1>
    <span id="status" class="disconnected">未连接</span>
  </div>

  <div id="video-container">
    <canvas id="canvas"></canvas>
    <div id="placeholder">📷 点击「开始检测」并授权摄像头</div>
    <div id="stats-overlay"></div>
  </div>

  <div id="controls">
    <select id="camera-select"></select>
    <button id="btn-start" onclick="startDetection()">▶ 开始检测</button>
    <button id="btn-stop" onclick="stopDetection()" disabled>⏹ 停止</button>
    <span id="fps">FPS: --</span>
  </div>
</div>

<script>
const STATE = {
  ws: null,
  stream: null,
  running: false,
  lastFrameTime: 0,
  fps: 0,
  frameCount: 0,
  fpsTimer: 0,
  targetFps: 15,        // send at most N frames/sec to server
  frameInterval: 1000 / 15,
};

const $ = id => document.getElementById(id);

// ── Camera enumeration ────────────────────────────────────────────────
async function listCameras() {
  try {
    // Need to request once to get labels on some browsers
    await navigator.mediaDevices.getUserMedia({video: true})
      .then(s => { s.getTracks().forEach(t => t.stop()); })
      .catch(() => {});
    const devices = await navigator.mediaDevices.enumerateDevices();
    const videos = devices.filter(d => d.kind === 'videoinput');
    const sel = $('camera-select');
    sel.innerHTML = '';
    videos.forEach((d, i) => {
      const opt = document.createElement('option');
      opt.value = d.deviceId;
      opt.textContent = d.label || `摄像头 ${i + 1}`;
      sel.appendChild(opt);
    });
  } catch (e) {
    console.warn('Cannot enumerate cameras:', e);
  }
}
listCameras();

// ── Start ──────────────────────────────────────────────────────────────
async function startDetection() {
  if (STATE.running) return;

  // Stop any existing stream
  if (STATE.stream) {
    STATE.stream.getTracks().forEach(t => t.stop());
  }

  const deviceId = $('camera-select').value;
  const constraints = {
    video: {
      deviceId: deviceId ? { exact: deviceId } : undefined,
      width: { ideal: 640 },
      height: { ideal: 480 },
      facingMode: 'environment',   // prefer rear camera on mobile
    },
    audio: false,
  };

  try {
    STATE.stream = await navigator.mediaDevices.getUserMedia(constraints);
  } catch (e) {
    alert('无法访问摄像头: ' + e.message);
    return;
  }

  // Connect WebSocket
  const proto = location.protocol === 'https:' ? 'wss' : 'ws';
  const wsUrl = `${proto}://${location.host}/ws`;
  STATE.ws = new WebSocket(wsUrl);
  STATE.ws.binaryType = 'arraybuffer';

  STATE.ws.onopen = () => {
    $('status').textContent = '已连接';
    $('status').className = 'connected';
    $('btn-start').disabled = true;
    $('btn-stop').disabled = false;
    $('placeholder').style.display = 'none';
    STATE.running = true;
    STATE.frameCount = 0;
    STATE.fpsTimer = performance.now();
    setupVideoPipeline();
    sendLoop();
  };

  STATE.ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    // Always update stats overlay immediately
    $('stats-overlay').textContent =
      `服务端 FPS: ${(data.server_fps || 0).toFixed(1)}  |  检出: ${(data.detections || []).length} 个目标`;
    // Keep last known boxes — only clear if empty after persistent expiry
    if (data.detections && data.detections.length > 0) {
      STATE.persistentDetections = data;
      STATE.lastBoxTime = performance.now();
    }
    // If empty, let old boxes stay for HOLD_DURATION before clearing
  };

  STATE.ws.onclose = () => {
    $('status').textContent = '未连接';
    $('status').className = 'disconnected';
    resetUI();
  };

  STATE.ws.onerror = () => {
    STATE.ws?.close();
  };
}

// ── Video pipeline ─────────────────────────────────────────────────────
function setupVideoPipeline() {
  const canvas = $('canvas');
  const ctx = canvas.getContext('2d');

  const video = document.createElement('video');
  video.srcObject = STATE.stream;
  video.playsInline = true;
  video.muted = true;
  video.play();

  // Resize canvas when video metadata loads
  video.addEventListener('loadedmetadata', () => {
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
  });

  // Draw video frame + persistent detection boxes each animation frame
  const HOLD_DURATION = 800;  // ms to keep old boxes after target disappears
  function drawVideo() {
    if (!STATE.running) return;
    if (video.readyState >= video.HAVE_CURRENT_DATA) {
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
      // Draw last known boxes — they persist between server responses
      if (STATE.persistentDetections) {
        const age = performance.now() - (STATE.lastBoxTime || 0);
        if (age < HOLD_DURATION) {
          drawBoxes(ctx, canvas.width, canvas.height, STATE.persistentDetections);
        } else {
          STATE.persistentDetections = null;
        }
      }
    }
    requestAnimationFrame(drawVideo);
  }
  drawVideo();

  // Store video reference for capture
  STATE.videoElement = video;
}

// ── Send frames to server (throttled) ──────────────────────────────────
function sendLoop() {
  if (!STATE.running) return;

  const now = performance.now();
  if (now - STATE.lastFrameTime >= STATE.frameInterval) {
    STATE.lastFrameTime = now;

    const canvas = $('canvas');
    canvas.toBlob(blob => {
      if (blob && STATE.ws?.readyState === WebSocket.OPEN) {
        STATE.ws.send(blob);
        STATE.frameCount++;
      }
    }, 'image/jpeg', 0.75);
  }

  requestAnimationFrame(sendLoop);
}

// ── FPS counter ────────────────────────────────────────────────────────
setInterval(() => {
  const now = performance.now();
  const elapsed = (now - STATE.fpsTimer) / 1000;
  if (elapsed >= 2.0) {
    STATE.fps = Math.round(STATE.frameCount / elapsed);
    STATE.frameCount = 0;
    STATE.fpsTimer = now;
    $('fps').textContent = `发送 FPS: ${STATE.fps}`;
  }
}, 2000);

// ── Draw detection boxes (called from animation loop) ─────────────────
const COLORS = ['#ef4444','#f97316','#eab308','#22c55e','#3b82f6',
                '#8b5cf6','#ec4899','#06b6d4','#84cc16','#f43f5e'];

function drawBoxes(ctx, w, h, data) {
  const detections = data.detections || [];
  for (const det of detections) {
    const [x1, y1, x2, y2] = det.bbox;
    const color = COLORS[det.class_id % COLORS.length];

    // Bounding box
    ctx.strokeStyle = color;
    ctx.lineWidth = Math.max(2, w / 400);
    ctx.strokeRect(x1 * w, y1 * h, (x2 - x1) * w, (y2 - y1) * h);

    // Label
    const label = `${det.name} ${(det.conf * 100).toFixed(0)}%`;
    const fontSize = Math.max(12, w / 50);
    ctx.font = `${fontSize}px system-ui, -apple-system, sans-serif`;
    const metrics = ctx.measureText(label);
    const lx = x1 * w;
    const ly = Math.max(0, y1 * h - fontSize * 1.6);
    ctx.fillStyle = color;
    ctx.fillRect(lx, ly, metrics.width + 8, fontSize * 1.6);
    ctx.fillStyle = '#fff';
    ctx.fillText(label, lx + 4, ly + fontSize * 1.1);
  }
}

// ── Stop ───────────────────────────────────────────────────────────────
function stopDetection() {
  STATE.running = false;
  if (STATE.ws) { STATE.ws.close(); STATE.ws = null; }
  if (STATE.stream) {
    STATE.stream.getTracks().forEach(t => t.stop());
    STATE.stream = null;
  }
  resetUI();
}

function resetUI() {
  $('btn-start').disabled = false;
  $('btn-stop').disabled = true;
  $('placeholder').style.display = 'block';
  $('stats-overlay').textContent = '';
  $('fps').textContent = 'FPS: --';
  STATE.persistentDetections = null;
  STATE.lastBoxTime = 0;
  // Clear canvas
  const ctx = $('canvas').getContext('2d');
  ctx.clearRect(0, 0, $('canvas').width, $('canvas').height);
}
</script>
</body>
</html>"""

# ═══════════════════════════════════════════════════════════════════════════════
# FastAPI Application
# ═══════════════════════════════════════════════════════════════════════════════

app = FastAPI(title="Small Target Detection Web Demo")

# Global model reference (set on startup)
MODEL: YOLO | None = None
CONF_THRESH: float = 0.25
IOU_THRESH: float = 0.5
IMG_SIZE: int = 640
COLORS: list[tuple[int, int, int]] = [
    (239, 68, 68), (249, 115, 22), (234, 179, 8),
    (34, 197, 94), (59, 130, 246), (139, 92, 246),
    (236, 72, 153), (6, 182, 212), (132, 204, 22),
    (244, 63, 94),
]


@app.get("/", response_class=HTMLResponse)
async def index():
    return HTMLResponse(content=FRONTEND_HTML)


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    logger.info(f"Client connected: {ws.client.host}")

    last_time = time.perf_counter()
    frame_count = 0
    fps = 0.0

    try:
        while True:
            # Receive JPEG frame bytes
            data = await ws.receive_bytes()

            # Decode image
            try:
                img = Image.open(io.BytesIO(data)).convert("RGB")
            except Exception:
                continue

            # Run inference
            results = MODEL(
                img,
                imgsz=IMG_SIZE,
                conf=CONF_THRESH,
                iou=IOU_THRESH,
                verbose=False,
            )

            # Build detection list
            detections = []
            if results and results[0].boxes is not None:
                boxes = results[0].boxes
                for i in range(len(boxes)):
                    xyxy = boxes.xyxy[i].cpu().tolist()
                    conf = float(boxes.conf[i])
                    cls_id = int(boxes.cls[i])
                    name = MODEL.names.get(cls_id, f"cls_{cls_id}")
                    # Normalize to [0, 1]
                    w, h = img.size
                    bbox_norm = [
                        xyxy[0] / w, xyxy[1] / h,
                        xyxy[2] / w, xyxy[3] / h,
                    ]
                    detections.append({
                        "bbox": bbox_norm,
                        "class_id": cls_id,
                        "name": name,
                        "conf": round(conf, 4),
                    })

            # Server-side FPS
            frame_count += 1
            now = time.perf_counter()
            if now - last_time >= 2.0:
                fps = frame_count / (now - last_time)
                frame_count = 0
                last_time = now

            await ws.send_text(json.dumps({
                "detections": detections,
                "server_fps": round(fps, 1),
            }))

    except WebSocketDisconnect:
        logger.info(f"Client disconnected: {ws.client.host}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        try:
            await ws.close()
        except Exception:
            pass


# ═══════════════════════════════════════════════════════════════════════════════
# CLI Entry Point
# ═══════════════════════════════════════════════════════════════════════════════

def _generate_ssl_cert(cert_file: Path, key_file: Path) -> None:
    """Generate a self-signed SSL certificate valid for local LAN IPs."""
    from cryptography import x509
    from cryptography.x509.oid import NameOID
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa

    # Generate private key
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)

    # Get local IPs for SAN
    hostname = socket.gethostname()
    try:
        local_ip = socket.gethostbyname(hostname)
    except Exception:
        local_ip = "127.0.0.1"

    subject = issuer = x509.Name([
        x509.NameAttribute(NameOID.COMMON_NAME, local_ip),
        x509.NameAttribute(NameOID.ORGANIZATION_NAME, "WebDemo"),
    ])

    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(datetime.datetime.now(datetime.UTC))
        .not_valid_after(datetime.datetime.now(datetime.UTC) + datetime.timedelta(days=365))
        .add_extension(
            x509.SubjectAlternativeName([
                x509.DNSName("localhost"),
                x509.IPAddress(ipaddress.IPv4Address("127.0.0.1")),
                x509.DNSName(local_ip),
                x509.IPAddress(ipaddress.IPv4Address(local_ip)),
            ]),
            critical=False,
        )
        .sign(key, hashes.SHA256())
    )

    # Write key
    key_file.write_bytes(key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.TraditionalOpenSSL,
        encryption_algorithm=serialization.NoEncryption(),
    ))

    # Write cert
    cert_file.write_bytes(cert.public_bytes(serialization.Encoding.PEM))

    logger.info(f"SSL certificate generated: {cert_file}")


def find_best_pt(stage: str) -> Path | None:
    """Find best.pt for a given stage name."""
    p = PROJECT_ROOT / "experiments" / stage / "train" / "weights" / "best.pt"
    if p.exists():
        return p
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Web-based real-time detection demo",
    )
    parser.add_argument(
        "--stage", type=str, default="stage1_baseline",
        help="Stage name (auto-discovers best.pt under experiments/). Default: stage1_baseline",
    )
    parser.add_argument(
        "--weights", type=Path, default=None,
        help="Direct path to .pt file (overrides --stage)",
    )
    parser.add_argument(
        "--port", type=int, default=8000,
        help="Server port. Default: 8000",
    )
    parser.add_argument(
        "--host", type=str, default="0.0.0.0",
        help="Bind address. Default: 0.0.0.0 (accessible from LAN)",
    )
    parser.add_argument(
        "--device", type=str, default="cpu",
        help="Inference device. Default: cpu (use '0' for CUDA, 'mps' for Mac)",
    )
    parser.add_argument(
        "--imgsz", type=int, default=640,
        help="Inference image size. Default: 640",
    )
    parser.add_argument(
        "--conf", type=float, default=0.25,
        help="Confidence threshold. Default: 0.25",
    )
    parser.add_argument(
        "--iou", type=float, default=0.5,
        help="IoU threshold for NMS. Default: 0.5",
    )
    parser.add_argument(
        "--no-ssl", action="store_true",
        help="Disable HTTPS (only safe for localhost dev; cameras will NOT work on remote devices)",
    )

    args = parser.parse_args()

    # ── Resolve weights ──────────────────────────────────────────────
    global MODEL, CONF_THRESH, IOU_THRESH, IMG_SIZE

    CONF_THRESH = args.conf
    IOU_THRESH = args.iou
    IMG_SIZE = args.imgsz

    if args.weights:
        weights_path = args.weights
    else:
        weights_path = find_best_pt(args.stage)
        if weights_path is None:
            logger.error(f"No best.pt found for stage '{args.stage}'")
            logger.info("Tip: use --weights to specify path directly, or run training first.")
            sys.exit(1)

    if not weights_path.exists():
        logger.error(f"Weights file not found: {weights_path}")
        sys.exit(1)

    # ── Load model ───────────────────────────────────────────────────
    logger.info(f"Loading model: {weights_path}")
    logger.info(f"Device: {args.device}, imgsz: {args.imgsz}, conf: {args.conf}")

    MODEL = YOLO(str(weights_path))
    if args.device != "cpu":
        MODEL.to(args.device)

    logger.info(f"Model ready. {MODEL.names}")

    # ── SSL certificate ──────────────────────────────────────────────
    ssl_certfile = None
    ssl_keyfile = None
    use_ssl = not args.no_ssl

    if use_ssl:
        cert_dir = PROJECT_ROOT / ".certs"
        cert_dir.mkdir(exist_ok=True)
        cert_file = cert_dir / "webdemo.pem"
        key_file = cert_dir / "webdemo.key"

        if not (cert_file.exists() and key_file.exists()):
            logger.info("Generating self-signed SSL certificate ...")
            try:
                _generate_ssl_cert(cert_file, key_file)
                cert_file.chmod(0o600)
                key_file.chmod(0o600)
            except Exception as e:
                logger.error(f"Failed to generate SSL cert: {e}")
                logger.info("Falling back to HTTP. Camera will only work on localhost.")
                use_ssl = False

        if use_ssl:
            ssl_certfile = str(cert_file)
            ssl_keyfile = str(key_file)

    # ── Get local IP ─────────────────────────────────────────────────
    hostname = socket.gethostname()
    try:
        local_ip = socket.gethostbyname(hostname)
    except Exception:
        local_ip = "127.0.0.1"

    scheme = "https" if use_ssl else "http"
    protocol = "wss" if use_ssl else "ws"

    # Update the frontend WebSocket URL to use the correct protocol
    global FRONTEND_HTML
    FRONTEND_HTML = FRONTEND_HTML.replace(
        "const proto = location.protocol === 'https:' ? 'wss' : 'ws';",
        f"const proto = '{protocol}';"
    )

    print()
    print("=" * 60)
    print("  🚦 Web Demo Server Ready")
    print("=" * 60)
    print(f"  Local:    {scheme}://localhost:{args.port}")
    if args.host == "0.0.0.0":
        print(f"  Network:  {scheme}://{local_ip}:{args.port}")
    print()
    print("  Open the "+("Network" if args.host == "0.0.0.0" else "Local")+" URL on any device with a camera.")
    if use_ssl:
        print("  ⚠️  Accept the self-signed certificate warning in your browser.")
    else:
        print("  ⚠️  HTTP mode — camera only works on localhost, NOT on phones/tablets.")
    print("=" * 60)
    print()

    # ── Start server ─────────────────────────────────────────────────
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        ssl_certfile=ssl_certfile,
        ssl_keyfile=ssl_keyfile,
        log_level="warning",
        ws_ping_interval=30,
        ws_ping_timeout=10,
    )


if __name__ == "__main__":
    main()
