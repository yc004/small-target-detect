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

FRONTEND_HTML = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0, user-scalable=no, viewport-fit=cover">
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
<title>交通标志检测</title>
<style>
  :root {
    --safe-top: env(safe-area-inset-top, 0px);
    --safe-bottom: env(safe-area-inset-bottom, 0px);
    --bg: #0a0a0f;
    --surface: rgba(20,20,35,0.92);
    --text: #e8e8f0;
    --muted: #8888a0;
    --accent: #3b82f6;
    --danger: #ef4444;
    --success: #22c55e;
    --radius: 14px;
  }

  * { margin:0; padding:0; box-sizing:border-box; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, 'SF Pro Display', 'Segoe UI', sans-serif;
    background: var(--bg); color: var(--text);
    overflow: hidden; height: 100dvh; touch-action: manipulation;
    -webkit-tap-highlight-color: transparent;
  }

  /* ════════════════════════════════════════════════════════════════════
     Video — full screen background
     ════════════════════════════════════════════════════════════════════ */
  #video-container {
    position: fixed; inset: 0; background: #000;
  }
  #live-video {
    width: 100%; height: 100%; object-fit: contain; display: block;
  }
  #box-overlay {
    position: absolute; inset: 0; width: 100%; height: 100%;
    object-fit: contain; pointer-events: none;
  }

  /* ════════════════════════════════════════════════════════════════════
     Top bar — status line
     ════════════════════════════════════════════════════════════════════ */
  #top-bar {
    position: fixed; top: 0; left: 0; right: 0; z-index: 20;
    padding: calc(8px + var(--safe-top)) 12px 8px;
    background: linear-gradient(rgba(0,0,0,0.6) 60%, transparent);
    display: flex; align-items: center; gap: 8px;
    transition: opacity 0.3s, transform 0.3s;
  }
  #top-bar.hidden { opacity: 0; transform: translateY(-100%); pointer-events: none; }

  #top-bar .title { font-size: 14px; font-weight: 600; letter-spacing: -0.01em; }
  #top-bar .badge {
    font-size: 11px; padding: 3px 9px; border-radius: 99px; font-weight: 600;
  }
  .badge-live { background: var(--success); color: #000; animation: pulse 2s infinite; }
  .badge-off  { background: rgba(255,255,255,0.12); color: var(--muted); }
  @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.6} }

  /* ════════════════════════════════════════════════════════════════════
     Stats pill — detection count + FPS
     ════════════════════════════════════════════════════════════════════ */
  #stats-pill {
    position: fixed; top: calc(52px + var(--safe-top)); left: 50%;
    transform: translateX(-50%); z-index: 20;
    background: var(--surface); backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 99px; padding: 6px 14px; font-size: 12px;
    display: flex; gap: 12px; align-items: center;
    white-space: nowrap; transition: opacity 0.3s;
  }
  #stats-pill.hidden { opacity: 0; pointer-events: none; }
  #stats-pill .det-count { color: var(--accent); font-weight: 700; }

  /* ════════════════════════════════════════════════════════════════════
     Placeholder
     ════════════════════════════════════════════════════════════════════ */
  #placeholder {
    position: fixed; inset: 0; display: flex; flex-direction: column;
    align-items: center; justify-content: center; z-index: 5;
    color: var(--muted); gap: 12px; text-align: center; padding: 20px;
  }
  #placeholder .icon { font-size: 48px; opacity: 0.5; }
  #placeholder .text { font-size: 15px; line-height: 1.5; }

  /* ════════════════════════════════════════════════════════════════════
     FAB — primary action button (bottom-right)
     ════════════════════════════════════════════════════════════════════ */
  #fab {
    position: fixed; bottom: calc(24px + var(--safe-bottom));
    right: 20px; z-index: 30;
    width: 60px; height: 60px; border-radius: 50%;
    border: none; cursor: pointer;
    display: flex; align-items: center; justify-content: center;
    font-size: 26px; color: #fff;
    box-shadow: 0 4px 24px rgba(0,0,0,0.5);
    transition: all 0.25s cubic-bezier(0.4,0,0.2,1);
    -webkit-tap-highlight-color: transparent;
  }
  #fab:active { transform: scale(0.92); }
  #fab.start { background: var(--accent); }
  #fab.stop  { background: var(--danger); }

  #fab-label {
    position: fixed; bottom: calc(34px + var(--safe-bottom));
    right: 88px; z-index: 30;
    font-size: 13px; font-weight: 600; color: #fff;
    background: rgba(0,0,0,0.7); padding: 6px 12px; border-radius: 99px;
    pointer-events: none; opacity: 0;
    transition: opacity 0.2s;
  }
  #fab:hover + #fab-label,
  #fab:active + #fab-label { opacity: 1; }

  /* ════════════════════════════════════════════════════════════════════
     Bottom sheet — camera picker + settings
     ════════════════════════════════════════════════════════════════════ */
  #bottom-sheet {
    position: fixed; bottom: 0; left: 0; right: 0; z-index: 25;
    background: var(--surface); backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border-top: 1px solid rgba(255,255,255,0.08);
    border-radius: 20px 20px 0 0;
    padding: 8px 16px calc(12px + var(--safe-bottom));
    display: flex; align-items: center; gap: 10px;
    transform: translateY(0);
    transition: transform 0.35s cubic-bezier(0.4,0,0.2,1);
  }
  #bottom-sheet.collapsed { transform: translateY(calc(100% - 52px)); }

  #bottom-sheet .handle {
    position: absolute; top: 6px; left: 50%; transform: translateX(-50%);
    width: 36px; height: 4px; border-radius: 99px;
    background: rgba(255,255,255,0.2);
  }

  #bottom-sheet select {
    flex: 1; min-width: 0;
    padding: 10px 12px; border-radius: var(--radius);
    border: 1px solid rgba(255,255,255,0.1);
    background: rgba(255,255,255,0.06); color: var(--text);
    font-size: 14px; -webkit-appearance: none;
    appearance: none;
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='12' height='12' viewBox='0 0 24 24' fill='none' stroke='%238888a0' stroke-width='2'%3E%3Cpath d='M6 9l6 6 6-6'/%3E%3C/svg%3E");
    background-repeat: no-repeat;
    background-position: right 10px center;
    padding-right: 30px;
  }

  #bottom-sheet button {
    padding: 10px 16px; border-radius: var(--radius);
    border: none; font-size: 13px; font-weight: 600;
    cursor: pointer; white-space: nowrap;
    background: rgba(255,255,255,0.08); color: var(--text);
    transition: all 0.15s;
  }
  #bottom-sheet button:active { transform: scale(0.96); }
  #bottom-sheet button.accent { background: var(--accent); color: #fff; }

  /* ════════════════════════════════════════════════════════════════════
     Toast
     ════════════════════════════════════════════════════════════════════ */
  #toast {
    position: fixed; top: 50%; left: 50%; transform: translate(-50%,-50%) scale(0.8);
    background: rgba(0,0,0,0.85); color: #fff;
    padding: 10px 20px; border-radius: 99px; font-size: 14px; font-weight: 600;
    z-index: 40; pointer-events: none; opacity: 0;
    transition: all 0.2s;
  }
  #toast.show { opacity: 1; transform: translate(-50%,-50%) scale(1); }
</style>
</head>
<body>

<div id="video-container">
  <video id="live-video" autoplay playsinline muted></video>
  <canvas id="box-overlay"></canvas>
</div>

<div id="top-bar">
  <span class="title">🚦 交通标志检测</span>
  <span class="badge badge-off" id="badge">待机</span>
</div>

<div id="stats-pill" class="hidden">
  <span>检出 <span class="det-count" id="det-num">0</span> 个目标</span>
  <span style="color:var(--muted)">|</span>
  <span id="sv-fps">0 fps</span>
</div>

<div id="placeholder">
  <span class="icon">📷</span>
  <span class="text">点击下方按钮开始检测<br>需要授权摄像头访问</span>
</div>

<button id="fab" class="start" onclick="handleFab()">▶</button>
<span id="fab-label">开始检测</span>

<div id="bottom-sheet" class="collapsed">
  <div class="handle" onclick="toggleSheet()"></div>
  <select id="camera-select"></select>
  <button id="btn-immersive" onclick="toggleImmersive()">⛶</button>
</div>

<div id="toast"></div>

<script>
const $ = id => document.getElementById(id);

const STATE = {
  ws: null, stream: null, running: false,
  frameCount: 0, fps: 0, fpsTimer: 0,
  frameInterval: 1000 / 15,
  immersive: false,
};

// ═══════════════════════════════════════════════════════════════════════
// Toast
// ═══════════════════════════════════════════════════════════════════════
function toast(msg, ms=1500) {
  const t = $('toast');
  t.textContent = msg; t.classList.add('show');
  clearTimeout(t._timer);
  t._timer = setTimeout(() => t.classList.remove('show'), ms);
}

// ═══════════════════════════════════════════════════════════════════════
// Bottom sheet
// ═══════════════════════════════════════════════════════════════════════
function toggleSheet() {
  $('bottom-sheet').classList.toggle('collapsed');
}

// ═══════════════════════════════════════════════════════════════════════
// Immersive mode
// ═══════════════════════════════════════════════════════════════════════
function toggleImmersive() {
  STATE.immersive = !STATE.immersive;
  const hide = STATE.immersive;
  $('top-bar').classList.toggle('hidden', hide);
  if (STATE.running) $('stats-pill').classList.toggle('hidden', hide);
  $('bottom-sheet').classList.toggle('collapsed', hide);
  $('btn-immersive').textContent = hide ? '⛶' : '⛶';
  $('btn-immersive').style.background = hide ? 'var(--accent)' : '';
  toast(hide ? '沉浸模式' : '退出沉浸');
}

// ═══════════════════════════════════════════════════════════════════════
// Camera list
// ═══════════════════════════════════════════════════════════════════════
async function listCameras() {
  try {
    await navigator.mediaDevices.getUserMedia({video:true})
      .then(s => s.getTracks().forEach(t => t.stop())).catch(()=>{});
    const devices = await navigator.mediaDevices.enumerateDevices();
    const sel = $('camera-select');
    sel.innerHTML = '';
    devices.filter(d => d.kind === 'videoinput').forEach((d,i) => {
      const o = document.createElement('option');
      o.value = d.deviceId;
      o.textContent = d.label || `摄像头 ${i+1}`;
      sel.appendChild(o);
    });
  } catch(e) { console.warn('camera enum:', e); }
}
listCameras();

// ═══════════════════════════════════════════════════════════════════════
// FAB handler
// ═══════════════════════════════════════════════════════════════════════
function handleFab() {
  STATE.running ? stopDetection() : startDetection();
}

// ═══════════════════════════════════════════════════════════════════════
// Start
// ═══════════════════════════════════════════════════════════════════════
async function startDetection() {
  if (STATE.running) return;
  if (STATE.stream) { STATE.stream.getTracks().forEach(t => t.stop()); }

  const constraints = {
    video: {
      deviceId: $('camera-select').value ? {exact:$('camera-select').value} : undefined,
      width: {ideal:640}, height: {ideal:480},
      facingMode: 'environment',
    },
    audio: false,
  };

  try {
    STATE.stream = await navigator.mediaDevices.getUserMedia(constraints);
  } catch(e) { alert('无法访问摄像头: '+e.message); return; }

  const proto = location.protocol==='https:'?'wss':'ws';
  STATE.ws = new WebSocket(`${proto}://${location.host}/ws`);
  STATE.ws.binaryType = 'arraybuffer';

  STATE.ws.onopen = () => {
    STATE.running = true; STATE.frameCount = 0;
    STATE.fpsTimer = performance.now();

    // UI state
    $('fab').className = 'stop'; $('fab').textContent = '⏹';
    $('fab-label').textContent = '停止检测';
    $('badge').textContent = '检测中'; $('badge').className = 'badge badge-live';
    $('placeholder').style.display = 'none';
    $('stats-pill').classList.remove('hidden');
    $('bottom-sheet').classList.add('collapsed');
    $('btn-immersive').style.display = '';

    setupVideo();
    sendLoop();
  };

  STATE.ws.onmessage = (e) => {
    const data = JSON.parse(e.data);
    const n = (data.detections||[]).length;
    $('det-num').textContent = n;
    $('sv-fps').textContent = (data.server_fps||0).toFixed(1)+' fps';

    const ctx = STATE.overlayCtx, ov = STATE.overlayEl;
    if (ctx && ov) {
      ctx.clearRect(0,0,ov.width,ov.height);
      drawBoxes(ctx, ov.width, ov.height, data);
    }
  };

  STATE.ws.onclose = () => { toast('连接断开'); resetAll(); };
  STATE.ws.onerror = () => STATE.ws?.close();
}

// ═══════════════════════════════════════════════════════════════════════
// Video setup
// ═══════════════════════════════════════════════════════════════════════
function setupVideo() {
  const v = $('live-video'), ov = $('box-overlay');
  v.srcObject = STATE.stream; v.playsInline = true; v.muted = true;
  v.play().catch(e => console.warn(e));
  v.addEventListener('loadedmetadata', () => {
    ov.width = v.videoWidth; ov.height = v.videoHeight;
  });
  STATE.videoEl = v; STATE.overlayEl = ov;
  STATE.overlayCtx = ov.getContext('2d');
}

// ═══════════════════════════════════════════════════════════════════════
// Send loop
// ═══════════════════════════════════════════════════════════════════════
let _sendTimer = null;
function sendLoop() {
  clearInterval(_sendTimer);
  _sendTimer = setInterval(() => {
    if (!STATE.running || STATE.ws?.readyState!==WebSocket.OPEN) return;
    const v = STATE.videoEl;
    if (!v || v.readyState < v.HAVE_CURRENT_DATA) return;
    const c = document.createElement('canvas');
    c.width=v.videoWidth; c.height=v.videoHeight;
    c.getContext('2d').drawImage(v,0,0);
    c.toBlob(b => {
      if (b && STATE.ws?.readyState===WebSocket.OPEN) {
        STATE.ws.send(b); STATE.frameCount++;
      }
    },'image/jpeg',0.75);
  }, STATE.frameInterval);
}

// ═══════════════════════════════════════════════════════════════════════
// FPS
// ═══════════════════════════════════════════════════════════════════════
setInterval(() => {
  if (!STATE.running) return;
  const now = performance.now(), el = (now-STATE.fpsTimer)/1000;
  if (el >= 2) {
    STATE.fps = Math.round(STATE.frameCount/el);
    STATE.frameCount=0; STATE.fpsTimer=now;
  }
},2000);

// ═══════════════════════════════════════════════════════════════════════
// Draw boxes
// ═══════════════════════════════════════════════════════════════════════
const COLORS = ['#ef4444','#f97316','#eab308','#22c55e','#3b82f6',
                '#8b5cf6','#ec4899','#06b6d4','#84cc16','#f43f5e'];

function drawBoxes(ctx, w, h, data) {
  for (const d of (data.detections||[])) {
    const [x1,y1,x2,y2] = d.bbox;
    const c = COLORS[d.class_id % COLORS.length];
    ctx.strokeStyle = c;
    ctx.lineWidth = Math.max(2.5, w/350);
    ctx.strokeRect(x1*w, y1*h, (x2-x1)*w, (y2-y1)*h);

    const label = `${d.name} ${(d.conf*100)|0}%`;
    const fs = Math.max(13, w/45);
    ctx.font = `600 ${fs}px -apple-system, sans-serif`;
    const m = ctx.measureText(label);
    const lx = x1*w, ly = Math.max(0, y1*h - fs*1.7);
    ctx.fillStyle = c; ctx.fillRect(lx, ly, m.width+8, fs*1.7);
    ctx.fillStyle = '#fff'; ctx.fillText(label, lx+4, ly+fs*1.2);
  }
}

// ═══════════════════════════════════════════════════════════════════════
// Stop
// ═══════════════════════════════════════════════════════════════════════
function stopDetection() {
  STATE.running = false;
  clearInterval(_sendTimer); _sendTimer = null;
  if (STATE.ws) { STATE.ws.close(); STATE.ws = null; }
  if (STATE.stream) { STATE.stream.getTracks().forEach(t => t.stop()); STATE.stream = null; }
  $('live-video').srcObject = null;
  resetAll();
}

function resetAll() {
  $('fab').className = 'start'; $('fab').textContent = '▶';
  $('fab-label').textContent = '开始检测';
  $('badge').textContent = '待机'; $('badge').className = 'badge badge-off';
  $('placeholder').style.display = '';
  $('stats-pill').classList.add('hidden');
  $('det-num').textContent = '0'; $('sv-fps').textContent = '0 fps';
  $('bottom-sheet').classList.remove('collapsed');
  STATE.immersive = false;
  $('top-bar').classList.remove('hidden');
  $('btn-immersive').style.background = '';
  const ov = $('box-overlay');
  if (ov) ov.getContext('2d').clearRect(0,0,ov.width,ov.height);
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
