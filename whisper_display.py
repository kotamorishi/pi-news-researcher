#!/usr/bin/env python3
"""Continuous speech recognition with Hailo Whisper → LED display + SQLite logging."""

import os
import sys
import time
import logging
import threading
import sqlite3
import json
import numpy as np
import sounddevice as sd
from datetime import datetime
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import io
import wave
import requests
import webrtcvad
import socket
from pathlib import Path
from resemblyzer import VoiceEncoder, preprocess_wav

sys.path.insert(0, "/home/kota/galactic-unicorn-horn")
from renderer import render_text_to_bitmap_payload

from hailo_platform import VDevice
from hailo_platform.genai import Speech2Text, Speech2TextTask, LLM

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

WHISPER_HEF = "/usr/local/hailo/resources/models/hailo10h/Whisper-Small.hef"
LLM_HEF = "/usr/local/hailo/resources/models/hailo10h/Qwen3-1.7B-Instruct.hef"
LED_DEVICE_IP = "192.168.2.61"
DISPLAY_CLEAR_DELAY = 10
_display_enabled = False  # Default: do not send to LED display
AUDIO_DEVICE_NAME = "USB PnP Audio Device"
TARGET_SR = 16000
NATIVE_SR = 48000
DOWNSAMPLE_FACTOR = NATIVE_SR // TARGET_SR
CHUNK_DURATION_MS = 30
NATIVE_CHUNK = int(NATIVE_SR * CHUNK_DURATION_MS / 1000)
TARGET_CHUNK = int(TARGET_SR * CHUNK_DURATION_MS / 1000)
SILENCE_TIMEOUT = 1.5
MIN_SPEECH_DURATION = 0.5
MAX_SPEECH_DURATION = 15.0  # Force split at 15 seconds
DB_PATH = "/home/kota/hailo-apps/whisper_log.db"
WEB_PORT = 8080
PROFILES_DIR = Path("/home/kota/hailo-apps/speaker_profiles")
SPEAKER_THRESHOLD = 0.75

# --- Database ---

def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS transcriptions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            text TEXT NOT NULL,
            duration_sec REAL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS daily_summaries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT UNIQUE NOT NULL,
            summary TEXT NOT NULL,
            total_entries INTEGER,
            total_duration_sec REAL
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_ts ON transcriptions(timestamp)")
    conn.commit()
    conn.close()
    logger.info("Database initialized: %s", DB_PATH)


# --- Speaker Identification ---

_voice_encoder = None
_speaker_profiles = {}

def init_speaker_id():
    global _voice_encoder, _speaker_profiles
    _voice_encoder = VoiceEncoder()
    _speaker_profiles = {}
    if PROFILES_DIR.exists():
        import numpy as _np
        for f in PROFILES_DIR.glob("*.npy"):
            _speaker_profiles[f.stem] = _np.load(f)
        logger.info("Loaded %d speaker profiles: %s", len(_speaker_profiles), list(_speaker_profiles.keys()))


def identify_speaker(audio_i16):
    """Identify speaker from int16 audio. Returns name or 'Unknown'."""
    if not _speaker_profiles or _voice_encoder is None:
        return "Unknown"
    try:
        audio_f32 = audio_i16.astype(np.float32) / 32768.0
        processed = preprocess_wav(audio_f32, source_sr=16000)
        if len(processed) < 1600:  # too short
            return "Unknown"
        embedding = _voice_encoder.embed_utterance(processed)
        best_name = "Unknown"
        best_score = 0.0
        for name, profile in _speaker_profiles.items():
            score = np.dot(embedding, profile) / (np.linalg.norm(embedding) * np.linalg.norm(profile))
            if score > best_score:
                best_score = score
                best_name = name
        if best_score >= SPEAKER_THRESHOLD:
            logger.info("Speaker: %s (%.2f)", best_name, best_score)
            return best_name
        else:
            logger.info("Speaker: Unknown (best: %s %.2f)", best_name, best_score)
            return "Unknown"
    except Exception:
        logger.exception("Speaker identification failed")
        return "Unknown"


def insert_transcription(text, duration_sec=None, speaker=None):
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.execute(
            "INSERT INTO transcriptions (timestamp, text, duration_sec, speaker) VALUES (?, ?, ?, ?)",
            (datetime.now().isoformat(), text, duration_sec, speaker),
        )
        conn.commit()
        conn.close()
    except Exception:
        logger.exception("Failed to insert transcription")


def query_logs(date=None, limit=200):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    if date:
        rows = conn.execute(
            "SELECT timestamp, text, duration_sec, speaker FROM transcriptions WHERE timestamp LIKE ? ORDER BY timestamp DESC LIMIT ?",
            (f"{date}%", limit),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT timestamp, text, duration_sec, speaker FROM transcriptions ORDER BY timestamp DESC LIMIT ?",
            (limit,),
        ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def query_dates():
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        "SELECT DISTINCT substr(timestamp, 1, 10) as date, COUNT(*) as count FROM transcriptions GROUP BY date ORDER BY date DESC LIMIT 30"
    ).fetchall()
    conn.close()
    return [{"date": r[0], "count": r[1]} for r in rows]


# --- Watchdog ---

def notify_watchdog():
    """Notify systemd watchdog that we are alive."""
    addr = os.environ.get("NOTIFY_SOCKET")
    if not addr:
        return
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
    try:
        sock.sendto(b"WATCHDOG=1", addr)
    finally:
        sock.close()


# --- Audio API ---

_utterance_queue = None  # Set in main()


def decode_audio_payload(body, content_type):
    """Decode incoming audio to int16 PCM at 16kHz mono."""
    if "wav" in content_type or body[:4] == b"RIFF":
        wf = wave.open(io.BytesIO(body), "rb")
        raw = wf.readframes(wf.getnframes())
        sr = wf.getframerate()
        ch = wf.getnchannels()
        pcm = np.frombuffer(raw, dtype=np.int16)
        # Mix to mono
        if ch > 1:
            pcm = pcm.reshape(-1, ch).mean(axis=1).astype(np.int16)
        # Resample to 16kHz if needed
        if sr != 16000:
            factor = sr / 16000
            indices = np.arange(0, len(pcm), factor).astype(int)
            indices = indices[indices < len(pcm)]
            pcm = pcm[indices]
        wf.close()
        return pcm
    else:
        # Assume raw PCM int16 mono 16kHz
        return np.frombuffer(body, dtype=np.int16)


# --- Web Server ---

class LogHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)
        params = parse_qs(parsed.query)

        if parsed.path == "/api/logs":
            date = params.get("date", [None])[0]
            limit = int(params.get("limit", [200])[0])
            data = query_logs(date=date, limit=limit)
            self._json_response(data)

        elif parsed.path == "/api/dates":
            data = query_dates()
            self._json_response(data)

        else:
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(WEB_PAGE.encode())

    def do_POST(self):
        parsed = urlparse(self.path)

        if parsed.path == "/api/display/toggle":
            global _display_enabled
            content_length = int(self.headers.get("Content-Length", 0))
            if content_length > 0:
                body = json.loads(self.rfile.read(content_length))
                _display_enabled = body.get("enabled", not _display_enabled)
            else:
                _display_enabled = not _display_enabled
            logger.info("Display toggle: %s", _display_enabled)
            self._json_response({"display_enabled": _display_enabled})
            return

        elif parsed.path == "/api/display/status":
            self._json_response({"display_enabled": _display_enabled})
            return

        elif parsed.path == "/api/audio":
            content_length = int(self.headers.get("Content-Length", 0))
            if content_length == 0:
                self._json_response({"error": "No audio data"}, status=400)
                return
            body = self.rfile.read(content_length)
            content_type = self.headers.get("Content-Type", "")
            source = self.headers.get("X-Source", "remote")

            try:
                pcm = decode_audio_payload(body, content_type)
            except Exception as e:
                self._json_response({"error": f"Failed to decode audio: {e}"}, status=400)
                return

            duration = len(pcm) / 16000
            if duration < 0.3:
                self._json_response({"error": "Audio too short", "duration": duration}, status=400)
                return

            if _utterance_queue is not None:
                _utterance_queue.put((pcm, source))
                logger.info("API: queued %.1fs audio from %s (queue: %d)", duration, source, _utterance_queue.qsize())
                self._json_response({"status": "queued", "duration": round(duration, 2), "queue_size": _utterance_queue.qsize()})
            else:
                self._json_response({"error": "Service not ready"}, status=503)
        else:
            self._json_response({"error": "Not found"}, status=404)

    def _json_response(self, data, status=200):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(data, ensure_ascii=False).encode())

    def log_message(self, format, *args):
        pass


WEB_PAGE = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Whisper Log</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
body{font-family:sans-serif;max-width:900px;margin:0 auto;padding:20px;background:#1a1a2e;color:#eee}
h1{color:#0ff}
.controls{display:flex;gap:10px;align-items:center;margin-bottom:15px;flex-wrap:wrap}
select,button{padding:6px 12px;border:1px solid #444;background:#2a2a4e;color:#eee;border-radius:4px;cursor:pointer}
button{background:#0ff;color:#000;font-weight:bold}
button.on{background:#0f0}
button.off{background:#f55;color:#fff}
button:hover{background:#0aa}
table{width:100%;border-collapse:collapse}
th,td{padding:8px 12px;text-align:left;border-bottom:1px solid #333}
th{color:#0ff;position:sticky;top:0;background:#1a1a2e}
td:first-child{white-space:nowrap;color:#888;width:100px}
td:nth-child(2){color:#0f0;width:80px}
td:nth-child(3){color:#ccc;width:50px;text-align:right;font-size:0.85em}
.status{color:#0f0;font-size:0.9em}
.live{display:inline-block;width:8px;height:8px;background:#0f0;border-radius:50%;margin-right:5px;animation:blink 2s infinite}
@keyframes blink{0%,100%{opacity:1}50%{opacity:0.3}}
</style></head><body>
<h1>Whisper Speech Log</h1>
<div class="controls">
  <select id="dateSelect"><option value="">Today (Live)</option></select>
  <button id="dispBtn" onclick="toggleDisplay()">Display: OFF</button>
  <button onclick="load()">Refresh</button>
  <span class="status" id="status"><span class="live"></span>Listening...</span>
</div>
<table><thead><tr><th>Time</th><th>Speaker</th><th>Sec</th><th>Text</th></tr></thead>
<tbody id="log"></tbody></table>
<script>
let autoRefresh=true;
const dateSelect=document.getElementById('dateSelect');

function loadDates(){
  fetch('/api/dates').then(r=>r.json()).then(dates=>{
    dates.forEach(d=>{
      const opt=document.createElement('option');
      opt.value=d.date;
      opt.textContent=d.date+' ('+d.count+')';
      dateSelect.appendChild(opt);
    });
  });
}

function load(){
  const date=dateSelect.value;
  const url=date?'/api/logs?date='+date:'/api/logs';
  fetch(url).then(r=>r.json()).then(data=>{
    const tbody=document.getElementById('log');
    tbody.innerHTML='';
    data.forEach(e=>{
      const d=new Date(e.timestamp);
      const ts=d.toLocaleTimeString('ja-JP');
      const dur=e.duration_sec?e.duration_sec.toFixed(1):'';
      const spk=e.speaker||'?';
      tbody.innerHTML+='<tr><td>'+ts+'</td><td>'+spk+'</td><td>'+dur+'</td><td>'+e.text+'</td></tr>';
    });
    document.getElementById('status').innerHTML=
      (date?'':'<span class="live"></span>')+
      data.length+' entries'+(!date?' (auto-refresh)':'')+
      ' - '+new Date().toLocaleTimeString('ja-JP');
  });
}

dateSelect.onchange=function(){autoRefresh=!this.value;load()};
function toggleDisplay(){
  fetch('/api/display/toggle',{method:'POST'}).then(r=>r.json()).then(d=>{
    updateDispBtn(d.display_enabled);
  });
}
function updateDispBtn(on){
  const b=document.getElementById('dispBtn');
  b.textContent='Display: '+(on?'ON':'OFF');
  b.className=on?'on':'off';
}
fetch('/api/display/status').then(r=>r.json()).then(d=>updateDispBtn(d.display_enabled));
loadDates();
load();
setInterval(()=>{if(autoRefresh)load()},5000);
</script></body></html>"""


# --- Display ---

DISPLAY_WIDTH_CHARS = 10  # Approximate characters that fit on display at once
WORD_DISPLAY_INTERVAL = 0.3  # seconds between each word group


def is_meaningful(llm, text):
    """Use LLM to judge if transcribed text is meaningful speech."""
    # Quick rule-based filter first
    stripped = text.strip().rstrip('.!?,')
    if len(stripped) <= 3:
        return False
    if text.strip() in ('...', '--', 'So,', 'Yeah.', 'OK.', 'Uh,', 'Um,'):
        return False

    prompt = [{
        "role": "user",
        "content": [{"type": "text", "text": f"""Judge if this transcribed speech is a meaningful sentence worth logging.
Reply ONLY "YES" or "NO".

- YES: complete thought, question, statement, or instruction
- NO: filler words, fragments, noise artifacts, meaningless sounds

Text: "{text}" """}]
    }]
    try:
        llm.clear_context()
        response = llm.generate_all(prompt=prompt, max_generated_tokens=5, temperature=0.1)
        answer = response.strip().upper().replace("<|IM_END|>", "").replace("<|im_end|>", "").strip()
        return answer.startswith("YES")
    except Exception:
        logger.exception("LLM filter failed")
        return True  # On error, keep the text


def send_to_display(text, color=None):
    """Display text word-by-word in quick succession, then show full text scrolling."""
    if color is None:
        color = {"r": 255, "g": 255, "b": 255}
    try:
        # Split into chunks that fit on display
        words = text.split()
        if not words:
            return

        # Group words into display-width chunks
        chunks = []
        current = ""
        for word in words:
            if current and len(current) + 1 + len(word) > DISPLAY_WIDTH_CHARS:
                chunks.append(current)
                current = word
            else:
                current = (current + " " + word).strip()
        if current:
            chunks.append(current)

        # Flash each chunk
        for chunk in chunks:
            payload = render_text_to_bitmap_payload(
                chunk, color=color, scroll_speed="slow",
                font_path="/home/kota/galactic-unicorn-horn/fonts/PixelMplus12-Regular.ttf",
                font_size=12,
            )
            payload["display_mode"] = "static"
            requests.post(f"http://{LED_DEVICE_IP}/api/bitmap", json=payload, timeout=5)
            time.sleep(WORD_DISPLAY_INTERVAL)

        logger.info("Display: %s", text[:80])
    except Exception:
        logger.exception("Failed to send to display")


def clear_display():
    try:
        requests.delete(f"http://{LED_DEVICE_IP}/api/bitmap", timeout=5)
    except Exception:
        pass


def find_audio_device():
    for i, dev in enumerate(sd.query_devices()):
        if AUDIO_DEVICE_NAME in dev["name"] and dev["max_input_channels"] > 0:
            logger.info("Audio device: %d - %s (native SR: %.0f)", i, dev["name"], dev["default_samplerate"])
            return i
    return None


# --- Main ---

def main():
    init_db()

    device_id = find_audio_device()
    if device_id is None:
        logger.error("Audio device '%s' not found", AUDIO_DEVICE_NAME)
        sys.exit(1)

    init_speaker_id()

    logger.info("Initializing Hailo device...")
    params = VDevice.create_params()
    params.group_id = "SHARED"
    vdevice = VDevice(params)

    logger.info("Loading Whisper-Small...")
    speech2text = Speech2Text(vdevice, WHISPER_HEF)
    logger.info("Whisper-Small loaded")

    logger.info("Loading Qwen3-1.7B for filtering...")
    llm = LLM(vdevice, LLM_HEF)
    logger.info("Qwen3-1.7B loaded")

    vad = webrtcvad.Vad(2)

    # Queue for completed utterances
    import queue
    utterance_queue = queue.Queue()

    speech_started = False
    silence_start = None
    audio_chunks = []
    clear_timer = None

    logger.info("Starting web server on port %d...", WEB_PORT)
    httpd = HTTPServer(("0.0.0.0", WEB_PORT), LogHandler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    logger.info("Web UI: http://192.168.2.55:%d", WEB_PORT)

    # Notify systemd we are ready
    addr = os.environ.get("NOTIFY_SOCKET")
    if addr:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
        sock.sendto(b"READY=1", addr)
        sock.close()
        logger.info("Notified systemd: READY")

    logger.info("Listening...")

    def process_worker():
        """Worker thread: processes utterances from queue sequentially."""
        nonlocal clear_timer
        while True:
            item = utterance_queue.get()
            if item is None:
                break

            if isinstance(item, tuple):
                audio_i16, source = item
            else:
                audio_i16, source = item, "mic"

            duration = len(audio_i16) / TARGET_SR
            logger.info("Transcribing %.1fs of audio (queue: %d remaining)...", duration, utterance_queue.qsize())

            try:
                audio_f32 = audio_i16.astype(np.float32) / 32768.0
                audio_le = audio_f32.astype("<f4")
                segments = speech2text.generate_all_segments(
                    audio_data=audio_le,
                    task=Speech2TextTask.TRANSCRIBE,
                    timeout_ms=30000,
                )

                if not segments:
                    continue

                text = "".join([seg.text for seg in segments]).strip()
                if not text or text in ("[BLANK_AUDIO]", "[Music]", "(Music)"):
                    logger.info("Skipped: %s", text)
                    continue

                # Check if result is Japanese or English, otherwise retry as English
                def is_ja_or_en(t):
                    for c in t:
                        cp = ord(c)
                        if 0x3040 <= cp <= 0x30FF or 0x4E00 <= cp <= 0x9FFF:
                            return True
                        if 0x41 <= cp <= 0x5A or 0x61 <= cp <= 0x7A:
                            return True
                    return False

                if not is_ja_or_en(text):
                    logger.info("Non ja/en detected (%s), retrying as English...", text[:30])
                    segments2 = speech2text.generate_all_segments(
                        audio_data=audio_le,
                        task=Speech2TextTask.TRANSCRIBE,
                        language="en",
                        timeout_ms=30000,
                    )
                    if segments2:
                        text2 = "".join([seg.text for seg in segments2]).strip()
                        if text2 and text2 not in ("[BLANK_AUDIO]", "[Music]", "(Music)"):
                            text = text2

                logger.info("Recognized: %s", text)

                if not is_meaningful(llm, text):
                    logger.info("Filtered (not meaningful): %s", text)
                    continue

                speaker = identify_speaker(audio_i16)
                insert_transcription(text, duration_sec=duration, speaker=speaker)
                if _display_enabled:
                    send_to_display(text, color={"r": 0, "g": 255, "b": 255})

                    if clear_timer:
                        clear_timer.cancel()
                    clear_timer = threading.Timer(DISPLAY_CLEAR_DELAY, clear_display)
                    clear_timer.start()








            except Exception:
                logger.exception("Transcription failed")

    global _utterance_queue
    _utterance_queue = utterance_queue

    # Start worker thread
    worker = threading.Thread(target=process_worker, daemon=True)
    worker.start()

    speech_start_time = [None]  # Use list for nonlocal mutation in callback

    def audio_callback(indata, frames, time_info, status):
        nonlocal speech_started, silence_start, audio_chunks

        if status and "overflow" not in str(status):
            logger.warning("Audio status: %s", status)

        # Notify watchdog
        notify_watchdog()

        # Always process audio - never skip
        mono = indata[:, 0]
        downsampled = mono[::DOWNSAMPLE_FACTOR]
        pcm = (downsampled * 32768).astype(np.int16)

        if len(pcm) < TARGET_CHUNK:
            pcm = np.pad(pcm, (0, TARGET_CHUNK - len(pcm)))
        else:
            pcm = pcm[:TARGET_CHUNK]

        try:
            is_speech = vad.is_speech(pcm.tobytes(), TARGET_SR)
        except Exception:
            is_speech = False

        if is_speech and not speech_started:
            speech_started = True
            silence_start = None
            speech_start_time[0] = time.time()
            audio_chunks = []
            logger.info("Speech detected")

        if speech_started:
            audio_chunks.append(pcm.copy())

            # Force split if speech is too long
            elapsed = time.time() - speech_start_time[0] if speech_start_time[0] else 0
            force_split = elapsed >= MAX_SPEECH_DURATION

            if not is_speech or force_split:
                if force_split:
                    if audio_chunks:
                        audio = np.concatenate(audio_chunks)
                        audio_chunks = []
                        speech_start_time[0] = time.time()
                        utterance_queue.put((audio, "mic"))
                        logger.info("Force split at %.0fs (queue: %d)", elapsed, utterance_queue.qsize())
                    # Keep speech_started=True to continue recording
                elif silence_start is None:
                    silence_start = time.time()
                elif time.time() - silence_start > SILENCE_TIMEOUT:
                    speech_started = False
                    silence_start = None
                    speech_start_time[0] = None
                    audio = np.concatenate(audio_chunks)
                    audio_chunks = []
                    if len(audio) / TARGET_SR >= MIN_SPEECH_DURATION:
                        utterance_queue.put((audio, "mic"))
                        logger.info("Queued utterance (%.1fs, queue: %d)", len(audio) / TARGET_SR, utterance_queue.qsize())
                    else:
                        logger.info("Too short, skipped")
            else:
                silence_start = None

    try:
        with sd.InputStream(device=device_id, channels=1, samplerate=NATIVE_SR,
                           dtype="float32", blocksize=NATIVE_CHUNK,
                           callback=audio_callback, latency="high"):
            while True:
                time.sleep(0.1)
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        utterance_queue.put(None)  # Signal worker to stop
        worker.join(timeout=5)
        llm.release()
        speech2text.release()
        vdevice.release()
        if clear_timer:
            clear_timer.cancel()


if __name__ == "__main__":
    main()
