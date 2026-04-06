# Pi News Researcher

Raspberry Pi 5 + Hailo-10H AI accelerator を使った常時音声認識システム。
認識結果をSQLiteに保存し、電光掲示板（Galactic Unicorn）にリアルタイム表示。

## System Architecture

```mermaid
graph TB
    subgraph "Raspberry Pi 5 (pi-llm)"
        MIC[USB Audio HAT<br/>SSS1629A5] -->|48kHz PCM| AD[Audio Downsampler<br/>48kHz → 16kHz]
        AD -->|16kHz PCM| VAD[Voice Activity Detector<br/>WebRTC VAD]
        VAD -->|Speech chunks| Q[Utterance Queue]
        Q --> WHISPER[Hailo-10H<br/>Whisper-Small]
        WHISPER --> SPEAKER[Speaker ID<br/>Resemblyzer]
        SPEAKER --> DB[(SQLite DB)]
        SPEAKER -->|if enabled| LED
        WEB[Web UI<br/>:8080] --> DB
        API[Audio API<br/>POST /api/audio] --> Q
    end

    subgraph "External Devices"
        LED[LED Display<br/>Galactic Unicorn<br/>192.168.2.61]
        ESP[ESP32<br/>Remote Mic] -->|HTTP POST| API
        BROWSER[Browser] --> WEB
    end

    style WHISPER fill:#0ff,color:#000
    style DB fill:#f90,color:#000
    style LED fill:#0f0,color:#000
```

## Processing Pipeline

```mermaid
sequenceDiagram
    participant Mic
    participant VAD
    participant Queue
    participant Whisper
    participant DB
    participant LED

    loop Every 30ms
        Mic->>VAD: Audio chunk (480 samples)
        VAD->>VAD: Speech detection
    end

    Note over VAD: Speech detected
    VAD->>VAD: Buffer audio chunks

    alt Silence > 1.5s
        VAD->>Queue: Complete utterance
    else Duration > 15s
        VAD->>Queue: Force split
    end

    Queue->>Whisper: Audio (Hailo-10H accelerated)
    Whisper->>Whisper: Auto language detect (ja/en)

    alt Non ja/en result
        Whisper->>Whisper: Retry as English
    end

    Whisper->>DB: Save (timestamp, text, duration, speaker)

    opt Display enabled
        Whisper->>LED: Word-by-word display (0.3s interval)
    end
```

## Hardware Requirements

| Component | Model | Role |
|-----------|-------|------|
| SBC | Raspberry Pi 5 (16GB) | Host |
| AI Accelerator | Hailo-10H (AI HAT+) | Whisper inference |
| Audio | USB Audio HAT (SSS1629A5) | Microphone input |
| Display | Galactic Unicorn (RPi Pico) | LED text display |
| Remote Mic (optional) | ESP32 + I2S mic | Remote audio input |

## Software Stack

| Component | Version |
|-----------|---------|
| HailoRT | 5.3.0 |
| Whisper Model | Whisper-Small (386MB HEF) |
| Speaker ID | Resemblyzer |
| VAD | WebRTC VAD |
| Web Server | Python http.server |
| Database | SQLite |

## Setup

### 1. Prerequisites

```bash
# HailoRT 5.3.0 must be installed with PCIe driver loaded
hailortcli fw-control identify
# Should show: Firmware Version: 5.3.0

# Whisper-Small model
ls /usr/local/hailo/resources/models/hailo10h/Whisper-Small.hef
```

### 2. Install Dependencies

```bash
cd /home/kota/hailo-apps
source venv_hailo_apps/bin/activate
pip install sounddevice webrtcvad resemblyzer
```

### 3. Clone Repository

```bash
cd /home/kota
git clone git@github.com:kotamorishi/pi-news-researcher.git
```

### 4. Audio HAT Setup

- Connect the USB Audio HAT (SSS1629A5) via Type-C cable to RPi USB port
- **DIP switch: Mic must be ON**
- Verify: `arecord -l` should show "USB PnP Audio Device"

### 5. Register Speaker (Optional)

```bash
# Stop the service first if running
sudo systemctl stop hailo-whisper-display

# Record 10 seconds of your voice
cd /home/kota/hailo-apps
source venv_hailo_apps/bin/activate
python /home/kota/pi-news-researcher/register_speaker.py <name>

# Restart the service
sudo systemctl start hailo-whisper-display
```

### 6. Install systemd Service

```bash
sudo tee /etc/systemd/system/hailo-whisper-display.service > /dev/null << 'EOF'
[Unit]
Description=Hailo Whisper Speech Recognition → LED Display
After=network.target

[Service]
Type=notify
User=kota
WorkingDirectory=/home/kota/hailo-apps
ExecStart=/home/kota/hailo-apps/venv_hailo_apps/bin/python /home/kota/pi-news-researcher/whisper_display.py
Restart=always
RestartSec=5
WatchdogSec=60
Environment=PYTHONUNBUFFERED=1
NotifyAccess=all

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable hailo-whisper-display
sudo systemctl start hailo-whisper-display
```

## Usage

### Web UI

Open http://192.168.2.55:8080 in your browser.

- View real-time transcription logs with timestamps and speaker labels
- Filter by date using the dropdown
- Toggle LED display output with the "Display" button (default: OFF)
- Auto-refreshes every 5 seconds

### API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Web UI |
| GET | `/api/logs?date=2026-04-06&limit=200` | Get transcription logs (JSON) |
| GET | `/api/dates` | Get available dates with entry counts |
| POST | `/api/audio` | Submit audio for transcription |
| POST | `/api/display/toggle` | Toggle LED display on/off |
| GET | `/api/display/status` | Get LED display status |

### Remote Audio (ESP32)

Send audio from remote microphones via HTTP:

```bash
curl -X POST http://192.168.2.55:8080/api/audio \
  -H "Content-Type: audio/wav" \
  -H "X-Source: esp32-kitchen" \
  --data-binary @recording.wav
```

Accepts WAV (any sample rate/channels, auto-converted) or raw PCM (16kHz, 16-bit, mono).

### Logs

```bash
# Real-time log
journalctl -u hailo-whisper-display -f

# Today's log
journalctl -u hailo-whisper-display --since today
```

## Database Schema

```sql
CREATE TABLE transcriptions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    text TEXT NOT NULL,
    duration_sec REAL,
    speaker TEXT
);

CREATE TABLE daily_summaries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    date TEXT UNIQUE NOT NULL,
    summary TEXT NOT NULL,
    total_entries INTEGER,
    total_duration_sec REAL
);
```

## OpenAI-Compatible API Server

A separate server (`openai_server.py`) provides OpenAI-compatible `/v1/chat/completions` endpoint using Hailo VLM (Qwen3-VL-2B-Instruct). Supports both text and image inputs.

```bash
# Start (separate from whisper service, cannot run simultaneously)
sudo systemctl start hailo-openai

# Text request
curl http://192.168.2.55:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3-vl-2b","messages":[{"role":"user","content":"Hello"}]}'
```

> Note: VLM server and Whisper service share the Hailo-10H device and cannot run simultaneously.
