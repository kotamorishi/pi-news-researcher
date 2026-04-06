#!/usr/bin/env python3
"""OpenAI API-compatible server for Hailo-10H VLM (Qwen2-VL-2B-Instruct)."""

import base64
import io
import time
import uuid
import threading

import cv2
import numpy as np
from flask import Flask, request, jsonify
from PIL import Image

from hailo_platform import VDevice
from hailo_platform.genai import VLM

app = Flask(__name__)

_vlm = None
_vlm_lock = threading.Lock()
_frame_size = (336, 336)

HEF_PATH = "/usr/local/hailo/resources/models/hailo10h/Qwen3-VL-2B-Instruct.hef"
MODEL_NAME = "qwen3-vl-2b"


def get_vlm():
    global _vlm
    if _vlm is None:
        params = VDevice.create_params()
        vdevice = VDevice(params)
        _vlm = VLM(vdevice, HEF_PATH)
        global _frame_size
        shape = _vlm.input_frame_shape()
        _frame_size = (shape[1], shape[0])
        print(f"VLM loaded: {HEF_PATH}")
    return _vlm


def decode_image(image_url: str) -> np.ndarray:
    """Decode base64 data URI or URL to numpy array."""
    if image_url.startswith("data:"):
        header, b64data = image_url.split(",", 1)
        img_bytes = base64.b64decode(b64data)
    else:
        import requests as req
        resp = req.get(image_url, timeout=10)
        resp.raise_for_status()
        img_bytes = resp.content

    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img_np = np.array(img)
    return resize_center_crop(img_np, _frame_size)


def resize_center_crop(image: np.ndarray, target_size: tuple = (336, 336)) -> np.ndarray:
    """Resize with center crop to target size."""
    h, w = image.shape[:2]
    target_w, target_h = target_size
    scale = max(target_w / w, target_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    x_start = (new_w - target_w) // 2
    y_start = (new_h - target_h) // 2
    return resized[y_start:y_start + target_h, x_start:x_start + target_w].astype(np.uint8)


def convert_messages(messages: list) -> tuple:
    """Convert OpenAI messages to Hailo prompt format. Returns (hailo_prompt, frames)."""
    hailo_prompt = []
    frames = []

    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")

        if isinstance(content, str):
            hailo_prompt.append({
                "role": role,
                "content": [{"type": "text", "text": content}]
            })
            continue

        hailo_content = []
        for part in content:
            if part.get("type") == "text":
                hailo_content.append({"type": "text", "text": part["text"]})
            elif part.get("type") == "image_url":
                url = part["image_url"]["url"]
                frame = decode_image(url)
                frames.append(frame)
                hailo_content.append({"type": "image"})

        hailo_prompt.append({"role": role, "content": hailo_content})

    return hailo_prompt, frames


@app.route("/v1/chat/completions", methods=["POST"])
def chat_completions():
    data = request.get_json()
    if not data or "messages" not in data:
        return jsonify({"error": {"message": "messages is required"}}), 400

    messages = data["messages"]
    max_tokens = data.get("max_tokens", 512)
    temperature = data.get("temperature", 0.7)

    try:
        hailo_prompt, frames = convert_messages(messages)
    except Exception as e:
        return jsonify({"error": {"message": f"Failed to parse messages: {e}"}}), 400

    try:
        with _vlm_lock:
            vlm = get_vlm()
            vlm.clear_context()
            response_text = vlm.generate_all(
                prompt=hailo_prompt,
                frames=frames,
                max_generated_tokens=max_tokens,
                temperature=temperature,
            )
            vlm.clear_context()
            vlm.clear_context()
    except Exception as e:
        return jsonify({"error": {"message": f"Inference failed: {e}"}}), 500

    return jsonify({
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": MODEL_NAME,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": response_text.replace("<|im_end|>", "").strip()},
            "finish_reason": "stop",
        }],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    })


@app.route("/v1/models", methods=["GET"])
def list_models():
    return jsonify({
        "object": "list",
        "data": [{"id": MODEL_NAME, "object": "model", "created": 0, "owned_by": "hailo"}],
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    print("Loading VLM model...")
    get_vlm()
    print("Starting OpenAI-compatible API server on 0.0.0.0:8000")
    app.run(host="0.0.0.0", port=8000, threaded=True)
