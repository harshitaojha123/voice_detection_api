from fastapi import FastAPI, Header, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import uuid
import base64

import librosa
import numpy as np

# ---------------- CONFIG ----------------
API_KEY = os.getenv("API_KEY")  # set in Railway Variables

app = FastAPI(title="AI Generated Voice Detection API")

# ---------------- CORS ----------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- ROOT ----------------
@app.get("/")
def home():
    return {
        "message": "AI Generated Voice Detection API is running",
        "endpoints": [
            "/detect-voice-file (multipart upload)",
            "/detect-voice-base64 (hackathon)"
        ]
    }

# ============================================================
# FILE UPLOAD ENDPOINT (Swagger / Demo)
# ============================================================

@app.post("/detect-voice-file")
async def detect_voice_file(
    file: UploadFile = File(...),
    language: str = "en",
    x_api_key: str = Header(None)
):
    if not API_KEY or x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")

    if language not in ["en", "hi", "ta", "ml", "te"]:
        raise HTTPException(status_code=400, detail="Unsupported language")

    if not file.filename.lower().endswith(".wav"):
        raise HTTPException(status_code=400, detail="Only WAV audio supported")

    filename = f"/tmp/{uuid.uuid4().hex}.wav"

    with open(filename, "wb") as f:
        f.write(await file.read())

    try:
        return analyze_audio(filename)
    finally:
        if os.path.exists(filename):
            os.remove(filename)


# ============================================================
# BASE64 ENDPOINT (HACKATHON REQUIRED)
# ============================================================

class Base64VoiceRequest(BaseModel):
    language: str
    audio_format: str
    audio_base64: str


@app.post("/detect-voice-base64")
def detect_voice_base64(
    data: Base64VoiceRequest,
    x_api_key: str = Header(None)
):
    if not API_KEY or x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")

    if data.language not in ["en", "hi", "ta", "ml", "te"]:
        raise HTTPException(status_code=400, detail="Unsupported language")

    if data.audio_format.lower() != "wav":
        raise HTTPException(status_code=400, detail="Only WAV audio supported")

    filename = f"/tmp/{uuid.uuid4().hex}.wav"

    try:
        audio_bytes = base64.b64decode(data.audio_base64)
        with open(filename, "wb") as f:
            f.write(audio_bytes)

        return analyze_audio(filename)

    except Exception:
        raise HTTPException(status_code=400, detail="Invalid base64 audio")

    finally:
        if os.path.exists(filename):
            os.remove(filename)


# ============================================================
# SHARED AUDIO ANALYSIS LOGIC
# ============================================================

def analyze_audio(filename: str):
    try:
        y, sr = librosa.load(filename, sr=None)

        rms = librosa.feature.rms(y=y)[0]
        energy_variance = np.var(rms)
        silence_ratio = np.sum(np.abs(y) < 0.01) / len(y)

        if energy_variance < 0.0005 and silence_ratio < 0.15:
            return {
                "classification": "AI_GENERATED",
                "confidence_score": 0.85,
                "explanation": "Low energy variation and uniform patterns detected"
            }

        return {
            "classification": "HUMAN",
            "confidence_score": 0.75,
            "explanation": "Natural speech variation detected"
        }

    except Exception:
        raise HTTPException(status_code=500, detail="Audio processing failed")
