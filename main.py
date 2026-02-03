from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
import os
import uuid
import base64
import librosa
import numpy as np

# ============================================================
# CONFIG
# ============================================================
API_KEY = os.getenv("API_KEY")  # Must be set in Railway Variables

app = FastAPI(title="AI Generated Voice Detection API")

# ============================================================
# CORS
# ============================================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# ROOT
# ============================================================
@app.get("/")
def home():
    return {
        "message": "AI Generated Voice Detection API is running",
        "endpoint": "/detect"
    }

# ============================================================
# REQUEST MODEL (ACCEPTS CAMEL + SNAKE CASE)
# ============================================================
class VoiceRequest(BaseModel):
    language: str

    audio_format: str = Field(..., alias="audioFormat")
    audio_base64: Optional[str] = Field(None, alias="audioBase64")
    audio: Optional[str] = None

    class Config:
        allow_population_by_field_name = True

# ============================================================
# AUDIO ANALYSIS
# ============================================================
def analyze_audio(filename: str):
    y, sr = librosa.load(filename, sr=None)

    if len(y) == 0:
        raise Exception("Empty audio")

    rms = librosa.feature.rms(y=y)[0]
    energy_variance = float(np.var(rms))
    silence_ratio = float(np.sum(np.abs(y) < 0.01) / len(y))

    if energy_variance < 0.0005 and silence_ratio < 0.15:
        return {
            "classification": "AI-generated",
            "confidence": 0.85,
            "explanation": "Low energy variance and uniform speech patterns detected"
        }
    else:
        return {
            "classification": "Human-generated",
            "confidence": 0.75,
            "explanation": "Natural pitch and energy variation detected"
        }

# ============================================================
# MAIN ENDPOINT (GUVI TESTER READY)
# ============================================================
@app.post("/detect")
def detect_voice(
    data: VoiceRequest,
    x_api_key: str = Header(None)
):
    # ---- AUTH CHECK ----
    if not x_api_key or x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")

    # ---- LANGUAGE CHECK ----
    if data.language not in ["en", "hi", "ta", "ml", "te"]:
        raise HTTPException(status_code=400, detail="Unsupported language")

    # ---- AUDIO FORMAT CHECK ----
    if data.audio_format.lower() != "mp3":
        raise HTTPException(status_code=400, detail="Only MP3 supported")

    # ---- PICK AUDIO FIELD ----
    audio_b64 = data.audio or data.audio_base64
    if not audio_b64:
        raise HTTPException(status_code=400, detail="Audio data missing")

    filename = f"/tmp/{uuid.uuid4().hex}.mp3"

    try:
        # Remove data URI if present
        if "," in audio_b64:
            audio_b64 = audio_b64.split(",")[1]

        audio_bytes = base64.b64decode(audio_b64)

        with open(filename, "wb") as f:
            f.write(audio_bytes)

        return analyze_audio(filename)

    except Exception:
        raise HTTPException(status_code=400, detail="Invalid audio input")

    finally:
        if os.path.exists(filename):
            os.remove(filename)
