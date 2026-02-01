from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
import uuid
import os
import requests

import librosa
import numpy as np

# ---------------- CONFIG ----------------
API_KEY = os.getenv("API_KEY")


app = FastAPI(title="AI Generated Voice Detection API")

# ---------------- ROOT ----------------
@app.get("/")
def home():
    return {
        "message": "AI Generated Voice Detection API is running",
        "endpoint": "/detect-voice"
    }

# ---------------- REQUEST MODEL ----------------
class VoiceRequest(BaseModel):
    audio_file_url: str
    language: str

# ---------------- API ENDPOINT ----------------
@app.post("/detect-voice")
def detect_voice(data: VoiceRequest, x_api_key: str = Header(None)):

    # 1️⃣ API KEY VALIDATION
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")

    # 2️⃣ LANGUAGE VALIDATION
    supported_languages = ["en", "hi", "ta", "ml", "te"]
    if data.language not in supported_languages:
        raise HTTPException(status_code=400, detail="Unsupported language")

    # 3️⃣ DOWNLOAD AUDIO
    try:
        response = requests.get(data.audio_file_url, timeout=10)
        if response.status_code != 200:
            raise HTTPException(status_code=400, detail="Unable to download audio")

        filename = f"audio_{uuid.uuid4().hex}.mp3"
        with open(filename, "wb") as f:
            f.write(response.content)

    except Exception:
        raise HTTPException(status_code=400, detail="Invalid audio URL")

    try:
        # 4️⃣ LOAD AUDIO
        y, sr = librosa.load(filename, sr=None)

        # 5️⃣ FEATURE EXTRACTION
        rms = librosa.feature.rms(y=y)[0]
        energy_variance = np.var(rms)

        silence_ratio = np.sum(np.abs(y) < 0.01) / len(y)

        # 6️⃣ DECISION
        if energy_variance < 0.0005 and silence_ratio < 0.15:
            classification = "AI_GENERATED"
            confidence_score = round(min(0.95, 0.6 + (0.0005 - energy_variance)), 2)
            explanation = "Low energy variation suggests AI-generated voice."
        else:
            classification = "HUMAN"
            confidence_score = round(min(0.95, 0.6 + energy_variance), 2)
            explanation = "Natural speech variations suggest human voice."

    except Exception:
        os.remove(filename)
        raise HTTPException(status_code=500, detail="Audio processing failed")

    os.remove(filename)

    return {
        "classification": classification,
        "confidence_score": confidence_score,
        "explanation": explanation
    }
