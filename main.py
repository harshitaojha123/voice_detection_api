from fastapi import FastAPI, Header, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import os
import uuid

import librosa
import numpy as np

# ---------------- CONFIG ----------------
API_KEY = os.getenv("API_KEY")

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
        "endpoint": "/detect-voice-file"
    }

# ---------------- API ENDPOINT (FILE UPLOAD) ----------------
@app.post("/detect-voice-file")
async def detect_voice(
    file: UploadFile = File(...),
    language: str = "en",
    x_api_key: str = Header(None)
):

    # 1️⃣ API KEY VALIDATION
    if not API_KEY or x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")

    # 2️⃣ LANGUAGE VALIDATION
    supported_languages = ["en", "hi", "ta", "ml", "te"]
    if language not in supported_languages:
        raise HTTPException(status_code=400, detail="Unsupported language")

    # 3️⃣ FILE TYPE VALIDATION
    if not file.filename.lower().endswith(".wav"):
        raise HTTPException(
            status_code=400,
            detail="Only WAV audio format is supported"
        )

    filename = f"audio_{uuid.uuid4().hex}.wav"

    # 4️⃣ SAVE FILE
    with open(filename, "wb") as f:
        f.write(await file.read())

    try:
        # 5️⃣ LOAD AUDIO (Railway-safe)
        y, sr = librosa.load(filename, sr=None, backend="soundfile")

        # 6️⃣ FEATURE EXTRACTION
        rms = librosa.feature.rms(y=y)[0]
        energy_variance = np.var(rms)
        silence_ratio = np.sum(np.abs(y) < 0.01) / len(y)

        # 7️⃣ DECISION LOGIC
        if energy_variance < 0.0005 and silence_ratio < 0.15:
            classification = "AI_GENERATED"
            confidence_score = 0.85
            explanation = "Low energy variation and uniform speech patterns detected."
        else:
            classification = "HUMAN"
            confidence_score = 0.75
            explanation = "Natural energy variations detected in speech."

    except Exception:
        os.remove(filename)
        raise HTTPException(status_code=500, detail="Audio processing failed")

    # 8️⃣ CLEANUP
    os.remove(filename)

    return {
        "classification": classification,
        "confidence_score": confidence_score,
        "explanation": explanation
    }
