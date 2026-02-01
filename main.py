from fastapi import FastAPI, Header, HTTPException
import base64, io
import librosa
import numpy as np
import torch

import os
API_KEY = os.getenv("API_KEY")


app = FastAPI()

# Dummy model (abhi ke liye)
def fake_model(features):
    return np.random.rand()

def load_audio_from_base64(b64_string):
    audio_bytes = base64.b64decode(b64_string)
    audio_buffer = io.BytesIO(audio_bytes)
    y, sr = librosa.load(audio_buffer, sr=16000)
    return y, sr

def extract_features(y, sr):
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return mel_db

def verify_key(auth: str):
    if auth != f"Bearer {API_KEY}":
        raise HTTPException(status_code=401, detail="Unauthorized")

@app.post("/detect")
def detect_voice(data: dict, Authorization: str = Header(None)):
    verify_key(Authorization)

    if "audio" not in data:
        raise HTTPException(status_code=400, detail="Missing audio")

    y, sr = load_audio_from_base64(data["audio"])
    features = extract_features(y, sr)

    prob = fake_model(features)

    result = "AI_GENERATED" if prob > 0.5 else "HUMAN"

    return {
        "classification": result,
        "confidence": float(round(prob, 3))
    }
