from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from faster_whisper import WhisperModel
import tempfile
import os

app = FastAPI(title="Whisper API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("Loading Whisper Model...")

model = WhisperModel(
    "base",
    device="cpu",
    compute_type="int8"
)

print("Model Loaded Successfully!")

@app.get("/")
def home():
    return {
        "status": "running",
        "model": "base"
    }

@app.get("/health")
def health():
    return {
        "success": True
    }

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):

    suffix = os.path.splitext(file.filename)[1] or ".webm"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp:
        temp.write(await file.read())
        temp_path = temp.name

    segments, info = model.transcribe(temp_path)

    text = " ".join(segment.text.strip() for segment in segments)

    os.remove(temp_path)

    return {
        "success": True,
        "language": info.language,
        "text": text
    }