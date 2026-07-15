from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from faster_whisper import WhisperModel
import tempfile
import os
import time

app = FastAPI(title="Whisper API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("Loading Whisper Medium Model...")

model = WhisperModel(
    "medium",
    device="cpu",
    compute_type="int8"
)

print("✅ Whisper Medium Loaded Successfully")


@app.get("/")
def home():
    return {
        "status": "running",
        "model": "medium"
    }


@app.get("/health")
def health():
    return {
        "success": True
    }


@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    start = time.time()

    suffix = os.path.splitext(file.filename)[1] or ".webm"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp:
        temp.write(await file.read())
        temp_path = temp.name

    try:
        segments, info = model.transcribe(
            temp_path,
            beam_size=5,
            best_of=5,
            temperature=0,
            vad_filter=True,
            condition_on_previous_text=True
        )

        text = " ".join(segment.text.strip() for segment in segments)

        return {
            "success": True,
            "language": info.language,
            "language_probability": round(info.language_probability, 3),
            "text": text,
            "time": round(time.time() - start, 2)
        }

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)