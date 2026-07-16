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

print("Loading Whisper Model...")

model = WhisperModel(
    "base",
    device="cpu",
    compute_type="int8",
    cpu_threads=1,
    num_workers=1
)

print("✅ Model Loaded Successfully!")

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
    start = time.time()

    suffix = os.path.splitext(file.filename)[1] or ".webm"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp:
        temp.write(await file.read())
        temp_path = temp.name

    try:
        segments, info = model.transcribe(
            temp_path,
            language="en",                # Remove if auto detect needed
            beam_size=1,
            best_of=1,
            temperature=0,
            vad_filter=False,
            condition_on_previous_text=False
        )

        text = " ".join(segment.text.strip() for segment in segments)

        elapsed = round(time.time() - start, 2)

        print(f"⚡ Transcribed in {elapsed}s")

        return {
            "success": True,
            "language": info.language,
            "text": text,
            "time": elapsed
        }

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)