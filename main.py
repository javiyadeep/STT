from fastapi import FastAPI, UploadFile, File, Form
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

# "base" (no ".en" suffix) is the multilingual checkpoint — it already
# understands Hindi and Gujarati alongside English. If accuracy on
# Hindi/Gujarati feels weak, bump this to "small" (still fine on CPU with
# int8, just a bit slower per chunk).
model = WhisperModel(
    "small",
    device="cpu",
    compute_type="int8"
)

print("✅ Model Loaded Successfully!")

@app.get("/")
def home():
    return {
        "status": "running",
        "model": "small"
    }

@app.get("/health")
def health():
    return {
        "success": True
    }


@app.post("/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    language: str = Form(None)   # "en" / "hi" / "gu" / None (=> auto-detect)
):
    start = time.time()

    suffix = os.path.splitext(file.filename)[1] or ".webm"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp:
        temp.write(await file.read())
        temp_path = temp.name

    try:
        # Treat empty string / "auto" the same as "let Whisper auto-detect"
        lang_param = None if (not language or language.lower() == "auto") else language

        segments, info = model.transcribe(
            temp_path,
            language=lang_param,          # None = auto-detect per chunk (handles code-mixing)
            beam_size=1,
            best_of=1,
            temperature=0,
            vad_filter=True,              # skip silence -> fewer hallucinations
            vad_parameters=dict(min_silence_duration_ms=300),
            condition_on_previous_text=False,
            no_speech_threshold=0.6,
            compression_ratio_threshold=2.4,
        )

        text = " ".join(
            segment.text.strip()
            for segment in segments
            if segment.no_speech_prob < 0.6
        )

        elapsed = round(time.time() - start, 2)

        print(f"⚡ Transcribed in {elapsed}s | detected language: {info.language}")

        return {
            "success": True,
            "language": info.language,     # tells the frontend what it detected
            "text": text,
            "time": elapsed
        }

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)