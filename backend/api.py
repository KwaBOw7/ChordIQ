from pathlib import Path
import shutil
import tempfile

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from main import analyze_song


app = FastAPI(
    title="ChordIQ API",
    version="1.0.0",
    description="Audio analysis API for ChordIQ.",
)

# Allow the React development server to communicate with FastAPI.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


ALLOWED_EXTENSIONS = {
    ".mp3",
    ".wav",
    ".flac",
    ".m4a",
    ".ogg",
    ".aac",
}


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "service": "ChordIQ",
    }


@app.post("/analyze")
async def analyze_audio(
    file: UploadFile = File(...),
    meter: str | None = None,
    bpm: float | None = None,
    key: str | None = None,
):
    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="No audio file was provided.",
        )

    extension = Path(
        file.filename
    ).suffix.lower()

    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported audio format: "
                f"{extension}. "
                f"Supported formats: "
                f"{', '.join(sorted(ALLOWED_EXTENSIONS))}"
            ),
        )

    temp_path = None

    try:
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=extension,
        ) as temp_file:

            shutil.copyfileobj(
                file.file,
                temp_file,
            )

            temp_path = Path(
                temp_file.name
            )

        result = analyze_song(
            str(temp_path),
            verbose=False,
            meter=meter,
            bpm=bpm,
            key=key,
        )

        # Include the original filename so the UI
        # knows which song was analyzed.
        result["filename"] = file.filename

        return result

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {error}",
        ) from error

    finally:
        if temp_path and temp_path.exists():
            temp_path.unlink()
