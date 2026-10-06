from pathlib import Path
import shutil
import tempfile
import uuid

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

from video_editor import PRETRAINED_STYLES, process_video

app = FastAPI(title="Video-AnimeGAN API", version="1.0.0")

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

@app.get("/health")
def health():
    return {"status": "ok", "styles": sorted(PRETRAINED_STYLES)}

@app.post("/stylize")
async def stylize_video(
    video: UploadFile = File(...),
    style: str = Form("paprika"),
    temporal_strength: float = Form(0.18),
    max_side: int = Form(1280),
):
    if style not in PRETRAINED_STYLES:
        raise HTTPException(status_code=400, detail=f"Unsupported style: {style}")

    suffix = Path(video.filename or "input.mp4").suffix or ".mp4"
    job_id = uuid.uuid4().hex
    output_path = OUTPUT_DIR / f"{job_id}.mp4"

    with tempfile.TemporaryDirectory(prefix="animegan_api_") as tmp:
        input_path = Path(tmp) / f"input{suffix}"
        with input_path.open("wb") as f:
            shutil.copyfileobj(video.file, f)

        try:
            process_video(
                input_video=str(input_path),
                output_video=str(output_path),
                style=style,
                max_side=None if max_side <= 0 else max_side,
                temporal_strength=temporal_strength,
            )
        except Exception as exc:
            if output_path.exists():
                output_path.unlink(missing_ok=True)
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    return FileResponse(
        output_path,
        media_type="video/mp4",
        filename=f"anime_{Path(video.filename or 'video').stem}.mp4",
    )
