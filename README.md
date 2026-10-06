# Video-AnimeGAN

A Colab-friendly video stylization pipeline using **real pretrained AnimeGANv2 weights** rather than randomly initialized generator/ConvLSTM blocks.

## What changed

- Uses pretrained AnimeGANv2 weights via `torch.hub`
- Removes the untrained ConvLSTM path
- Preserves source aspect ratio and original output dimensions
- Correctly maps AnimeGAN output from `[-1, 1]` back to 8-bit video
- Adds CUDA automatic mixed precision
- Adds optical-flow temporal smoothing without requiring training
- Preserves/remuxes original audio with FFmpeg
- Adds CPU/CUDA/MPS device selection
- Adds input validation and a reproducible dependency file

## Supported pretrained styles

- `paprika`
- `celeba_distill`
- `face_paint_512_v1`
- `face_paint_512_v2`

## Google Colab

```bash
!git clone https://github.com/chrisndirangu54/Video-AnimeGAN.git
%cd Video-AnimeGAN
!pip install -r requirements.txt
!apt-get -qq update && apt-get -qq install -y ffmpeg
```

Upload a video to Colab, then run:

```bash
!python video_editor.py input.mp4 output.mp4 --style paprika --device auto
```

For lower VRAM usage:

```bash
!python video_editor.py input.mp4 output.mp4 --style paprika --max-side 720
```

Disable temporal smoothing:

```bash
!python video_editor.py input.mp4 output.mp4 --temporal-strength 0
```

## Python API

```python
from video_editor import process_video

process_video(
    input_video="input.mp4",
    output_video="output.mp4",
    style="paprika",
    max_side=1280,
    temporal_strength=0.18,
)
```

## Advanced editing roadmap

This repository now has a clean inference foundation for adding specialized pretrained editors without retraining the cartoon model:

- **Object selection/tracking:** Grounded SAM 2
- **Object removal / video inpainting:** ProPainter
- **Object recoloring:** SAM 2 masks + deterministic HSV/LAB transforms
- **Text detection/replacement:** PaddleOCR + tracked masks + ProPainter + OpenCV/Pillow rendering
- **Historical B&W colorization:** DeOldify

These should remain separate modules because they solve different vision tasks and have different model/license/runtime requirements.

## Notes

The first run downloads the selected AnimeGANv2 checkpoint through PyTorch Hub. For long or 4K footage, use `--max-side 720` or `--max-side 960` in Colab to reduce GPU memory use.

## Attribution

The pretrained cartoon generator is loaded from the open-source `bryandlee/animegan2-pytorch` implementation of AnimeGANv2. Review upstream licenses before commercial deployment.


## Android UI (Gradle + Jetpack Compose)

A native Android client lives in `android-ui/`.

### Start the backend

```bash
pip install -r requirements.txt
uvicorn backend_api:app --host 0.0.0.0 --port 8000
```

### Run the Android app

Open `android-ui/` in Android Studio and run the `app` configuration.

The Android emulator uses `http://10.0.2.2:8000/` to reach a backend running on the development machine. For a physical device or hosted GPU, set `API_BASE_URL` in `android-ui/app/build.gradle.kts` to the appropriate HTTPS endpoint.

The UI supports video selection, AnimeGANv2 style selection, upload/processing state, error feedback, and output preview using Media3/ExoPlayer.
