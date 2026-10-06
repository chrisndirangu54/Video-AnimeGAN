# Video-AnimeGAN

A Colab-friendly video stylization and AI-video workspace built around **real pretrained AnimeGANv2 weights**.

## Current architecture

- **Gradio web UI** as the single user interface
- pretrained AnimeGANv2 inference through `torch.hub`
- aspect-ratio-preserving video processing
- correct AnimeGAN `[-1, 1]` output conversion
- CUDA automatic mixed precision
- optical-flow temporal smoothing without untrained temporal networks
- original-audio preservation with FFmpeg
- CPU/CUDA/MPS device selection

The previous Android client and its dedicated FastAPI bridge have been removed so the repository stays focused on a lightweight Colab/browser workflow.

## Gradio tabs

### Anime Conversion — operational

Upload a normal video and convert it into an anime/cartoon-styled video using a pretrained AnimeGANv2 checkpoint.

Available styles:

- `paprika`
- `celeba_distill`
- `face_paint_512_v1`
- `face_paint_512_v2`

Controls include inference resolution, optical-flow temporal smoothing and CUDA mixed precision.

### Object Removal — UI ready

Intended backend:

`text prompt → Grounded SAM 2 tracking → ProPainter video inpainting`

### Recolor Object — UI ready

Intended backend:

`text prompt → Grounded SAM 2 mask → LAB/HSV recoloring → temporal compositing`

### Replace Video Text — UI ready

Intended backend:

`PaddleOCR → tracked text mask → ProPainter → perspective-aware replacement rendering`

### B&W Colorization — UI ready

Intended backend:

`DeOldify-compatible colorizer → temporal post-processing → FFmpeg audio remux`

### Generative Anime — experimental UI

This is reserved for a heavier video-diffusion backend that can reinterpret characters, backgrounds, clothing, lighting and visual style rather than only applying fast AnimeGAN style transfer.

The non-AnimeGAN tabs deliberately report their backend status instead of silently falling back to low-quality fake edits.

## Google Colab

```bash
!git clone https://github.com/chrisndirangu54/Video-AnimeGAN.git
%cd Video-AnimeGAN
!pip install -r requirements.txt
!apt-get -qq update && apt-get -qq install -y ffmpeg
```

Launch the UI:

```bash
!python app.py --share
```

Open the generated Gradio URL.

## CLI anime conversion

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

## Notes

The first AnimeGAN run downloads the selected checkpoint through PyTorch Hub. For long or 4K footage, use `--max-side 720` or `--max-side 960` in Colab to reduce GPU-memory use.

## Attribution

The pretrained cartoon generator is loaded from the open-source `bryandlee/animegan2-pytorch` implementation of AnimeGANv2. Review upstream model and code licenses before commercial deployment.
