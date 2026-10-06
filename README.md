# Video-AnimeGAN

A Colab-friendly video stylization and AI-video editing workspace built around pretrained open-source models.

## Operational features

### Anime Conversion

Uses pretrained AnimeGANv2 weights with aspect-ratio-preserving inference, CUDA mixed precision, optical-flow temporal smoothing, original-audio preservation, and browser preview/download through Gradio.

### Grounded Object Removal

The Object Removal tab is wired to a real three-stage pipeline:

    natural-language target
            ↓
    Grounding DINO
    open-vocabulary detection
            ↓
    SAM 2
    video mask propagation/tracking
            ↓
    ProPainter
    temporal video inpainting
            ↓
    FFmpeg
    original audio remux

The detector samples frames until it finds the requested object. SAM 2 then propagates that object mask forward and backward through the clip, and ProPainter reconstructs the selected region across time.

Example prompts:

- person in the background
- bottle on the table
- red car
- microphone

## Google Colab

Clone:

    !git clone https://github.com/chrisndirangu54/Video-AnimeGAN.git
    %cd Video-AnimeGAN

Install the full editing stack:

    !bash setup_editing_models.sh
    !apt-get -qq update && apt-get -qq install -y ffmpeg

Launch Gradio:

    !python app.py --share

The first object-removal run downloads Grounding DINO and SAM 2 weights. ProPainter downloads its pretrained weights automatically on first inference.

## Gradio tabs

- Anime Conversion — operational
- Object Removal — operational with Grounding DINO + SAM 2 + ProPainter
- Recolor Object — UI ready; intended to reuse SAM 2 masks
- Replace Video Text — UI ready; intended to reuse masks + ProPainter
- B&W Colorization — UI ready
- Generative Anime — experimental UI

## Object-removal controls

- Grounding box threshold: lower it if the target is missed; raise it to reject weak detections.
- Grounding text threshold: controls prompt-region matching confidence.
- Detection scan stride: controls how frequently frames are searched for a grounding frame.
- Mask dilation: expands the removed region around object boundaries.
- ProPainter processing scale: lower values reduce VRAM demand.
- FP16: recommended on NVIDIA Colab GPUs.

## Python API

    from object_removal import remove_object_from_video

    remove_object_from_video(
        input_video="input.mp4",
        output_video="removed.mp4",
        prompt="person in the background",
        box_threshold=0.35,
        text_threshold=0.25,
        scan_stride=30,
        mask_dilation=6,
        resize_ratio=1.0,
    )

## Design choices

Grounding DINO is loaded through Hugging Face Transformers, which avoids compiling the original Grounding DINO custom deformable-attention extension. SAM 2 is installed from Meta's official repository and performs video mask propagation. ProPainter remains the temporal inpainting stage instead of performing independent per-frame fills.

For long or high-resolution videos on Colab, lower the ProPainter scale to about 0.5–0.75 to reduce GPU-memory pressure.

## Attribution and licensing

This project orchestrates third-party pretrained models. Review the upstream code and model licenses for AnimeGANv2, Grounding DINO, SAM 2, and ProPainter before commercial deployment.
