from __future__ import annotations

import os
import uuid
from pathlib import Path

import gradio as gr

from object_removal import remove_object_from_video
from video_editor import PRETRAINED_STYLES, process_video


OUTPUT_DIR = Path(os.environ.get("VIDEO_ANIMEGAN_OUTPUT_DIR", "outputs"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _require_video(video: str | None) -> Path:
    if not video:
        raise gr.Error("Please upload a video first.")
    path = Path(video)
    if not path.exists():
        raise gr.Error("The uploaded video could not be found.")
    return path


def stylize_video_ui(input_video, style, max_side, temporal_strength, use_amp, progress=gr.Progress(track_tqdm=True)):
    src = _require_video(input_video)
    output_path = OUTPUT_DIR / f"anime_{uuid.uuid4().hex[:10]}.mp4"
    progress(0.05, desc="Preparing AnimeGANv2")
    try:
        process_video(
            input_video=str(src),
            output_video=str(output_path),
            style=style,
            device="auto",
            max_side=None if int(max_side) == 0 else int(max_side),
            temporal_strength=float(temporal_strength),
            use_amp=bool(use_amp),
        )
    except Exception as exc:
        raise gr.Error(f"Anime conversion failed: {exc}") from exc
    progress(1.0, desc="Finished")
    return str(output_path), str(output_path)


def remove_object_ui(
    input_video,
    object_prompt,
    box_threshold,
    text_threshold,
    scan_stride,
    mask_dilation,
    resize_ratio,
    fp16,
    progress=gr.Progress(track_tqdm=True),
):
    src = _require_video(input_video)
    if not object_prompt or not object_prompt.strip():
        raise gr.Error("Describe the object you want removed.")

    output_path = OUTPUT_DIR / f"removed_{uuid.uuid4().hex[:10]}.mp4"
    progress(0.05, desc="Grounding object prompt")
    try:
        result = remove_object_from_video(
            input_video=str(src),
            output_video=str(output_path),
            prompt=object_prompt,
            box_threshold=float(box_threshold),
            text_threshold=float(text_threshold),
            scan_stride=int(scan_stride),
            mask_dilation=int(mask_dilation),
            resize_ratio=float(resize_ratio),
            fp16=bool(fp16),
        )
    except Exception as exc:
        raise gr.Error(f"Object removal failed: {exc}") from exc
    progress(1.0, desc="Finished")
    return result, result


def planned_tool_status(video: str | None, operation: str, details: str) -> str:
    _require_video(video)
    return (
        f"### {operation}\n{details}\n\n"
        "**Status:** UI is ready, but this specialized backend is not enabled yet. "
        "It will reuse the Grounded SAM 2 masks where appropriate."
    )


with gr.Blocks(title="Video AnimeGAN Studio", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # Video AnimeGAN Studio
        Pretrained anime conversion and AI-assisted video editing for Colab.

        **Operational now:** Anime Conversion and Grounded Object Removal.
        """
    )

    with gr.Tab("Anime Conversion"):
        with gr.Row():
            with gr.Column():
                anime_input = gr.Video(label="Input video", sources=["upload"], format=None)
                anime_style = gr.Dropdown(sorted(PRETRAINED_STYLES), value="paprika", label="Anime style")
                anime_max_side = gr.Slider(0, 1920, value=960, step=32, label="Inference max side")
                anime_temporal = gr.Slider(0.0, 0.6, value=0.18, step=0.01, label="Temporal smoothing")
                anime_amp = gr.Checkbox(value=True, label="Use CUDA mixed precision")
                anime_run = gr.Button("Create Anime Video", variant="primary")
            with gr.Column():
                anime_output = gr.Video(label="Anime output")
                anime_download = gr.File(label="Download MP4")

        anime_run.click(
            stylize_video_ui,
            [anime_input, anime_style, anime_max_side, anime_temporal, anime_amp],
            [anime_output, anime_download],
            concurrency_limit=1,
        )

    with gr.Tab("Object Removal"):
        gr.Markdown(
            "Uses **Grounding DINO → SAM 2 tracking → ProPainter**. "
            "Describe the target with ordinary text, such as person in the background."
        )
        with gr.Row():
            with gr.Column():
                removal_video = gr.Video(label="Input video", sources=["upload"], format=None)
                removal_prompt = gr.Textbox(
                    label="Object to remove",
                    placeholder="e.g. bottle on the table",
                )
                with gr.Accordion("Advanced controls", open=False):
                    box_threshold = gr.Slider(0.1, 0.8, value=0.35, step=0.01, label="Grounding box threshold")
                    text_threshold = gr.Slider(0.1, 0.8, value=0.25, step=0.01, label="Grounding text threshold")
                    scan_stride = gr.Slider(1, 120, value=30, step=1, label="Detection scan stride (frames)")
                    mask_dilation = gr.Slider(0, 20, value=6, step=1, label="Mask dilation")
                    resize_ratio = gr.Slider(0.25, 1.0, value=1.0, step=0.05, label="ProPainter processing scale")
                    removal_fp16 = gr.Checkbox(value=True, label="Use ProPainter FP16")
                removal_run = gr.Button("Remove Object", variant="primary")
            with gr.Column():
                removal_output = gr.Video(label="Object-removed output")
                removal_download = gr.File(label="Download MP4")

        removal_run.click(
            remove_object_ui,
            [
                removal_video,
                removal_prompt,
                box_threshold,
                text_threshold,
                scan_stride,
                mask_dilation,
                resize_ratio,
                removal_fp16,
            ],
            [removal_output, removal_download],
            concurrency_limit=1,
        )

    with gr.Tab("Recolor Object"):
        recolor_video = gr.Video(label="Input video", sources=["upload"], format=None)
        recolor_prompt = gr.Textbox(label="Object", placeholder="e.g. jacket")
        recolor_color = gr.Textbox(label="Target color", placeholder="e.g. metallic blue")
        recolor_run = gr.Button("Prepare Recolor")
        recolor_info = gr.Markdown()
        recolor_run.click(
            lambda v, o, c: planned_tool_status(
                v,
                "Recolor Object",
                f"Target: **{o or 'not specified'}** → **{c or 'not specified'}**. "
                "The next backend will reuse Grounded SAM 2 masks and apply lighting-aware LAB/HSV recoloring.",
            ),
            [recolor_video, recolor_prompt, recolor_color],
            recolor_info,
        )

    with gr.Tab("Replace Video Text"):
        text_video = gr.Video(label="Input video", sources=["upload"], format=None)
        old_text = gr.Textbox(label="Existing text", placeholder="Leave blank for OCR detection")
        new_text = gr.Textbox(label="Replacement text", placeholder="e.g. TECHRUPTORS")
        text_run = gr.Button("Prepare Text Replacement")
        text_info = gr.Markdown()
        text_run.click(
            lambda v, o, n: planned_tool_status(
                v,
                "Replace Video Text",
                f"Replace **{o or 'detected text'}** with **{n or 'new text'}**. "
                "Planned backend: OCR + Grounded/SAM masks + ProPainter + perspective-aware rendering.",
            ),
            [text_video, old_text, new_text],
            text_info,
        )

    with gr.Tab("B&W Colorization"):
        color_video = gr.Video(label="Black-and-white video", sources=["upload"], format=None)
        color_run = gr.Button("Prepare Colorization")
        color_info = gr.Markdown()
        color_run.click(
            lambda v: planned_tool_status(
                v,
                "B&W Colorization",
                "Planned backend: DeOldify-compatible colorization with temporal post-processing.",
            ),
            color_video,
            color_info,
        )

    with gr.Tab("Generative Anime"):
        gen_video = gr.Video(label="Input video", sources=["upload"], format=None)
        gen_prompt = gr.Textbox(label="Prompt", placeholder="cinematic hand-drawn anime, detailed cel shading")
        gen_strength = gr.Slider(0.1, 1.0, value=0.65, step=0.05, label="Transformation strength")
        gen_run = gr.Button("Prepare Generative Anime")
        gen_info = gr.Markdown()
        gen_run.click(
            lambda v, p, s: planned_tool_status(
                v,
                "Generative Anime",
                f"Prompt: **{p or 'anime reinterpretation'}**; strength: **{s:.2f}**. "
                "This will use a temporal video-diffusion backend rather than frame-only stylization.",
            ),
            [gen_video, gen_prompt, gen_strength],
            gen_info,
        )

    gr.Markdown(
        """
        ### Colab setup
        Anime-only setup: pip install -r requirements.txt

        Complete object-removal stack: bash setup_editing_models.sh

        Launch: python app.py --share
        """
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Launch the Video-AnimeGAN Gradio UI.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    demo.queue(default_concurrency_limit=1).launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        show_error=True,
    )
