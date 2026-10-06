from __future__ import annotations

import os
import uuid
from pathlib import Path

import gradio as gr

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


def stylize_video_ui(
    input_video: str | None,
    style: str,
    max_side: int,
    temporal_strength: float,
    use_amp: bool,
    progress=gr.Progress(track_tqdm=True),
):
    src = _require_video(input_video)
    output_path = OUTPUT_DIR / f"anime_{uuid.uuid4().hex[:10]}.mp4"

    progress(0.05, desc="Preparing model and video")
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
        raise gr.Error(f"Video processing failed: {exc}") from exc

    progress(1.0, desc="Finished")
    return str(output_path), str(output_path)


def planned_tool_status(video: str | None, operation: str, details: str) -> str:
    _require_video(video)
    return (
        f"### {operation}\n"
        f"{details}\n\n"
        "**Status:** UI is wired and ready for the specialized pretrained backend, "
        "but this operation is intentionally not faked with frame-by-frame heuristics. "
        "The required model integration will be added as a separate optional module so "
        "the working AnimeGANv2 path remains lightweight and reliable in Colab."
    )


def removal_status(video, object_prompt):
    return planned_tool_status(
        video,
        "Object Removal",
        f"Target: **{object_prompt or 'not specified'}**. "
        "Planned backend: Grounded SAM 2 for prompt-based tracking + ProPainter for temporally consistent video inpainting.",
    )


def recolor_status(video, object_prompt, target_color):
    return planned_tool_status(
        video,
        "Object Recolor",
        f"Target: **{object_prompt or 'not specified'}** → **{target_color or 'not specified'}**. "
        "Planned backend: Grounded SAM 2 masks + LAB/HSV recoloring while preserving lighting and texture.",
    )


def text_status(video, old_text, new_text):
    return planned_tool_status(
        video,
        "Replace Video Text",
        f"Replace **{old_text or 'detected text'}** with **{new_text or 'new text'}**. "
        "Planned backend: PaddleOCR + tracked text masks + ProPainter background reconstruction + perspective-aware rendering.",
    )


def colorize_status(video):
    return planned_tool_status(
        video,
        "B&W Colorization",
        "Planned backend: DeOldify-compatible colorization module with temporal post-processing and original-audio remuxing.",
    )


def generative_status(video, prompt, strength):
    return planned_tool_status(
        video,
        "Generative Video-to-Video Anime",
        f"Prompt: **{prompt or 'anime reinterpretation'}**; transformation strength: **{strength:.2f}**. "
        "Planned backend: a video diffusion model with temporal consistency controls. "
        "This is separate from the fast AnimeGANv2 style-transfer mode.",
    )


with gr.Blocks(title="Video AnimeGAN Studio", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # Video AnimeGAN Studio
        A Colab-friendly AI video workspace centered on pretrained AnimeGANv2.

        **Anime Conversion is operational now.** The other tabs expose the intended workflow
        and are kept explicit about model availability rather than pretending unsupported edits work.
        """
    )

    with gr.Tab("Anime Conversion"):
        with gr.Row():
            with gr.Column(scale=1):
                anime_input = gr.Video(label="Input video", sources=["upload"], format=None)
                anime_style = gr.Dropdown(
                    choices=sorted(PRETRAINED_STYLES),
                    value="paprika",
                    label="Anime style",
                )
                anime_max_side = gr.Slider(
                    minimum=0,
                    maximum=1920,
                    value=960,
                    step=32,
                    label="Inference max side",
                    info="Lower values use less VRAM. Set to 0 for original resolution.",
                )
                anime_temporal = gr.Slider(
                    minimum=0.0,
                    maximum=0.6,
                    value=0.18,
                    step=0.01,
                    label="Temporal smoothing",
                )
                anime_amp = gr.Checkbox(value=True, label="Use CUDA mixed precision")
                anime_run = gr.Button("Create Anime Video", variant="primary")
            with gr.Column(scale=1):
                anime_output = gr.Video(label="Anime output")
                anime_download = gr.File(label="Download MP4")

        anime_run.click(
            fn=stylize_video_ui,
            inputs=[anime_input, anime_style, anime_max_side, anime_temporal, anime_amp],
            outputs=[anime_output, anime_download],
            concurrency_limit=1,
        )

    with gr.Tab("Object Removal"):
        removal_video = gr.Video(label="Input video", sources=["upload"], format=None)
        removal_prompt = gr.Textbox(
            label="Object to remove",
            placeholder="e.g. bottle on the table, person in the background",
        )
        removal_run = gr.Button("Prepare Object Removal", variant="primary")
        removal_info = gr.Markdown()
        removal_run.click(removal_status, [removal_video, removal_prompt], removal_info)

    with gr.Tab("Recolor Object"):
        recolor_video = gr.Video(label="Input video", sources=["upload"], format=None)
        recolor_prompt = gr.Textbox(label="Object", placeholder="e.g. jacket, red car")
        recolor_color = gr.Textbox(label="Target color", placeholder="e.g. matte black, metallic blue")
        recolor_run = gr.Button("Prepare Recolor", variant="primary")
        recolor_info = gr.Markdown()
        recolor_run.click(recolor_status, [recolor_video, recolor_prompt, recolor_color], recolor_info)

    with gr.Tab("Replace Video Text"):
        text_video = gr.Video(label="Input video", sources=["upload"], format=None)
        with gr.Row():
            old_text = gr.Textbox(label="Existing text", placeholder="Leave blank for OCR detection")
            new_text = gr.Textbox(label="Replacement text", placeholder="e.g. TECHRUPTORS")
        text_run = gr.Button("Prepare Text Replacement", variant="primary")
        text_info = gr.Markdown()
        text_run.click(text_status, [text_video, old_text, new_text], text_info)

    with gr.Tab("B&W Colorization"):
        color_video = gr.Video(label="Black-and-white video", sources=["upload"], format=None)
        color_run = gr.Button("Prepare Colorization", variant="primary")
        color_info = gr.Markdown()
        color_run.click(colorize_status, color_video, color_info)

    with gr.Tab("Generative Anime"):
        gen_video = gr.Video(label="Input video", sources=["upload"], format=None)
        gen_prompt = gr.Textbox(
            label="Prompt",
            placeholder="e.g. cinematic hand-drawn anime, detailed cel shading, dramatic evening light",
        )
        gen_strength = gr.Slider(0.1, 1.0, value=0.65, step=0.05, label="Transformation strength")
        gen_run = gr.Button("Prepare Generative Anime", variant="primary")
        gen_info = gr.Markdown()
        gen_run.click(generative_status, [gen_video, gen_prompt, gen_strength], gen_info)

    gr.Markdown(
        """
        ### Google Colab
        Install dependencies, then run:

        `python app.py --share`

        The generated Gradio URL opens the complete browser interface.
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
