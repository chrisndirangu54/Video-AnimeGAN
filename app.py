from __future__ import annotations

import os
import tempfile
import uuid
from pathlib import Path

import gradio as gr

from video_editor import PRETRAINED_STYLES, process_video


OUTPUT_DIR = Path(os.environ.get("VIDEO_ANIMEGAN_OUTPUT_DIR", "outputs"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def stylize_video_ui(
    input_video: str | None,
    style: str,
    max_side: int,
    temporal_strength: float,
    use_amp: bool,
    progress=gr.Progress(track_tqdm=True),
):
    if not input_video:
        raise gr.Error("Please upload a video first.")

    src = Path(input_video)
    if not src.exists():
        raise gr.Error("The uploaded video could not be found.")

    suffix = src.suffix.lower() if src.suffix else ".mp4"
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


with gr.Blocks(title="Video AnimeGAN Studio", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # Video AnimeGAN Studio
        Convert ordinary video into anime/cartoon-styled footage with pretrained AnimeGANv2.

        The app preserves the original video dimensions and remuxes the source audio when FFmpeg is available.
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            input_video = gr.Video(
                label="Input video",
                sources=["upload"],
                format=None,
            )
            style = gr.Dropdown(
                choices=sorted(PRETRAINED_STYLES),
                value="paprika",
                label="Anime style",
            )
            max_side = gr.Slider(
                minimum=0,
                maximum=1920,
                value=960,
                step=32,
                label="Inference max side",
                info="Lower values use less VRAM. Set to 0 for original resolution.",
            )
            temporal_strength = gr.Slider(
                minimum=0.0,
                maximum=0.6,
                value=0.18,
                step=0.01,
                label="Temporal smoothing",
                info="Higher values reduce flicker but may soften fast motion.",
            )
            use_amp = gr.Checkbox(
                value=True,
                label="Use CUDA mixed precision",
                info="Recommended on Colab GPU.",
            )
            run_button = gr.Button("Create Anime Video", variant="primary")

        with gr.Column(scale=1):
            output_video = gr.Video(label="Anime output")
            download_file = gr.File(label="Download MP4")

    run_button.click(
        fn=stylize_video_ui,
        inputs=[
            input_video,
            style,
            max_side,
            temporal_strength,
            use_amp,
        ],
        outputs=[output_video, download_file],
        concurrency_limit=1,
    )

    gr.Markdown(
        """
        ### Colab tip
        Start the app with:

        `python app.py --share`

        Gradio will provide a temporary public link that you can open from any browser.
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
