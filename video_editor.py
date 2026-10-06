from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm
from torchvision.transforms.functional import to_tensor


PRETRAINED_STYLES = {
    "paprika",
    "celeba_distill",
    "face_paint_512_v1",
    "face_paint_512_v2",
}


def pick_device(requested: str = "auto") -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_pretrained_animegan(style: str, device: torch.device) -> torch.nn.Module:
    if style not in PRETRAINED_STYLES:
        raise ValueError(f"Unknown style '{style}'. Choose from: {sorted(PRETRAINED_STYLES)}")

    model = torch.hub.load(
        "bryandlee/animegan2-pytorch:main",
        "generator",
        pretrained=style,
        trust_repo=True,
    )
    return model.to(device).eval()


def _resize_for_inference(frame_rgb: np.ndarray, max_side: int | None) -> tuple[np.ndarray, tuple[int, int]]:
    h, w = frame_rgb.shape[:2]
    if not max_side or max(h, w) <= max_side:
        return frame_rgb, (w, h)

    scale = max_side / float(max(h, w))
    nw = max(32, int(round(w * scale)))
    nh = max(32, int(round(h * scale)))
    # AnimeGAN works best when dimensions are divisible by 32.
    nw = max(32, (nw // 32) * 32)
    nh = max(32, (nh // 32) * 32)
    resized = cv2.resize(frame_rgb, (nw, nh), interpolation=cv2.INTER_AREA)
    return resized, (w, h)


def stylize_frame(
    frame_bgr: np.ndarray,
    model: torch.nn.Module,
    device: torch.device,
    max_side: int | None = 1280,
    use_amp: bool = True,
) -> np.ndarray:
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    resized, original_size = _resize_for_inference(frame_rgb, max_side)

    tensor = to_tensor(resized).unsqueeze(0).to(device)
    tensor = tensor * 2.0 - 1.0

    amp_enabled = device.type == "cuda" and use_amp
    with torch.inference_mode():
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp_enabled):
            output = model(tensor)

    output = output[0].float().cpu()
    output = ((output * 0.5 + 0.5).clamp(0, 1) * 255.0).byte()
    out_rgb = output.permute(1, 2, 0).numpy()

    if (out_rgb.shape[1], out_rgb.shape[0]) != original_size:
        out_rgb = cv2.resize(out_rgb, original_size, interpolation=cv2.INTER_CUBIC)

    return cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)


def _optical_flow(prev_bgr: np.ndarray, curr_bgr: np.ndarray) -> np.ndarray:
    prev_gray = cv2.cvtColor(prev_bgr, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr_bgr, cv2.COLOR_BGR2GRAY)
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST)
    return dis.calc(prev_gray, curr_gray, None)


def _warp_previous(previous_bgr: np.ndarray, flow: np.ndarray) -> np.ndarray:
    h, w = flow.shape[:2]
    grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
    map_x = (grid_x + flow[..., 0]).astype(np.float32)
    map_y = (grid_y + flow[..., 1]).astype(np.float32)
    return cv2.remap(
        previous_bgr,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT101,
    )


def temporal_smooth(
    previous_source: np.ndarray | None,
    current_source: np.ndarray,
    previous_stylized: np.ndarray | None,
    current_stylized: np.ndarray,
    strength: float,
) -> np.ndarray:
    if previous_source is None or previous_stylized is None or strength <= 0:
        return current_stylized

    flow = _optical_flow(previous_source, current_source)
    warped_previous = _warp_previous(previous_stylized, flow)

    # Keep new scene content dominant while damping frame-to-frame flicker.
    alpha = float(np.clip(strength, 0.0, 0.95))
    return cv2.addWeighted(current_stylized, 1.0 - alpha, warped_previous, alpha, 0.0)


def _mux_original_audio(silent_video: Path, original_video: Path, output_video: Path) -> None:
    if shutil.which("ffmpeg") is None:
        shutil.copy2(silent_video, output_video)
        print("ffmpeg was not found; wrote video without remuxed audio.")
        return

    command = [
        "ffmpeg", "-y",
        "-i", str(silent_video),
        "-i", str(original_video),
        "-map", "0:v:0",
        "-map", "1:a?",
        "-c:v", "libx264",
        "-preset", "medium",
        "-crf", "18",
        "-c:a", "aac",
        "-b:a", "192k",
        "-shortest",
        str(output_video),
    ]
    subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def process_video(
    input_video: str,
    output_video: str,
    style: str = "paprika",
    device: str = "auto",
    max_side: int | None = 1280,
    temporal_strength: float = 0.18,
    use_amp: bool = True,
) -> None:
    source = Path(input_video)
    target = Path(output_video)
    if not source.exists():
        raise FileNotFoundError(source)

    dev = pick_device(device)
    print(f"Using device: {dev}")
    print(f"Loading pretrained AnimeGANv2 style: {style}")
    model = load_pretrained_animegan(style, dev)

    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open input video: {source}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None

    target.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="video_animegan_") as tmp:
        silent_path = Path(tmp) / "silent.mp4"
        writer = cv2.VideoWriter(
            str(silent_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
        if not writer.isOpened():
            cap.release()
            raise RuntimeError("Could not initialize output video writer.")

        previous_source = None
        previous_stylized = None

        try:
            iterator = tqdm(total=total, desc="Stylizing", unit="frame")
            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                stylized = stylize_frame(
                    frame,
                    model=model,
                    device=dev,
                    max_side=max_side,
                    use_amp=use_amp,
                )
                stylized = temporal_smooth(
                    previous_source,
                    frame,
                    previous_stylized,
                    stylized,
                    temporal_strength,
                )
                writer.write(stylized)

                previous_source = frame
                previous_stylized = stylized
                iterator.update(1)
            iterator.close()
        finally:
            cap.release()
            writer.release()

        _mux_original_audio(silent_path, source, target)

    print(f"Saved: {target}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Pretrained AnimeGANv2 video stylizer.")
    parser.add_argument("input", help="Input video path")
    parser.add_argument("output", help="Output MP4 path")
    parser.add_argument("--style", default="paprika", choices=sorted(PRETRAINED_STYLES))
    parser.add_argument("--device", default="auto", help="auto, cuda, cpu, or mps")
    parser.add_argument("--max-side", type=int, default=1280, help="Inference max image side; 0 disables resizing")
    parser.add_argument("--temporal-strength", type=float, default=0.18, help="0 disables optical-flow smoothing")
    parser.add_argument("--no-amp", action="store_true", help="Disable CUDA mixed precision")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    process_video(
        input_video=args.input,
        output_video=args.output,
        style=args.style,
        device=args.device,
        max_side=None if args.max_side == 0 else args.max_side,
        temporal_strength=args.temporal_strength,
        use_amp=not args.no_amp,
    )


if __name__ == "__main__":
    main()
