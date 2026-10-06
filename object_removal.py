from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image


GROUNDING_MODEL_ID = os.environ.get("GROUNDING_DINO_MODEL_ID", "IDEA-Research/grounding-dino-tiny")
SAM2_MODEL_ID = os.environ.get("SAM2_MODEL_ID", "facebook/sam2-hiera-large")
PROPAINTER_DIR = Path(os.environ.get("PROPAINTER_DIR", "third_party/ProPainter"))


def _require_editing_dependencies() -> None:
    missing = []
    try:
        import transformers  # noqa: F401
    except Exception:
        missing.append("transformers")
    try:
        import sam2  # noqa: F401
    except Exception:
        missing.append("sam2")

    if missing:
        raise RuntimeError(
            "Missing object-removal dependencies: "
            + ", ".join(missing)
            + ". Run: bash setup_editing_models.sh"
        )

    if not (PROPAINTER_DIR / "inference_propainter.py").exists():
        raise RuntimeError(
            f"ProPainter was not found at {PROPAINTER_DIR}. Run: bash setup_editing_models.sh"
        )


def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _extract_frames(video_path: Path, frame_dir: Path) -> int:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    count = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        cv2.imwrite(str(frame_dir / f"{count:06d}.jpg"), frame)
        count += 1

    cap.release()
    if count == 0:
        raise RuntimeError("The video contains no readable frames.")
    return count


def _load_grounding_dino():
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

    device = _device()
    processor = AutoProcessor.from_pretrained(GROUNDING_MODEL_ID)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(GROUNDING_MODEL_ID).to(device)
    model.eval()
    return processor, model, device


def _detect_boxes(
    image_path: Path,
    prompt: str,
    processor,
    model,
    device: str,
    box_threshold: float,
    text_threshold: float,
) -> np.ndarray:
    prompt = prompt.strip().lower()
    if not prompt:
        raise ValueError("Object prompt cannot be empty.")
    if not prompt.endswith("."):
        prompt += "."

    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)

    with torch.inference_mode():
        outputs = model(**inputs)

    kwargs = dict(
        outputs=outputs,
        input_ids=inputs.input_ids,
        text_threshold=float(text_threshold),
        target_sizes=[(image.height, image.width)],
    )

    try:
        results = processor.post_process_grounded_object_detection(
            threshold=float(box_threshold), **kwargs
        )
    except TypeError:
        results = processor.post_process_grounded_object_detection(
            box_threshold=float(box_threshold), **kwargs
        )

    boxes = results[0].get("boxes")
    if boxes is None or len(boxes) == 0:
        return np.empty((0, 4), dtype=np.float32)
    return boxes.detach().cpu().numpy().astype(np.float32)


def _find_detection_frame(
    frame_dir: Path,
    frame_count: int,
    prompt: str,
    processor,
    model,
    device: str,
    box_threshold: float,
    text_threshold: float,
    scan_stride: int,
) -> tuple[int, np.ndarray]:
    stride = max(1, int(scan_stride))
    candidates = list(range(0, frame_count, stride))
    if frame_count - 1 not in candidates:
        candidates.append(frame_count - 1)

    for idx in candidates:
        boxes = _detect_boxes(
            frame_dir / f"{idx:06d}.jpg",
            prompt,
            processor,
            model,
            device,
            box_threshold,
            text_threshold,
        )
        if len(boxes):
            return idx, boxes

    raise RuntimeError(
        f'Grounding DINO could not find "{prompt}" in sampled frames. '
        "Try a simpler noun phrase, lower the thresholds, or reduce scan stride."
    )


def _build_masks_with_sam2(
    frame_dir: Path,
    frame_count: int,
    start_idx: int,
    boxes: np.ndarray,
    mask_dir: Path,
) -> None:
    from sam2.sam2_video_predictor import SAM2VideoPredictor

    predictor = SAM2VideoPredictor.from_pretrained(SAM2_MODEL_ID)
    state = predictor.init_state(
        str(frame_dir),
        offload_video_to_cpu=True,
        offload_state_to_cpu=False,
        async_loading_frames=False,
    )

    union_masks: dict[int, np.ndarray] = {}

    def absorb(frame_idx, mask_logits):
        masks = (mask_logits > 0.0).detach().cpu().numpy()
        merged = np.any(masks[:, 0], axis=0).astype(np.uint8) * 255
        union_masks[int(frame_idx)] = merged

    if torch.cuda.is_available():
        amp_context = torch.autocast("cuda", dtype=torch.bfloat16)
    else:
        from contextlib import nullcontext
        amp_context = nullcontext()

    with torch.inference_mode(), amp_context:
        for obj_id, box in enumerate(boxes, start=1):
            frame_idx, _, mask_logits = predictor.add_new_points_or_box(
                inference_state=state,
                frame_idx=int(start_idx),
                obj_id=int(obj_id),
                box=np.asarray(box, dtype=np.float32),
            )
            absorb(frame_idx, mask_logits)

        for frame_idx, _, mask_logits in predictor.propagate_in_video(
            state, start_frame_idx=int(start_idx)
        ):
            absorb(frame_idx, mask_logits)

        if start_idx > 0:
            for frame_idx, _, mask_logits in predictor.propagate_in_video(
                state, start_frame_idx=int(start_idx), reverse=True
            ):
                absorb(frame_idx, mask_logits)

    if not union_masks:
        raise RuntimeError("SAM 2 produced no tracking masks.")

    sample = next(iter(union_masks.values()))
    h, w = sample.shape
    for idx in range(frame_count):
        mask = union_masks.get(idx, np.zeros((h, w), dtype=np.uint8))
        cv2.imwrite(str(mask_dir / f"{idx:06d}.png"), mask)


def _run_propainter(
    video_path: Path,
    mask_dir: Path,
    output_root: Path,
    fp16: bool,
    mask_dilation: int,
    resize_ratio: float,
) -> Path:
    script = (PROPAINTER_DIR / "inference_propainter.py").resolve()
    cmd = [
        sys.executable,
        str(script),
        "--video",
        str(video_path.resolve()),
        "--mask",
        str(mask_dir.resolve()),
        "--output",
        str(output_root.resolve()),
        "--mask_dilation",
        str(int(mask_dilation)),
        "--resize_ratio",
        str(float(resize_ratio)),
    ]
    if fp16 and torch.cuda.is_available():
        cmd.append("--fp16")

    subprocess.run(cmd, cwd=str(PROPAINTER_DIR.resolve()), check=True)

    result = output_root / video_path.stem / "inpaint_out.mp4"
    if not result.exists():
        matches = list(output_root.rglob("inpaint_out.mp4"))
        if len(matches) == 1:
            result = matches[0]
        else:
            raise RuntimeError("ProPainter finished but no inpaint_out.mp4 was found.")
    return result


def _mux_audio(video_only: Path, original: Path, output: Path) -> None:
    if shutil.which("ffmpeg") is None:
        shutil.copy2(video_only, output)
        return

    subprocess.run(
        [
            "ffmpeg", "-y",
            "-i", str(video_only),
            "-i", str(original),
            "-map", "0:v:0",
            "-map", "1:a?",
            "-c:v", "copy",
            "-c:a", "aac",
            "-b:a", "192k",
            "-shortest",
            str(output),
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def remove_object_from_video(
    input_video: str,
    output_video: str,
    prompt: str,
    box_threshold: float = 0.35,
    text_threshold: float = 0.25,
    scan_stride: int = 30,
    mask_dilation: int = 6,
    resize_ratio: float = 1.0,
    fp16: bool = True,
) -> str:
    """Remove a text-described object using Grounding DINO + SAM 2 + ProPainter."""
    _require_editing_dependencies()

    source = Path(input_video)
    target = Path(output_video)
    if not source.exists():
        raise FileNotFoundError(source)
    target.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="grounded_propainter_") as tmp:
        root = Path(tmp)
        frame_dir = root / "frames"
        mask_dir = root / "masks"
        propainter_out = root / "propainter"
        frame_dir.mkdir()
        mask_dir.mkdir()
        propainter_out.mkdir()

        frame_count = _extract_frames(source, frame_dir)

        processor, detector, device = _load_grounding_dino()
        start_idx, boxes = _find_detection_frame(
            frame_dir,
            frame_count,
            prompt,
            processor,
            detector,
            device,
            box_threshold,
            text_threshold,
            scan_stride,
        )

        del detector
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        _build_masks_with_sam2(
            frame_dir,
            frame_count,
            start_idx,
            boxes,
            mask_dir,
        )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        inpainted = _run_propainter(
            source,
            mask_dir,
            propainter_out,
            fp16,
            mask_dilation,
            resize_ratio,
        )
        _mux_audio(inpainted, source, target)

    return str(target)
