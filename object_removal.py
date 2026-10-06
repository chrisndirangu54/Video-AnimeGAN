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

_GROUNDING_CACHE = None
_SAM2_CACHE = None


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
    global _GROUNDING_CACHE
    if _GROUNDING_CACHE is not None:
        return _GROUNDING_CACHE

    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

    device = _device()
    processor = AutoProcessor.from_pretrained(GROUNDING_MODEL_ID)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(GROUNDING_MODEL_ID).to(device)
    model.eval()
    _GROUNDING_CACHE = (processor, model, device)
    return _GROUNDING_CACHE


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
    global _SAM2_CACHE
    from sam2.sam2_video_predictor import SAM2VideoPredictor

    if _SAM2_CACHE is None:
        _SAM2_CACHE = SAM2VideoPredictor.from_pretrained(SAM2_MODEL_ID)
    predictor = _SAM2_CACHE
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
        major, _ = torch.cuda.get_device_capability()
        amp_dtype = torch.bfloat16 if major >= 8 else torch.float16
        amp_context = torch.autocast("cuda", dtype=amp_dtype)
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



def _compute_global_roi(mask_dir: Path, frame_count: int, padding: int, width: int, height: int):
    xs, ys, xe, ye = [], [], [], []
    for idx in range(frame_count):
        mask = cv2.imread(str(mask_dir / f"{idx:06d}.png"), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        points = cv2.findNonZero(mask)
        if points is None:
            continue
        x, y, w, h = cv2.boundingRect(points)
        xs.append(x)
        ys.append(y)
        xe.append(x + w)
        ye.append(y + h)

    if not xs:
        return 0, 0, width, height

    x1 = max(0, min(xs) - padding)
    y1 = max(0, min(ys) - padding)
    x2 = min(width, max(xe) + padding)
    y2 = min(height, max(ye) + padding)

    # Align ROI dimensions for common encoder/model constraints.
    roi_w = x2 - x1
    roi_h = y2 - y1
    x2 = min(width, x1 + max(32, ((roi_w + 31) // 32) * 32))
    y2 = min(height, y1 + max(32, ((roi_h + 31) // 32) * 32))
    return x1, y1, x2, y2


def _write_cropped_video_and_masks(
    source: Path,
    mask_dir: Path,
    crop_video: Path,
    crop_mask_dir: Path,
    roi,
):
    x1, y1, x2, y2 = roi
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open source video: {source}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    writer = cv2.VideoWriter(
        str(crop_video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (x2 - x1, y2 - y1),
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError("Could not create cropped ROI video.")

    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        crop = frame[y1:y2, x1:x2]
        writer.write(crop)

        mask = cv2.imread(str(mask_dir / f"{idx:06d}.png"), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.imwrite(str(crop_mask_dir / f"{idx:06d}.png"), mask[y1:y2, x1:x2])
        idx += 1

    cap.release()
    writer.release()


def _composite_roi_back(
    source: Path,
    inpainted_crop: Path,
    mask_dir: Path,
    roi,
    output_video: Path,
    feather: int = 7,
):
    x1, y1, x2, y2 = roi
    source_cap = cv2.VideoCapture(str(source))
    crop_cap = cv2.VideoCapture(str(inpainted_crop))
    if not source_cap.isOpened() or not crop_cap.isOpened():
        raise RuntimeError("Could not open source or inpainted ROI for compositing.")

    fps = source_cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(source_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(source_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(
        str(output_video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        source_cap.release()
        crop_cap.release()
        raise RuntimeError("Could not create composited output video.")

    idx = 0
    kernel = max(1, int(feather) * 2 + 1)
    while True:
        ok_src, frame = source_cap.read()
        ok_crop, crop = crop_cap.read()
        if not ok_src:
            break
        if not ok_crop:
            crop = frame[y1:y2, x1:x2].copy()

        mask = cv2.imread(str(mask_dir / f"{idx:06d}.png"), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            mask_roi = np.zeros((y2 - y1, x2 - x1), dtype=np.uint8)
        else:
            mask_roi = mask[y1:y2, x1:x2]

        if feather > 0:
            alpha = cv2.GaussianBlur(mask_roi, (kernel, kernel), 0).astype(np.float32) / 255.0
        else:
            alpha = mask_roi.astype(np.float32) / 255.0
        alpha = alpha[..., None]

        base = frame[y1:y2, x1:x2].astype(np.float32)
        filled = crop.astype(np.float32)
        blended = filled * alpha + base * (1.0 - alpha)
        frame[y1:y2, x1:x2] = np.clip(blended, 0, 255).astype(np.uint8)
        writer.write(frame)
        idx += 1

    source_cap.release()
    crop_cap.release()
    writer.release()


def _adaptive_resize_ratio(roi, target_long_side: int, user_ratio: float) -> float:
    x1, y1, x2, y2 = roi
    longest = max(x2 - x1, y2 - y1)
    if longest <= 0:
        return float(user_ratio)
    auto_ratio = min(1.0, float(target_long_side) / float(longest))
    return max(0.25, min(float(user_ratio), auto_ratio))


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
    roi_enabled: bool = True,
    roi_padding: int = 96,
    roi_target_long_side: int = 768,
    feather: int = 7,
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
        cap_meta = cv2.VideoCapture(str(source))
        width = int(cap_meta.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap_meta.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap_meta.release()

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

        # Keep Grounding DINO cached across Gradio jobs; avoid repeated checkpoint loads.
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

        if roi_enabled:
            roi = _compute_global_roi(mask_dir, frame_count, int(roi_padding), width, height)
            crop_video = root / "roi.mp4"
            crop_mask_dir = root / "roi_masks"
            crop_mask_dir.mkdir()

            _write_cropped_video_and_masks(
                source,
                mask_dir,
                crop_video,
                crop_mask_dir,
                roi,
            )

            effective_ratio = _adaptive_resize_ratio(
                roi,
                int(roi_target_long_side),
                float(resize_ratio),
            )
            inpainted_crop = _run_propainter(
                crop_video,
                crop_mask_dir,
                propainter_out,
                fp16,
                mask_dilation,
                effective_ratio,
            )
            composited = root / "composited.mp4"
            _composite_roi_back(
                source,
                inpainted_crop,
                mask_dir,
                roi,
                composited,
                int(feather),
            )
            _mux_audio(composited, source, target)
        else:
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
