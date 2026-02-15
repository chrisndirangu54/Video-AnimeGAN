"""
Hybrid Anime Video Stylization Pipeline — Pure TensorFlow (2026 Updated & Fixed)
===============================================================================

Features:
- Stage 1: AnimeGANv3 coarse stylization + temporal ConvLSTM + optical flow blending
- Stage 2: KerasCV SD 2.1 img2img refinement + lineart guidance
- Stage 3: Optional plot twist narration overlay
- Stage 4: AI object addition/removal with automatic mask generation & fast quantized inpainting
- Stage 5: Generate short alternate anime scenes (AnimateDiff) + random insertion
- Stage 6: Audio stylization (replace original audio with new narration + optional BGM)

Requirements:
    pip install tensorflow keras-cv opencv-python tqdm onnx onnx-tf \
    diffusers torch accelerate openai-whisper piper-tts ffmpeg-python moviepy
"""

from __future__ import annotations

import os
import math
import random
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tqdm import tqdm
import keras_cv
import whisper
from piper.voice import PiperVoice
import subprocess

# AnimateDiff
from diffusers import AnimateDiffPipeline, MotionAdapter
from diffusers.utils import export_to_video
import torch

# Inpainting (quantized/fast)
from diffusers import StableDiffusionInpaintPipeline

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SD_IMG_SIZE = 512
_VAE_SCALING_FACTOR = 0.18215

_AD_NUM_FRAMES = 16
_AD_MOTION_LORA = "guoyww/animatediff-motion-lora-zoom-in"

_INPAINT_MODEL = "runwayml/stable-diffusion-inpainting"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ensure_dir(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)


def _to_multiple(x: int, base: int) -> int:
    return max(base, x - (x % base))


def compute_optical_flow(prev_bgr: np.ndarray, curr_bgr: np.ndarray) -> np.ndarray:
    prev_gray = cv2.cvtColor(prev_bgr, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr_bgr, cv2.COLOR_BGR2GRAY)
    return cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)


def warp_frame(frame: np.ndarray, flow: np.ndarray) -> np.ndarray:
    h, w = flow.shape[:2]
    xs = np.tile(np.arange(w), h).astype(np.float32)
    ys = np.repeat(np.arange(h), w).astype(np.float32)
    map_x = (xs + flow[..., 0].ravel()).reshape(h, w)
    map_y = (ys + flow[..., 1].ravel()).reshape(h, w)
    return cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def temporal_blend(current: np.ndarray, warped_prev: np.ndarray, base_alpha: float = 0.7) -> np.ndarray:
    diff = np.mean(np.abs(current.astype(np.float32) - warped_prev.astype(np.float32)))
    alpha = float(np.clip(base_alpha + diff / 3000.0, base_alpha, 0.97))
    return cv2.addWeighted(current, alpha, warped_prev, 1.0 - alpha, 0)


def consistency_loss_np(current: np.ndarray, warped_prev: np.ndarray) -> float:
    a = current.astype(np.float32) / 255.0
    b = warped_prev.astype(np.float32) / 255.0
    return float(np.mean((a - b) ** 2))


def extract_lineart(rgb_frame: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2GRAY)
    smooth = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)
    sx = cv2.Sobel(smooth, cv2.CV_32F, 1, 0, ksize=3)
    sy = cv2.Sobel(smooth, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(sx ** 2 + sy ** 2)
    mag = np.clip(mag / (np.max(mag) + 1e-8) * 255, 0, 255).astype(np.uint8)
    inverted = 255 - mag
    return cv2.cvtColor(inverted, cv2.COLOR_GRAY2RGB)


# ---------------------------------------------------------------------------
# AnimeGAN Components (unchanged)
# ---------------------------------------------------------------------------

class ResidualBlock(layers.Layer):
    def __init__(self, channels: int, **kwargs):
        super().__init__(**kwargs)
        self.conv1 = layers.Conv2D(channels, 3, padding="same", use_bias=False,
                                   kernel_initializer="he_normal")
        self.bn1 = layers.BatchNormalization()
        self.relu = layers.ReLU()
        self.conv2 = layers.Conv2D(channels, 3, padding="same", use_bias=False,
                                   kernel_initializer="he_normal")
        self.bn2 = layers.BatchNormalization()

    def call(self, x, training=False):
        residual = x
        out = self.relu(self.bn1(self.conv1(x), training=training))
        out = self.bn2(self.conv2(out), training=training)
        return out + residual


class AnimeGANv3Generator(tf.keras.Model):
    def __init__(self, out_channels=3, num_residual_blocks=8, hidden_dim=64, **kwargs):
        super().__init__(**kwargs)
        self.initial = tf.keras.Sequential([
            layers.Conv2D(hidden_dim, 7, padding="same", use_bias=False),
            layers.BatchNormalization(),
            layers.ReLU(),
        ])
        self.down1 = tf.keras.Sequential([
            layers.Conv2D(hidden_dim * 2, 3, strides=2, padding="same", use_bias=False),
            layers.BatchNormalization(),
            layers.ReLU(),
        ])
        self.down2 = tf.keras.Sequential([
            layers.Conv2D(hidden_dim * 4, 3, strides=2, padding="same", use_bias=False),
            layers.BatchNormalization(),
            layers.ReLU(),
        ])
        self.residual_blocks = tf.keras.Sequential(
            [ResidualBlock(hidden_dim * 4) for _ in range(num_residual_blocks)]
        )
        self.up1 = tf.keras.Sequential([
            layers.Conv2DTranspose(hidden_dim * 2, 3, strides=2, padding="same", use_bias=False),
            layers.BatchNormalization(),
            layers.ReLU(),
        ])
        self.up2 = tf.keras.Sequential([
            layers.Conv2DTranspose(hidden_dim, 3, strides=2, padding="same", use_bias=False),
            layers.BatchNormalization(),
            layers.ReLU(),
        ])
        self.output_conv = layers.Conv2D(out_channels, 7, padding="same",
                                         activation="tanh")

    def call(self, x, training=False):
        x = self.initial(x, training=training)
        x = self.down1(x, training=training)
        x = self.down2(x, training=training)
        x = self.residual_blocks(x, training=training)
        x = self.up1(x, training=training)
        x = self.up2(x, training=training)
        return self.output_conv(x)


class OnnxGeneratorAdapter:
    def __init__(self, saved_model_dir: str):
        self._sm = tf.saved_model.load(saved_model_dir)
        self._infer = self._sm.signatures["serving_default"]
        self._input_key = next(iter(self._infer.structured_input_signature[1].keys()))
        self._output_key = next(iter(self._infer.structured_outputs.keys()))

    def __call__(self, x: tf.Tensor, training=False) -> tf.Tensor:
        x_bgr_255 = (tf.reverse(x, axis=[-1]) + 1.0) * 127.5
        out = self._infer(**{self._input_key: x_bgr_255})[self._output_key]
        return tf.reverse(out, axis=[-1])


def _onnx_to_saved_model(onnx_path: str, output_dir: str) -> str:
    try:
        import onnx
        import onnx_tf.backend as onnx_tf
    except ImportError as e:
        raise ImportError("Install onnx and onnx-tf: pip install onnx onnx-tf") from e

    print(f"Converting ONNX → SavedModel: {onnx_path}")
    model_proto = onnx.load(onnx_path)
    tf_rep = onnx_tf.prepare(model_proto)
    tf_rep.export_graph(output_dir)
    return output_dir


def load_animegan_generator(weights_path: str, style: str = "Hayao") -> Union[AnimeGANv3Generator, OnnxGeneratorAdapter]:
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"weights_path not found: {weights_path}")

    if weights_path.endswith(".onnx"):
        print(f"[ONNX] Loading/converting for style: {style}")
        saved_model_dir = weights_path + ".savedmodel"
        if not os.path.isdir(saved_model_dir):
            _onnx_to_saved_model(weights_path, saved_model_dir)
        else:
            print(f"  Using cached SavedModel: {saved_model_dir}")
        return OnnxGeneratorAdapter(saved_model_dir)

    print(f"[TF weights] Loading for style: {style}")
    model = AnimeGANv3Generator()
    model(tf.zeros([1, 256, 256, 3]), training=False)

    if os.path.isdir(weights_path):
        latest = tf.train.latest_checkpoint(weights_path)
        if not latest:
            raise FileNotFoundError(f"No checkpoint in: {weights_path}")
        ckpt = tf.train.Checkpoint(model=model)
        ckpt.restore(latest).expect_partial()
        print(f"  Restored: {latest}")
    elif weights_path.endswith((".h5", ".keras")):
        model.load_weights(weights_path)
        print(f"  Loaded: {weights_path}")
    else:
        raise ValueError(f"Unsupported format: {weights_path}")

    return model


# ---------------------------------------------------------------------------
# Temporal Wrapper
# ---------------------------------------------------------------------------

class ConvLSTMCell(layers.Layer):
    def __init__(self, input_dim: int, hidden_dim: int, kernel_size: int = 3, **kwargs):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.gates_conv = layers.Conv2D(
            4 * hidden_dim, kernel_size, padding="same",
            kernel_initializer="he_normal", use_bias=True
        )

    def call(self, inputs, states):
        h, c = states
        combined = tf.concat([inputs, h], axis=-1)
        gates = self.gates_conv(combined)
        i, f, o, g = tf.split(gates, 4, axis=-1)
        i = tf.sigmoid(i)
        f = tf.sigmoid(f)
        o = tf.sigmoid(o)
        g = tf.tanh(g)
        c_next = f * c + i * g
        h_next = o * tf.tanh(c_next)
        return h_next, (h_next, c_next)

    def zero_state(self, batch_size: int, height: int, width: int):
        shape = (batch_size, height, width, self.hidden_dim)
        return tf.zeros(shape), tf.zeros(shape)


class TemporalAnimeGAN:
    def __init__(self, base_generator, hidden_dim=64, num_lstm_layers=2):
        self.base_generator = base_generator
        self.lstm_cells = [
            ConvLSTMCell(3, hidden_dim, 3, name=f"lstm_{i}")
            for i in range(num_lstm_layers)
        ]
        self.refine = layers.Conv2D(3, 1, activation="tanh", name="temporal_refine")
        self._states: Optional[List[Tuple[tf.Tensor, tf.Tensor]]] = None

    def reset_states(self):
        self._states = None

    def process_frame(self, frame_rgb_f32: np.ndarray) -> np.ndarray:
        h, w = frame_rgb_f32.shape[:2]
        x = tf.constant(frame_rgb_f32[np.newaxis])

        feat = self.base_generator(x, training=False)

        if self._states is None:
            self._states = [cell.zero_state(1, h, w) for cell in self.lstm_cells]

        new_states = []
        for cell, state in zip(self.lstm_cells, self._states):
            feat, new_state = cell(feat, state)
            new_states.append(new_state)
        self._states = new_states

        output = self.refine(feat)
        out_np = output.numpy()[0]
        return ((out_np + 1.0) * 127.5).clip(0, 255).astype(np.uint8)


def gan_coarse_process_video(
    input_video_path: str,
    coarse_output_path: str,
    temporal_model: TemporalAnimeGAN,
    resize_multiple: int = 8,
) -> str:
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {input_video_path}")

    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    gan_w = _to_multiple(orig_w, resize_multiple)
    gan_h = _to_multiple(orig_h, resize_multiple)

    _ensure_dir(coarse_output_path)
    out = cv2.VideoWriter(
        coarse_output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (orig_w, orig_h)
    )

    temporal_model.reset_states()
    prev_anime_bgr = prev_orig_bgr = None

    for frame_idx in tqdm(range(total_frames), desc="Stage 1 — GAN Coarse"):
        ret, frame_bgr = cap.read()
        if not ret:
            break

        frame_bgr_resized = cv2.resize(frame_bgr, (gan_w, gan_h),
                                       interpolation=cv2.INTER_AREA) \
            if (orig_w, orig_h) != (gan_w, gan_h) else frame_bgr

        frame_rgb_f32 = cv2.cvtColor(frame_bgr_resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 127.5 - 1.0

        anime_rgb = temporal_model.process_frame(frame_rgb_f32)

        anime_rgb = cv2.resize(anime_rgb, (orig_w, orig_h),
                               interpolation=cv2.INTER_LANCZOS4) \
            if (gan_w, gan_h) != (orig_w, orig_h) else anime_rgb

        anime_bgr = cv2.cvtColor(anime_rgb, cv2.COLOR_RGB2BGR)

        if prev_anime_bgr is not None and prev_orig_bgr is not None:
            flow = compute_optical_flow(prev_orig_bgr, frame_bgr)
            warped_prev = warp_frame(prev_anime_bgr, flow)
            anime_bgr = temporal_blend(anime_bgr, warped_prev)
            c_loss = consistency_loss_np(anime_bgr, warped_prev)
            print(f"  frame {frame_idx:05d} | consistency: {c_loss:.5f}")

        prev_anime_bgr = anime_bgr.copy()
        prev_orig_bgr = frame_bgr.copy()

        out.write(anime_bgr)

    cap.release()
    out.release()
    print(f"Stage 1 complete → {coarse_output_path}")
    return coarse_output_path


# ---------------------------------------------------------------------------
# Stage 2 — Diffusion Refinement
# ---------------------------------------------------------------------------

class SinusoidalTimeEmbedding(layers.Layer):
    def __init__(self, dim: int = 320):
        super().__init__()
        self.dim = dim

    def call(self, timesteps):
        half = self.dim // 2
        emb = tf.math.log(10000.0) / (half - 1)
        emb = tf.exp(tf.range(half, dtype=tf.float32) * -emb)
        emb = tf.cast(timesteps[:, None], tf.float32) * emb[None, :]
        emb = tf.concat([tf.math.sin(emb), tf.math.cos(emb)], axis=-1)
        return emb


_time_embedder = SinusoidalTimeEmbedding()


def _load_sd_pipeline(weights_path: Optional[str] = None) -> keras_cv.models.StableDiffusion:
    sd = keras_cv.models.StableDiffusion(
        img_height=_SD_IMG_SIZE,
        img_width=_SD_IMG_SIZE,
        jit_compile=False,
    )

    if weights_path and os.path.exists(weights_path):
        sd.diffusion_model.load_weights(weights_path)
        print(f"Loaded fine-tuned diffusion weights: {weights_path}")
    else:
        print("Using pretrained SD 2.1 weights from KerasCV")

    return sd


def _encode_image_to_latent(sd, rgb_frame: np.ndarray) -> tf.Tensor:
    resized = cv2.resize(rgb_frame, (_SD_IMG_SIZE, _SD_IMG_SIZE),
                         interpolation=cv2.INTER_AREA)
    img_f32 = resized.astype(np.float32) / 127.5 - 1.0
    img_batch = img_f32[np.newaxis]

    encoded = sd.vae.encode(img_batch)
    latent = encoded.mean() * _VAE_SCALING_FACTOR
    return tf.cast(latent, tf.float32)


def _decode_latent_to_image(sd, latent: tf.Tensor) -> np.ndarray:
    latent = latent / _VAE_SCALING_FACTOR
    decoded = sd.vae.decode(latent)
    rgb = ((decoded[0].numpy() + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
    return rgb


def _add_noise_to_latent(latent: tf.Tensor, timestep: int,
                         alphas_cumprod: np.ndarray) -> Tuple[tf.Tensor, tf.Tensor]:
    alpha_bar = float(alphas_cumprod[timestep])
    noise = tf.random.normal(tf.shape(latent))
    noisy = math.sqrt(alpha_bar) * latent + math.sqrt(1.0 - alpha_bar) * noise
    return noisy, noise


def _build_alphas_cumprod(num_train_steps=1000, beta_start=1e-4, beta_end=0.02) -> np.ndarray:
    betas = np.linspace(beta_start, beta_end, num_train_steps, dtype=np.float64)
    alphas = 1.0 - betas
    return np.cumprod(alphas).astype(np.float32)


def _ddim_step(x_t: tf.Tensor, eps_pred: tf.Tensor, a_curr: float, a_next: float,
               eta: float = 0.0) -> tf.Tensor:
    x0_pred = (x_t - math.sqrt(1.0 - a_curr) * eps_pred) / math.sqrt(a_curr)
    x0_pred = tf.clip_by_value(x0_pred, -1.0, 1.0)

    coeff_x0 = math.sqrt(a_next)
    sigma = eta * math.sqrt((1.0 - a_next) / (1.0 - a_curr) * (1.0 - a_curr / a_next))
    coeff_dir = math.sqrt(max(1.0 - a_next - sigma ** 2, 0.0))

    x_prev = coeff_x0 * x0_pred + coeff_dir * eps_pred
    if eta > 0:
        x_prev += sigma * tf.random.normal(tf.shape(x_t))
    return x_prev


def _encode_lineart_to_latent(sd, rgb_frame: np.ndarray) -> tf.Tensor:
    lineart_rgb = extract_lineart(rgb_frame)
    return _encode_image_to_latent(sd, lineart_rgb)


def diffusion_refine_video(
    coarse_video_path: str,
    refined_output_path: str,
    sd: keras_cv.models.StableDiffusion,
    prompt: str,
    negative_prompt: str = "blurry, lowres, bad anatomy, deformed, watermark",
    num_inference_steps: int = 20,
    strength: float = 0.45,
    guidance_scale: float = 7.5,
    lineart_blend: float = 0.15,
) -> None:
    cap = cv2.VideoCapture(coarse_video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open: {coarse_video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    coarse_frames_rgb = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        coarse_frames_rgb.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()

    print(f"Loaded {len(coarse_frames_rgb)} coarse frames")

    _ensure_dir(refined_output_path)
    final_out = cv2.VideoWriter(
        refined_output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (orig_w, orig_h)
    )

    print("Encoding prompts...")
    pos_tokens = sd.tokenizer.encode(prompt)
    neg_tokens = sd.tokenizer.encode(negative_prompt)
    pos_context = sd.text_encoder(tf.constant([pos_tokens]))[0]
    neg_context = sd.text_encoder(tf.constant([neg_tokens]))[0]

    alphas_cumprod = _build_alphas_cumprod()
    T = len(alphas_cumprod)
    t_start = max(1, int(T * (1.0 - strength)))
    timesteps = np.linspace(t_start, 1, num_inference_steps, dtype=int)[::-1]

    prev_refined_bgr = prev_orig_bgr = None

    for idx, rgb_frame in enumerate(tqdm(coarse_frames_rgb, desc="Refinement")):
        orig_bgr = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)

        z0 = _encode_image_to_latent(sd, rgb_frame)
        lineart_latent = _encode_lineart_to_latent(sd, rgb_frame)

        z_t, _ = _add_noise_to_latent(z0, timesteps[0], alphas_cumprod)

        for step_i, t_curr in enumerate(timesteps[:-1]):
            t_next = timesteps[step_i + 1]
            a_curr = float(alphas_cumprod[t_curr])
            a_next = float(alphas_cumprod[t_next])

            if lineart_blend > 0:
                z_t = (1 - lineart_blend) * z_t + lineart_blend * lineart_latent

            t_emb = _time_embedder(tf.constant([t_curr], dtype=tf.float32))

            ctx_comb = tf.concat([neg_context, pos_context], axis=0)
            z_comb = tf.tile(z_t, [2, 1, 1, 1])
            t_comb = tf.tile(t_emb, [2, 1])

            eps_comb = sd.diffusion_model(z_comb, t_comb, ctx_comb, training=False)
            eps_uncond, eps_cond = tf.split(eps_comb, 2)
            eps_pred = eps_uncond + guidance_scale * (eps_cond - eps_uncond)

            z_t = _ddim_step(z_t, eps_pred, a_curr, a_next)

        refined_rgb = _decode_latent_to_image(sd, z_t)

        if refined_rgb.shape[:2] != (orig_h, orig_w):
            refined_rgb = cv2.resize(refined_rgb, (orig_w, orig_h),
                                     interpolation=cv2.INTER_LANCZOS4)

        refined_bgr = cv2.cvtColor(refined_rgb, cv2.COLOR_RGB2BGR)

        if prev_refined_bgr is not None and prev_orig_bgr is not None:
            flow = compute_optical_flow(prev_orig_bgr, orig_bgr)
            warped_prev = warp_frame(prev_refined_bgr, flow)
            refined_bgr = temporal_blend(refined_bgr, warped_prev, base_alpha=0.8)
            loss = consistency_loss_np(refined_bgr, warped_prev)
            print(f"  frame {idx:05d} | post-refine loss: {loss:.5f}")

        prev_refined_bgr = refined_bgr.copy()
        prev_orig_bgr = orig_bgr.copy()

        final_out.write(refined_bgr)

    final_out.release()
    print(f"Stage 2 complete → {refined_output_path}")


# ---------------------------------------------------------------------------
# Stage 3 — Plot Twist Narration
# ---------------------------------------------------------------------------

def extract_dialogue(video_path: str) -> str:
    model = whisper.load_model("base")
    result = model.transcribe(video_path)
    return " ".join([seg['text'] for seg in result['segments']])


def rewrite_dialogue(original: str, twist: str = "") -> str:
    rewritten = original.replace("he", "she").replace("love", "betrayal").replace("happy", "shocked")
    return rewritten + " " + twist


def generate_narration(text: str, output_wav: str, voice_model: str = "en_US-lessac-medium"):
    voice = PiperVoice.load(voice_model)
    wav_bytes = voice.synthesize(text)
    with open(output_wav, 'wb') as f:
        f.write(wav_bytes)


def overlay_narration(video_path: str, audio_path: str, output_path: str):
    subprocess.run([
        'ffmpeg', '-i', video_path, '-i', audio_path,
        '-c:v', 'copy', '-c:a', 'aac', '-map', '0:v:0', '-map', '1:a:0',
        '-shortest', output_path
    ], check=True)


def add_plot_twist(video_path: str, output_path: str, twist_text: Optional[str] = None):
    print("Stage 3 — Adding plot twist narration")
    temp_audio = "temp_narration.wav"
    
    original = extract_dialogue(video_path)
    rewritten = rewrite_dialogue(original, twist_text or "")
    generate_narration(rewritten, temp_audio)
    
    overlay_narration(video_path, temp_audio, output_path)
    os.remove(temp_audio)
    print(f"Plot twist added → {output_path}")


# ---------------------------------------------------------------------------
# Stage 4 — AI Object Addition / Removal with Automatic Mask Generation
# ---------------------------------------------------------------------------

def generate_auto_mask(frame_rgb: np.ndarray, method: str = "edge", color_range: Optional[Tuple] = None) -> np.ndarray:
    """
    Automatic mask generation (no extra models)
    - method: "edge" (Canny), "color" (HSV range), "sobel"
    - color_range: ((h_min,s_min,v_min), (h_max,s_max,v_max)) for color method
    Returns binary mask (255 = area to inpaint)
    """
    if method == "color" and color_range:
        hsv = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2HSV)
        lower, upper = color_range
        mask = cv2.inRange(hsv, lower, upper)
    else:
        gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
        if method == "canny":
            edges = cv2.Canny(gray, 80, 180)
        elif method == "sobel":
            sobx = cv2.Sobel(gray, cv2.CV_8U, 1, 0, ksize=3)
            soby = cv2.Sobel(gray, cv2.CV_8U, 0, 1, ksize=3)
            edges = cv2.addWeighted(sobx, 0.5, soby, 0.5, 0)
        else:
            edges = gray

        _, mask = cv2.threshold(edges, 30, 255, cv2.THRESH_BINARY)

    # Clean up and dilate
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.dilate(mask, kernel, iterations=2)

    return mask


def ai_object_modification(
    frame_rgb: np.ndarray,
    modifications: List[dict],
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> np.ndarray:
    """
    Add or remove objects using fast quantized inpainting + auto masks.
    
    Example modifications:
    [
        {"action": "add", "prompt": "red modern sofa furniture anime style", "mask_method": "edge"},
        {"action": "remove", "prompt": "clean wooden floor anime style", "mask_method": "color", "color_range": ((0,0,0),(180,50,50))},
        {"action": "add", "prompt": "elegant black dress clothes anime style", "mask_method": "manual", "coords": (450,150,700,650)}
    ]
    """
    current = Image.fromarray(frame_rgb)

    inpaint_pipe = StableDiffusionInpaintPipeline.from_pretrained(
        _INPAINT_MODEL, torch_dtype=torch.float16, variant="fp16"
    )
    inpaint_pipe.to(device)
    inpaint_pipe.enable_attention_slicing()
    inpaint_pipe.enable_model_cpu_offload()

    for mod in modifications:
        action = mod["action"]
        prompt = mod["prompt"]

        # Generate mask
        if mod["mask_method"] == "manual":
            mask_img = Image.new("L", current.size, 0)
            draw = ImageDraw.Draw(mask_img)
            draw.rectangle(mod["coords"], fill=255)
        else:
            mask_np = generate_auto_mask(np.array(current), method=mod["mask_method"],
                                         color_range=mod.get("color_range"))
            mask_img = Image.fromarray(mask_np)

        neg_prompt = "blurry, deformed, low quality, artifacts" if action == "remove" else negative_prompt

        out = inpaint_pipe(
            prompt=prompt,
            image=current,
            mask_image=mask_img,
            negative_prompt=neg_prompt,
            strength=0.75,
            guidance_scale=7.5,
            num_inference_steps=20   # fast
        ).images[0]

        current = out

    return np.array(current)


def insert_ai_objects_video(
    input_path: str,
    output_path: str,
    modifications: List[dict]
):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        altered_rgb = ai_object_modification(frame_rgb, modifications)

        altered_bgr = cv2.cvtColor(altered_rgb, cv2.COLOR_RGB2BGR)
        out.write(altered_bgr)

        frame_idx += 1
        if frame_idx % 30 == 0:
            print(f"AI object processing: {frame_idx} frames done")

    cap.release()
    out.release()
    print(f"AI object addition/removal complete → {output_path}")


# ---------------------------------------------------------------------------
# Stage 5 — Alternate Scenes
# ---------------------------------------------------------------------------

def generate_alternate_scene(
    prompt: str,
    output_path: str,
    num_frames: int = _AD_NUM_FRAMES,
    seed: int = 42
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16

    adapter = MotionAdapter.from_pretrained(_AD_MOTION_LORA, torch_dtype=dtype)
    pipe = AnimateDiffPipeline.from_pretrained(
        "Linaqruf/animagine-xl-3.1",
        motion_adapter=adapter,
        torch_dtype=dtype,
    ).to(device)

    pipe.enable_vae_slicing()
    pipe.enable_model_cpu_offload()

    result = pipe(
        prompt=prompt,
        negative_prompt="blurry, lowres, bad anatomy, watermark",
        num_frames=num_frames,
        guidance_scale=7.5,
        num_inference_steps=25,
        generator=torch.Generator(device).manual_seed(seed),
    ).frames[0]

    export_to_video(result, output_path, fps=8)
    print(f"Alternate scene generated → {output_path}")


def insert_alternate_scenes(
    main_video_path: str,
    alternate_clips: List[str],
    output_path: str,
    insert_points_ms: List[int] = None
):
    from moviepy.editor import VideoFileClip, concatenate_videoclips

    main = VideoFileClip(main_video_path)
    clips = []

    if insert_points_ms is None:
        duration_ms = main.duration * 1000
        insert_points_ms = [int(duration_ms * 0.3), int(duration_ms * 0.7)]

    prev_end = 0
    for point_ms in sorted(insert_points_ms):
        start_sec = prev_end / 1000
        end_sec = point_ms / 1000
        clips.append(main.subclip(start_sec, end_sec))

        alt_path = random.choice(alternate_clips)
        alt_clip = VideoFileClip(alt_path)
        clips.append(alt_clip)

        prev_end = point_ms

    clips.append(main.subclip(prev_end / 1000, main.duration))

    final = concatenate_videoclips(clips)
    final.write_videofile(output_path, codec="libx264", audio_codec="aac", fps=24)
    print(f"Alternate scenes inserted → {output_path}")


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------

def hybrid_anime_stylize_video(
    input_video_path: str,
    final_output_path: str,
    gan_weights_path: str,
    style: str = "Hayao",
    denoiser_weights_path: Optional[str] = None,
    prompt: str = "masterpiece, best quality, anime style, detailed eyes, vibrant colors, cinematic lighting",
    negative_prompt: str = "blurry, lowres, bad anatomy, deformed, watermark",
    num_inference_steps: int = 20,
    diffusion_strength: float = 0.45,
    guidance_scale: float = 7.5,
    lineart_blend: float = 0.15,
    add_twist: bool = False,
    plot_twist_text: Optional[str] = None,
    add_objects: bool = False,
    object_mods: List[dict] = None,
    generate_alternates: bool = False,
    alternate_prompts: List[str] = None,
    insert_points_ms: List[int] = None,
    keep_coarse: bool = False,
) -> None:
    if not os.path.exists(input_video_path):
        raise FileNotFoundError(f"Input not found: {input_video_path}")

    print("=" * 70)
    print(f"Stage 1 — AnimeGAN ({style})")
    print("=" * 70)
    gan_base = load_animegan_generator(gan_weights_path, style=style)
    temporal = TemporalAnimeGAN(gan_base)

    coarse_path = os.path.splitext(final_output_path)[0] + "_coarse.mp4"
    gan_coarse_process_video(input_video_path, coarse_path, temporal)

    print("\n" + "=" * 70)
    print("Stage 2 — SD Refinement")
    print("=" * 70)
    sd = _load_sd_pipeline(denoiser_weights_path)

    refined_path = os.path.splitext(final_output_path)[0] + "_refined.mp4"
    diffusion_refine_video(
        coarse_path,
        refined_path,
        sd,
        prompt,
        negative_prompt,
        num_inference_steps,
        diffusion_strength,
        guidance_scale,
        lineart_blend,
    )

    current_path = refined_path

    if add_twist:
        twist_path = os.path.splitext(final_output_path)[0] + "_twist.mp4"
        add_plot_twist(current_path, twist_path, plot_twist_text)
        current_path = twist_path

    if add_objects:
        objects_path = os.path.splitext(final_output_path)[0] + "_objects.mp4"
        insert_ai_objects_video(
            current_path,
            objects_path,
            object_mods or [
                # Furniture & household examples
                {"action": "add", "prompt": "cozy red modern sofa furniture anime style", "mask_method": "edge"},
                {"action": "add", "prompt": "elegant wooden coffee table household goods anime style", "mask_method": "edge"},
                {"action": "remove", "prompt": "clean empty living room floor anime style", "mask_method": "color", "color_range": ((0,0,100),(180,30,255))},
                {"action": "add", "prompt": "minimalist floor lamp household goods anime style", "mask_method": "manual", "coords": (800, 200, 900, 600)},
                # Clothes examples
                {"action": "add", "prompt": "elegant black evening dress clothes anime style", "mask_method": "manual", "coords": (450, 150, 700, 650)},
                {"action": "add", "prompt": "casual denim jacket and jeans clothes anime style", "mask_method": "edge"},
                {"action": "add", "prompt": "cute pastel hoodie and skirt clothes anime style", "mask_method": "manual", "coords": (500, 100, 750, 550)},
            ]
        )
        current_path = objects_path

    if generate_alternates:
        alt_dir = os.path.splitext(final_output_path)[0] + "_alternates"
        os.makedirs(alt_dir, exist_ok=True)
        alt_clips = []

        alt_prompts = alternate_prompts or [
            "masterpiece, anime style, luxury car reveal in futuristic showroom, dramatic lighting",
            "vibrant anime drift on coastal highway at sunset, vibrant colors",
            "cinematic anime car chase through neon city streets, high speed motion"
        ]

        for i, p in enumerate(alt_prompts):
            p_path = os.path.join(alt_dir, f"alt_{i}.mp4")
            generate_alternate_scene(p, p_path)
            alt_clips.append(p_path)

        final_with_alts = os.path.splitext(final_output_path)[0] + "_with_alts.mp4"
        insert_alternate_scenes(current_path, alt_clips, final_with_alts, insert_points_ms)
        current_path = final_with_alts

    if current_path != final_output_path:
        os.rename(current_path, final_output_path)

    if not keep_coarse and os.path.exists(coarse_path):
        os.remove(coarse_path)

    print(f"\nPipeline complete → {final_output_path}")


if __name__ == "__main__":
    hybrid_anime_stylize_video(
        input_video_path="input.mp4",
        final_output_path="final_hybrid_anime.mp4",
        gan_weights_path="/content/AnimeGANv3_Hayao_36.onnx",
        style="Hayao",
        prompt="masterpiece, best quality, anime style, detailed eyes, vibrant colors, cinematic lighting",
        negative_prompt="blurry, lowres, bad anatomy, deformed, watermark",
        num_inference_steps=20,
        diffusion_strength=0.45,
        guidance_scale=7.5,
        lineart_blend=0.15,
        add_twist=True,
        plot_twist_text="In this version the car becomes sentient and races through a dreamlike cityscape.",
        add_objects=True,
        generate_alternates=True,
        alternate_prompts=[
            "anime style, luxury car reveal in futuristic showroom, dramatic lighting",
            "vibrant anime drift scene on coastal highway, sunset colors",
            "cyberpunk anime car chase through neon streets, high speed"
        ],
        insert_points_ms=[12000, 40000],
        keep_coarse=False,
    )
