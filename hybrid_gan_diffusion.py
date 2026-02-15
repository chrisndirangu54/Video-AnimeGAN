"""
Hybrid Anime Video Stylization Pipeline — Pure TensorFlow (Fixed & Improved)
===========================================================================

Stage 1 — Coarse stylization
    Loads AnimeGANv3 (preferred via ONNX conversion) or AnimeGANv2 (fallback)
    Wraps generator with ConvLSTM for temporal consistency
    Applies optical-flow adaptive blending

Stage 2 — Diffusion refinement
    Uses KerasCV Stable Diffusion 2.1 (auto-downloads pretrained weights)
    True img2img via custom DDIM in latent space
    Sobel lineart blended into latent for structure guidance
    Post-refinement temporal blending to reduce flicker

Requirements:
    pip install tensorflow keras-cv opencv-python tqdm onnx onnx-tf
"""

from __future__ import annotations

import os
import math
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tqdm import tqdm
import keras_cv

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SD_LATENT_SCALE = 8
_SD_IMG_SIZE = 512
_VAE_SCALING_FACTOR = 0.18215  # Standard for SD 2.1

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ensure_dir(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)


def _to_multiple(x: int, base: int) -> int:
    return max(base, x - (x % base))


# ---------------------------------------------------------------------------
# Optical flow & blending
# ---------------------------------------------------------------------------

def compute_optical_flow(prev_bgr: np.ndarray, curr_bgr: np.ndarray) -> np.ndarray:
    prev_gray = cv2.cvtColor(prev_bgr, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr_bgr, cv2.COLOR_BGR2GRAY)
    return cv2.calcOpticalFlowFarneback(
        prev_gray, curr_gray, None,
        pyr_scale=0.5, levels=3, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.2, flags=0
    )


def warp_frame(frame: np.ndarray, flow: np.ndarray) -> np.ndarray:
    h, w = flow.shape[:2]
    xs = np.tile(np.arange(w), h).astype(np.float32)
    ys = np.repeat(np.arange(h), w).astype(np.float32)
    map_x = (xs + flow[..., 0].ravel()).reshape(h, w)
    map_y = (ys + flow[..., 1].ravel()).reshape(h, w)
    return cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REPLICATE)


def temporal_blend(current: np.ndarray, warped_prev: np.ndarray,
                   base_alpha: float = 0.7) -> np.ndarray:
    diff = np.mean(np.abs(current.astype(np.float32) - warped_prev.astype(np.float32)))
    alpha = float(np.clip(base_alpha + diff / 3000.0, base_alpha, 0.97))
    return cv2.addWeighted(current, alpha, warped_prev, 1.0 - alpha, 0)


def consistency_loss_np(current: np.ndarray, warped_prev: np.ndarray) -> float:
    a = current.astype(np.float32) / 255.0
    b = warped_prev.astype(np.float32) / 255.0
    return float(np.mean((a - b) ** 2))


# ---------------------------------------------------------------------------
# Lineart extractor
# ---------------------------------------------------------------------------

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
# AnimeGAN Components
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
        # RGB [-1,1] → BGR [0,255]
        x_bgr_255 = (tf.reverse(x, axis=[-1]) + 1.0) * 127.5
        out = self._infer(**{self._input_key: x_bgr_255})[self._output_key]
        # BGR [-1,1] → RGB [-1,1]
        return tf.reverse(out, axis=[-1])


def _onnx_to_saved_model(onnx_path: str, output_dir: str) -> str:
    try:
        import onnx
        import onnx_tf.backend as onnx_tf
    except ImportError as e:
        raise ImportError("Install onnx and onnx-tf:  pip install onnx onnx-tf") from e

    print(f"Converting ONNX → SavedModel: {onnx_path}")
    model_proto = onnx.load(onnx_path)
    tf_rep = onnx_tf.prepare(model_proto)
    tf_rep.export_graph(output_dir)
    return output_dir


def load_animegan_generator(weights_path: str, style: str = "Hayao") -> Union[AnimeGANv3Generator, OnnxGeneratorAdapter]:
    if not os.path.exists(weights_path):
        raise FileNotFoundError(
            f"weights_path not found: {weights_path}\n\n"
            "Supported options:\n"
            "  • AnimeGANv3 ONNX: https://github.com/TachibanaYoshino/AnimeGANv3\n"
            "    Example: AnimeGANv3_Hayao_36.onnx\n"
            "  • AnimeGANv2 TF checkpoint: search forks or releases\n"
        )

    if weights_path.endswith(".onnx"):
        print(f"[ONNX] Loading/converting for style: {style}")
        saved_model_dir = weights_path + ".savedmodel"
        if not os.path.isdir(saved_model_dir):
            _onnx_to_saved_model(weights_path, saved_model_dir)
        else:
            print(f"  Using cached SavedModel: {saved_model_dir}")
        return OnnxGeneratorAdapter(saved_model_dir)

    # TF native fallback
    print(f"[TF weights] Loading for style: {style}")
    model = AnimeGANv3Generator()
    model(tf.zeros([1, 256, 256, 3]), training=False)  # build

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
        x = tf.constant(frame_rgb_f32[np.newaxis])  # [1, H, W, 3]

        feat = self.base_generator(x, training=False)

        if self._states is None:
            self._states = [
                cell.zero_state(1, h, w) for cell in self.lstm_cells
            ]

        new_states = []
        for cell, state in zip(self.lstm_cells, self._states):
            feat, new_state = cell(feat, state)
            new_states.append(new_state)
        self._states = new_states

        output = self.refine(feat)
        out_np = output.numpy()[0]
        return ((out_np + 1.0) * 127.5).clip(0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Stage 1 — Coarse Processing
# ---------------------------------------------------------------------------

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
    prev_anime_bgr = None
    prev_orig_bgr = None
    frames_written = 0

    for frame_idx in tqdm(range(total_frames), desc="Stage 1 — GAN Coarse"):
        ret, frame_bgr = cap.read()
        if not ret:
            break

        frame_bgr_resized = cv2.resize(frame_bgr, (gan_w, gan_h),
                                       interpolation=cv2.INTER_AREA) \
            if (orig_w, orig_h) != (gan_w, gan_h) else frame_bgr

        frame_rgb_f32 = cv2.cvtColor(frame_bgr_resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 127.5 - 1.0

        anime_rgb = temporal_model.process_frame(frame_rgb_f32)

        if (gan_w, gan_h) != (orig_w, orig_h):
            anime_rgb = cv2.resize(anime_rgb, (orig_w, orig_h),
                                   interpolation=cv2.INTER_LANCZOS4)

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
        frames_written += 1

    cap.release()
    out.release()
    print(f"Stage 1 complete — {frames_written} frames written")
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
    final_output_path: str,
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

    _ensure_dir(final_output_path)
    final_out = cv2.VideoWriter(
        final_output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (orig_w, orig_h)
    )

    # Text embeddings
    print("Encoding prompts...")
    pos_tokens = sd.tokenizer.encode(prompt)
    neg_tokens = sd.tokenizer.encode(negative_prompt)
    pos_context = sd.text_encoder(tf.constant([pos_tokens]))[0]  # [1, 77, 768]
    neg_context = sd.text_encoder(tf.constant([neg_tokens]))[0]

    # DDIM schedule
    alphas_cumprod = _build_alphas_cumprod()
    T = len(alphas_cumprod)
    t_start = max(1, int(T * (1.0 - strength)))
    timesteps = np.linspace(t_start, 1, num_inference_steps, dtype=int)[::-1]

    # Temporal state
    prev_refined_bgr = None
    prev_orig_bgr = None

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

            # Batched CFG
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

        # Post-refinement blending
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
    print(f"Stage 2 complete → {final_output_path}")


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
    keep_coarse: bool = False,
) -> None:
    if not os.path.exists(input_video_path):
        raise FileNotFoundError(f"Input not found: {input_video_path}")

    print("=" * 70)
    print(f"Stage 1 — Loading AnimeGAN ({style})")
    print("=" * 70)
    gan_base = load_animegan_generator(gan_weights_path, style=style)
    temporal = TemporalAnimeGAN(gan_base)

    coarse_path = os.path.splitext(final_output_path)[0] + "_coarse.mp4"
    gan_coarse_process_video(input_video_path, coarse_path, temporal)

    print("\n" + "=" * 70)
    print("Stage 2 — KerasCV Stable Diffusion Refinement")
    print("=" * 70)
    sd = _load_sd_pipeline(denoiser_weights_path)

    diffusion_refine_video(
        coarse_path,
        final_output_path,
        sd,
        prompt,
        negative_prompt,
        num_inference_steps,
        diffusion_strength,
        guidance_scale,
        lineart_blend,
    )

    if not keep_coarse and os.path.exists(coarse_path):
        os.remove(coarse_path)
        print(f"Cleaned up: removed {coarse_path}")

    print(f"\nFinished → {final_output_path}")


if __name__ == "__main__":
    hybrid_anime_stylize_video(
        input_video_path="input.mp4",
        final_output_path="final_hybrid_anime.mp4",
        gan_weights_path="/content/AnimeGANv3_Hayao_36.onnx",  # or TF checkpoint dir / .h5
        style="Hayao",
        # denoiser_weights_path="/content/anime_finetuned.h5",  # optional
        prompt="masterpiece, best quality, anime style, detailed eyes, vibrant colors, cinematic lighting",
        negative_prompt="blurry, lowres, bad anatomy, deformed, watermark",
        num_inference_steps=20,
        diffusion_strength=0.45,
        guidance_scale=7.5,
        lineart_blend=0.15,
        keep_coarse=False,
    )
