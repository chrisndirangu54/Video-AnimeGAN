"""Backward-compatible entry point for the optimized pretrained video pipeline.

The old implementation contained randomly initialized ConvLSTM layers in front of
AnimeGAN. That path has been removed because it could distort inference without
training. Use video_editor.py for the actual implementation.
"""

from video_editor import (
    PRETRAINED_STYLES,
    load_pretrained_animegan,
    main,
    process_video,
    stylize_frame,
    temporal_smooth,
)

__all__ = [
    "PRETRAINED_STYLES",
    "load_pretrained_animegan",
    "process_video",
    "stylize_frame",
    "temporal_smooth",
]

if __name__ == "__main__":
    main()
