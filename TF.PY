"""Legacy compatibility entry point.

The repository previously maintained a second TensorFlow AnimeGAN-like model that
was randomly initialized in the default execution path. The active implementation
now uses published pretrained AnimeGANv2 weights through PyTorch.

Run:
    python video_editor.py input.mp4 output.mp4 --style paprika
"""

from video_editor import main, process_video

__all__ = ["process_video"]

if __name__ == "__main__":
    main()
