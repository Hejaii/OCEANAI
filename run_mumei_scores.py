#!/usr/bin/env python
"""Quick script to install dependencies, download model weights, and
compute personality trait scores for ``mumei.mp4``.

Running this file will:
1. Install the project and its dependencies using ``pip``.
2. Download all required model weights.
3. Compute and print the Big Five personality trait scores for the
   bundled ``mumei.mp4`` sample video.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path


def install_package() -> None:
    """Install the current project with its dependencies via ``pip``.

    If installation fails (for example, when network access is
    unavailable) the script continues to run with the existing
    environment.
    """

    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "."])
    except subprocess.CalledProcessError as err:  # pragma: no cover - best effort
        print(f"Warning: failed to install package via pip: {err}")


def prepare_dataset(video: Path) -> Path:
    """Place ``video`` in a temporary dataset directory and return its path."""

    root = Path(__file__).resolve().parent
    dataset_root = root / "_mumei_dataset"
    person_dir = dataset_root / "sample"
    person_dir.mkdir(parents=True, exist_ok=True)
    target = person_dir / video.name
    if not target.exists():
        shutil.copy2(video, target)
    return dataset_root


def main() -> int:
    install_package()

    from oceanai.modules.lab.build import Run

    run = Run(lang="en")
    run.path_to_save_ = os.path.normpath("./models")
    run.chunk_size_ = 2_000_000

    # Load neural network architectures and weights
    run.load_video_model_hc(lang="en", show_summary=False, out=True, run=True)
    run.load_video_model_weights_hc(
        url=run.weights_for_big5_["video"]["fi"]["hc"]["googledisk"],
        force_reload=False,
        out=True,
        run=True,
    )

    run.load_video_model_deep_fe(show_summary=False, out=True, run=True)
    run.load_video_model_weights_deep_fe(
        url=run.weights_for_big5_["video"]["fi"]["fe"]["googledisk"],
        force_reload=False,
        out=True,
        run=True,
    )

    run.load_video_model_nn(show_summary=False, out=True, run=True)
    run.load_video_model_weights_nn(
        url=run.weights_for_big5_["video"]["fi"]["nn"]["googledisk"],
        force_reload=False,
        out=True,
        run=True,
    )

    run.load_video_models_b5(show_summary=False, out=True, run=True)
    b5_urls = run.weights_for_big5_["video"]["fi"]["b5"]
    run.load_video_models_weights_b5(
        url_openness=b5_urls["openness"]["googledisk"],
        url_conscientiousness=b5_urls["conscientiousness"]["googledisk"],
        url_extraversion=b5_urls["extraversion"]["googledisk"],
        url_agreeableness=b5_urls["agreeableness"]["googledisk"],
        url_non_neuroticism=b5_urls["non_neuroticism"]["googledisk"],
        force_reload=False,
        out=True,
        run=True,
    )

    video_path = Path(__file__).resolve().parent / "mumei.mp4"
    dataset_root = prepare_dataset(video_path)

    run.path_to_dataset_ = os.path.normpath(str(dataset_root))
    run.ignore_dirs_ = []
    run.keys_dataset_ = [
        "Path",
        "Openness",
        "Conscientiousness",
        "Extraversion",
        "Agreeableness",
        "Non-Neuroticism",
    ]
    run.ext_ = [".mp4"]
    run.path_to_logs_ = os.path.normpath("./logs")

    run.get_video_union_predictions(
        depth=1,
        recursive=False,
        reduction_fps=5,
        window=10,
        step=5,
        lang="en",
        accuracy=False,
        logs=False,
        out=True,
        run=True,
    )

    print(run.df_files_)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
