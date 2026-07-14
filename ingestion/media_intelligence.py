"""Optional subtitle, speech, and vision enrichment for offline media."""

from __future__ import annotations

import gc
import importlib.util
import os
import shutil
import subprocess
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional


def ffmpeg_path() -> Optional[str]:
    executable = shutil.which("ffmpeg")
    if executable:
        return executable
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def extract_embedded_subtitle(video_path: Path, output_dir: Path) -> Optional[Path]:
    """Extract the first embedded subtitle stream when FFmpeg can read one."""

    executable = ffmpeg_path()
    if not executable:
        return None
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / f"{video_path.stem}.embedded.srt"
    command = [
        executable,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(video_path),
        "-map",
        "0:s:0",
        str(target),
    ]
    try:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=180, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if completed.returncode != 0 or not target.exists() or target.stat().st_size == 0:
        target.unlink(missing_ok=True)
        return None
    return target


def _srt_timestamp(seconds: float) -> str:
    milliseconds = max(0, int(round(seconds * 1000)))
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1_000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


@lru_cache(maxsize=2)
def _whisper_model(model_size: str):
    from faster_whisper import WhisperModel

    local_model = Path("models") / f"faster-whisper-{model_size}"
    model_reference = str(local_model) if (local_model / "model.bin").exists() else model_size
    preferred_device = os.getenv("CINESCENE_WHISPER_DEVICE", "auto").lower()
    allow_cuda = preferred_device == "cuda"
    if preferred_device == "auto":
        try:
            import torch

            allow_cuda = torch.cuda.is_available() and torch.cuda.get_device_properties(0).total_memory >= 6 * 1024**3
        except Exception:
            allow_cuda = False
    try:
        import ctranslate2

        if allow_cuda and ctranslate2.get_cuda_device_count() > 0:
            try:
                return WhisperModel(model_reference, device="cuda", compute_type="int8_float16")
            except Exception:
                pass
    except Exception:
        pass
    return WhisperModel(model_reference, device="cpu", compute_type="int8")


def transcribe_to_srt(video_path: Path, output_dir: Path, model_size: str = "small") -> Optional[Path]:
    """Generate timestamped subtitles locally with faster-whisper."""

    if importlib.util.find_spec("faster_whisper") is None:
        return None
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / f"{video_path.stem}.generated.srt"
    try:
        model = _whisper_model(model_size)
        segments, _ = model.transcribe(
            str(video_path),
            beam_size=5,
            vad_filter=True,
            condition_on_previous_text=True,
        )
    except Exception:
        return None
    lines = []
    for index, segment in enumerate(segments, start=1):
        text = str(segment.text or "").strip()
        if not text:
            continue
        lines.extend(
            [
                str(index),
                f"{_srt_timestamp(float(segment.start))} --> {_srt_timestamp(float(segment.end))}",
                text,
                "",
            ]
        )
    if not lines:
        return None
    target.write_text("\n".join(lines), encoding="utf-8")
    return target


@lru_cache(maxsize=1)
def _vision_components(model_name: str):
    import torch
    from transformers import BlipForConditionalGeneration, BlipProcessor

    processor = BlipProcessor.from_pretrained(model_name)
    model = BlipForConditionalGeneration.from_pretrained(model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    return processor, model, device


def caption_keyframe(path: Path, model_name: str = "Salesforce/blip-image-captioning-base") -> Optional[str]:
    """Caption a keyframe with BLIP. The model is loaded only in deep mode."""

    try:
        import torch
        from PIL import Image

        processor, model, device = _vision_components(model_name)
        image = Image.open(path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt")
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.inference_mode():
            output = model.generate(**inputs, max_new_tokens=36, num_beams=4)
        caption = processor.decode(output[0], skip_special_tokens=True).strip()
        return caption or None
    except Exception:
        return None


def release_media_models():
    """Release optional media models before loading the embedding model again."""

    _whisper_model.cache_clear()
    _vision_components.cache_clear()
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def resolve_subtitle_files(
    video_path: Path,
    sidecars: List[Path],
    output_dir: Path,
    extract_embedded: bool = True,
    transcribe: bool = False,
    whisper_model: str = "small",
) -> Dict:
    sources = [path for path in sidecars if path.exists()]
    embedded = None
    generated = None

    if extract_embedded:
        embedded = extract_embedded_subtitle(video_path, output_dir)
        if embedded:
            sources.append(embedded)
    if transcribe and not sources:
        generated = transcribe_to_srt(video_path, output_dir, model_size=whisper_model)
        if generated:
            sources.append(generated)

    deduped = []
    seen = set()
    for path in sources:
        key = str(path.resolve())
        if key not in seen:
            seen.add(key)
            deduped.append(path)
    return {
        "paths": deduped,
        "embedded": str(embedded) if embedded else None,
        "generated": str(generated) if generated else None,
        "transcript_mode": "generated" if generated else "embedded" if embedded else "sidecar" if deduped else "none",
    }


def intelligence_capabilities() -> Dict:
    vision_model = os.getenv("CINESCENE_VISION_MODEL", "Salesforce/blip-image-captioning-base")
    whisper_model = os.getenv("CINESCENE_WHISPER_MODEL", "small")
    local_whisper = Path("models") / f"faster-whisper-{whisper_model}" / "model.bin"
    return {
        "ffmpeg": ffmpeg_path() is not None,
        "embedded_subtitles": ffmpeg_path() is not None,
        "faster_whisper": importlib.util.find_spec("faster_whisper") is not None,
        "vision_captioning": importlib.util.find_spec("transformers") is not None,
        "vision_model": vision_model,
        "whisper_model": whisper_model,
        "whisper_model_cached": local_whisper.exists(),
    }
