import base64
import os
from io import BytesIO

import audioread
import av
import librosa
import numpy as np


SAMPLE_RATE=16000

ENV_AUDIO_WAV_ROOT = "QWEN_OMNI_AUDIO_WAV_ROOT"

def _resolve_wav_for_video_path(video_path: str) -> str | None:
    """
    Return a .wav path if we can map `video_path` to a pre-extracted audio file.

    Priority:
    1) If QWEN_OMNI_AUDIO_WAV_ROOT is set, mirror the relative video path under that root.
       Example:
         video_path: /data/armaan/purs/videos/video-mme/action_reasoning/video.mp4
         root:       /data/armaan/purs/videos_audio_wav
         ->          /data/armaan/purs/videos_audio_wav/video-mme/action_reasoning/video.wav
    2) Same directory as the video: replace extension with .wav
    """
    if video_path.startswith(("http://", "https://")):
        return None
    if video_path.startswith("file://"):
        video_path = video_path[len("file://") :]

    wav_root = os.environ.get(ENV_AUDIO_WAV_ROOT)
    if wav_root:
        # Mirror relative to the "videos/" directory if present (matches our ffmpeg cache layout).
        normalized = video_path.replace("\\", "/")
        marker = "/videos/"
        rel = None
        if marker in normalized:
            rel = normalized.split(marker, 1)[1]
        if rel is None:
            # Fallback: just use basename under wav_root
            rel = os.path.basename(normalized)

        rel_wav = os.path.splitext(rel)[0] + ".wav"
        candidate = os.path.join(wav_root, rel_wav)
        if os.path.exists(candidate):
            return candidate

    local_candidate = os.path.splitext(video_path)[0] + ".wav"
    if os.path.exists(local_candidate):
        return local_candidate
    return None

def _check_if_video_has_audio(video_path):
    container = av.open(video_path)
    audio_streams = [stream for stream in container.streams if stream.type == "audio"]
    if not audio_streams:
        return False
    return True


def process_audio_info(conversations: list[dict] | list[list[dict]], use_audio_in_video: bool):
    """
    Read and process audio info

    Support dict keys:

    type = audio
    - audio
    - audio_start
    - audio_end

    type = video
    - video
    - video_start
    - video_end
    """
    audios = []
    if isinstance(conversations[0], dict):
        conversations = [conversations]
    for conversation in conversations:
        for message in conversation:
            if not isinstance(message["content"], list):
                continue
            for ele in message["content"]:
                if ele["type"] == "audio":
                    if "audio" in ele or "audio_url" in ele:
                        path = ele.get("audio", ele.get("audio_url"))
                        audio_start = ele.get("audio_start", 0.0)
                        audio_end = ele.get("audio_end", None)
                        if isinstance(path, np.ndarray):
                            if path.ndim > 1:
                                raise ValueError("Support only mono audio")
                            audios.append(
                                path[int(SAMPLE_RATE * audio_start) : None if audio_end is None else int(SAMPLE_RATE * audio_end)]
                            )
                            continue
                        elif path.startswith("data:audio"):
                            _, base64_data = path.split("base64,", 1)
                            data = BytesIO(base64.b64decode(base64_data))
                        elif path.startswith("http://") or path.startswith("https://"):
                            data = audioread.ffdec.FFmpegAudioFile(path)
                        elif path.startswith("file://"):
                            data = path[len("file://") :]
                        else:
                            data = path
                    else:
                        raise ValueError("Unknown audio {}".format(ele))
                elif use_audio_in_video and ele["type"] == "video":
                    if "video" in ele or "video_url" in ele:
                        path = ele.get("video", ele.get("video_url"))
                        audio_start = ele.get("video_start", 0.0)
                        audio_end = ele.get("video_end", None)
                        wav_path = _resolve_wav_for_video_path(path) if isinstance(path, str) else None
                        if wav_path is not None:
                            data = wav_path
                        else:
                            assert _check_if_video_has_audio(
                                path
                            ), "Video must has audio track when use_audio_in_video=True"
                            if path.startswith("http://") or path.startswith("https://"):
                                data = audioread.ffdec.FFmpegAudioFile(path)
                            elif path.startswith("file://"):
                                data = path[len("file://") :]
                            elif isinstance(path, str) and path.lower().endswith(
                                (".mp4", ".mkv", ".webm", ".avi", ".mov", ".m4v")
                            ):
                                # Local container formats: same as remote — plain paths confuse soundfile; use ffmpeg.
                                data = audioread.ffdec.FFmpegAudioFile(path)
                            else:
                                data = path
                    else:
                        raise ValueError("Unknown video {}".format(ele))
                else:
                    continue
                audios.append(
                    librosa.load(
                        data,
                        sr=SAMPLE_RATE,
                        offset=audio_start,
                        duration=(audio_end - audio_start) if audio_end is not None else None,
                    )[0]
                )
    if len(audios) == 0:
        audios = None
    return audios
