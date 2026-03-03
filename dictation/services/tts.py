"""Text-to-speech service for agent mode.

Local-first: defaults to Qwen3-TTS (fully offline, GPU-accelerated).
Cloud optional: set ELEVENLABS_API_KEY to use ElevenLabs streaming.

Adapted from ~/agent-relay/matrix/tts.py — same Qwen3-TTS voice
cloning / VoiceDesign pipeline.
"""

from __future__ import annotations

import io
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
from typing import Callable, Optional

# ── Configuration ────────────────────────────────────────────────────

VOICEDESIGN_MODEL_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
BASE_MODEL_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"

DEFAULT_VOICE_DESCRIPTION = (
    "Precise, measured male voice with a flat, neutral affect. Slightly "
    "formal register, no emotional inflection or contractions. Clear "
    "enunciation of every syllable. Fast speaking pace — rapid but still "
    "perfectly intelligible, like a technical briefing delivered efficiently. "
    "Resembles a highly articulate android reading a technical report at speed."
)




def _get_backend() -> str:
    """Determine TTS backend: 'elevenlabs' or 'qwen'."""
    if os.environ.get("ELEVENLABS_API_KEY"):
        return "elevenlabs"
    return "qwen"


# ── Module-level model state (loaded once) ───────────────────────────

_model = None
_model_lock = threading.Lock()
_load_error: Optional[str] = None

_clone_model = None
_voice_prompt = None
_voice_prompt_lock = threading.Lock()


def _load_model(
    voice_sample_path: Optional[str] = None,
) -> None:
    """Load the Qwen3-TTS model (called once, lazily, thread-safe)."""
    global _model, _load_error
    with _model_lock:
        if _model is not None or _load_error is not None:
            return
        try:
            import torch
            from qwen_tts import Qwen3TTSModel

            device = "cuda:0" if torch.cuda.is_available() else "cpu"
            dtype = torch.bfloat16

            if voice_sample_path:
                sys.stdout.write(
                    f"[TTS] Loading Base model for voice cloning ({BASE_MODEL_ID})...\n"
                )
                _model = Qwen3TTSModel.from_pretrained(
                    BASE_MODEL_ID, device_map=device, dtype=dtype,
                )
            else:
                sys.stdout.write(
                    f"[TTS] Loading VoiceDesign model ({VOICEDESIGN_MODEL_ID})...\n"
                )
                _model = Qwen3TTSModel.from_pretrained(
                    VOICEDESIGN_MODEL_ID, device_map=device, dtype=dtype,
                )
            sys.stdout.write(f"[TTS] Model ready on {device}\n")
            sys.stdout.flush()

        except ImportError as e:
            _load_error = f"qwen-tts or torch not installed: {e}"
            sys.stdout.write(f"[TTS] Unavailable: {_load_error}\n")
            sys.stdout.flush()
        except Exception as e:
            _load_error = str(e)
            sys.stdout.write(f"[TTS] Failed to load model: {e}\n")
            sys.stdout.flush()


def _get_or_build_voice_prompt(
    voice_sample_path: Optional[str] = None,
    voice_sample_transcript: str = "",
    voice_description: str = DEFAULT_VOICE_DESCRIPTION,
):
    """Build and cache the voice clone prompt. Called once on first synthesis.

    If TTS_CLONE_PROMPT env var points to a .pt file (saved by clone-voice.py),
    it is loaded directly — skipping the expensive feature extraction step.
    """
    global _voice_prompt, _clone_model
    with _voice_prompt_lock:
        if _voice_prompt is not None:
            return _voice_prompt

        try:
            import soundfile as sf
            import torch

            # Fast path: load pre-built clone prompt from disk
            clone_prompt_path = os.environ.get("TTS_CLONE_PROMPT", "")
            if clone_prompt_path and os.path.exists(clone_prompt_path):
                sys.stdout.write(
                    f"[TTS] Loading cached clone prompt from {clone_prompt_path}...\n"
                )
                _voice_prompt = torch.load(
                    clone_prompt_path, map_location="cpu", weights_only=False
                )
                _clone_model = _model
                sys.stdout.write("[TTS] Clone prompt loaded — ready.\n")
                sys.stdout.flush()
                return _voice_prompt

            if voice_sample_path:
                sys.stdout.write(
                    f"[TTS] Building voice clone prompt from {voice_sample_path}...\n"
                )
                ref_audio, ref_sr = sf.read(voice_sample_path, dtype="float32")
                if ref_sr != 24000:
                    import librosa
                    ref_audio = librosa.resample(
                        ref_audio, orig_sr=ref_sr, target_sr=24000
                    )
                    ref_sr = 24000

                _voice_prompt = _model.create_voice_clone_prompt(
                    ref_audio=(ref_audio, ref_sr),
                    ref_text=voice_sample_transcript or "",
                )
                _clone_model = _model
                sys.stdout.write("[TTS] Voice clone prompt ready.\n")

            else:
                # VoiceDesign mode: generate reference, then cache clone prompt
                sys.stdout.write(
                    "[TTS] Building voice prompt from description...\n"
                )
                ref_text = "Hello. I am ready to assist you."
                wavs, sr = _model.generate_voice_design(
                    text=ref_text,
                    language="English",
                    instruct=voice_description,
                )
                # Load Base model to cache the clone prompt for fast reuse
                from qwen_tts import Qwen3TTSModel

                device = "cuda:0" if torch.cuda.is_available() else "cpu"
                clone_model = Qwen3TTSModel.from_pretrained(
                    BASE_MODEL_ID, device_map=device, dtype=torch.bfloat16,
                )
                _voice_prompt = clone_model.create_voice_clone_prompt(
                    ref_audio=(wavs[0], sr),
                    ref_text=ref_text,
                )
                _clone_model = clone_model
                sys.stdout.write("[TTS] Voice prompt cached — fast synthesis ready.\n")

            sys.stdout.flush()
        except Exception as e:
            sys.stdout.write(f"[TTS] Failed to build voice prompt: {e}\n")
            sys.stdout.flush()
            _voice_prompt = None

        return _voice_prompt


# ── Text cleanup (from agent-relay/matrix/tts.py) ───────────────────

def _prepare_text(text: str) -> str:
    """Strip markdown, URLs, code blocks, and truncate for TTS."""

    # Fenced code blocks
    text = re.sub(r"```[\s\S]*?```", "[code omitted]", text)

    # URLs → bare domain
    _LOCAL_HOSTS = {"localhost", "127.0.0.1", "0.0.0.0", "::1"}

    def _url_to_spoken(m):
        url = m.group(0)
        host_m = re.search(r"https?://([^/:?#\s]+)", url)
        if not host_m:
            return ""
        host = host_m.group(1)
        if host in _LOCAL_HOSTS:
            return ""
        return re.sub(r"^www\.", "", host)

    text = re.sub(r"https?://\S+", _url_to_spoken, text)

    # File paths → filename only
    def _path_to_filename(m):
        path = m.group(0)
        filename = path.rstrip("/").rsplit("/", 1)[-1]
        if re.fullmatch(r"[0-9a-f]{7,}", filename):
            return ""
        return filename if filename else ""

    text = re.sub(r"(?<!\w)(?:~|\.{1,2})?/[\w.\-/]+", _path_to_filename, text)

    # Hex hashes
    text = re.sub(r"\b[0-9a-f]{7,64}\b", "", text)

    # snake_case → spaces
    text = re.sub(
        r"\b([a-z][a-z0-9]*)(?:_([a-z0-9]+)){2,}\b",
        lambda m: m.group(0).replace("_", " "),
        text,
    )
    # camelCase → spaces
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    text = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", text)

    # Inline code backticks
    text = re.sub(r"`([^`]+)`", r"\1", text)
    # Bold/italic
    text = re.sub(r"\*{1,3}([^*]+)\*{1,3}", r"\1", text)
    # Markdown links
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    # HTML tags
    text = re.sub(r"<[^>]+>", "", text)
    # Heading markers
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)

    # Cleanup whitespace
    text = re.sub(r"\s+([.,;:!?])", r"\1", text)
    text = re.sub(r" {2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = text.strip()

    return text


# ── TTSService class ────────────────────────────────────────────────

class TTSService:
    """Text-to-speech with local speaker playback.

    Collects text chunks (streamed from LLM) and synthesizes speech.
    Uses Qwen3-TTS locally (GPU) or ElevenLabs (cloud).
    """

    def __init__(self, on_done: Callable[[], None]):
        self._on_done = on_done
        self._backend = _get_backend()
        self._cancelled = False
        self._text_buffer = ""
        self._task: Optional[threading.Thread] = None
        self._playback_process: Optional[subprocess.Popen] = None

        # Voice config from env vars
        self._voice_sample_path = os.environ.get("TTS_VOICE_SAMPLE") or None
        self._voice_sample_transcript = os.environ.get("TTS_VOICE_SAMPLE_TRANSCRIPT", "")
        self._voice_description = os.environ.get(
            "TTS_VOICE_DESCRIPTION", DEFAULT_VOICE_DESCRIPTION
        )

        sys.stdout.write(f"[TTS] Backend: {self._backend}\n")
        sys.stdout.flush()

        # Preload model in background if using Qwen
        if self._backend == "qwen":
            t = threading.Thread(
                target=_load_model,
                args=(self._voice_sample_path,),
                daemon=True,
            )
            t.start()

    def send(self, text: str) -> None:
        """Buffer text for synthesis.

        Text arrives token by token from the LLM.
        We buffer until we have a sentence boundary, then synthesize.
        """
        self._text_buffer += text

        # Synthesize on sentence boundaries for natural phrasing
        if any(self._text_buffer.rstrip().endswith(p) for p in ".!?:"):
            sentence = self._text_buffer.strip()
            self._text_buffer = ""
            if sentence and not self._cancelled:
                self._synthesize_and_play(sentence)

    def flush(self) -> None:
        """Synthesize any remaining buffered text, then signal done."""
        remaining = self._text_buffer.strip()
        self._text_buffer = ""
        if remaining and not self._cancelled:
            self._synthesize_and_play(remaining)
        # Wait for playback to finish before signaling done
        if self._task and self._task.is_alive():
            self._task.join(timeout=30)
        if not self._cancelled:
            self._on_done()

    def cancel(self) -> None:
        """Immediately stop all synthesis and playback."""
        self._cancelled = True
        self._text_buffer = ""
        if self._playback_process:
            try:
                self._playback_process.terminate()
            except Exception:
                pass

    def _synthesize_and_play(self, text: str) -> None:
        """Synthesize text and play audio."""
        if self._cancelled:
            return

        # Wait for previous chunk to finish before starting next
        if self._task and self._task.is_alive():
            self._task.join(timeout=30)

        self._task = threading.Thread(
            target=self._do_synthesize, args=(text,), daemon=True
        )
        self._task.start()

    def _do_synthesize(self, text: str) -> None:
        """Actually run synthesis + playback."""
        if self._cancelled:
            return

        if self._backend == "qwen":
            self._synthesize_qwen(text)
        elif self._backend == "elevenlabs":
            self._synthesize_elevenlabs(text)

    def _synthesize_qwen(self, text: str) -> None:
        """Synthesize using Qwen3-TTS (local, GPU-accelerated)."""
        if _load_error is not None:
            sys.stdout.write(f"[TTS] Qwen3-TTS unavailable: {_load_error}\n")
            sys.stdout.flush()
            return
        if _model is None:
            _load_model(self._voice_sample_path)
            if _model is None:
                return

        cleaned = _prepare_text(text)
        if not cleaned:
            return

        tmp_path = None
        try:
            import numpy as np
            import soundfile as sf

            voice_prompt = _get_or_build_voice_prompt(
                voice_sample_path=self._voice_sample_path,
                voice_sample_transcript=self._voice_sample_transcript,
                voice_description=self._voice_description,
            )

            if voice_prompt is not None and _clone_model is not None:
                wavs, sr = _clone_model.generate_voice_clone(
                    text=cleaned,
                    language="English",
                    voice_clone_prompt=voice_prompt,
                )
            else:
                # Fallback: VoiceDesign directly
                wavs, sr = _model.generate_voice_design(
                    text=cleaned,
                    language="English",
                    instruct=self._voice_description,
                )

            wav_array = wavs[0]

            # Trim leading artifact (breath/click before speech)
            frame_size = int(0.02 * sr)
            mandatory_skip = int(0.15 * sr)
            speech_threshold = np.max(np.abs(wav_array)) * 0.08
            trim_frames = mandatory_skip
            for i in range(
                mandatory_skip,
                min(len(wav_array) - frame_size, int(0.8 * sr)),
                frame_size,
            ):
                frame_rms = np.sqrt(np.mean(wav_array[i : i + frame_size] ** 2))
                if frame_rms > speech_threshold:
                    trim_frames = max(mandatory_skip, i - frame_size)
                    break

            wav_array = wav_array[trim_frames:]

            # Normalize volume — Qwen3-TTS output is often very quiet.
            # Target -1 dBFS (peak ~0.89) for loud, clear audio.
            peak = np.max(np.abs(wav_array))
            if peak > 1e-6:  # avoid division by near-zero (silence)
                target_peak = 0.89  # -1 dBFS
                wav_array = wav_array * (target_peak / peak)

            if self._cancelled:
                return

            # Write to temp file and play
            with tempfile.NamedTemporaryFile(
                suffix=".wav", delete=False
            ) as tmp:
                tmp_path = tmp.name

            buf = io.BytesIO()
            sf.write(buf, wav_array, sr, format="WAV")
            buf.seek(0)
            with open(tmp_path, "wb") as f:
                f.write(buf.read())

            if not self._cancelled:
                self._play_audio(tmp_path)

        except Exception as e:
            if not self._cancelled:
                sys.stdout.write(f"[TTS] Qwen3-TTS error: {e}\n")
                sys.stdout.flush()
        finally:
            if tmp_path:
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass

    def _synthesize_elevenlabs(self, text: str) -> None:
        """Synthesize using ElevenLabs API (cloud, streaming)."""
        tmp_path = None
        try:
            from elevenlabs import ElevenLabs as ELClient

            api_key = os.environ["ELEVENLABS_API_KEY"]
            voice_id = os.environ.get(
                "ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM"
            )

            client = ELClient(api_key=api_key)

            with tempfile.NamedTemporaryFile(
                suffix=".mp3", delete=False
            ) as tmp:
                tmp_path = tmp.name

            audio = client.text_to_speech.convert(
                voice_id=voice_id,
                text=text,
                model_id="eleven_turbo_v2_5",
            )

            with open(tmp_path, "wb") as f:
                for chunk in audio:
                    if self._cancelled:
                        break
                    f.write(chunk)

            if not self._cancelled:
                self._play_audio(tmp_path)

        except ImportError:
            sys.stdout.write(
                "[TTS] elevenlabs package not found. "
                "Install with: pip install elevenlabs\n"
            )
            sys.stdout.flush()
        except Exception as e:
            if not self._cancelled:
                sys.stdout.write(f"[TTS] ElevenLabs error: {e}\n")
                sys.stdout.flush()
        finally:
            if tmp_path:
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass

    def _play_audio(self, path: str) -> None:
        """Play an audio file through the system speaker."""
        if self._cancelled:
            return

        try:
            import platform

            if platform.system() == "Darwin":
                cmd = ["afplay", path]
            else:
                # paplay (PulseAudio) for WAV, ffplay as fallback
                cmd = ["paplay", path]

            self._playback_process = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            self._playback_process.wait()
            self._playback_process = None

        except FileNotFoundError:
            # Fallback to ffplay then aplay
            for fallback_cmd in [
                ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", path],
                ["aplay", "-q", path],
            ]:
                try:
                    self._playback_process = subprocess.Popen(
                        fallback_cmd,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                    self._playback_process.wait()
                    self._playback_process = None
                    return
                except FileNotFoundError:
                    continue
            sys.stdout.write("[TTS] No audio player found (paplay/ffplay/aplay)\n")
            sys.stdout.flush()
        except Exception as e:
            if not self._cancelled:
                sys.stdout.write(f"[TTS] Playback error: {e}\n")
                sys.stdout.flush()
