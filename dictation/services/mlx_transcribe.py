"""mlx-whisper transcription for macOS Metal GPU acceleration.

On Apple Silicon Macs, this replaces the faster-whisper (CTranslate2) backend
with mlx-whisper, which runs Whisper natively on the Metal GPU.

This module monkey-patches RealtimeSTT's `transcribe()` method so that the
final transcription uses mlx-whisper while RealtimeSTT still handles audio
capture, VAD, wake word detection, and realtime preview (tiny.en on CPU).
"""

import copy
import sys
import time


# Default MLX model -- large-v3-turbo is the best speed/quality tradeoff.
# Can be overridden via WHISPER_MODEL env var.
MLX_MODEL_MAP = {
    # Map stock faster-whisper model names to MLX equivalents
    "tiny.en": "mlx-community/whisper-tiny.en-mlx",
    "base.en": "mlx-community/whisper-base.en-mlx",
    "small.en": "mlx-community/whisper-small.en-mlx",
    "medium.en": "mlx-community/whisper-medium.en-mlx",
    "large-v2": "mlx-community/whisper-large-v3-turbo",
    "large-v3": "mlx-community/whisper-large-v3-mlx",
    "large-v3-turbo": "mlx-community/whisper-large-v3-turbo",
}

DEFAULT_MLX_MODEL = "mlx-community/whisper-large-v3-turbo"


def _resolve_mlx_model(model_name: str) -> str:
    """Convert a faster-whisper model name to an MLX model repo ID."""
    if model_name.startswith("mlx-community/"):
        return model_name
    return MLX_MODEL_MAP.get(model_name, DEFAULT_MLX_MODEL)


def patch_recorder_for_mlx(recorder, model_name: str, initial_prompt: str = "") -> None:
    """Monkey-patch a RealtimeSTT recorder to use mlx-whisper for final transcription.

    Parameters
    ----------
    recorder : AudioToTextRecorder
        The RealtimeSTT recorder instance.
    model_name : str
        The Whisper model name (e.g. "large-v2") or MLX repo ID.
    initial_prompt : str
        Initial prompt to bias transcription toward specific vocabulary.
    """
    import mlx_whisper

    mlx_model = _resolve_mlx_model(model_name)

    # Preload the model so the first transcription isn't slow
    sys.stdout.write(f"Loading MLX model: {mlx_model} (Metal GPU)...\n")
    sys.stdout.flush()

    def _mlx_transcribe(self_recorder):
        """Replacement for AudioToTextRecorder.transcribe() using mlx-whisper."""
        self_recorder._set_state("transcribing")
        audio_copy = copy.deepcopy(self_recorder.audio)
        start_time = time.time()

        try:
            with self_recorder.transcription_lock:
                # Drain any pending faster-whisper requests from the pipe
                # (early transcription may have queued one)
                while self_recorder.transcribe_count > 0:
                    try:
                        self_recorder.parent_transcription_pipe.recv()
                    except Exception:
                        pass
                    self_recorder.transcribe_count -= 1

                self_recorder.allowed_to_early_transcribe = True
                self_recorder._set_state("inactive")

                # Transcribe with mlx-whisper on Metal GPU
                result = mlx_whisper.transcribe(
                    self_recorder.audio,
                    path_or_hf_repo=mlx_model,
                    language="en",
                    initial_prompt=initial_prompt or None,
                    condition_on_previous_text=False,
                )

                transcription = result.get("text", "").strip()
                self_recorder.last_transcription_bytes = audio_copy

                # Apply RealtimeSTT's text post-processing
                transcription = self_recorder._preprocess_output(transcription)

                end_time = time.time()
                elapsed = end_time - start_time
                if self_recorder.print_transcription_time:
                    print(f"MLX {mlx_model.split('/')[-1]} completed transcription in {elapsed:.2f} seconds")

                return transcription

        except Exception as e:
            self_recorder._set_state("inactive")
            sys.stdout.write(f"MLX transcription error: {e}\n")
            sys.stdout.flush()
            raise

    # Bind the replacement method to the recorder instance
    import types
    recorder.transcribe = types.MethodType(_mlx_transcribe, recorder)
    recorder.main_model_type = mlx_model.split("/")[-1]
