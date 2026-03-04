"""
S3Tokenizer — Speech-to-Token encoder for Chatterbox TTS.

Converts raw audio waveforms into discrete speech tokens used by both T3 (as targets)
and S3Gen (as input for mel generation).

Constants
---------
S3_SR = 16,000 Hz          — REQUIRED input sample rate. Audio MUST be resampled to 16kHz.
S3_HOP = 160               — STFT hop length (10ms at 16kHz) → 100 mel frames/sec
S3_TOKEN_HOP = 640          — Token hop length (40ms at 16kHz) → 25 tokens/sec
S3_TOKEN_RATE = 25           — Output token rate: 25 tokens per second of audio
SPEECH_VOCAB_SIZE = 6561     — Number of valid speech tokens (IDs 0-6560)

Token special values (defined in __init__.py):
    SOS = 6561               — Start-of-speech (used by T3)
    EOS = 6562               — End-of-speech (used by T3)

Pipeline
--------
    16kHz audio → log_mel_spectrogram → S3TokenizerV2.quantize → speech tokens

    Mel spectrogram: n_fft=400, hop=160, 128 mel bands (from s3tokenizer config)
    Produces 100 mel frames/sec, quantized to 25 tokens/sec (4 mel frames per token)

Example
-------
    tokenizer = S3Tokenizer("speech_tokenizer_v2_25hz")
    # Input: 16kHz audio tensor, shape (1, num_samples) or list of numpy arrays
    tokens, token_lens = tokenizer(audio_16k)
    # tokens: (B, T) long, where T = ceil(audio_duration_sec * 25)
    # token_lens: (B,) long — actual token count per sample

    For 6 seconds of audio: T = 6 * 25 = 150 tokens
    For 10 seconds of audio: T = 10 * 25 = 250 tokens
"""
from typing import List, Tuple

import numpy as np
import librosa
import torch
import torch.nn.functional as F
from s3tokenizer.utils import padding
from s3tokenizer.model_v2 import (
    S3TokenizerV2,
    ModelConfig,
)


# ⚠ CRITICAL: S3Tokenizer ONLY accepts 16kHz audio. Feeding 24kHz produces garbage tokens.
S3_SR = 16_000              # Required input sample rate
S3_HOP = 160                # STFT hop length → 100 mel frames/sec at 16kHz
S3_TOKEN_HOP = 640          # Token hop length → 25 tokens/sec at 16kHz
S3_TOKEN_RATE = 25           # Output: 25 discrete speech tokens per second
SPEECH_VOCAB_SIZE = 6561     # Valid speech token IDs: [0, 6560]


class S3Tokenizer(S3TokenizerV2):
    """S3Tokenizer — Converts 16kHz audio to discrete speech tokens at 25 tokens/sec.

    Inherits from s3tokenizer.S3TokenizerV2 with additions:
    - Integrated forward() that handles list-of-arrays input
    - log_mel_spectrogram() using registered buffer mel filters (GPU-friendly)

    Mel spectrogram config:
        n_fft:    400 (25ms window at 16kHz)
        hop:      160 (10ms hop → 100 frames/sec)
        n_mels:   128 (from ModelConfig)
        window:   Hann

    Quantization: 100 mel frames → 25 tokens (4:1 compression by the VQ encoder)
    """

    ignore_state_dict_missing = ("_mel_filters", "window")

    def __init__(
        self,
        name: str="speech_tokenizer_v2_25hz",
        config: ModelConfig = ModelConfig()
    ):
        super().__init__(name)

        self.n_fft = 400
        _mel_filters = librosa.filters.mel(
            sr=S3_SR,
            n_fft=self.n_fft,
            n_mels=config.n_mels
        )
        self.register_buffer(
            "_mel_filters",
            torch.FloatTensor(_mel_filters),
        )

        self.register_buffer(
            "window",
            torch.hann_window(self.n_fft),
        )

    def pad(self, wavs, sr) -> List[torch.Tensor]:
        """
        Given a list of wavs with the same `sample_rate`, pad them so that the length is multiple of 40ms (S3 runs at 25 token/sec).
        """
        processed_wavs = []
        for wav in wavs:
            if isinstance(wav, np.ndarray):
                wav = torch.from_numpy(wav)
            if wav.dim() == 1:
                wav = wav.unsqueeze(0)

            n_tokens = (wav.shape[1] / sr) * S3_TOKEN_RATE
            n_tokens = np.ceil(n_tokens)
            intended_wav_len = n_tokens * (sr / S3_TOKEN_RATE)
            intended_wav_len = int(intended_wav_len)
            wav = torch.nn.functional.pad(
                wav,
                (0, intended_wav_len - wav.shape[-1]),
                mode="constant",
                value=0
            )
            processed_wavs.append(wav)
        return processed_wavs

    def _prepare_audio(self, wavs):
        """Prepare a list of audios for s3tokenizer processing."""
        processed_wavs = []
        for wav in wavs:
            if isinstance(wav, np.ndarray):
                wav = torch.from_numpy(wav)
            if wav.dim() == 1:
                wav = wav.unsqueeze(0)

            processed_wavs.append(wav)
        return processed_wavs

    @torch.no_grad()
    def forward(
        self,
        wavs: torch.Tensor,
        accelerator: 'Accelerator'=None,
        max_len: int=None,
    ) -> Tuple[torch.Tensor, torch.LongTensor]:
        """Tokenize 16kHz audio into discrete speech tokens.

        ⚠ Input MUST be 16kHz. Feeding other sample rates produces garbage tokens.

        Args:
            wavs: 16kHz speech audio. Accepts:
                  - torch.Tensor of shape (1, num_samples) or (num_samples,)
                  - List of numpy arrays [np.ndarray(num_samples,), ...]
                  - List of tensors [(1, num_samples), ...]
            max_len: Maximum output token sequence length (at 25 tok/sec).
                     If set, truncates mel to max_len*4 frames before quantization.
                     For 6s reference: max_len=150. For 10s: max_len=250.

        Returns:
            speech_tokens: (B, T) long — discrete token IDs in range [0, 6560]
                           T = ceil(audio_duration_sec * 25)
            speech_token_lens: (B,) long — actual token count per sample

        Pipeline:
            1. wavs → log_mel_spectrogram() → (1, 128, num_frames)  [100 frames/sec]
            2. Optional truncation: mel[:, :, :max_len*4]
            3. padding() → batched mels
            4. quantize(mels, mel_lens) → (B, T) tokens at 25 tok/sec

        Examples:
            1 second audio (16000 samples)  → ~25 tokens
            6 seconds audio (96000 samples) → ~150 tokens
            10 seconds audio (160000 samples) → ~250 tokens
        """
        processed_wavs = self._prepare_audio(wavs)
        mels, mel_lens = [], []
        for wav in processed_wavs:
            wav = wav.to(self.device)
            mel = self.log_mel_spectrogram(wav)  # [B=1, F, T]
            if max_len is not None:
                mel = mel[..., :max_len * 4]  # num_mel_frames = 4 * num_tokens
            mels.append(mel.squeeze(0))

        mels, mel_lens = padding(mels)
        if accelerator is None:
            tokenizer = self
        else:
            tokenizer = accelerator.unwrap_model(self)

        speech_tokens, speech_token_lens = tokenizer.quantize(mels, mel_lens.to(self.device))
        return (
            speech_tokens.long().detach(),
            speech_token_lens.long().detach(),
        )

    def log_mel_spectrogram(
        self,
        audio: torch.Tensor,
        padding: int = 0,
    ):
        """Compute log-mel spectrogram for S3Tokenizer quantization.

        ⚠ Input MUST be 16kHz audio.

        Args:
            audio: torch.Tensor — 16kHz waveform
                   Shape: (num_samples,) or (1, num_samples) or (B, num_samples)
            padding: int — number of zero samples to pad to the right

        Returns:
            log_mel: torch.Tensor, shape (B, 128, n_frames)
                     n_frames ≈ num_samples / S3_HOP = num_samples / 160
                     At 16kHz: 100 mel frames per second of audio.

        STFT config:
            n_fft:      400 (25ms window at 16kHz)
            hop_length: 160 (10ms hop at 16kHz → 100 frames/sec)
            window:     Hann (registered buffer)
            mel_bands:  128 (via registered _mel_filters)

        Post-processing:
            1. magnitude² of STFT
            2. mel filterbank projection (128 bands)
            3. log10, clamped to min=1e-10
            4. max-normalized: clamp to (max - 8.0)
            5. shifted and scaled: (log_spec + 4.0) / 4.0
        """
        if not torch.is_tensor(audio):
            audio = torch.from_numpy(audio)

        audio = audio.to(self.device)
        if padding > 0:
            audio = F.pad(audio, (0, padding))
        stft = torch.stft(
            audio, self.n_fft, S3_HOP,
            window=self.window.to(self.device),
            return_complex=True
        )
        magnitudes = stft[..., :-1].abs()**2

        mel_spec = self._mel_filters.to(self.device) @ magnitudes

        log_spec = torch.clamp(mel_spec, min=1e-10).log10()
        log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
        log_spec = (log_spec + 4.0) / 4.0
        return log_spec
