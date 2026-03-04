"""
T3 Conditioning — Dataclass + Encoder for all non-text conditioning signals.

Architecture Overview
=====================
T3Cond holds all conditioning inputs. T3CondEnc projects them into a sequence of
conditioning tokens that are prepended to the transformer input.

Conditioning tokens breakdown (default config):
    speaker_emb      → Linear(256 → 1024)           → (B, 1, 1024)    [1 token]
    clap_emb         → (unused, always None)         → (B, 0, 1024)    [0 tokens]
    speech_cond_emb  → Perceiver(150→32 queries)     → (B, 32, 1024)   [32 tokens]
    emotion_adv      → Linear(1 → 1024, no bias)     → (B, 1, 1024)    [1 token]
                                                        ──────────────
    Total conditioning sequence:                        (B, 34, 1024)   [34 tokens]

    ⚠ Without cond_prompt_speech_tokens: only (B, 2, 1024) — 2 tokens!
    This causes a massive train/inference mismatch if omitted during training.

The full T3 transformer input is: [cond_tokens | text_tokens | speech_tokens]
"""
from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn, Tensor

from .perceiver import Perceiver
from .t3_config import T3Config


@dataclass
class T3Cond:
    """Container for all T3 conditioning signals.

    Fields
    ------
    speaker_emb : Tensor
        Shape: (B, 256) or (1, 256)
        L2-normalized speaker embedding from VoiceEncoder.
        VoiceEncoder is a 3-layer LSTM (40-mel → 256-dim hidden → 256-dim projection + L2 norm).
        Input audio must be 16kHz. The VE internally computes 40-band mel spectrograms.

    clap_emb : Optional[Tensor]
        Not implemented. Always None.

    cond_prompt_speech_tokens : Optional[Tensor]
        Shape: (B, 150) dtype=long
        S3Tokenizer tokens from reference audio (first 6 seconds, at 25 tokens/sec = 150 tokens).
        Created by: s3gen.tokenizer(ref_16k_wav[:6*16000]) → pad/truncate to 150.
        These tokens are embedded by T3.speech_emb and position-encoded, then compressed
        by the Perceiver Resampler (150 → 32 tokens).

    cond_prompt_speech_emb : Optional[Tensor]
        Shape: (B, 150, 1024) — computed by T3.prepare_conditioning(), not set by user.
        = T3.speech_emb(cond_prompt_speech_tokens) + T3.speech_pos_emb(cond_prompt_speech_tokens)
        This is then fed through Perceiver → (B, 32, 1024).

    emotion_adv : Optional[Tensor]
        Shape: (B, 1, 1) or scalar float (default 0.5)
        Emotion exaggeration control. 0.0 = neutral, 1.0 = maximum expressiveness.
        Projected by: Linear(1 → 1024, no bias) → (B, 1, 1024).
    """

    speaker_emb: Tensor
    clap_emb: Optional[Tensor] = None
    cond_prompt_speech_tokens: Optional[Tensor] = None
    cond_prompt_speech_emb: Optional[Tensor] = None
    emotion_adv: Optional[Tensor] = 0.5

    def to(self, *, device=None, dtype=None):
        "Cast to a device and dtype. Dtype casting is ignored for long/int tensors."
        for k, v in self.__dict__.items():
            if torch.is_tensor(v):
                is_fp = type(v.view(-1)[0].item()) is not int
                setattr(self, k, v.to(device=device, dtype=dtype if is_fp else None))
        return self

    def save(self, fpath):
        torch.save(self.__dict__, fpath)

    @staticmethod
    def load(fpath, map_location="cpu"):
        kwargs = torch.load(fpath, map_location=map_location, weights_only=True)
        return T3Cond(**kwargs)


class T3CondEnc(nn.Module):
    """Conditioning encoder — projects all non-text conditioning into transformer-compatible tokens.

    Modules
    -------
    spkr_enc : Linear(256 → 1024)
        Projects VoiceEncoder speaker embedding to transformer dim.

    emotion_adv_fc : Linear(1 → 1024, no bias)
        Projects emotion exaggeration scalar to transformer dim.

    perceiver : Perceiver (32 learned queries, cross-attention + self-attention)
        Compresses 150 speech conditioning embeddings into 32 tokens.
        Input: (B, 150, 1024) → Output: (B, 32, 1024)

    Forward pass shapes
    -------------------
    Input:  T3Cond (see dataclass above)
    Output: (B, L_cond, 1024) where L_cond = 1 + 0 + {0 or 32} + {0 or 1}

    With all conditioning (typical inference):
        cond_spkr:              (B, 1, 1024)   — from speaker_emb
        cond_clap:              (B, 0, 1024)   — unused
        cond_prompt_speech_emb: (B, 32, 1024)  — Perceiver output
        cond_emotion_adv:       (B, 1, 1024)   — from emotion_adv
        → concat → (B, 34, 1024)

    Without speech conditioning tokens (buggy training):
        cond_spkr:              (B, 1, 1024)
        cond_clap:              (B, 0, 1024)
        cond_prompt_speech_emb: (B, 0, 1024)   — empty!
        cond_emotion_adv:       (B, 1, 1024)
        → concat → (B, 2, 1024)                — only 2 tokens vs 34!
    """

    def __init__(self, hp: T3Config):
        super().__init__()
        self.hp = hp
        if hp.encoder_type == "voice_encoder":
            # speaker_embed_size=256, n_channels=1024 (hidden_size from Llama config)
            self.spkr_enc = nn.Linear(hp.speaker_embed_size, hp.n_channels)
        else:
            raise NotImplementedError(str(hp.encoder_type))

        # emotion adv — Linear(1 → 1024, no bias)
        self.emotion_adv_fc = None
        if hp.emotion_adv:
            self.emotion_adv_fc = nn.Linear(1, hp.n_channels, bias=False)

        # Perceiver Resampler — compresses 150 speech tokens into 32 query tokens
        # Uses 32 learned queries of dim 1024, cross-attention with 4 heads
        self.perceiver = None
        if hp.use_perceiver_resampler:
            self.perceiver = Perceiver()

    def forward(self, cond: T3Cond):
        """Project all conditioning signals and concatenate into a token sequence.

        Args:
            cond: T3Cond dataclass with all conditioning fields.
                  NOTE: cond.cond_prompt_speech_emb must already be computed
                  by T3.prepare_conditioning() before calling this method.

        Returns:
            cond_embeds: (B, L_cond, 1024) — conditioning token sequence.
                L_cond = 34 with full conditioning, 2 without speech tokens.
        """
        # Validate: tokens and embeddings must be both present or both absent
        assert (cond.cond_prompt_speech_tokens is None) == (cond.cond_prompt_speech_emb is None), \
            "no embeddings for cond_prompt_speech_tokens"

        # Speaker embedding: (B, 256) → Linear → (B, 1, 1024)
        cond_spkr = self.spkr_enc(cond.speaker_emb.view(-1, self.hp.speaker_embed_size))[:, None]  # (B, 1, dim)
        empty = torch.zeros_like(cond_spkr[:, :0])  # (B, 0, dim) — zero-length placeholder

        # CLAP embedding: not implemented, always empty
        assert cond.clap_emb is None, "clap_embed not implemented"
        cond_clap = empty  # (B, 0, dim)

        # Speech conditioning prompt:
        #   If present: (B, 150, 1024) → Perceiver → (B, 32, 1024)
        #   If absent:  (B, 0, 1024) — no speech conditioning tokens
        cond_prompt_speech_emb = cond.cond_prompt_speech_emb
        if cond_prompt_speech_emb is None:
            cond_prompt_speech_emb = empty  # (B, 0, dim)
        elif self.hp.use_perceiver_resampler:
            cond_prompt_speech_emb = self.perceiver(cond_prompt_speech_emb)  # (B, 32, 1024)

        # Emotion exaggeration: scalar → Linear(1 → 1024) → (B, 1, 1024)
        cond_emotion_adv = empty  # (B, 0, dim)
        if self.hp.emotion_adv:
            assert cond.emotion_adv is not None
            cond_emotion_adv = self.emotion_adv_fc(cond.emotion_adv.view(-1, 1, 1))  # (B, 1, 1024)

        # Concatenate all conditioning: (B, L_cond, 1024)
        # Order: [speaker(1) | clap(0) | speech(32) | emotion(1)] = 34 tokens total
        cond_embeds = torch.cat((
            cond_spkr,
            cond_clap,
            cond_prompt_speech_emb,
            cond_emotion_adv,
        ), dim=1)
        return cond_embeds
