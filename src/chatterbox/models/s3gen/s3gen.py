# Modified from CosyVoice https://github.com/FunAudioLLM/CosyVoice
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging

import numpy as np
import torch
import torchaudio as ta
from functools import lru_cache
from typing import Optional

from ..s3tokenizer import S3_SR, SPEECH_VOCAB_SIZE, S3Tokenizer
from .const import S3GEN_SR
from .flow import CausalMaskedDiffWithXvec
from .xvector import CAMPPlus
from .utils.mel import mel_spectrogram
from .f0_predictor import ConvRNNF0Predictor
from .hifigan import HiFTGenerator
from .transformer.upsample_encoder import UpsampleConformerEncoder
from .flow_matching import CausalConditionalCFM
from .decoder import ConditionalDecoder
from .configs import CFM_PARAMS


def drop_invalid_tokens(x):
    assert len(x.shape) <= 2 and x.shape[0] == 1, "only batch size of one allowed for now"
    return x[x < SPEECH_VOCAB_SIZE]


# TODO: global resampler cache
@lru_cache(100)
def get_resampler(src_sr, dst_sr, device):
    return ta.transforms.Resample(src_sr, dst_sr).to(device)


class S3Token2Mel(torch.nn.Module):
    """S3Gen's token-to-mel decoder — converts discrete speech tokens to mel spectrograms.

    This is the first half of S3Gen. It takes speech tokens (from T3 or S3Tokenizer)
    and reference audio conditioning, and generates mel spectrograms via Conditional
    Flow Matching (CFM).

    Components
    ----------
    tokenizer : S3Tokenizer
        Speech tokenizer (16kHz → 25 tok/sec). Used for encoding reference audio.
        Input: 16kHz audio → Output: (B, T) tokens, T = 25 * duration_sec

    speaker_encoder : CAMPPlus
        X-vector speaker encoder for S3Gen's flow model conditioning.
        Input: 16kHz audio (B, num_samples) → Output: (B, 192) speaker embedding
        NOTE: This is DIFFERENT from VoiceEncoder (which produces 256-dim for T3).

    flow : CausalMaskedDiffWithXvec
        The main CFM decoder. Maps speech tokens → mel spectrograms with speaker conditioning.
        - input_embedding: Embedding(6561, 512) — embeds speech tokens
        - encoder: UpsampleConformerEncoder — upsamples tokens 2x (25 tok/s → 50 mel frames/s??)
        - decoder: CausalConditionalCFM — flow matching with speaker conditioning
        - token_mel_ratio: 2 (each token → 2 mel frames)

    Reference conditioning (embed_ref output)
    ------------------------------------------
    embed_ref() produces a dict with:
        prompt_token:     (1, T_ref) long — S3 tokens of reference audio (from 16kHz)
        prompt_token_len: (1,) long — reference token count
        prompt_feat:      (1, T_mel, 80) float — mel spectrogram of reference (from 24kHz)
        prompt_feat_len:  None
        embedding:        (1, 192) float — CAMPPlus x-vector speaker embedding

    The mel/token ratio MUST be 2:1 (enforced with warning + truncation).
    """
    def __init__(self, meanflow=False):
        super().__init__()
        self.tokenizer = S3Tokenizer("speech_tokenizer_v2_25hz")
        self.mel_extractor = mel_spectrogram # TODO: make it a torch module?
        self.speaker_encoder = CAMPPlus(
            # NOTE: This doesn't affect inference. It turns off activation checkpointing
            # (a training optimization), which causes a crazy DDP error with accelerate
            memory_efficient=False,
        )
        self.meanflow = meanflow

        encoder = UpsampleConformerEncoder(
            output_size=512,
            attention_heads=8,
            linear_units=2048,
            num_blocks=6,
            dropout_rate=0.1,
            positional_dropout_rate=0.1,
            attention_dropout_rate=0.1,
            normalize_before=True,
            input_layer='linear',
            pos_enc_layer_type='rel_pos_espnet',
            selfattention_layer_type='rel_selfattn',
            input_size=512,
            use_cnn_module=False,
            macaron_style=False,
        )

        estimator = ConditionalDecoder(
            in_channels=320,
            out_channels=80,
            causal=True,
            channels=[256],
            dropout=0.0,
            attention_head_dim=64,
            n_blocks=4,
            num_mid_blocks=12,
            num_heads=8,
            act_fn='gelu',
            meanflow=self.meanflow,
        )
        cfm_params = CFM_PARAMS
        decoder = CausalConditionalCFM(
            spk_emb_dim=80,
            cfm_params=cfm_params,
            estimator=estimator,
        )

        self.flow = CausalMaskedDiffWithXvec(
            encoder=encoder,
            decoder=decoder
        )

        self.resamplers = {}

    @property
    def device(self):
        params = self.tokenizer.parameters()
        return next(params).device

    @property
    def dtype(self):
        params = self.flow.parameters()
        return next(params).dtype

    def embed_ref(
        self,
        ref_wav: torch.Tensor,
        ref_sr: int,
        device="auto",
        ref_fade_out=True,
    ):
        """Extract reference audio conditioning for S3Gen's flow model.

        Processes reference audio into three conditioning signals:
        1. Mel spectrogram at 24kHz (for acoustic conditioning)
        2. S3 tokens at 16kHz (for token-level conditioning)
        3. CAMPPlus x-vector at 16kHz (for speaker conditioning)

        Args:
            ref_wav: Reference waveform — numpy array or tensor, shape (num_samples,) or (1, num_samples)
                     Max 10 seconds (DEC_COND_LEN = 10 * 24000 = 240,000 samples at 24kHz).
            ref_sr:  Sample rate of ref_wav (will be resampled to 24kHz and 16kHz internally)
            device:  Target device ("auto" uses self.device)

        Returns:
            dict with:
                prompt_token:     (1, T_ref) long — S3 tokens from 16kHz reference
                                  T_ref = ceil(duration_sec * 25)
                prompt_token_len: (1,) long — actual token count
                prompt_feat:      (1, T_mel, 80) float — mel spectrogram from 24kHz reference
                                  T_mel should equal 2 * T_ref (enforced with warning)
                prompt_feat_len:  None
                embedding:        (1, 192) float — CAMPPlus x-vector speaker embedding

        Internal pipeline:
            ref_wav → resample to 24kHz → mel_spectrogram → prompt_feat (1, T_mel, 80)
            ref_wav → resample to 16kHz → S3Tokenizer → prompt_token (1, T_ref)
            ref_wav → resample to 16kHz → CAMPPlus → embedding (1, 192)
        """
        device = self.device if device == "auto" else device
        if isinstance(ref_wav, np.ndarray):
            ref_wav = torch.from_numpy(ref_wav).float()

        if ref_wav.device != device:
            ref_wav = ref_wav.to(device)

        if len(ref_wav.shape) == 1:
            ref_wav = ref_wav.unsqueeze(0)  # (B, L)

        if ref_wav.size(1) > 10 * ref_sr:
            print("WARNING: s3gen received ref longer than 10s")

        ref_wav_24 = ref_wav
        if ref_sr != S3GEN_SR:
            ref_wav_24 = get_resampler(ref_sr, S3GEN_SR, device)(ref_wav)
        ref_wav_24 = ref_wav_24.to(device=device, dtype=self.dtype)

        ref_mels_24 = self.mel_extractor(ref_wav_24).transpose(1, 2).to(dtype=self.dtype)
        ref_mels_24_len = None

        # Resample to 16kHz
        ref_wav_16 = ref_wav
        if ref_sr != S3_SR:
            ref_wav_16 = get_resampler(ref_sr, S3_SR, device)(ref_wav)

        # Speaker embedding
        ref_x_vector = self.speaker_encoder.inference(ref_wav_16.to(dtype=self.dtype))

        # Tokenize 16khz reference
        ref_speech_tokens, ref_speech_token_lens = self.tokenizer(ref_wav_16.float())

        # Make sure mel_len = 2 * stoken_len (happens when the input is not padded to multiple of 40ms)
        if ref_mels_24.shape[1] != 2 * ref_speech_tokens.shape[1]:
            logging.warning(
                "Reference mel length is not equal to 2 * reference token length.\n"
            )
            ref_speech_tokens = ref_speech_tokens[:, :ref_mels_24.shape[1] // 2]
            ref_speech_token_lens[0] = ref_speech_tokens.shape[1]

        return dict(
            prompt_token=ref_speech_tokens.to(device),
            prompt_token_len=ref_speech_token_lens,
            prompt_feat=ref_mels_24,
            prompt_feat_len=ref_mels_24_len,
            embedding=ref_x_vector,
        )

    def forward(
        self,
        speech_tokens: torch.LongTensor,
        # locally-computed ref embedding (mutex with ref_dict)
        ref_wav: Optional[torch.Tensor],
        ref_sr: Optional[int],
        # pre-computed ref embedding (prod API)
        ref_dict: Optional[dict] = None,
        n_cfm_timesteps = None,
        finalize: bool = False,
        speech_token_lens=None,
        noised_mels=None,
    ):
        """
        Generate waveforms from S3 speech tokens and a reference waveform, which the speaker timbre is inferred from.

        NOTE:
        - The speaker encoder accepts 16 kHz waveform.
        - S3TokenizerV2 accepts 16 kHz waveform.
        - The mel-spectrogram for the reference assumes 24 kHz input signal.
        - This function is designed for batch_size=1 only.

        Args
        ----
        - `speech_tokens`: S3 speech tokens [B=1, T]
        - `ref_wav`: reference waveform (`torch.Tensor` with shape=[B=1, T])
        - `ref_sr`: reference sample rate
        - `finalize`: whether streaming is finished or not. Note that if False, the last 3 tokens will be ignored.
        """
        assert (ref_wav is None) ^ (ref_dict is None), f"Must provide exactly one of ref_wav or ref_dict (got {ref_wav} and {ref_dict})"

        if ref_dict is None:
            ref_dict = self.embed_ref(ref_wav, ref_sr)
        else:
            # type/device casting (all values will be numpy if it's from a prod API call)
            for rk in list(ref_dict):
                if isinstance(ref_dict[rk], np.ndarray):
                    ref_dict[rk] = torch.from_numpy(ref_dict[rk])
                if torch.is_tensor(ref_dict[rk]):
                    ref_dict[rk] = ref_dict[rk].to(device=self.device, dtype=self.dtype)

        speech_tokens = torch.atleast_2d(speech_tokens)

        # backcompat
        if speech_token_lens is None:
            speech_token_lens = torch.LongTensor([st.size(-1) for st in speech_tokens]).to(self.device)

        output_mels, _ = self.flow.inference(
            token=speech_tokens,
            token_len=speech_token_lens,
            finalize=finalize,
            noised_mels=noised_mels,
            n_timesteps=n_cfm_timesteps,
            meanflow=self.meanflow,
            **ref_dict,
        )
        return output_mels


class S3Token2Wav(S3Token2Mel):
    """S3Gen full decoder — speech tokens → mel → waveform.

    Combines token-to-mel (CFM flow model) with mel-to-waveform (HiFi-GAN vocoder).
    This is the class instantiated as `S3Gen()` in Chatterbox.

    Components (inherited from S3Token2Mel):
        tokenizer:        S3Tokenizer — 16kHz audio → tokens
        speaker_encoder:  CAMPPlus — 16kHz audio → (B, 192) x-vector
        flow:             CausalMaskedDiffWithXvec — tokens → mel

    Additional components:
        mel2wav:          HiFTGenerator — mel spectrogram → 24kHz waveform
                          Upsample rates: [8, 5, 3] → total 120x
                          With f0 predictor (ConvRNNF0Predictor)
        trim_fade:        Buffer — cosine fade-in to reduce reference spillover

    Full inference pipeline:
        speech_tokens (B, T) → flow_inference → mel (B, 80, T*2) → hift_inference → wav (B, T*240)

    Output sample rate: 24,000 Hz (S3GEN_SR)

    Example:
        s3gen = S3Gen()  # alias for S3Token2Wav
        wav, sources = s3gen.inference(
            speech_tokens=tokens,       # (1, N) long, N speech tokens
            ref_dict=ref_conditioning,  # from embed_ref()
        )
        # wav: (1, N * 960) float — 24kHz waveform (each token → 40ms → 960 samples)
    """

    ignore_state_dict_missing = ("tokenizer._mel_filters", "tokenizer.window")

    def __init__(self, meanflow=False):
        super().__init__(meanflow)

        f0_predictor = ConvRNNF0Predictor()
        self.mel2wav = HiFTGenerator(
            sampling_rate=S3GEN_SR,
            upsample_rates=[8, 5, 3],
            upsample_kernel_sizes=[16, 11, 7],
            source_resblock_kernel_sizes=[7, 7, 11],
            source_resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
            f0_predictor=f0_predictor,
        )

        # silence out a few ms and fade audio in to reduce artifacts
        n_trim = S3GEN_SR // 50  # 20ms = half of a frame
        trim_fade = torch.zeros(2 * n_trim)
        trim_fade[n_trim:] = (torch.cos(torch.linspace(torch.pi, 0, n_trim)) + 1) / 2
        self.register_buffer("trim_fade", trim_fade, persistent=False) # (buffers get automatic device casting)
        self.estimator_dtype = "fp32"

    def forward(
        self,
        speech_tokens,
        # locally-computed ref embedding (mutex with ref_dict)
        ref_wav: Optional[torch.Tensor],
        ref_sr: Optional[int],
        # pre-computed ref embedding (prod API)
        ref_dict: Optional[dict] = None,
        finalize: bool = False,
        speech_token_lens=None,
        skip_vocoder=False,
        n_cfm_timesteps=None,
        noised_mels=None,

    ):
        """
        Generate waveforms from S3 speech tokens and a reference waveform, which the speaker timbre is inferred from.
        NOTE: used for sync synthesis only. Please use `S3GenStreamer` for streaming synthesis.
        """
        output_mels = super().forward(
            speech_tokens, speech_token_lens=speech_token_lens, ref_wav=ref_wav,
            ref_sr=ref_sr, ref_dict=ref_dict, finalize=finalize,
            n_cfm_timesteps=n_cfm_timesteps, noised_mels=noised_mels,
        )

        if skip_vocoder:
            return output_mels

        # TODO jrm: ignoring the speed control (mel interpolation) and the HiFTGAN caching mechanisms for now.
        hift_cache_source = torch.zeros(1, 1, 0).to(self.device)

        output_wavs, *_ = self.mel2wav.inference(speech_feat=output_mels, cache_source=hift_cache_source)

        if not self.training:
            # NOTE: ad-hoc method to reduce "spillover" from the reference clip.
            output_wavs[:, :len(self.trim_fade)] *= self.trim_fade

        return output_wavs

    @torch.inference_mode()
    def flow_inference(
        self,
        speech_tokens,
        # locally-computed ref embedding (mutex with ref_dict)
        ref_wav: Optional[torch.Tensor] = None,
        ref_sr: Optional[int] = None,
        # pre-computed ref embedding (prod API)
        ref_dict: Optional[dict] = None,
        n_cfm_timesteps = None,
        finalize: bool = False,
        speech_token_lens=None,
    ):
        n_cfm_timesteps = n_cfm_timesteps or (2 if self.meanflow else 10)
        noise = None
        if self.meanflow:
            noise = torch.randn(1, 80, speech_tokens.size(-1) * 2, dtype=self.dtype, device=self.device)
        output_mels = super().forward(
            speech_tokens, speech_token_lens=speech_token_lens, ref_wav=ref_wav, ref_sr=ref_sr, ref_dict=ref_dict,
            n_cfm_timesteps=n_cfm_timesteps, finalize=finalize, noised_mels=noise,
        )
        return output_mels

    @torch.inference_mode()
    def hift_inference(self, speech_feat, cache_source: torch.Tensor = None):
        if cache_source is None:
            cache_source = torch.zeros(1, 1, 0).to(device=self.device, dtype=self.dtype)
        return self.mel2wav.inference(speech_feat=speech_feat, cache_source=cache_source)

    @torch.inference_mode()
    def inference(
        self,
        speech_tokens,
        # locally-computed ref embedding (mutex with ref_dict)
        ref_wav: Optional[torch.Tensor] = None,
        ref_sr: Optional[int] = None,
        # pre-computed ref embedding (prod API)
        ref_dict: Optional[dict] = None,
        # left as a kwarg because this can change input/output size ratio
        drop_invalid_tokens=True,
        n_cfm_timesteps=None,
        speech_token_lens=None,
    ):
        """Full inference: speech tokens → waveform.

        This is the main entry point for converting T3-generated speech tokens into audio.

        Args:
            speech_tokens:    (1, N) or (N,) long — speech token IDs in [0, 6560]
                              ⚠ Tokens >= 6561 will cause CUDA crash in flow.input_embedding!
                              Use drop_invalid_tokens or clamp before calling.
            ref_wav:          Optional (1, num_samples) float — reference waveform (any sample rate)
                              Mutually exclusive with ref_dict.
            ref_sr:           int — sample rate of ref_wav
            ref_dict:         Optional dict — pre-computed reference conditioning from embed_ref()
                              Keys: prompt_token, prompt_token_len, prompt_feat, prompt_feat_len, embedding
            n_cfm_timesteps:  int — CFM denoising steps (default: 2 for meanflow, 10 otherwise)
            speech_token_lens: Optional (B,) long — token lengths for batched input

        Returns:
            output_wavs:    (1, num_audio_samples) float — 24kHz waveform
                            Duration ≈ N_tokens * 40ms (each token → 2 mel frames → 960 audio samples)
            output_sources: (1, 1, num_audio_samples) — HiFi-GAN source signal (for f0)

        Pipeline:
            1. flow_inference(tokens, ref_dict) → mel (1, 80, N*2)
            2. hift_inference(mel) → wav (1, num_samples) at 24kHz
            3. Cosine fade-in (20ms) to reduce reference spillover artifact
        """

        output_mels = self.flow_inference(
            speech_tokens,
            speech_token_lens=speech_token_lens,
            ref_wav=ref_wav,
            ref_sr=ref_sr,
            ref_dict=ref_dict,
            n_cfm_timesteps=n_cfm_timesteps,
            finalize=True,
        )
        output_mels = output_mels.to(dtype=self.dtype) # FIXME (fp16 mode) is this still needed?
        output_wavs, output_sources = self.hift_inference(output_mels, None)

        # NOTE: ad-hoc method to reduce "spillover" from the reference clip.
        output_wavs[:, :len(self.trim_fade)] *= self.trim_fade

        return output_wavs, output_sources
