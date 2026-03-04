"""
ChatterboxMultilingualTTS — Multilingual text-to-speech inference pipeline.

Supports 23 languages (see SUPPORTED_LANGUAGES dict). Uses the same 4-component
architecture as the English-only ChatterboxTTS but with:
    - MTLTokenizer (grapheme-based, vocab=2454) instead of EnTokenizer (vocab=704)
    - T3 with T3Config.multilingual() (text_tokens_dict_size=2454)
    - A `language_id` parameter in generate() to select target language

Architecture (same components, different text tokenizer + T3 config)
====================================================================
    VoiceEncoder  →  speaker_emb (B, 256)          — speaker identity
    S3Tokenizer   →  cond_prompt_speech_tokens (B, 150)  — speech style conditioning
    T3            →  speech_tokens (1, T)           — autoregressive text→token generation
    S3Gen         →  waveform (1, num_samples)      — token→audio synthesis at 24kHz

Inference pipeline
==================
    1. Load reference audio → resample to 24kHz (S3Gen) and 16kHz (VE + S3Tokenizer)
    2. VoiceEncoder: 16kHz ref → speaker_emb (1, 256)
    3. S3Tokenizer: 16kHz ref[:6s] → cond_prompt_speech_tokens (1, 150)
    4. S3Gen.embed_ref: 24kHz ref[:10s] → ref_dict (prompt conditioning for CFM decoder)
    5. MTLTokenizer: text + language_id → text_tokens (1, N)
    6. T3.inference: [cond(34) | text(N+2) | speech(?)] → speech_tokens (1, T)
    7. S3Gen.inference: speech_tokens + ref_dict → wav (1, num_samples) at 24kHz
    8. Perth watermarking → watermarked wav

Key differences from English-only ChatterboxTTS (tts.py)
=========================================================
    - text_tokens_dict_size: 2454 vs 704
    - Tokenizer: MTLTokenizer (grapheme) vs EnTokenizer (phoneme-based)
    - generate() takes `language_id` param (ISO 639-1 code, e.g. "el", "fr", "zh")
    - T3 checkpoint: t3_mtl23ls_v2.safetensors vs t3_cfg.safetensors
    - Conditionals cache file: conds.pt (shared with English version)

Constants
---------
    S3_SR = 16,000 Hz       — S3Tokenizer + VoiceEncoder sample rate
    S3GEN_SR = 24,000 Hz    — S3Gen output sample rate
    ENC_COND_LEN = 96,000   — 6 seconds at 16kHz (S3Tokenizer conditioning input)
    DEC_COND_LEN = 240,000  — 10 seconds at 24kHz (S3Gen reference input)
"""
from dataclasses import dataclass
from pathlib import Path
import os

import librosa
import torch
import perth
import torch.nn.functional as F
from safetensors.torch import load_file as load_safetensors
from huggingface_hub import snapshot_download

from .models.t3 import T3
from .models.t3.modules.t3_config import T3Config
from .models.s3tokenizer import S3_SR, drop_invalid_tokens
from .models.s3gen import S3GEN_SR, S3Gen
from .models.tokenizers import MTLTokenizer
from .models.voice_encoder import VoiceEncoder
from .models.t3.modules.cond_enc import T3Cond


REPO_ID = "ResembleAI/chatterbox"

# Supported languages for the multilingual model
SUPPORTED_LANGUAGES = {
  "ar": "Arabic",
  "da": "Danish",
  "de": "German",
  "el": "Greek",
  "en": "English",
  "es": "Spanish",
  "fi": "Finnish",
  "fr": "French",
  "he": "Hebrew",
  "hi": "Hindi",
  "it": "Italian",
  "ja": "Japanese",
  "ko": "Korean",
  "ms": "Malay",
  "nl": "Dutch",
  "no": "Norwegian",
  "pl": "Polish",
  "pt": "Portuguese",
  "ru": "Russian",
  "sv": "Swedish",
  "sw": "Swahili",
  "tr": "Turkish",
  "zh": "Chinese",
}


def punc_norm(text: str) -> str:
    """
        Quick cleanup func for punctuation from LLMs or
        containing chars not seen often in the dataset
    """
    if len(text) == 0:
        return "You need to add some text for me to talk."

    # Capitalise first letter
    if text[0].islower():
        text = text[0].upper() + text[1:]

    # Remove multiple space chars
    text = " ".join(text.split())

    # Replace uncommon/llm punc
    punc_to_replace = [
        ("...", ", "),
        ("…", ", "),
        (":", ","),
        (" - ", ", "),
        (";", ", "),
        ("—", "-"),
        ("–", "-"),
        (" ,", ","),
        ("“", "\""),
        ("”", "\""),
        ("‘", "'"),
        ("’", "'"),
    ]
    for old_char_sequence, new_char in punc_to_replace:
        text = text.replace(old_char_sequence, new_char)

    # Add full stop if no ending punc
    text = text.rstrip(" ")
    sentence_enders = {".", "!", "?", "-", ",","、","，","。","？","！"}
    if not any(text.endswith(p) for p in sentence_enders):
        text += "."

    return text


@dataclass
class Conditionals:
    """Container for all pre-computed conditioning signals used during inference.

    Holds both T3 (autoregressive token generation) and S3Gen (mel synthesis) conditioning.

    Fields
    ------
    t3 : T3Cond
        T3 conditioning dataclass containing:
            speaker_emb              : (1, 256) — L2-normalized VoiceEncoder embedding
            cond_prompt_speech_tokens: (1, 150) — S3Tokenizer tokens from 6s reference
            cond_prompt_speech_emb   : None (computed lazily by T3.prepare_conditioning)
            emotion_adv              : (1, 1, 1) — emotion exaggeration scalar [0.0-1.0]

    gen : dict
        S3Gen reference conditioning dict from S3Gen.embed_ref(), containing:
            prompt_token     : (1, T_ref) long — S3 tokens from reference audio
            prompt_token_len : (1,) long — length of prompt tokens
            prompt_feat      : (1, T_mel, 100) — mel spectrogram of reference
            prompt_feat_len  : (1,) long — length of mel features
            embedding        : (1, 192) — speaker verification embedding (ECAPA-TDNN)

    Serialization
    -------------
        conds.save("conds.pt")           — saves t3.__dict__ + gen dict
        Conditionals.load("conds.pt")    — restores from file
    """
    t3: T3Cond
    gen: dict

    def to(self, device):
        self.t3 = self.t3.to(device=device)
        for k, v in self.gen.items():
            if torch.is_tensor(v):
                self.gen[k] = v.to(device=device)
        return self

    def save(self, fpath: Path):
        arg_dict = dict(
            t3=self.t3.__dict__,
            gen=self.gen
        )
        torch.save(arg_dict, fpath)

    @classmethod
    def load(cls, fpath, map_location="cpu"):
        kwargs = torch.load(fpath, map_location=map_location, weights_only=True)
        return cls(T3Cond(**kwargs['t3']), kwargs['gen'])


class ChatterboxMultilingualTTS:
    """Multilingual text-to-speech model supporting 23 languages.

    Components (same as English-only, different T3 config + tokenizer)
    ------------------------------------------------------------------
    t3        : T3 — Autoregressive text→speech token model (LlamaModel backbone)
                Config: T3Config.multilingual() → text_tokens_dict_size=2454
    s3gen     : S3Gen — Speech token→waveform synthesis (CFM flow + HiFi-GAN vocoder)
    ve        : VoiceEncoder — 3-layer LSTM speaker encoder (40-mel → 256-dim)
    tokenizer : MTLTokenizer — Grapheme-based multilingual tokenizer (vocab=2454)

    Class constants
    ---------------
    ENC_COND_LEN = 96,000   — 6 seconds * 16kHz — max reference audio for S3Tokenizer
    DEC_COND_LEN = 240,000  — 10 seconds * 24kHz — max reference audio for S3Gen

    Attributes
    ----------
    sr          : int = 24,000 — output audio sample rate (S3GEN_SR)
    device      : str — torch device ("cuda", "cpu")
    conds       : Conditionals — pre-computed conditioning (set by prepare_conditionals)
    watermarker : PerthImplicitWatermarker — applies Perth audio watermark to output
    """
    ENC_COND_LEN = 6 * S3_SR
    DEC_COND_LEN = 10 * S3GEN_SR

    def __init__(
        self,
        t3: T3,
        s3gen: S3Gen,
        ve: VoiceEncoder,
        tokenizer: MTLTokenizer,
        device: str,
        conds: Conditionals = None,
    ):
        self.sr = S3GEN_SR  # sample rate of synthesized audio
        self.t3 = t3
        self.s3gen = s3gen
        self.ve = ve
        self.tokenizer = tokenizer
        self.device = device
        self.conds = conds
        self.watermarker = perth.PerthImplicitWatermarker()

    @classmethod
    def get_supported_languages(cls):
        """Return dictionary of supported language codes and names."""
        return SUPPORTED_LANGUAGES.copy()

    @classmethod
    def from_local(cls, ckpt_dir, device) -> 'ChatterboxMultilingualTTS':
        ckpt_dir = Path(ckpt_dir)

        ve = VoiceEncoder()
        ve.load_state_dict(
            torch.load(ckpt_dir / "ve.pt", weights_only=True)
        )
        ve.to(device).eval()

        t3 = T3(T3Config.multilingual())
        t3_state = load_safetensors(ckpt_dir / "t3_mtl23ls_v2.safetensors")
        if "model" in t3_state.keys():
            t3_state = t3_state["model"][0]
        t3.load_state_dict(t3_state)
        t3.to(device).eval()

        s3gen = S3Gen()
        s3gen.load_state_dict(
            torch.load(ckpt_dir / "s3gen.pt", weights_only=True)
        )
        s3gen.to(device).eval()

        tokenizer = MTLTokenizer(
            str(ckpt_dir / "grapheme_mtl_merged_expanded_v1.json")
        )

        conds = None
        if (builtin_voice := ckpt_dir / "conds.pt").exists():
            conds = Conditionals.load(builtin_voice).to(device)

        return cls(t3, s3gen, ve, tokenizer, device, conds=conds)

    @classmethod
    def from_pretrained(cls, device: torch.device) -> 'ChatterboxMultilingualTTS':
        ckpt_dir = Path(
            snapshot_download(
                repo_id=REPO_ID,
                repo_type="model",
                revision="main", 
                allow_patterns=["ve.pt", "t3_mtl23ls_v2.safetensors", "s3gen.pt", "grapheme_mtl_merged_expanded_v1.json", "conds.pt", "Cangjie5_TC.json"],
                token=os.getenv("HF_TOKEN"),
            )
        )
        return cls.from_local(ckpt_dir, device)
    
    def prepare_conditionals(self, wav_fpath, exaggeration=0.5):
        """Compute all conditioning signals from a reference audio file.

        Identical to ChatterboxTTS.prepare_conditionals(). Loads reference audio,
        extracts speaker embedding, speech conditioning tokens, and S3Gen reference features.

        Args:
            wav_fpath: Path to reference audio file (any format librosa supports).
            exaggeration: float [0.0-1.0] — emotion exaggeration level (default 0.5).

        Internal pipeline:
            1. librosa.load(wav_fpath, sr=24000) → s3gen_ref_wav (24kHz numpy)
            2. Resample 24kHz → 16kHz → ref_16k_wav
            3. S3Gen.embed_ref(ref_24k[:10s]) → s3gen_ref_dict
               Keys: prompt_token (1, T), prompt_feat (1, T, 100), embedding (1, 192)
            4. S3Tokenizer(ref_16k[:6s], max_len=150) → cond_prompt_speech_tokens (1, 150)
            5. VoiceEncoder.embeds_from_wavs([ref_16k]) → speaker_emb (1, 256)
            6. Assemble T3Cond + Conditionals → self.conds

        Sets:
            self.conds : Conditionals — ready for generate() calls.
        """
        ## Load reference wav
        s3gen_ref_wav, _sr = librosa.load(wav_fpath, sr=S3GEN_SR)

        ref_16k_wav = librosa.resample(s3gen_ref_wav, orig_sr=S3GEN_SR, target_sr=S3_SR)

        s3gen_ref_wav = s3gen_ref_wav[:self.DEC_COND_LEN]
        s3gen_ref_dict = self.s3gen.embed_ref(s3gen_ref_wav, S3GEN_SR, device=self.device)

        # Speech cond prompt tokens
        t3_cond_prompt_tokens = None
        if plen := self.t3.hp.speech_cond_prompt_len:
            s3_tokzr = self.s3gen.tokenizer
            t3_cond_prompt_tokens, _ = s3_tokzr.forward([ref_16k_wav[:self.ENC_COND_LEN]], max_len=plen)
            t3_cond_prompt_tokens = torch.atleast_2d(t3_cond_prompt_tokens).to(self.device)

        # Voice-encoder speaker embedding
        ve_embed = torch.from_numpy(self.ve.embeds_from_wavs([ref_16k_wav], sample_rate=S3_SR))
        ve_embed = ve_embed.mean(axis=0, keepdim=True).to(self.device)

        t3_cond = T3Cond(
            speaker_emb=ve_embed,
            cond_prompt_speech_tokens=t3_cond_prompt_tokens,
            emotion_adv=exaggeration * torch.ones(1, 1, 1),
        ).to(device=self.device)
        self.conds = Conditionals(t3_cond, s3gen_ref_dict)

    def generate(
        self,
        text,
        language_id,
        audio_prompt_path=None,
        exaggeration=0.5,
        cfg_weight=0.5,
        temperature=0.8,
        repetition_penalty=2.0,
        min_p=0.05,
        top_p=1.0,
    ):
        """Generate speech audio from text in a specified language.

        Args:
            text: str — Input text to synthesize.
            language_id: str — ISO 639-1 language code (e.g. "el", "fr", "zh").
                Must be in SUPPORTED_LANGUAGES (23 languages).
            audio_prompt_path: Optional[str] — Path to reference audio for voice cloning.
                If provided, calls prepare_conditionals() first.
                If None, uses previously cached self.conds.
            exaggeration: float — Emotion exaggeration [0.0-1.0] (default 0.5).
            cfg_weight: float — Classifier-free guidance weight (default 0.5).
                Higher = more text adherence, lower = more natural variation.
            temperature: float — Sampling temperature for T3 (default 0.8).
            repetition_penalty: float — Token repetition penalty (default 2.0).
            min_p: float — Minimum probability threshold for sampling (default 0.05).
            top_p: float — Nucleus sampling threshold (default 1.0 = disabled).

        Returns:
            torch.Tensor — Watermarked audio, shape (1, num_samples) at 24kHz.

        Pipeline:
            1. Validate language_id against SUPPORTED_LANGUAGES
            2. prepare_conditionals() if audio_prompt_path given
            3. Update emotion exaggeration if changed
            4. punc_norm(text) → MTLTokenizer.text_to_tokens(text, language_id) → (1, N)
            5. Duplicate text_tokens for CFG: (2, N) — [conditional, unconditional]
            6. Prepend SOT (255), append EOT (0) → (2, N+2)
            7. T3.inference(t3_cond, text_tokens) → speech_tokens (1, T)
            8. drop_invalid_tokens() — remove tokens >= SPEECH_VOCAB_SIZE (6561)
            9. S3Gen.inference(speech_tokens, ref_dict) → wav at 24kHz
           10. Perth watermark → return (1, num_samples)
        """
        # Validate language_id
        if language_id and language_id.lower() not in SUPPORTED_LANGUAGES:
            supported_langs = ", ".join(SUPPORTED_LANGUAGES.keys())
            raise ValueError(
                f"Unsupported language_id '{language_id}'. "
                f"Supported languages: {supported_langs}"
            )
        
        if audio_prompt_path:
            self.prepare_conditionals(audio_prompt_path, exaggeration=exaggeration)
        else:
            assert self.conds is not None, "Please `prepare_conditionals` first or specify `audio_prompt_path`"

        # Update exaggeration if needed
        if float(exaggeration) != float(self.conds.t3.emotion_adv[0, 0, 0].item()):
            _cond: T3Cond = self.conds.t3
            self.conds.t3 = T3Cond(
                speaker_emb=_cond.speaker_emb,
                cond_prompt_speech_tokens=_cond.cond_prompt_speech_tokens,
                emotion_adv=exaggeration * torch.ones(1, 1, 1),
            ).to(device=self.device)

        # Norm and tokenize text
        text = punc_norm(text)
        text_tokens = self.tokenizer.text_to_tokens(text, language_id=language_id.lower() if language_id else None).to(self.device)
        text_tokens = torch.cat([text_tokens, text_tokens], dim=0)  # Need two seqs for CFG

        sot = self.t3.hp.start_text_token
        eot = self.t3.hp.stop_text_token
        text_tokens = F.pad(text_tokens, (1, 0), value=sot)
        text_tokens = F.pad(text_tokens, (0, 1), value=eot)

        with torch.inference_mode():
            speech_tokens = self.t3.inference(
                t3_cond=self.conds.t3,
                text_tokens=text_tokens,
                max_new_tokens=1000,  # TODO: use the value in config
                temperature=temperature,
                cfg_weight=cfg_weight,
                repetition_penalty=repetition_penalty,
                min_p=min_p,
                top_p=top_p,
            )
            # Extract only the conditional batch.
            speech_tokens = speech_tokens[0]

            # TODO: output becomes 1D
            speech_tokens = drop_invalid_tokens(speech_tokens)
            speech_tokens = speech_tokens.to(self.device)

            wav, _ = self.s3gen.inference(
                speech_tokens=speech_tokens,
                ref_dict=self.conds.gen,
            )
            wav = wav.squeeze(0).detach().cpu().numpy()
            watermarked_wav = self.watermarker.apply_watermark(wav, sample_rate=self.sr)
        return torch.from_numpy(watermarked_wav).unsqueeze(0)
