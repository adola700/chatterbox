"""
ChatterboxTTS — English-only Text-to-Speech pipeline.

Full inference pipeline:
========================
    1. prepare_conditionals(audio_prompt_path):
       │
       ├─ librosa.load(wav, sr=24000) → s3gen_ref_wav
       ├─ resample to 16kHz → ref_16k_wav
       │
       ├─ S3Gen.embed_ref(s3gen_ref_wav[:10s], 24kHz):
       │   ├─ mel_spectrogram(24kHz audio) → prompt_feat (1, T_mel, 80)
       │   ├─ CAMPPlus(16kHz audio) → embedding (1, 192) [x-vector]
       │   └─ S3Tokenizer(16kHz audio) → prompt_token (1, T_ref)
       │   → Returns ref_dict for S3Gen decoder
       │
       ├─ S3Tokenizer(ref_16k[:6s], max_len=150):
       │   → cond_prompt_speech_tokens (1, 150)     [for T3 conditioning]
       │
       ├─ VoiceEncoder.embeds_from_wavs([ref_16k]):
       │   → speaker_emb (1, 256) [L2-normalized]   [for T3 conditioning]
       │
       └─ Build T3Cond + Conditionals(t3_cond, s3gen_ref_dict)

    2. generate(text):
       │
       ├─ punc_norm(text) → cleaned text
       ├─ tokenizer.text_to_tokens(text) → (1, L_text)
       ├─ Duplicate for CFG: (2, L_text)
       ├─ Add start(255)/stop(0) tokens → (2, L_text+2)
       │
       ├─ T3.inference(t3_cond, text_tokens):
       │   → speech_tokens (1, N) — N speech tokens at 25 tok/sec
       │
       ├─ drop_invalid_tokens → remove SOS/EOS
       ├─ Filter: speech_tokens[speech_tokens < 6561]
       │
       ├─ S3Gen.inference(speech_tokens, ref_dict):
       │   → wav (1, num_samples) at 24kHz
       │
       └─ Perth watermark → final audio tensor

    Output: (1, num_samples) tensor at 24,000 Hz

Key constants:
    ENC_COND_LEN = 6 * 16000 = 96,000 samples  — T3 reference audio cap (6 seconds at 16kHz)
    DEC_COND_LEN = 10 * 24000 = 240,000 samples — S3Gen reference audio cap (10 seconds at 24kHz)

Default voice:
    When no audio_prompt_path is given, uses conds.pt (pre-computed Conditionals)
    from the HuggingFace repo. This is a fixed English female voice.
"""
from dataclasses import dataclass
from pathlib import Path

import librosa
import torch
import perth
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from .models.t3 import T3
from .models.s3tokenizer import S3_SR, drop_invalid_tokens
from .models.s3gen import S3GEN_SR, S3Gen
from .models.tokenizers import EnTokenizer
from .models.voice_encoder import VoiceEncoder
from .models.t3.modules.cond_enc import T3Cond


REPO_ID = "ResembleAI/chatterbox"


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
    sentence_enders = {".", "!", "?", "-", ","}
    if not any(text.endswith(p) for p in sentence_enders):
        text += "."

    return text


@dataclass
class Conditionals:
    """
    Conditionals for T3 and S3Gen
    - T3 conditionals:
        - speaker_emb
        - clap_emb
        - cond_prompt_speech_tokens
        - cond_prompt_speech_emb
        - emotion_adv
    - S3Gen conditionals:
        - prompt_token
        - prompt_token_len
        - prompt_feat
        - prompt_feat_len
        - embedding
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
        if isinstance(map_location, str):
            map_location = torch.device(map_location)
        kwargs = torch.load(fpath, map_location=map_location, weights_only=True)
        return cls(T3Cond(**kwargs['t3']), kwargs['gen'])


class ChatterboxTTS:
    ENC_COND_LEN = 6 * S3_SR
    DEC_COND_LEN = 10 * S3GEN_SR

    def __init__(
        self,
        t3: T3,
        s3gen: S3Gen,
        ve: VoiceEncoder,
        tokenizer: EnTokenizer,
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
    def from_local(cls, ckpt_dir, device) -> 'ChatterboxTTS':
        ckpt_dir = Path(ckpt_dir)

        # Always load to CPU first for non-CUDA devices to handle CUDA-saved models
        if device in ["cpu", "mps"]:
            map_location = torch.device('cpu')
        else:
            map_location = None

        ve = VoiceEncoder()
        ve.load_state_dict(
            load_file(ckpt_dir / "ve.safetensors")
        )
        ve.to(device).eval()

        t3 = T3()
        t3_state = load_file(ckpt_dir / "t3_cfg.safetensors")
        if "model" in t3_state.keys():
            t3_state = t3_state["model"][0]
        t3.load_state_dict(t3_state)
        t3.to(device).eval()

        s3gen = S3Gen()
        s3gen.load_state_dict(
            load_file(ckpt_dir / "s3gen.safetensors"), strict=False
        )
        s3gen.to(device).eval()

        tokenizer = EnTokenizer(
            str(ckpt_dir / "tokenizer.json")
        )

        conds = None
        if (builtin_voice := ckpt_dir / "conds.pt").exists():
            conds = Conditionals.load(builtin_voice, map_location=map_location).to(device)

        return cls(t3, s3gen, ve, tokenizer, device, conds=conds)

    @classmethod
    def from_pretrained(cls, device) -> 'ChatterboxTTS':
        # Check if MPS is available on macOS
        if device == "mps" and not torch.backends.mps.is_available():
            if not torch.backends.mps.is_built():
                print("MPS not available because the current PyTorch install was not built with MPS enabled.")
            else:
                print("MPS not available because the current MacOS version is not 12.3+ and/or you do not have an MPS-enabled device on this machine.")
            device = "cpu"

        for fpath in ["ve.safetensors", "t3_cfg.safetensors", "s3gen.safetensors", "tokenizer.json", "conds.pt"]:
            local_path = hf_hub_download(repo_id=REPO_ID, filename=fpath)

        return cls.from_local(Path(local_path).parent, device)

    def prepare_conditionals(self, wav_fpath, exaggeration=0.5):
        """Extract all conditioning from a reference audio file.

        This prepares both T3 conditioning (for text→token generation) and
        S3Gen conditioning (for token→waveform synthesis).

        Args:
            wav_fpath: str — path to reference WAV file (any sample rate, resampled internally)
            exaggeration: float — emotion exaggeration (0.0=neutral, 1.0=max). Default 0.5.

        Side effects:
            Sets self.conds = Conditionals(t3_cond, s3gen_ref_dict)

        T3 conditioning (stored in self.conds.t3):
            speaker_emb:                (1, 256) — VoiceEncoder L2-normed embedding from 16kHz ref
            cond_prompt_speech_tokens:  (1, 150) — S3Tokenizer tokens from first 6s of 16kHz ref
            emotion_adv:                (1, 1, 1) — exaggeration scalar

        S3Gen conditioning (stored in self.conds.gen):
            prompt_token:     (1, T_ref) — S3 tokens from 16kHz ref (up to 10s)
            prompt_token_len: (1,) — token count
            prompt_feat:      (1, T_mel, 80) — mel from 24kHz ref (up to 10s)
            embedding:        (1, 192) — CAMPPlus x-vector from 16kHz ref

        Internal flow:
            1. Load wav at 24kHz (for S3Gen mel conditioning)
            2. Resample to 16kHz (for S3Tokenizer + VoiceEncoder)
            3. Truncate 24kHz to 10s (DEC_COND_LEN) → S3Gen.embed_ref()
            4. Truncate 16kHz to 6s (ENC_COND_LEN) → S3Tokenizer(max_len=150)
            5. Full 16kHz → VoiceEncoder.embeds_from_wavs() → speaker_emb
        """
        ## Load reference wav at 24kHz (S3Gen's native rate)
        s3gen_ref_wav, _sr = librosa.load(wav_fpath, sr=S3GEN_SR)  # (num_samples,) at 24kHz

        # Resample to 16kHz for S3Tokenizer and VoiceEncoder
        ref_16k_wav = librosa.resample(s3gen_ref_wav, orig_sr=S3GEN_SR, target_sr=S3_SR)  # (num_samples_16k,)

        # S3Gen reference conditioning (up to 10s of 24kHz audio)
        s3gen_ref_wav = s3gen_ref_wav[:self.DEC_COND_LEN]  # truncate to 10s * 24kHz = 240,000 samples
        s3gen_ref_dict = self.s3gen.embed_ref(s3gen_ref_wav, S3GEN_SR, device=self.device)

        # T3 speech conditioning tokens (up to 6s of 16kHz audio → 150 tokens)
        if plen := self.t3.hp.speech_cond_prompt_len:  # plen = 150
            s3_tokzr = self.s3gen.tokenizer
            # ref_16k_wav[:ENC_COND_LEN] = first 6s at 16kHz = 96,000 samples
            # max_len=150 → truncate to 150 tokens (6s * 25 tok/s)
            t3_cond_prompt_tokens, _ = s3_tokzr.forward([ref_16k_wav[:self.ENC_COND_LEN]], max_len=plen)
            t3_cond_prompt_tokens = torch.atleast_2d(t3_cond_prompt_tokens).to(self.device)  # (1, 150)

        # VoiceEncoder speaker embedding from full 16kHz reference
        # embeds_from_wavs: internally computes mel → LSTM → L2-norm → (1, 256)
        ve_embed = torch.from_numpy(self.ve.embeds_from_wavs([ref_16k_wav], sample_rate=S3_SR))
        ve_embed = ve_embed.mean(axis=0, keepdim=True).to(self.device)  # (1, 256)

        t3_cond = T3Cond(
            speaker_emb=ve_embed,                           # (1, 256)
            cond_prompt_speech_tokens=t3_cond_prompt_tokens, # (1, 150)
            emotion_adv=exaggeration * torch.ones(1, 1, 1),  # (1, 1, 1)
        ).to(device=self.device)
        self.conds = Conditionals(t3_cond, s3gen_ref_dict)

    def generate(
        self,
        text,
        repetition_penalty=1.2,
        min_p=0.05,
        top_p=1.0,
        audio_prompt_path=None,
        exaggeration=0.5,
        cfg_weight=0.5,
        temperature=0.8,
    ):
        """Generate speech audio from text.

        Args:
            text: str — input text to synthesize
            audio_prompt_path: Optional str — reference audio for voice cloning.
                              If None, uses self.conds (built-in voice or previous prepare_conditionals).
            exaggeration: float — emotion exaggeration (0.0-1.0, default 0.5)
            cfg_weight: float — Classifier-Free Guidance weight (default 0.5)
                       Higher = more faithful to text, lower = more natural
            temperature: float — sampling temperature (default 0.8)
            repetition_penalty: float — penalty for repeated tokens (default 1.2)
            min_p: float — minimum probability filter (default 0.05)
            top_p: float — nucleus sampling (default 1.0 = disabled)

        Returns:
            wav: (1, num_samples) tensor — watermarked 24kHz audio waveform

        Pipeline:
            1. prepare_conditionals(audio_prompt_path) if provided
            2. punc_norm(text) → text cleanup
            3. tokenizer.text_to_tokens(text) → (1, L_text)
            4. Duplicate for CFG → (2, L_text)
            5. Add start(255)/stop(0) → (2, L_text+2)
            6. T3.inference(t3_cond, text_tokens) → speech_tokens (1, N)
            7. drop_invalid_tokens → remove SOS(6561)/EOS(6562)
            8. Filter tokens < 6561 (safety against out-of-vocab)
            9. S3Gen.inference(speech_tokens, ref_dict) → wav (1, samples)
            10. Perth watermark → final audio
        """
        if audio_prompt_path:
            self.prepare_conditionals(audio_prompt_path, exaggeration=exaggeration)
        else:
            assert self.conds is not None, "Please `prepare_conditionals` first or specify `audio_prompt_path`"

        # Update exaggeration if needed
        if exaggeration != self.conds.t3.emotion_adv[0, 0, 0]:
            _cond: T3Cond = self.conds.t3
            self.conds.t3 = T3Cond(
                speaker_emb=_cond.speaker_emb,
                cond_prompt_speech_tokens=_cond.cond_prompt_speech_tokens,
                emotion_adv=exaggeration * torch.ones(1, 1, 1),
            ).to(device=self.device)

        # Norm and tokenize text
        text = punc_norm(text)
        text_tokens = self.tokenizer.text_to_tokens(text).to(self.device)

        if cfg_weight > 0.0:
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
            
            speech_tokens = speech_tokens[speech_tokens < 6561]

            speech_tokens = speech_tokens.to(self.device)

            wav, _ = self.s3gen.inference(
                speech_tokens=speech_tokens,
                ref_dict=self.conds.gen,
            )
            wav = wav.squeeze(0).detach().cpu().numpy()
            watermarked_wav = self.watermarker.apply_watermark(wav, sample_rate=self.sr)
        return torch.from_numpy(watermarked_wav).unsqueeze(0)
