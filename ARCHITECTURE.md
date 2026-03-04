# Chatterbox TTS — Complete Architecture & Computational Graph

## Table of Contents

1. [System Overview](#1-system-overview)
2. [End-to-End Inference Flow Diagram](#2-end-to-end-inference-flow-diagram)
3. [Component 1: VoiceEncoder (Speaker Identity)](#3-component-1-voiceencoder)
4. [Component 2: S3Tokenizer (Speech-to-Tokens)](#4-component-2-s3tokenizer)
5. [Component 3: T3 (Text-to-Speech-Tokens)](#5-component-3-t3)
6. [Component 4: S3Gen (Speech-Tokens-to-Waveform)](#6-component-4-s3gen)
7. [Constants & Token ID Reference](#7-constants--token-id-reference)
8. [Dimension Reference Table](#8-dimension-reference-table)
9. [Training vs Inference Differences](#9-training-vs-inference-differences)

---

## 1. System Overview

Chatterbox is a two-stage TTS system with voice cloning:

```
                          ┌──────────────────────┐
    Reference Audio ──────┤  Conditioning Stage   ├──── T3Cond + S3Gen ref_dict
                          └──────────────────────┘
                                     │
    Text Input ──────┐               │
                     ▼               ▼
              ┌─────────────────────────────┐
              │     Stage 1: T3             │
              │  Text → Speech Tokens       │
              │  (Autoregressive LlamaModel)│
              └──────────┬──────────────────┘
                         │ speech_tokens (1, N) long
                         ▼
              ┌─────────────────────────────┐
              │     Stage 2: S3Gen          │
              │  Speech Tokens → Waveform   │
              │  (CFM Flow + HiFi-GAN)      │
              └──────────┬──────────────────┘
                         │ wav (1, N*240) float @ 24kHz
                         ▼
                   Output Audio
```

**4 Neural Network Components:**

| Component | Architecture | Input | Output | Params |
|-----------|-------------|-------|--------|--------|
| VoiceEncoder | 3-layer LSTM | 16kHz audio | (1, 256) speaker embedding | ~1M |
| S3Tokenizer | VQ Encoder (S3TokenizerV2) | 16kHz audio | (1, T) speech tokens | ~100M |
| T3 | LlamaModel (30 layers) | text tokens + conditioning | speech tokens | ~536M |
| S3Gen | CFM Flow + HiFi-GAN | speech tokens + ref conditioning | 24kHz waveform | ~300M |

---

## 2. End-to-End Inference Flow Diagram

```
REFERENCE AUDIO FILE (any sample rate, any length)
│
│  librosa.load(wav_fpath, sr=24000)
│  ─────────────────────────────────
▼
s3gen_ref_wav: numpy (T_24k,)    ←── resampled to 24kHz
│
├────────────────────────────── 24kHz PATH ──────────────────────────────────┐
│                                                                            │
│  s3gen_ref_wav[:240000]  ←── truncate to 10 seconds max                   │
│         │                                                                  │
│         ├──── mel_spectrogram(24kHz, n_fft=1024, hop=256, 80 bands)       │
│         │        └──► ref_mel: (1, T_mel_24, 80) float                    │
│         │             T_mel_24 ≈ T_24k / 256 ≈ duration_sec * 93.75      │
│         │                                                                  │
│         ├──── resample 24kHz → 16kHz                                      │
│         │        └──► ref_wav_16k: (1, T_16k) float                      │
│         │                                                                  │
│         ├──── CAMPPlus.inference(ref_wav_16k)                             │
│         │        └──► x_vector: (1, 192) float   ── "embedding"           │
│         │                                                                  │
│         └──── S3Tokenizer(ref_wav_16k)                                    │
│                  STFT(n_fft=400, hop=160) → mel(128 bands) → VQ          │
│                  └──► ref_speech_tokens: (1, T_ref) long                  │
│                       ref_speech_token_lens: (1,) long                    │
│                       T_ref ≈ duration_sec * 25                           │
│                                                                            │
│  ENFORCE: ref_mel.shape[1] == 2 * ref_speech_tokens.shape[1]             │
│                                                                            │
│  ┌─────────────────────────────────────────────────────────────┐          │
│  │ s3gen_ref_dict = {                                          │          │
│  │     "prompt_token":     (1, T_ref) long,                    │          │
│  │     "prompt_token_len": (1,) long,                          │          │
│  │     "prompt_feat":      (1, T_mel_24, 80) float,            │          │
│  │     "prompt_feat_len":  None,                                │          │
│  │     "embedding":        (1, 192) float                      │          │
│  │ }                                                            │          │
│  └─────────────────────────────────────────────────────────────┘          │
│                                                                            │
├────────────────────────────── 16kHz PATH ──────────────────────────────────┤
│                                                                            │
│  librosa.resample(s3gen_ref_wav, 24000 → 16000)                           │
│         └──► ref_16k_wav: numpy (T_16k,)                                  │
│                                                                            │
│  ┌─── VoiceEncoder ───────────────────────────────────────────────┐       │
│  │                                                                 │       │
│  │  ref_16k_wav: numpy (T_16k,)                                   │       │
│  │       │                                                         │       │
│  │       ├── librosa.effects.trim(top_db=20) → trimmed wav        │       │
│  │       ├── melspectrogram(sr=16k, 40 bands) → (T_mel, 40)      │       │
│  │       ├── stride_as_partials(overlap=0.5, rate=1.3)            │       │
│  │       │      └──► partials: (N_partials, P, 40)                │       │
│  │       │                                                         │       │
│  │       ├── LSTM(input=40, hidden=256, layers=3, batch_first)    │       │
│  │       │      input:  (N_partials, P, 40)                       │       │
│  │       │      output: _, (hidden, _)                             │       │
│  │       │              hidden: (3, N_partials, 256)               │       │
│  │       │              hidden[-1]: (N_partials, 256)  ← last layer│      │
│  │       │                                                         │       │
│  │       ├── Linear(256 → 256)                                    │       │
│  │       │      └──► raw_embeds: (N_partials, 256)                │       │
│  │       │                                                         │       │
│  │       ├── L2 normalize: raw / ||raw||₂                          │       │
│  │       │      └──► partial_embeds: (N_partials, 256)            │       │
│  │       │                                                         │       │
│  │       ├── mean(dim=0) → (1, 256)                               │       │
│  │       └── L2 normalize → speaker_emb: (1, 256) float          │       │
│  │                                                                 │       │
│  └─────────────────────────────────────────────────────────────────┘       │
│                                                                            │
│  ┌─── S3Tokenizer (for T3 conditioning) ──────────────────────────┐       │
│  │                                                                 │       │
│  │  ref_16k_wav[:96000]  ←── truncate to 6 seconds (ENC_COND_LEN)│       │
│  │       │                                                         │       │
│  │  S3Tokenizer.forward([ref_16k], max_len=150):                  │       │
│  │       │                                                         │       │
│  │       ├── log_mel_spectrogram(ref_16k):                        │       │
│  │       │      STFT(n_fft=400, hop=160, Hann window)             │       │
│  │       │        input:  (1, 96000)                               │       │
│  │       │        output: complex (1, 201, 600)                    │       │
│  │       │      stft[..., :-1].abs()² → magnitudes: (1, 200, 599)│       │
│  │       │      mel_filters(128×201) @ magnitudes → (1, 128, 599)│       │
│  │       │      log10, clamp, max-normalize, scale                │       │
│  │       │        └──► mel: (1, 128, 600) float                   │       │
│  │       │                                                         │       │
│  │       ├── mel[..., :150*4] = mel[..., :600]  ←── max_len trim │       │
│  │       │      └──► mel: (1, 128, 600)                           │       │
│  │       │                                                         │       │
│  │       ├── padding(mels) → batched: (1, 128, 600)              │       │
│  │       │                                                         │       │
│  │       └── S3TokenizerV2.quantize(mel, mel_lens)                │       │
│  │              VQ codebook lookup (4 mel frames → 1 token)       │       │
│  │              └──► cond_prompt_speech_tokens: (1, 150) long     │       │
│  │                   token values ∈ [0, 6560]                      │       │
│  │                                                                 │       │
│  └─────────────────────────────────────────────────────────────────┘       │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘

ASSEMBLE CONDITIONING:
┌──────────────────────────────────────────────────────┐
│ t3_cond = T3Cond(                                    │
│     speaker_emb              = (1, 256)   float,     │
│     cond_prompt_speech_tokens = (1, 150)  long,      │
│     emotion_adv              = (1, 1, 1)  float,     │
│     # Fields set to None (computed later):           │
│     clap_emb                 = None,                 │
│     cond_prompt_speech_emb   = None,                 │
│ )                                                    │
│                                                      │
│ conds = Conditionals(t3=t3_cond, gen=s3gen_ref_dict) │
└──────────────────────────────────────────────────────┘


TEXT INPUT
│
│  text: str  (e.g., "Hello, how are you?")
│
├── punc_norm(text) → cleaned text (capitalize, normalize punctuation, add period)
│
├── EnTokenizer.text_to_tokens(text)   [English, vocab=704]
│   OR MTLTokenizer.text_to_tokens(text, language_id)  [Multilingual, vocab=2454]
│      └──► text_tokens: (1, L_text) long
│
├── torch.cat([text_tokens, text_tokens], dim=0)       ←── CFG duplication
│      └──► text_tokens: (2, L_text)   [batch 0 = conditional, batch 1 = unconditional]
│
├── F.pad(text_tokens, (1, 0), value=255)              ←── prepend SOT token
│      └──► text_tokens: (2, L_text + 1)
│
└── F.pad(text_tokens, (0, 1), value=0)                ←── append EOT token
       └──► text_tokens: (2, L_text + 2)


═══════════════════════════════════════════════════════════════════════════════
 STAGE 1: T3 AUTOREGRESSIVE GENERATION
═══════════════════════════════════════════════════════════════════════════════

T3.inference(t3_cond, text_tokens=(2, L+2), cfg_weight=0.5, temperature=0.8)
│
│ ┌─── T3.prepare_conditioning(t3_cond) ─────────────────────────────────┐
│ │                                                                       │
│ │  t3_cond.cond_prompt_speech_tokens: (1, 150) long                    │
│ │       │                                                               │
│ │       ├── T3.speech_emb: Embedding(8194, 1024)                       │
│ │       │      (1, 150) → lookup → (1, 150, 1024)                     │
│ │       │                                                               │
│ │       ├── T3.speech_pos_emb: LearnedPositionEmbeddings(4100, 1024)   │
│ │       │      arange(0, 150) → Embedding → (150, 1024)               │
│ │       │      broadcast → (1, 150, 1024)                              │
│ │       │                                                               │
│ │       └── sum: speech_emb + speech_pos_emb                           │
│ │              └──► cond_prompt_speech_emb: (1, 150, 1024)             │
│ │              stored in t3_cond.cond_prompt_speech_emb                 │
│ │                                                                       │
│ │  T3CondEnc.forward(t3_cond):                                         │
│ │       │                                                               │
│ │       ├── Speaker projection:                                        │
│ │       │      spkr_enc: Linear(256 → 1024)                           │
│ │       │      speaker_emb.view(-1, 256): (1, 256)                     │
│ │       │      → (1, 1024) → [:, None] → (1, 1, 1024)                │
│ │       │      └──► cond_spkr: (1, 1, 1024)                           │
│ │       │                                                               │
│ │       ├── CLAP (unused):                                             │
│ │       │      └──► cond_clap: (1, 0, 1024)  ← zero-length tensor    │
│ │       │                                                               │
│ │       ├── Perceiver Resampler:                                       │
│ │       │      ┌─────────────────────────────────────────────────┐     │
│ │       │      │ Input: cond_prompt_speech_emb (1, 150, 1024)   │     │
│ │       │      │                                                 │     │
│ │       │      │ Step 1 — Cross-Attention:                       │     │
│ │       │      │   Q = learned queries: (1, 32, 1024)            │     │
│ │       │      │   K = LayerNorm → Linear(1024→1024): from input │     │
│ │       │      │   V = LayerNorm → Linear(1024→1024): from input │     │
│ │       │      │   Q = LayerNorm → Linear(1024→1024): from query │     │
│ │       │      │   Split heads: 4 heads × 256 dim                │     │
│ │       │      │   Attention(Q:(1,4,32,256), K:(1,4,150,256))   │     │
│ │       │      │   → (1, 4, 32, 256) → combine → (1, 32, 1024) │     │
│ │       │      │   proj_out: Linear(1024→1024) → (1, 32, 1024)  │     │
│ │       │      │   + residual (from queries)                     │     │
│ │       │      │   └──► pre_att: (1, 32, 1024)                  │     │
│ │       │      │                                                 │     │
│ │       │      │ Step 2 — Self-Attention:                        │     │
│ │       │      │   Q = K = V = pre_att: (1, 32, 1024)           │     │
│ │       │      │   Same AttentionBlock2 (shared weights)         │     │
│ │       │      │   Attention(Q:(1,4,32,256), K:(1,4,32,256))   │     │
│ │       │      │   → (1, 32, 1024) + residual                   │     │
│ │       │      │   └──► output: (1, 32, 1024)                   │     │
│ │       │      └─────────────────────────────────────────────────┘     │
│ │       │      └──► cond_speech: (1, 32, 1024)                        │
│ │       │                                                               │
│ │       ├── Emotion projection:                                        │
│ │       │      emotion_adv_fc: Linear(1 → 1024, no bias)              │
│ │       │      emotion_adv.view(-1, 1, 1): (1, 1, 1)                  │
│ │       │      → (1, 1, 1024)                                         │
│ │       │      └──► cond_emotion: (1, 1, 1024)                        │
│ │       │                                                               │
│ │       └── Concatenate along sequence dim:                            │
│ │              cat([spkr(1), clap(0), speech(32), emotion(1)], dim=1) │
│ │              └──► cond_emb: (1, 34, 1024)                           │
│ │                                                                       │
│ └───────────────────────────────────────────────────────────────────────┘
│       └──► cond_emb: (1, 34, 1024)
│
│ ┌─── T3.prepare_input_embeds ───────────────────────────────────────────┐
│ │                                                                       │
│ │  cond_emb: (1, 34, 1024)                                             │
│ │       expand → (2, 34, 1024)  ←── repeat for CFG batch               │
│ │                                                                       │
│ │  text_tokens: (2, L+2) long                                          │
│ │       ├── text_emb: Embedding(704, 1024)  [or 2454 for MTL]         │
│ │       │      (2, L+2) → (2, L+2, 1024)                              │
│ │       ├── text_pos_emb: LearnedPositionEmbeddings(2050, 1024)        │
│ │       │      arange(0, L+2) → Embedding → (L+2, 1024)               │
│ │       │      broadcast → (2, L+2, 1024)                              │
│ │       ├── sum: text_emb + text_pos_emb → (2, L+2, 1024)             │
│ │       └── text_emb[1].zero_()  ←── zero out unconditional batch      │
│ │              └──► text_embeds: (2, L+2, 1024)                        │
│ │                                                                       │
│ │  BOS token: [[6561], [6561]]  (2, 1) long                           │
│ │       ├── speech_emb: Embedding(8194, 1024)                          │
│ │       │      (2, 1) → (2, 1, 1024)                                  │
│ │       ├── speech_pos_emb.get_fixed_embedding(0)                      │
│ │       │      → (1, 1, 1024) → expand → (2, 1, 1024)                │
│ │       └── sum → bos_embed: (2, 1, 1024)                             │
│ │                                                                       │
│ │  cat([cond_emb, text_embeds, bos_embed], dim=1)                      │
│ │       └──► input_embeds: (2, 34 + L+2 + 1, 1024)                    │
│ │                        = (2, L+37, 1024)                              │
│ │                                                                       │
│ └───────────────────────────────────────────────────────────────────────┘
│
│ ┌─── LlamaModel Forward (first pass) ──────────────────────────────────┐
│ │                                                                       │
│ │  LlamaModel config (Llama_520M):                                     │
│ │       hidden_size          = 1024                                     │
│ │       num_hidden_layers    = 30                                       │
│ │       num_attention_heads  = 16                                       │
│ │       head_dim             = 64                                       │
│ │       intermediate_size    = 4096                                     │
│ │       hidden_act           = "silu"                                   │
│ │       attention_impl       = "sdpa"                                   │
│ │       RoPE: llama3, theta=500000, factor=8.0                         │
│ │                                                                       │
│ │  input_embeds: (2, L+37, 1024)                                       │
│ │       │                                                               │
│ │       ├── 30 × LlamaDecoderLayer:                                    │
│ │       │      ┌── RMSNorm(1024, eps=1e-5) ──────────────────────┐    │
│ │       │      │   Self-Attention (causal):                       │    │
│ │       │      │     Q: Linear(1024 → 1024) → (2, L+37, 16, 64)│    │
│ │       │      │     K: Linear(1024 → 1024) → (2, L+37, 16, 64)│    │
│ │       │      │     V: Linear(1024 → 1024) → (2, L+37, 16, 64)│    │
│ │       │      │     + RoPE positional encoding                  │    │
│ │       │      │     SDPA attention → (2, L+37, 1024)            │    │
│ │       │      │     O: Linear(1024 → 1024)                     │    │
│ │       │      │   + residual                                    │    │
│ │       │      │                                                  │    │
│ │       │      │   RMSNorm(1024, eps=1e-5)                       │    │
│ │       │      │   MLP:                                           │    │
│ │       │      │     gate_proj: Linear(1024 → 4096)              │    │
│ │       │      │     up_proj:   Linear(1024 → 4096)              │    │
│ │       │      │     silu(gate) * up → (2, L+37, 4096)           │    │
│ │       │      │     down_proj: Linear(4096 → 1024)              │    │
│ │       │      │   + residual → (2, L+37, 1024)                  │    │
│ │       │      └──────────────────────────────────────────────────┘    │
│ │       │                                                               │
│ │       ├── RMSNorm(1024) → hidden_states: (2, L+37, 1024)            │
│ │       └── past_key_values: KV cache for all 30 layers               │
│ │                                                                       │
│ │  speech_head: Linear(1024 → 8194, no bias)                          │
│ │       hidden_states[:, -1, :] → logits_step: (2, 8194)              │
│ │                                                                       │
│ └───────────────────────────────────────────────────────────────────────┘
│
│ ┌─── Autoregressive Generation Loop (up to max_new_tokens=1000) ───────┐
│ │                                                                       │
│ │  FOR i = 0, 1, 2, ... until EOS or max_new_tokens:                   │
│ │       │                                                               │
│ │       ├── Classifier-Free Guidance (CFG):                            │
│ │       │      cond_logits   = logits_step[0:1, :]: (1, 8194)         │
│ │       │      uncond_logits = logits_step[1:2, :]: (1, 8194)         │
│ │       │      logits = cond + cfg_weight * (cond - uncond)            │
│ │       │      └──► guided_logits: (1, 8194)                          │
│ │       │                                                               │
│ │       ├── Repetition Penalty (default 1.2):                          │
│ │       │      For each previously generated token t:                  │
│ │       │        logits[t] /= penalty (if positive)                    │
│ │       │        logits[t] *= penalty (if negative)                    │
│ │       │                                                               │
│ │       ├── Temperature scaling:                                       │
│ │       │      logits /= temperature (default 0.8)                     │
│ │       │                                                               │
│ │       ├── min_p filtering (default 0.05):                            │
│ │       │      probs < 0.05 * max(probs) → set to -inf                │
│ │       │                                                               │
│ │       ├── top_p nucleus sampling (default 1.0 = disabled):           │
│ │       │      cumulative probs > top_p → set to -inf                  │
│ │       │                                                               │
│ │       ├── softmax(logits) → probs: (1, 8194)                        │
│ │       ├── multinomial(probs, 1) → next_token: (1, 1) long           │
│ │       │                                                               │
│ │       ├── STOP if next_token == 6562 (EOS)                           │
│ │       │                                                               │
│ │       ├── Prepare next input embedding:                              │
│ │       │      speech_emb(next_token): (1, 1, 1024)                   │
│ │       │      + speech_pos_emb.get_fixed_embedding(i+1): (1, 1, 1024)│
│ │       │      → next_embed: (1, 1, 1024)                             │
│ │       │      cat([next_embed, next_embed]) → (2, 1, 1024)  [CFG]   │
│ │       │                                                               │
│ │       └── LlamaModel.forward(                                       │
│ │              inputs_embeds=(2, 1, 1024),                              │
│ │              past_key_values=cached_KV,                               │
│ │              use_cache=True                                           │
│ │           )                                                           │
│ │           → logits_step: (2, 1, 8194) → squeeze → (2, 8194)        │
│ │           → updated past_key_values                                   │
│ │                                                                       │
│ │  cat(all predicted tokens, dim=1)                                    │
│ │       └──► predicted_tokens: (1, N) long                             │
│ │                                                                       │
│ └───────────────────────────────────────────────────────────────────────┘
│
│ Post-processing:
│       speech_tokens = predicted_tokens[0]        → (N,) long
│       drop_invalid_tokens(speech_tokens):
│           remove SOS (6561) and EOS (6562)
│           keep only tokens in [0, 6560]
│       speech_tokens.unsqueeze(0)                 → (1, N_clean) long
│
▼

═══════════════════════════════════════════════════════════════════════════════
 STAGE 2: S3Gen WAVEFORM SYNTHESIS
═══════════════════════════════════════════════════════════════════════════════

S3Gen.inference(speech_tokens=(1, N), ref_dict=s3gen_ref_dict)
│
│ ┌─── flow_inference ────────────────────────────────────────────────────┐
│ │                                                                       │
│ │  n_cfm_timesteps = 10 (default) or 2 (meanflow)                     │
│ │  if meanflow: noise = randn(1, 80, N*2) → noised_mels              │
│ │                                                                       │
│ │  CausalMaskedDiffWithXvec.inference(                                 │
│ │       token           = speech_tokens: (1, N) long,                  │
│ │       token_len       = (1,) long,                                   │
│ │       prompt_token    = ref_dict["prompt_token"]: (1, T_ref) long,  │
│ │       prompt_token_len= ref_dict["prompt_token_len"]: (1,) long,    │
│ │       prompt_feat     = ref_dict["prompt_feat"]: (1, T_mel, 80),    │
│ │       embedding       = ref_dict["embedding"]: (1, 192),             │
│ │       finalize        = True,                                        │
│ │       n_timesteps     = 10,                                          │
│ │  ):                                                                   │
│ │       │                                                               │
│ │       ├── Speaker embedding projection:                              │
│ │       │      F.normalize(embedding, dim=1): (1, 192)                │
│ │       │      spk_embed_affine_layer: Linear(192 → 80)               │
│ │       │      └──► spk_emb: (1, 80)                                  │
│ │       │                                                               │
│ │       ├── Concatenate prompt + target tokens:                        │
│ │       │      cat([prompt_token, token], dim=1)                       │
│ │       │      └──► all_tokens: (1, T_ref + N) long                   │
│ │       │      token_len = prompt_token_len + token_len                │
│ │       │                                                               │
│ │       ├── Padding mask:                                              │
│ │       │      ~make_pad_mask(token_len): (1, T_ref+N, 1) float      │
│ │       │                                                               │
│ │       ├── Token embedding:                                           │
│ │       │      input_embedding: Embedding(6561, 512)                   │
│ │       │      ⚠ TOKENS MUST BE < 6561 — larger values cause CUDA     │
│ │       │        device-side assert crash                               │
│ │       │      all_tokens.long() → (1, T_ref+N, 512)                  │
│ │       │      * mask → (1, T_ref+N, 512)                              │
│ │       │                                                               │
│ │       ├── UpsampleConformerEncoder:                                  │
│ │       │      ┌──────────────────────────────────────────────────┐    │
│ │       │      │ Config:                                          │    │
│ │       │      │   input_size     = 512                           │    │
│ │       │      │   output_size    = 512                           │    │
│ │       │      │   attention_heads = 8                            │    │
│ │       │      │   linear_units   = 2048                          │    │
│ │       │      │   num_blocks     = 6                             │    │
│ │       │      │   dropout        = 0.1                           │    │
│ │       │      │   pos_enc        = rel_pos_espnet                │    │
│ │       │      │   upsample factor = token_mel_ratio = 2          │    │
│ │       │      │                                                  │    │
│ │       │      │ Input:  (1, T_ref+N, 512)                       │    │
│ │       │      │   Linear(512→512) input layer                   │    │
│ │       │      │   6 × ConformerBlock:                            │    │
│ │       │      │     LayerNorm → MultiHeadAttn(8 heads, 64 dim)  │    │
│ │       │      │     + FeedForward(512→2048→512)                 │    │
│ │       │      │   2x upsample (repeat interleave)               │    │
│ │       │      │ Output: (1, 2*(T_ref+N), 512)                   │    │
│ │       │      │ Masks:  (1, 1, 2*(T_ref+N))                    │    │
│ │       │      └──────────────────────────────────────────────────┘    │
│ │       │                                                               │
│ │       ├── Encoder projection:                                        │
│ │       │      encoder_proj: Linear(512 → 80)                         │
│ │       │      (1, 2*(T_ref+N), 512) → (1, 2*(T_ref+N), 80)         │
│ │       │      └──► mu (after transpose): (1, 80, 2*(T_ref+N))       │
│ │       │                                                               │
│ │       ├── Build conditioning tensor:                                 │
│ │       │      conds = zeros(1, 80, 2*(T_ref+N))                     │
│ │       │      conds[:, :, :T_mel] = prompt_feat.T                    │
│ │       │      └──► conds: (1, 80, 2*(T_ref+N))                      │
│ │       │                                                               │
│ │       ├── CausalConditionalCFM (Flow Matching Decoder):             │
│ │       │      ┌──────────────────────────────────────────────────┐    │
│ │       │      │                                                  │    │
│ │       │      │ ConditionalDecoder config:                       │    │
│ │       │      │   in_channels  = 320 (= 80*3 + 80 spk)         │    │
│ │       │      │   out_channels = 80                              │    │
│ │       │      │   channels     = [256]                           │    │
│ │       │      │   n_blocks     = 4                               │    │
│ │       │      │   num_mid_blocks = 12                            │    │
│ │       │      │   num_heads    = 8                               │    │
│ │       │      │   head_dim     = 64                              │    │
│ │       │      │   act_fn       = "gelu"                          │    │
│ │       │      │   causal       = True                            │    │
│ │       │      │                                                  │    │
│ │       │      │ ODE solver (n_timesteps=10 Euler steps):         │    │
│ │       │      │   t: [0.0, 0.1, 0.2, ..., 1.0]                 │    │
│ │       │      │   For each timestep:                             │    │
│ │       │      │     x_t = (1-t)*noise + t*mu                    │    │
│ │       │      │     cat([x_t, mu, mu-x_t, spk_emb]) → (1,320,T)│    │
│ │       │      │     ConditionalDecoder forward:                  │    │
│ │       │      │       4 down blocks (Conv1D + Attn)             │    │
│ │       │      │       12 mid blocks (Conv1D + Attn)             │    │
│ │       │      │       4 up blocks (Conv1D + Attn)               │    │
│ │       │      │     → velocity: (1, 80, T)                      │    │
│ │       │      │     x_{t+dt} = x_t + dt * velocity              │    │
│ │       │      │                                                  │    │
│ │       │      │ Output: (1, 80, 2*(T_ref+N))                    │    │
│ │       │      └──────────────────────────────────────────────────┘    │
│ │       │                                                               │
│ │       └── Slice target portion:                                      │
│ │              feat[:, :, T_mel:]                                      │
│ │              └──► output_mels: (1, 80, N*2) float                   │
│ │                   N tokens × 2 mel frames/token = N*2 mel frames    │
│ │                                                                       │
│ └───────────────────────────────────────────────────────────────────────┘
│
│ ┌─── hift_inference (HiFi-GAN Vocoder) ────────────────────────────────┐
│ │                                                                       │
│ │  HiFTGenerator config:                                                │
│ │       sampling_rate       = 24000                                     │
│ │       upsample_rates      = [8, 5, 3]    → total = 8×5×3 = 120x     │
│ │       upsample_kernel_sizes = [16, 11, 7]                            │
│ │       f0_predictor        = ConvRNNF0Predictor                       │
│ │                                                                       │
│ │  Input: output_mels (1, 80, N*2)                                     │
│ │       │                                                               │
│ │       ├── F0 prediction (ConvRNNF0Predictor):                        │
│ │       │      output_mels → conv layers + GRU → f0: (1, 1, N*2)     │
│ │       │                                                               │
│ │       ├── Source excitation from f0:                                  │
│ │       │      harmonic oscillator → source signal: (1, 1, N*2*120)   │
│ │       │                                                               │
│ │       ├── Upsampling chain:                                          │
│ │       │      ConvTranspose1d(×8):  (1, C, N*2)    → (1, C, N*16)   │
│ │       │      + residual blocks                                       │
│ │       │      ConvTranspose1d(×5):  (1, C, N*16)   → (1, C, N*80)   │
│ │       │      + residual blocks                                       │
│ │       │      ConvTranspose1d(×3):  (1, C, N*80)   → (1, C, N*240)  │
│ │       │      + residual blocks                                       │
│ │       │      Final conv → (1, 1, N*240)                              │
│ │       │                                                               │
│ │       └── output:                                                    │
│ │              output_wavs:    (1, N*240) float @ 24kHz                │
│ │              output_sources: (1, 1, N*240) float (source signal)     │
│ │                                                                       │
│ │  N tokens → N*240 audio samples → N*10ms audio                      │
│ │  Example: 100 tokens = 4 seconds → 96,000 samples @ 24kHz           │
│ │                                                                       │
│ └───────────────────────────────────────────────────────────────────────┘
│
│ Post-processing:
│       output_wavs[:, :480] *= trim_fade    ←── cosine fade-in (20ms)
│       trim_fade: (480,) = cos(π..0) mapped to 0..1
│
▼

FINAL OUTPUT
│
│  wav = output_wavs.squeeze(0).detach().cpu().numpy()
│       → numpy (N*240,)
│
│  perth.PerthImplicitWatermarker.apply_watermark(wav, sample_rate=24000)
│       → watermarked_wav: numpy (N*240,)
│
│  torch.from_numpy(watermarked_wav).unsqueeze(0)
│       └──► FINAL: (1, N*240) float tensor @ 24kHz
│
│  Duration: N tokens × 40ms/token = N/25 seconds
│  Example: "Hello world" → ~50 tokens → 2 seconds → 48,000 samples
```

---

## 3. Component 1: VoiceEncoder

**Purpose:** Extract L2-normalized 256-dim speaker identity embedding from reference audio.

```
FILE: src/chatterbox/models/voice_encoder/voice_encoder.py

┌─────────────────────────────────────────────────────────────────────────┐
│ VoiceEncoder(nn.Module)                                                 │
│                                                                         │
│ Layers:                                                                 │
│   lstm: LSTM(input_size=40, hidden_size=256, num_layers=3, batch_first)│
│   proj: Linear(256 → 256)                                              │
│   similarity_weight: Parameter(scalar, init=10.0)                      │
│   similarity_bias:   Parameter(scalar, init=-5.0)                      │
│                                                                         │
│ Mel config (internal, NOT S3Tokenizer's mel):                          │
│   sample_rate = 16000                                                   │
│   n_mels      = 40       ← different from S3Tokenizer's 128!          │
│   hop/win     = from VoiceEncConfig                                    │
│                                                                         │
│ embeds_from_wavs(wavs: List[ndarray], sample_rate=16000):              │
│   INPUT:  List of numpy arrays, each (num_samples,) at any SR         │
│                                                                         │
│   For each wav:                                                         │
│     if sr != 16000: librosa.resample → 16kHz                           │
│     librosa.effects.trim(top_db=20) → trimmed wav                     │
│     melspectrogram(wav, hp) → (T_i, 40) numpy                         │
│                                                                         │
│   stride_as_partials(overlap=0.5, rate=1.3):                           │
│     sliding windows → (N_partials, P, 40) per utterance               │
│                                                                         │
│   forward(partials_batch):                                              │
│     (N_partials, P, 40)                                                │
│       → LSTM → hidden: (3, N_partials, 256)                           │
│       → hidden[-1]: (N_partials, 256)                                  │
│       → proj: Linear(256→256) → (N_partials, 256)                     │
│       → ReLU (optional)                                                │
│       → L2 normalize → (N_partials, 256)                               │
│                                                                         │
│   mean(partial_embeds, dim=0) → (1, 256)                               │
│   L2 normalize → speaker_emb: (1, 256)                                 │
│                                                                         │
│ OUTPUT: numpy (1, 256) — L2-normalized, values in [-1, 1]             │
│         Used as T3Cond.speaker_emb                                     │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Component 2: S3Tokenizer

**Purpose:** Convert 16kHz audio to discrete speech tokens at 25 tokens/sec.

```
FILE: src/chatterbox/models/s3tokenizer/s3tokenizer.py

┌─────────────────────────────────────────────────────────────────────────┐
│ S3Tokenizer(S3TokenizerV2)  — inherits VQ quantizer from s3tokenizer  │
│                                                                         │
│ Constants:                                                              │
│   S3_SR          = 16,000 Hz    ⚠ REQUIRED input sample rate           │
│   S3_HOP         = 160          STFT hop (10ms → 100 mel frames/sec)   │
│   S3_TOKEN_HOP   = 640          token hop (40ms → 25 tokens/sec)       │
│   S3_TOKEN_RATE  = 25           output token rate                      │
│   SPEECH_VOCAB_SIZE = 6561      valid token IDs: [0, 6560]             │
│                                                                         │
│ Registered buffers:                                                     │
│   _mel_filters: (128, 201)      librosa mel filterbank                 │
│   window:       (400,)          Hann window for STFT                   │
│                                                                         │
│ forward(wavs, max_len=None):                                           │
│   INPUT:  wavs — Tensor(1, T) or List[ndarray(T,)]                    │
│           ⚠ MUST be 16kHz. 24kHz input → garbage tokens.              │
│                                                                         │
│   Per-wav pipeline:                                                     │
│     wav: (1, T) float @ 16kHz                                          │
│       │                                                                 │
│       ├── STFT(n_fft=400, hop=160, Hann window)                        │
│       │     → complex: (1, 201, n_frames+1)                            │
│       │     n_frames = T / 160                                         │
│       │                                                                 │
│       ├── stft[..., :-1].abs()²                                        │
│       │     → magnitudes: (1, 200, n_frames)                           │
│       │                                                                 │
│       ├── _mel_filters @ magnitudes                                     │
│       │     (128, 201) @ (1, 200, n_frames) → (1, 128, n_frames)      │
│       │                                                                 │
│       ├── Post-process:                                                 │
│       │     clamp(min=1e-10).log10()                                   │
│       │     max(log_spec, log_spec.max() - 8.0)                        │
│       │     (log_spec + 4.0) / 4.0                                     │
│       │     → mel: (1, 128, n_frames)                                  │
│       │                                                                 │
│       ├── if max_len: mel = mel[..., :max_len*4]                       │
│       │     max_len=150 → keep first 600 mel frames (6s audio)         │
│       │                                                                 │
│       └── squeeze(0) → mel: (128, n_frames)                            │
│                                                                         │
│   padding(mels_list) → batched_mels: (B, 128, T_max)                  │
│                         mel_lens: (B,) long                            │
│                                                                         │
│   S3TokenizerV2.quantize(batched_mels, mel_lens):                      │
│     VQ codebook lookup                                                  │
│     4 mel frames → 1 token (100 frames/sec ÷ 4 = 25 tokens/sec)       │
│     → speech_tokens: (B, T_tok) long, T_tok = ceil(n_frames / 4)      │
│     → speech_token_lens: (B,) long                                    │
│                                                                         │
│ OUTPUT:                                                                 │
│   speech_tokens:     (B, T) long — token IDs in [0, 6560]             │
│   speech_token_lens: (B,) long — actual count per sample               │
│                                                                         │
│ Duration examples:                                                      │
│   1s audio  (16000 samples)  → 100 mel frames → 25 tokens             │
│   6s audio  (96000 samples)  → 600 mel frames → 150 tokens            │
│   10s audio (160000 samples) → 1000 mel frames → 250 tokens           │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 5. Component 3: T3

**Purpose:** Autoregressive generation of speech tokens from text + conditioning.

```
FILE: src/chatterbox/models/t3/t3.py

┌─────────────────────────────────────────────────────────────────────────┐
│ T3(nn.Module) — ~536M parameters                                       │
│                                                                         │
│ Submodules & dimensions:                                                │
│   tfmr:          LlamaModel (Llama_520M)                               │
│                    30 layers, 1024 hidden, 16 heads, 64 head_dim       │
│                    4096 intermediate, SiLU, SDPA, RoPE (llama3)        │
│                                                                         │
│   cond_enc:      T3CondEnc                                             │
│                    spkr_enc: Linear(256 → 1024)                        │
│                    emotion_adv_fc: Linear(1 → 1024, no bias)           │
│                    perceiver: Perceiver(32 queries, 1024 dim, 4 heads) │
│                                                                         │
│   text_emb:      Embedding(704, 1024)  [EN]                           │
│                  Embedding(2454, 1024) [MTL]                           │
│   speech_emb:    Embedding(8194, 1024)                                 │
│   text_pos_emb:  LearnedPositionEmbeddings(2050, 1024)                 │
│                    emb: Embedding(2050, 1024)                          │
│   speech_pos_emb:LearnedPositionEmbeddings(4100, 1024)                 │
│                    emb: Embedding(4100, 1024)                          │
│   text_head:     Linear(1024 → 704, no bias) [EN]                     │
│                  Linear(1024 → 2454, no bias) [MTL]                    │
│   speech_head:   Linear(1024 → 8194, no bias)                         │
│                                                                         │
│ ─────────────────────────────────────────────────────────────────────── │
│                                                                         │
│ Transformer input sequence layout (inference):                          │
│                                                                         │
│ ┌────────────┬─────────────────┬──────────────────────┐                │
│ │ Cond (34)  │ Text (L_text+2) │ Speech (1..N)        │                │
│ │            │                 │ (generated one by one)│                │
│ ├────────────┼─────────────────┼──────────────────────┤                │
│ │ spkr  (1)  │ SOT=255  (1)   │ SOS=6561 (1) ← start│                │
│ │ clap  (0)  │ text...  (L)   │ tok_0               │                │
│ │ perc (32)  │ EOT=0    (1)   │ tok_1               │                │
│ │ emot  (1)  │                 │ ...                  │                │
│ │            │                 │ tok_N               │                │
│ │            │                 │ EOS=6562 ← stop     │                │
│ └────────────┴─────────────────┴──────────────────────┘                │
│  positions:  0..33    34..34+L+1    34+L+2..                           │
│                                                                         │
│ Each position: (B, 1, 1024) = content_emb + position_emb              │
│ Conditioning tokens have NO position embeddings (implicit from concat) │
│ Text tokens:   text_emb(token) + text_pos_emb(0..L+1)                 │
│ Speech tokens: speech_emb(token) + speech_pos_emb(0..N)               │
│                                                                         │
│ CFG: B=2, batch[0]=conditional, batch[1]=unconditional                 │
│      text_emb[1] is zeroed out (unconditional has no text info)        │
│      cond_emb is shared (same conditioning in both batches)            │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### T3CondEnc Computational Graph

```
FILE: src/chatterbox/models/t3/modules/cond_enc.py

┌─────────────────────────────────────────────────────────────────────────┐
│ T3CondEnc.forward(cond: T3Cond) → (B, 34, 1024)                       │
│                                                                         │
│ INPUT: T3Cond dataclass                                                 │
│   speaker_emb:              (B, 256) float                             │
│   cond_prompt_speech_emb:   (B, 150, 1024) float  [pre-computed]       │
│   emotion_adv:              (B, 1, 1) float or scalar                  │
│                                                                         │
│ COMPUTATION:                                                            │
│                                                                         │
│   speaker_emb ─── Linear(256→1024) ─── unsqueeze(1) ──► (B, 1, 1024)  │
│                                                                         │
│   (nothing) ──────────────────────────────────────────► (B, 0, 1024)   │
│                                                          [CLAP unused] │
│                                                                         │
│   cond_prompt_speech_emb ─── Perceiver ──────────────► (B, 32, 1024)   │
│     (B, 150, 1024)            │                                        │
│                               ├── Cross-Attn: Q(32)×KV(150) → (32)    │
│                               └── Self-Attn:  Q(32)×KV(32)  → (32)    │
│                                                                         │
│   emotion_adv ─── view(-1,1,1) ─── Linear(1→1024) ──► (B, 1, 1024)   │
│                                      (no bias)                          │
│                                                                         │
│ CONCATENATE (dim=1):                                                    │
│   ┌─────────┬──────┬───────────────┬──────────┐                        │
│   │ spkr(1) │ ·(0) │ perceiver(32) │ emot(1)  │  = 34 tokens          │
│   └─────────┴──────┴───────────────┴──────────┘                        │
│                                                                         │
│ OUTPUT: (B, 34, 1024)                                                   │
└─────────────────────────────────────────────────────────────────────────┘
```

### Perceiver Resampler Computational Graph

```
FILE: src/chatterbox/models/t3/modules/perceiver.py

┌─────────────────────────────────────────────────────────────────────────┐
│ Perceiver.forward(h) → (B, 32, 1024)                                   │
│                                                                         │
│ INPUT: h: (B, 150, 1024)                                               │
│                                                                         │
│ Parameters:                                                             │
│   pre_attention_query: nn.Parameter(1, 32, 1024)  ← learned queries    │
│   attn: AttentionBlock2(channels=1024, num_heads=4)                    │
│     norm:      LayerNorm(1024)                                          │
│     to_q:      Linear(1024 → 1024)                                     │
│     to_k:      Linear(1024 → 1024)                                     │
│     to_v:      Linear(1024 → 1024)                                     │
│     attention: AttentionQKV(4 heads, head_dim=256, flash=True)         │
│     proj_out:  Linear(1024 → 1024)                                     │
│                                                                         │
│ ─── Step 1: Cross-Attention ───────────────────────────────────────── │
│                                                                         │
│   query_ = pre_attention_query.expand(B, -1, -1): (B, 32, 1024)       │
│                                                                         │
│   x1_norm = LayerNorm(query_): (B, 32, 1024)                          │
│   x2_norm = LayerNorm(h):      (B, 150, 1024)                         │
│                                                                         │
│   Q = to_q(x1_norm): (B, 32, 1024) → split → (B, 4, 32, 256)        │
│   K = to_k(x2_norm): (B, 150, 1024) → split → (B, 4, 150, 256)      │
│   V = to_v(x2_norm): (B, 150, 1024) → split → (B, 4, 150, 256)      │
│                                                                         │
│   Attention: QKᵀ/√256 → softmax → @V                                  │
│     (B, 4, 32, 256) × (B, 4, 256, 150) = (B, 4, 32, 150) [scores]   │
│     softmax → (B, 4, 32, 150)                                         │
│     × (B, 4, 150, 256) = (B, 4, 32, 256)                              │
│     combine_heads → (B, 32, 1024)                                      │
│                                                                         │
│   proj_out: Linear(1024→1024) → (B, 32, 1024)                         │
│   + residual (query_) → pre_att: (B, 32, 1024)                        │
│                                                                         │
│ ─── Step 2: Self-Attention ────────────────────────────────────────── │
│                                                                         │
│   x1_norm = LayerNorm(pre_att): (B, 32, 1024)                         │
│   x2_norm = LayerNorm(pre_att): (B, 32, 1024)                         │
│                                                                         │
│   Q = to_q(x1_norm): (B, 32, 1024) → (B, 4, 32, 256)                │
│   K = to_k(x2_norm): (B, 32, 1024) → (B, 4, 32, 256)                │
│   V = to_v(x2_norm): (B, 32, 1024) → (B, 4, 32, 256)                │
│                                                                         │
│   Attention: (B, 4, 32, 256) × (B, 4, 256, 32) = (B, 4, 32, 32)     │
│   softmax → × V → (B, 4, 32, 256) → combine → (B, 32, 1024)         │
│   proj_out → (B, 32, 1024)                                             │
│   + residual (pre_att) → output: (B, 32, 1024)                        │
│                                                                         │
│ OUTPUT: (B, 32, 1024)                                                   │
│                                                                         │
│ NOTE: Both cross-attn and self-attn share the SAME AttentionBlock2     │
│       weights (single self.attn module used twice)                      │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 6. Component 4: S3Gen

**Purpose:** Convert discrete speech tokens to 24kHz audio waveform.

```
FILE: src/chatterbox/models/s3gen/s3gen.py + flow.py

┌─────────────────────────────────────────────────────────────────────────┐
│ S3Gen = S3Token2Wav(S3Token2Mel)                                       │
│                                                                         │
│ Submodules:                                                             │
│                                                                         │
│ ┌─── S3Token2Mel ──────────────────────────────────────────────────┐   │
│ │   tokenizer:       S3Tokenizer (for reference tokenization)      │   │
│ │   mel_extractor:    mel_spectrogram (24kHz → 80-band mel)        │   │
│ │   speaker_encoder:  CAMPPlus(memory_efficient=False)             │   │
│ │                       Input: 16kHz audio → Output: (B, 192)      │   │
│ │   flow:             CausalMaskedDiffWithXvec                     │   │
│ │     input_embedding: Embedding(6561, 512)                        │   │
│ │     encoder: UpsampleConformerEncoder                            │   │
│ │       6 blocks, 512 dim, 8 attn heads, 2048 FFN                 │   │
│ │       2x upsample (token_mel_ratio=2)                            │   │
│ │     encoder_proj: Linear(512 → 80)                               │   │
│ │     decoder: CausalConditionalCFM                                │   │
│ │       ConditionalDecoder(in=320, out=80, 4+12+4 blocks)         │   │
│ │     spk_embed_affine_layer: Linear(192 → 80)                    │   │
│ └──────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│ ┌─── S3Token2Wav additions ────────────────────────────────────────┐   │
│ │   mel2wav: HiFTGenerator                                         │   │
│ │     upsample_rates = [8, 5, 3] → 120x total                     │   │
│ │     f0_predictor: ConvRNNF0Predictor                             │   │
│ │   trim_fade: buffer(480,) — cosine fade-in ramp                  │   │
│ └──────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│ ─────────────────────────────────────────────────────────────────────── │
│                                                                         │
│ embed_ref(ref_wav, ref_sr) → ref_dict                                  │
│                                                                         │
│   ref_wav: numpy (T,) at any SR                                        │
│     │                                                                   │
│     ├── resample → 24kHz: (1, T_24k)                                  │
│     │     mel_extractor(24kHz wav)                                     │
│     │     → ref_mel: (1, T_mel, 80)                                   │
│     │     T_mel ≈ T_24k / hop_size                                    │
│     │                                                                   │
│     ├── resample → 16kHz: (1, T_16k)                                  │
│     │     CAMPPlus.inference(16kHz wav)                                │
│     │     → x_vector: (1, 192)                                        │
│     │                                                                   │
│     └── S3Tokenizer(16kHz wav)                                         │
│           → ref_tokens: (1, T_ref), T_ref ≈ duration * 25             │
│           → ref_token_lens: (1,)                                       │
│                                                                         │
│   ENFORCE: T_mel == 2 * T_ref (truncate tokens if needed)             │
│                                                                         │
│   ref_dict = {                                                          │
│       prompt_token:     (1, T_ref) long                                │
│       prompt_token_len: (1,) long                                      │
│       prompt_feat:      (1, T_mel, 80) float                           │
│       prompt_feat_len:  None                                            │
│       embedding:        (1, 192) float                                 │
│   }                                                                     │
│                                                                         │
│ ─────────────────────────────────────────────────────────────────────── │
│                                                                         │
│ inference(speech_tokens, ref_dict) → (wav, sources)                    │
│                                                                         │
│   speech_tokens: (1, N) long, values in [0, 6560]                     │
│                                                                         │
│   ┌─── CausalMaskedDiffWithXvec.inference ─────────────────────────┐  │
│   │                                                                 │  │
│   │ 1. Speaker projection:                                          │  │
│   │    embedding (1,192) → normalize → Linear(192→80) → (1, 80)   │  │
│   │                                                                 │  │
│   │ 2. Concat tokens:                                               │  │
│   │    [prompt_token(T_ref) | speech_token(N)]                      │  │
│   │    → all_tokens: (1, T_ref+N) long                             │  │
│   │                                                                 │  │
│   │ 3. Token embedding:                                             │  │
│   │    Embedding(6561, 512)                                         │  │
│   │    all_tokens → (1, T_ref+N, 512)                              │  │
│   │    * padding_mask → (1, T_ref+N, 512)                          │  │
│   │                                                                 │  │
│   │ 4. UpsampleConformerEncoder:                                    │  │
│   │    (1, T_ref+N, 512) → 6 Conformer blocks → 2x upsample      │  │
│   │    → (1, 2*(T_ref+N), 512)                                    │  │
│   │                                                                 │  │
│   │ 5. Encoder projection:                                          │  │
│   │    Linear(512→80)                                               │  │
│   │    → mu: (1, 2*(T_ref+N), 80) → transpose → (1, 80, T_total) │  │
│   │    where T_total = 2*(T_ref+N)                                 │  │
│   │                                                                 │  │
│   │ 6. Build conditioning:                                          │  │
│   │    conds = zeros(1, 80, T_total)                               │  │
│   │    conds[:, :, :T_mel] = prompt_feat.transpose(1,2)            │  │
│   │    → conds: (1, 80, T_total)                                   │  │
│   │                                                                 │  │
│   │ 7. CausalConditionalCFM (10 Euler ODE steps):                  │  │
│   │    ┌─────────────────────────────────────────────────────────┐ │  │
│   │    │ For t in [0.0, 0.1, 0.2, ..., 0.9]:                    │ │  │
│   │    │   x_t = (1-t)*noise + t*mu                              │ │  │
│   │    │   input = cat([x_t, mu, mu-x_t, spk_emb])              │ │  │
│   │    │         → (1, 320, T_total)                             │ │  │
│   │    │                                                         │ │  │
│   │    │   ConditionalDecoder:                                   │ │  │
│   │    │     4 down blocks:   (1, 320, T) → (1, 256, T)         │ │  │
│   │    │       each: GroupNorm + Conv1D + causal Attn(8h, 64d)  │ │  │
│   │    │     12 mid blocks:   (1, 256, T) → (1, 256, T)         │ │  │
│   │    │       each: GroupNorm + Conv1D + causal Attn(8h, 64d)  │ │  │
│   │    │     4 up blocks:     (1, 256, T) → (1, 80, T)          │ │  │
│   │    │       each: GroupNorm + Conv1D + causal Attn(8h, 64d)  │ │  │
│   │    │                                                         │ │  │
│   │    │   → velocity: (1, 80, T_total)                          │ │  │
│   │    │   x_{t+0.1} = x_t + 0.1 * velocity                    │ │  │
│   │    └─────────────────────────────────────────────────────────┘ │  │
│   │                                                                 │  │
│   │ 8. Slice target portion:                                        │  │
│   │    feat[:, :, T_mel:] → (1, 80, N*2)                          │  │
│   │    (strip prompt mel frames, keep only generated)              │  │
│   │                                                                 │  │
│   │ OUTPUT: output_mels: (1, 80, N*2) float                       │  │
│   └─────────────────────────────────────────────────────────────────┘  │
│                                                                         │
│   ┌─── HiFTGenerator (vocoder) ────────────────────────────────────┐  │
│   │                                                                 │  │
│   │ INPUT: output_mels (1, 80, N*2)                                │  │
│   │                                                                 │  │
│   │ F0 prediction:                                                  │  │
│   │   ConvRNNF0Predictor(mels) → f0: (1, 1, N*2)                  │  │
│   │                                                                 │  │
│   │ Source excitation:                                               │  │
│   │   harmonic_oscillator(f0) → source: (1, 1, N*2*120)           │  │
│   │                                                                 │  │
│   │ Upsample chain (mel frames → audio samples):                   │  │
│   │   ConvTranspose1d(×8):                                          │  │
│   │     (1, C, N*2) ──────────→ (1, C, N*16)                      │  │
│   │     + ResBlock + source mixing                                  │  │
│   │                                                                 │  │
│   │   ConvTranspose1d(×5):                                          │  │
│   │     (1, C, N*16) ─────────→ (1, C, N*80)                      │  │
│   │     + ResBlock + source mixing                                  │  │
│   │                                                                 │  │
│   │   ConvTranspose1d(×3):                                          │  │
│   │     (1, C, N*80) ─────────→ (1, C, N*240)                     │  │
│   │     + ResBlock + source mixing                                  │  │
│   │                                                                 │  │
│   │   Final conv → (1, 1, N*240) → squeeze → (1, N*240)           │  │
│   │                                                                 │  │
│   │ OUTPUT:                                                         │  │
│   │   wav:     (1, N*240)   float @ 24kHz                          │  │
│   │   sources: (1, 1, N*240) float (excitation signal)             │  │
│   │                                                                 │  │
│   │ Conversion: N tokens → N*2 mel frames → N*240 audio samples   │  │
│   │   Each token = 40ms = 960 audio samples @ 24kHz                │  │
│   │   mel→audio: 120x upsample (8×5×3)                            │  │
│   │   token→mel: 2x (token_mel_ratio)                              │  │
│   │   total: 240x (2×120)                                          │  │
│   └─────────────────────────────────────────────────────────────────┘  │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 7. Constants & Token ID Reference

### Sample Rates

| Constant | Value | Used By |
|----------|-------|---------|
| `S3_SR` | 16,000 Hz | S3Tokenizer, VoiceEncoder |
| `S3GEN_SR` | 24,000 Hz | S3Gen output, mel_extractor |

### Time-to-Token Conversions

| Duration | S3Tokenizer (16kHz) | S3Gen Mel (24kHz) | Audio Samples (24kHz) |
|----------|--------------------|--------------------|----------------------|
| 1 second | 25 tokens | 50 mel frames | 24,000 samples |
| 6 seconds | 150 tokens | 300 mel frames | 144,000 samples |
| 10 seconds | 250 tokens | 500 mel frames | 240,000 samples |
| 40 ms (1 token) | 1 token | 2 mel frames | 960 samples |

### Token ID Map

```
Speech Token IDs (S3Tokenizer output / T3 speech vocabulary):
┌────────────────────────────────────────────────────────┐
│ 0 - 6560    Valid speech tokens (SPEECH_VOCAB_SIZE=6561)│
│ 6561        SOS (Start of Speech) — T3 generation start │
│ 6562        EOS (End of Speech) — T3 generation stop    │
│ 6563 - 8193 Padding (unused, in speech_tokens_dict_size)│
└────────────────────────────────────────────────────────┘

Text Token IDs:
┌────────────────────────────────────────────────────────┐
│ 0           EOT (End of Text) — stop_text_token         │
│ 1 - 254     Text tokens                                 │
│ 255         SOT (Start of Text) — start_text_token      │
│ 256 - 703   Text tokens (English only, vocab=704)       │
│ 256 - 2453  Text tokens (Multilingual, vocab=2454)      │
└────────────────────────────────────────────────────────┘

⚠ S3Gen.flow.input_embedding: Embedding(6561, 512)
  Tokens ≥ 6561 cause CUDA device-side assert crash!
  Always filter with drop_invalid_tokens() before S3Gen.
```

### Reference Audio Limits

| Constant | Value | Description |
|----------|-------|-------------|
| `ENC_COND_LEN` | 96,000 samples | 6s @ 16kHz — max ref for S3Tokenizer → 150 tokens |
| `DEC_COND_LEN` | 240,000 samples | 10s @ 24kHz — max ref for S3Gen mel conditioning |

---

## 8. Dimension Reference Table

### All Tensor Shapes in Inference

| Signal | Shape | dtype | Component | Notes |
|--------|-------|-------|-----------|-------|
| Reference audio (loaded) | `(T_24k,)` | float32 | librosa | max 240,000 (10s@24kHz) |
| Reference audio (16kHz) | `(T_16k,)` | float32 | librosa | max 96,000 (6s@16kHz) |
| VE mel spectrogram | `(T, 40)` | float32 | VoiceEncoder | 40 mel bands, NOT 128 |
| VE partials | `(N_part, P, 40)` | float32 | VoiceEncoder | windowed mel chunks |
| VE LSTM hidden | `(3, N_part, 256)` | float32 | VoiceEncoder | 3 LSTM layers |
| **speaker_emb** | `(1, 256)` | float32 | VoiceEncoder→T3 | L2-normalized |
| S3 mel spectrogram | `(1, 128, n_frames)` | float32 | S3Tokenizer | 128 mel bands, 100 fps |
| **cond_prompt_speech_tokens** | `(1, 150)` | long | S3Tokenizer→T3 | 6s × 25 tok/s |
| cond_prompt_speech_emb | `(1, 150, 1024)` | float32 | T3 internal | speech_emb + pos_emb |
| Perceiver Q (learned) | `(1, 32, 1024)` | float32 | Perceiver | nn.Parameter |
| Perceiver cross-attn scores | `(1, 4, 32, 150)` | float32 | Perceiver | 4 heads |
| Perceiver output | `(1, 32, 1024)` | float32 | Perceiver | compressed conditioning |
| **cond_emb** (all conditioning) | `(1, 34, 1024)` | float32 | T3CondEnc | 1+0+32+1 tokens |
| text_tokens (raw) | `(1, L_text)` | long | Tokenizer | before SOT/EOT/CFG |
| text_tokens (prepared) | `(2, L_text+2)` | long | T3 | +SOT+EOT, CFG doubled |
| text_embeds | `(2, L_text+2, 1024)` | float32 | T3 | emb + pos_emb |
| BOS embed | `(2, 1, 1024)` | float32 | T3 | speech_emb(6561) + pos |
| **T3 input_embeds** (full) | `(2, L+37, 1024)` | float32 | T3→Llama | cond+text+bos |
| Llama Q/K/V per head | `(2, 16, seq, 64)` | bf16/f32 | LlamaModel | 16 heads × 64 dim |
| Llama hidden states | `(2, seq, 1024)` | float32 | LlamaModel | per layer output |
| Llama MLP intermediate | `(2, seq, 4096)` | float32 | LlamaModel | gate/up projection |
| speech_head logits | `(2, 8194)` | float32 | T3 | per generation step |
| CFG guided logits | `(1, 8194)` | float32 | T3 | cond+w*(cond-uncond) |
| sampling probs | `(1, 8194)` | float32 | T3 | after temp/min_p/top_p |
| next_token | `(1, 1)` | long | T3 | multinomial sample |
| **predicted_tokens** | `(1, N)` | long | T3→S3Gen | N ≤ 1000 |
| S3Gen prompt_token | `(1, T_ref)` | long | S3Gen ref_dict | ref speech tokens |
| S3Gen prompt_feat | `(1, T_mel, 80)` | float32 | S3Gen ref_dict | ref mel spectrogram |
| S3Gen embedding (x-vector) | `(1, 192)` | float32 | S3Gen ref_dict | CAMPPlus speaker |
| S3Gen spk_emb (projected) | `(1, 80)` | float32 | flow | Linear(192→80) |
| all_tokens (concat) | `(1, T_ref+N)` | long | flow | prompt+target |
| flow token_emb | `(1, T_ref+N, 512)` | float32 | flow | Embedding(6561,512) |
| Conformer output | `(1, 2*(T_ref+N), 512)` | float32 | flow | 2x upsampled |
| encoder_proj output (mu) | `(1, 80, T_total)` | float32 | flow | T_total=2*(T_ref+N) |
| flow conditioning | `(1, 80, T_total)` | float32 | flow | prompt mel + zeros |
| CFM decoder input | `(1, 320, T_total)` | float32 | flow | cat[x_t,mu,mu-x_t,spk] |
| CFM velocity | `(1, 80, T_total)` | float32 | flow | per ODE step |
| **output_mels** (target only) | `(1, 80, N*2)` | float32 | flow→vocoder | N tokens × 2 frames |
| HiFT f0 prediction | `(1, 1, N*2)` | float32 | vocoder | pitch contour |
| HiFT source signal | `(1, 1, N*240)` | float32 | vocoder | harmonic excitation |
| HiFT after upsample ×8 | `(1, C, N*16)` | float32 | vocoder | |
| HiFT after upsample ×5 | `(1, C, N*80)` | float32 | vocoder | |
| HiFT after upsample ×3 | `(1, C, N*240)` | float32 | vocoder | |
| **output_wav** | `(1, N*240)` | float32 | vocoder | @ 24kHz |
| trim_fade | `(480,)` | float32 | S3Gen | cosine 0→1, 20ms |
| **final output** | `(1, N*240)` | float32 | watermarker | @ 24kHz, watermarked |

### Concrete Example: 4-second utterance, 20-char text

```
Text: "Hello, how are you?" → 12 text tokens (English phoneme tokenizer)

Reference: 8 seconds of audio

Conditioning:
  speaker_emb:          (1, 256)
  cond_speech_tokens:   (1, 150)    ← first 6s of ref
  S3Gen prompt_token:   (1, 200)    ← 8s × 25 tok/s
  S3Gen prompt_feat:    (1, 400, 80) ← 8s × ~50 mel/s

T3 sequence:
  cond:   34 tokens
  text:   14 tokens (12 + SOT + EOT)
  speech: ~100 tokens (4s × 25 tok/s)
  total:  ~148 positions through Llama

S3Gen:
  all_tokens: (1, 200+100) = (1, 300)
  Conformer:  (1, 600, 512)
  mu:         (1, 80, 600)
  output_mel: (1, 80, 200)   ← 100 tokens × 2 mel frames
  wav:        (1, 24000)     ← 200 mel × 120 upsample
  duration:   24000/24000 = 1.0 second... wait

  Actually: 100 tokens × 240 samples/token = 24,000 samples = 1.0s
  But we said 4 seconds? Let me recalculate:
  4 seconds → 100 tokens → 100 × 240 = 24,000 samples...

  Correction: at 24kHz, 4 seconds = 96,000 samples
  So 4 seconds → 100 tokens → but 100 × 960 samples/token = 96,000 ✓

  mel: 100 tokens × 2 = 200 mel frames
  HiFT: 200 frames × 480 hop = 96,000 samples...

  Actually the upsampling is: 200 mel × 120 = 24,000.

  Let me recalculate the hop:
  S3Gen mel_extractor uses n_fft=1024, hop=256 at 24kHz
  So mel frame rate = 24000/256 ≈ 93.75 fps
  But flow uses token_mel_ratio=2, so 25 tok/s × 2 = 50 mel frames/sec
  At 24kHz, 50 mel frames/sec → each mel frame = 480 samples
  HiFT upsamples mel by 120x: but each mel corresponds to 480 samples
  So total upsample for tokens: 2 × 480 = 960 samples/token ✓

Corrected:
  4 seconds → 100 tokens
  100 tokens → 200 mel frames (×2)
  200 mel frames → HiFT → 96,000 samples (200 × 480)
  96,000 / 24,000 = 4.0 seconds ✓
```

---

## 9. Training vs Inference Differences

### T3 Training

```
Training forward pass (T3.forward + T3.loss):

  Input:
    text_tokens:   (B, L_text) long — teacher-forced
    speech_tokens: (B, L_speech) long — ground truth from S3Tokenizer
    t3_cond:       T3Cond with all conditioning

  Sequence: [cond(34) | text(L) | speech(L_s)]
  All tokens fed at once (no autoregressive loop)

  Output:
    text_logits:   (B, L_text, 704/2454)   — text prediction head
    speech_logits: (B, L_speech, 8194)      — speech prediction head

  Loss:
    loss_text   = cross_entropy(text_logits, text_targets, ignore=-100)
    loss_speech = cross_entropy(speech_logits, speech_targets, ignore=-100)

  No CFG during training (B=actual batch size, not doubled)
  No KV cache (full sequence processed at once)
```

### Critical Training Bugs (Historical)

```
Bug 1: S3Tokenizer fed 24kHz audio instead of required 16kHz
  → Garbage speech token targets
  → Fix: resample to 16kHz before tokenization

Bug 2: Missing cond_prompt_speech_tokens in T3Cond during training
  → Only 2 conditioning tokens (speaker + emotion) instead of 34
  → Massive train/inference distribution mismatch
  → Fix: Extract and pass cond_prompt_speech_tokens (150 tokens)
```
