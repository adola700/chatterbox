# Greek Inference Trace — Complete Shape Walkthrough

Concrete example tracing every tensor shape through the entire Chatterbox multilingual pipeline.

**Text:** `"Γεια σου, πώς είσαι;"` (~22 chars)
**Reference audio:** 5 seconds of speech
**Language:** `"el"` (Greek)

---

## Step 1: Reference Audio Loading

```
librosa.load(wav_fpath, sr=24000)
  → s3gen_ref_wav: numpy (120000,)          # 5s × 24kHz

librosa.resample(24000 → 16000)
  → ref_16k_wav: numpy (80000,)             # 5s × 16kHz
```

---

## Step 2: VoiceEncoder

```
INPUT:  [ref_16k_wav]  — List[numpy (80000,)]

  trim(top_db=20)         → trimmed wav ~(78000,)
  melspectrogram(40 bands) → (T, 40)  e.g. (487, 40)
  stride_as_partials       → (N_partials, P, 40)  e.g. (6, 160, 40)
  LSTM(40→256, 3 layers)   → hidden: (3, 6, 256)
  hidden[-1]               → (6, 256)
  Linear(256→256)          → (6, 256)
  L2 normalize             → (6, 256)
  mean(dim=0)              → (1, 256)
  L2 normalize             → (1, 256)

OUTPUT: speaker_emb: (1, 256) float
```

---

## Step 3: S3Tokenizer (for T3 conditioning)

```
INPUT:  ref_16k_wav[:96000] = ref_16k_wav[:80000]  — only 5s available
        → [numpy (80000,)]

  log_mel_spectrogram:
    STFT(n_fft=400, hop=160)  → complex (1, 201, 501)
    [..., :-1].abs()²         → (1, 200, 500)
    mel_filters(128,201) @    → (1, 128, 500)
    log10 + normalize         → (1, 128, 500)

  max_len=150 → mel[..., :600] → stays (1, 128, 500)  # 500 < 600, no trim

  quantize(mel, mel_lens):
    VQ codebook (4 frames → 1 token)
    500 frames / 4 = 125 tokens

OUTPUT: cond_prompt_speech_tokens: (1, 125) long
        cond_prompt_speech_token_lens: (1,) = [125]

  torch.atleast_2d → (1, 125) long   # NOTE: only 125, not 150 (5s not 6s)
```

---

## Step 4: S3Gen.embed_ref (for S3Gen conditioning)

```
INPUT:  s3gen_ref_wav[:240000] = (120000,)  — 5s, all of it

  mel_extractor(24kHz, n_fft=1024, hop=256, 80 bands):
    → ref_mel: (1, 80, 468) → transpose → (1, 468, 80)

  resample → 16kHz: (1, 80000)
  CAMPPlus.inference:
    → x_vector: (1, 192)

  S3Tokenizer(16kHz wav):
    → ref_speech_tokens: (1, 125) long
    → ref_speech_token_lens: (1,) = [125]

  Enforce: T_mel == 2 * T_ref → truncate mels to 2*125 = 250
    → ref_mel: (1, 250, 80)

OUTPUT: s3gen_ref_dict = {
    "prompt_token":     (1, 125) long,
    "prompt_token_len": (1,) = [125],
    "prompt_feat":      (1, 250, 80) float,
    "prompt_feat_len":  None,
    "embedding":        (1, 192) float,
}
```

---

## Step 5: Assemble T3Cond

```
T3Cond(
    speaker_emb              = (1, 256) float,
    cond_prompt_speech_tokens = (1, 125) long,
    emotion_adv              = (1, 1, 1) float = 0.5,
    clap_emb                 = None,
    cond_prompt_speech_emb   = None,          # computed later by prepare_conditioning
)
```

---

## Step 6: MTLTokenizer

```
INPUT:  text="Γεια σου, πώς είσαι;", language_id="el"

  punc_norm:
    capitalize, normalize punc → "Γεια σου, πώς είσαι;"
    (semicolon is valid ender, no period added)

  encode(text, language_id="el"):
    lowercase         → "γεια σου, πώς είσαι;"
    NFKD normalize    → decomposes accents (ώ → ω + combining accent, etc.)
    NO special Greek processor (el has no language-specific handler)
    prepend "[el]"    → "[el]γεια σου, πώς είσαι;"
    replace spaces    → "[el]γεια[SPACE]σου,[SPACE]πώς[SPACE]είσαι;"
    tokenizer.encode  → e.g. [412, 45, 22, 31, 8, 3, 67, ...]
                         ~25 token IDs (grapheme-level, vocab=2454)

  text_to_tokens → torch.IntTensor([...]).unsqueeze(0)

OUTPUT: text_tokens: (1, 25) long    # approximate

  cat([text_tokens, text_tokens], dim=0)  → (2, 25)     # CFG duplication
  F.pad(SOT=255 left)                     → (2, 26)
  F.pad(EOT=0 right)                      → (2, 27)

FINAL: text_tokens: (2, 27) long
```

---

## Step 7: T3.prepare_conditioning

```
INPUT: t3_cond with cond_prompt_speech_tokens: (1, 125) long

  speech_emb: Embedding(8194, 1024)
    (1, 125) → (1, 125, 1024)

  speech_pos_emb: LearnedPositionEmbeddings(4100, 1024)
    arange(0, 125) → Embedding → (125, 1024) → broadcast → (1, 125, 1024)

  sum → cond_prompt_speech_emb: (1, 125, 1024)

OUTPUT: t3_cond.cond_prompt_speech_emb = (1, 125, 1024)
```

---

## Step 8: T3CondEnc.forward

```
INPUT: t3_cond

  spkr_enc: Linear(256→1024)
    speaker_emb (1, 256) → (1, 1024) → unsqueeze → (1, 1, 1024)

  cond_clap: (1, 0, 1024)   # empty, unused

  Perceiver:
    INPUT: (1, 125, 1024)    # NOTE: 125 not 150 (5s ref, not 6s)
    cross-attn: Q=(1,32,1024), K=V=(1,125,1024)
      split heads: Q=(1,4,32,256), K=(1,4,125,256), V=(1,4,125,256)
      scores: QKᵀ/√256 → (1, 4, 32, 125)
      softmax → × V → (1, 4, 32, 256) → combine → (1, 32, 1024)
      proj_out: Linear(1024→1024) → (1, 32, 1024)
      + residual → pre_att: (1, 32, 1024)
    self-attn: Q=K=V=(1,32,1024)
      split heads: (1, 4, 32, 256) each
      scores: (1, 4, 32, 32)
      softmax → × V → (1, 4, 32, 256) → combine → (1, 32, 1024)
      proj_out → (1, 32, 1024)
      + residual → output: (1, 32, 1024)
    OUTPUT: (1, 32, 1024)

  emotion_adv_fc: Linear(1→1024, no bias)
    (1, 1, 1) → (1, 1, 1024)

  cat([spkr(1), clap(0), perceiver(32), emotion(1)], dim=1)

OUTPUT: cond_emb: (1, 34, 1024)
```

---

## Step 9: T3.prepare_input_embeds

```
  cond_emb: (1, 34, 1024) → expand → (2, 34, 1024)

  text_emb: Embedding(2454, 1024)      ← MTL vocab (not 704)
    text_tokens (2, 27) → (2, 27, 1024)
  text_pos_emb: LearnedPositionEmbeddings(2050, 1024)
    arange(0, 27) → Embedding → (27, 1024) → broadcast → (2, 27, 1024)
  sum → (2, 27, 1024)
  text_emb[1].zero_()                  ← unconditional CFG batch zeroed

  BOS = [[6561], [6561]]: (2, 1) long
  speech_emb: Embedding(8194, 1024)
    (2, 1) → (2, 1, 1024)
  + speech_pos_emb.get_fixed_embedding(0) → (1, 1, 1024) → expand → (2, 1, 1024)
  → bos_embed: (2, 1, 1024)

  cat([cond_emb, text_embeds, bos_embed], dim=1)

OUTPUT: input_embeds: (2, 34+27+1, 1024) = (2, 62, 1024)
```

---

## Step 10: LlamaModel (first pass)

```
INPUT:  input_embeds: (2, 62, 1024)

  LlamaModel config (Llama_520M):
    hidden_size       = 1024
    num_hidden_layers = 30
    num_attention_heads = 16
    head_dim          = 64
    intermediate_size = 4096
    hidden_act        = "silu"
    attention_impl    = "sdpa"
    RoPE: llama3, theta=500000, factor=8.0

  30 × LlamaDecoderLayer:
    RMSNorm(1024, eps=1e-5)
    Self-Attention (causal):
      Q: Linear(1024→1024) → reshape → (2, 16, 62, 64)
      K: Linear(1024→1024) → reshape → (2, 16, 62, 64)
      V: Linear(1024→1024) → reshape → (2, 16, 62, 64)
      + RoPE positional encoding
      SDPA attention → (2, 16, 62, 64) → reshape → (2, 62, 1024)
      O: Linear(1024→1024)
    + residual

    RMSNorm(1024, eps=1e-5)
    MLP:
      gate_proj: Linear(1024→4096)
      up_proj:   Linear(1024→4096)
      silu(gate) * up → (2, 62, 4096)
      down_proj: Linear(4096→1024)
    + residual → (2, 62, 1024)

  Final RMSNorm → hidden_states: (2, 62, 1024)
  KV cache: 30 layers × 2 tensors × (2, 16, 62, 64)

  speech_head: Linear(1024→8194, no bias)
    hidden_states[:, -1, :] → logits: (2, 8194)

OUTPUT: logits: (2, 8194), past_key_values
```

---

## Step 11: Autoregressive Generation Loop

Generates ~100 tokens ≈ 4 seconds of speech.

```
FOR i = 0 to 99:

  Classifier-Free Guidance (CFG):
    cond_logits   = logits[0:1, :]: (1, 8194)
    uncond_logits = logits[1:2, :]: (1, 8194)
    logits = cond + 0.5 * (cond - uncond): (1, 8194)

  Repetition penalty (1.2):
    For each previously generated token t:
      logits[t] /= 1.2 (if positive)
      logits[t] *= 1.2 (if negative)

  Temperature scaling:
    logits /= 0.8

  min_p filtering (0.05):
    probs < 0.05 * max(probs) → set to -inf

  top_p nucleus sampling (1.0 = disabled)

  softmax(logits) → probs: (1, 8194)
  multinomial(probs, 1) → next_token: (1, 1) long

  STOP if next_token == 6562 (EOS)

  Prepare next input embedding:
    speech_emb(next_token): (1, 1, 1024)
    + speech_pos_emb.get_fixed_embedding(i+1): (1, 1, 1024)
    → next_embed: (1, 1, 1024)
    cat([next_embed, next_embed]) → (2, 1, 1024)   # CFG batch

  LlamaModel.forward(
    inputs_embeds=(2, 1, 1024),
    past_key_values=cached_KV,
    use_cache=True
  )
  → logits: (2, 1, 8194) → squeeze → (2, 8194)
  → updated KV cache: 30 × 2 × (2, 16, 62+i+1, 64)

cat(all predicted tokens, dim=1)

OUTPUT: predicted_tokens: (1, 100) long
```

---

## Step 12: Post-process Tokens

```
  speech_tokens = predicted_tokens[0]: (100,) long
  drop_invalid_tokens:
    remove any tokens >= 6561 (SOS/EOS)
    → speech_tokens: (N_clean,) long, e.g. (98,)
  unsqueeze → (1, 98) long

OUTPUT: speech_tokens: (1, 98) long, values in [0, 6560]
```

---

## Step 13: S3Gen Flow (CausalMaskedDiffWithXvec)

```
INPUT:
  speech_tokens:   (1, 98) long
  prompt_token:    (1, 125) long       (from ref_dict)
  prompt_feat:     (1, 250, 80) float  (from ref_dict)
  embedding:       (1, 192) float      (from ref_dict)

  Speaker embedding projection:
    F.normalize(embedding, dim=1): (1, 192)
    spk_embed_affine_layer: Linear(192→80)
    → spk_emb: (1, 80)

  Concatenate prompt + target tokens:
    cat([prompt_token(125), speech_token(98)], dim=1)
    → all_tokens: (1, 223) long
    token_len = 125 + 98 = 223

  Padding mask:
    ~make_pad_mask(token_len): (1, 223, 1) float

  Token embedding:
    input_embedding: Embedding(6561, 512)
    ⚠ ALL tokens must be < 6561
    (1, 223) → (1, 223, 512)
    × mask → (1, 223, 512)

  UpsampleConformerEncoder:
    input_size=512, output_size=512
    6 ConformerBlocks (8 attn heads, 2048 FFN)
    2x upsample (token_mel_ratio=2)
    (1, 223, 512) → (1, 446, 512)
    masks: (1, 1, 446)

  Encoder projection:
    Linear(512→80)
    (1, 446, 512) → (1, 446, 80) → transpose → mu: (1, 80, 446)

  Build conditioning tensor:
    conds = zeros(1, 80, 446)
    conds[:, :, :250] = prompt_feat.transpose(1,2)   # fill prompt mel
    → conds: (1, 80, 446)

  CausalConditionalCFM (10 Euler ODE steps):
    For t in [0.0, 0.1, 0.2, ..., 0.9]:
      x_t = (1-t)*noise + t*mu                     (1, 80, 446)
      input = cat([x_t, mu, mu-x_t, spk_emb])      (1, 320, 446)

      ConditionalDecoder:
        in_channels=320, out_channels=80
        4 down blocks (Conv1D + causal Attn, 8 heads, 64 dim)
        12 mid blocks (Conv1D + causal Attn, 8 heads, 64 dim)
        4 up blocks   (Conv1D + causal Attn, 8 heads, 64 dim)
        → velocity: (1, 80, 446)

      x_{t+0.1} = x_t + 0.1 * velocity

    final: feat: (1, 80, 446)

  Slice target portion:
    feat[:, :, 250:]
    → output_mels: (1, 80, 196)     # 446 - 250 = 196 = 98 tokens × 2

OUTPUT: output_mels: (1, 80, 196) float
```

---

## Step 14: HiFTGenerator (Vocoder)

```
INPUT: output_mels: (1, 80, 196)

  HiFTGenerator config:
    sampling_rate        = 24000
    upsample_rates       = [8, 5, 3]    → total = 8×5×3 = 120x
    upsample_kernel_sizes = [16, 11, 7]
    f0_predictor         = ConvRNNF0Predictor

  F0 prediction:
    ConvRNNF0Predictor(mels) → f0: (1, 1, 196)

  Source excitation:
    harmonic_oscillator(f0) → source: (1, 1, 23520)   # 196 × 120

  Upsample chain (mel frames → audio samples):
    ConvTranspose1d(×8):  (1, C, 196)   → (1, C, 1568)   + ResBlocks
    ConvTranspose1d(×5):  (1, C, 1568)  → (1, C, 7840)   + ResBlocks
    ConvTranspose1d(×3):  (1, C, 7840)  → (1, C, 23520)  + ResBlocks
    Final conv          → (1, 1, 23520) → squeeze → (1, 23520)

  Fade-in:
    output[:, :480] *= trim_fade     # cosine ramp 0→1 over 20ms

  Note on duration:
    Each token = 40ms = 960 audio samples @ 24kHz
    98 tokens × 960 = 94,080 samples = 3.92 seconds
    The HiFT effective hop per mel frame = 480 samples
    196 mel frames × 480 = 94,080 samples ✓

OUTPUT: wav: (1, 94080) float @ 24kHz
        sources: (1, 1, 94080) float
```

---

## Step 15: Final Output

```
  wav.squeeze(0).detach().cpu().numpy()   → numpy (94080,)
  perth.apply_watermark(wav, sr=24000)    → numpy (94080,)
  torch.from_numpy(...).unsqueeze(0)      → (1, 94080) float tensor

FINAL OUTPUT: (1, 94080) tensor @ 24kHz
              duration: 94,080 / 24,000 = 3.92 seconds
```

---

## Summary Table

| Step | Component | Input Shape | Output Shape |
|------|-----------|-------------|--------------|
| Load ref | librosa | wav file | `(120000,)` numpy @24kHz |
| Resample | librosa | `(120000,)` @24kHz | `(80000,)` @16kHz |
| Speaker emb | VoiceEncoder | `(80000,)` @16kHz | `(1, 256)` float |
| Cond tokens | S3Tokenizer | `(1, 80000)` @16kHz | `(1, 125)` long |
| Ref mel | mel_extractor | `(1, 120000)` @24kHz | `(1, 250, 80)` float |
| Ref x-vector | CAMPPlus | `(1, 80000)` @16kHz | `(1, 192)` float |
| Ref tokens | S3Tokenizer | `(1, 80000)` @16kHz | `(1, 125)` long |
| Text tokenize | MTLTokenizer | `"Γεια σου..."` + `"el"` | `(1, 25)` long |
| CFG duplicate | cat+pad | `(1, 25)` | `(2, 27)` (+SOT+EOT) |
| Cond speech emb | speech_emb+pos | `(1, 125)` long | `(1, 125, 1024)` |
| Perceiver | cross+self attn | `(1, 125, 1024)` | `(1, 32, 1024)` |
| T3CondEnc | concat all | spkr+perc+emot | `(1, 34, 1024)` |
| T3 input embeds | concat | cond+text+bos | `(2, 62, 1024)` |
| Llama (first) | 30-layer transformer | `(2, 62, 1024)` | `(2, 62, 1024)` + KV cache |
| Llama (per step) | with KV cache | `(2, 1, 1024)` | `(2, 8194)` logits |
| CFG guidance | logit arithmetic | `(2, 8194)` | `(1, 8194)` guided |
| Sampling | multinomial | `(1, 8194)` probs | `(1, 1)` next token |
| Generated tokens | autoregressive ×100 | — | `(1, 100)` long |
| Post-process | drop_invalid | `(1, 100)` | `(1, 98)` long |
| Flow token embed | Embedding(6561,512) | `(1, 223)` long | `(1, 223, 512)` |
| Conformer | 6 blocks + 2x up | `(1, 223, 512)` | `(1, 446, 512)` |
| Encoder proj | Linear(512→80) | `(1, 446, 512)` | `(1, 80, 446)` |
| CFM decoder | 10 ODE steps | `(1, 320, 446)` | `(1, 80, 446)` |
| Slice target | strip prompt | `(1, 80, 446)` | `(1, 80, 196)` |
| HiFT vocoder | 120x upsample | `(1, 80, 196)` | `(1, 94080)` @24kHz |
| Watermark | perth | `(94080,)` numpy | `(1, 94080)` tensor |

---

## Greek-Specific Notes

1. **No special Greek preprocessor** — Greek (`el`) only gets `lowercase` + `NFKD normalize` (accent decomposition). Compare with Chinese (Cangjie), Japanese (hiragana), Hebrew (diacritics), Korean (Jamo), Russian (stress marks) which all have dedicated processors.

2. **Language identity** is encoded **solely** by the `[el]` token prepended to the text sequence. The T3 model learned during training that this prefix signals Greek phonetics/prosody.

3. **All conditioning is language-agnostic** — speaker_emb, cond_prompt_speech_tokens, S3Gen ref_dict capture voice identity and style, not language. You can use an English reference voice and generate Greek speech.

4. **Grapheme tokenizer** — Greek characters (α, β, γ, ...) are tokenized as individual graphemes from the 2454-token multilingual vocabulary, unlike English which uses a 704-token phoneme-based tokenizer.
