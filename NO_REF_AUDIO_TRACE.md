# No-Reference-Audio Inference Trace — What Changes?

When you call `generate()` **without** `audio_prompt_path`, the model uses a **pre-computed built-in voice** from `conds.pt`. This document explains exactly what changes compared to the [full reference audio trace](GREEK_INFERENCE_TRACE.md).

**Text:** `"Γεια σου, πώς είσαι;"`
**Reference audio:** **NONE** (using built-in `conds.pt`)
**Language:** `"el"` (Greek)

---

## What Gets Skipped Entirely

When no reference audio is provided, **Steps 1–5 are completely skipped**:

```
┌──────────────────────────────────────────────────────────────────────┐
│ SKIPPED — These only run when audio_prompt_path is provided:        │
│                                                                      │
│   Step 1: librosa.load(wav_fpath)              ← NOT CALLED         │
│   Step 2: VoiceEncoder                          ← NOT CALLED         │
│   Step 3: S3Tokenizer (T3 conditioning)         ← NOT CALLED         │
│   Step 4: S3Gen.embed_ref                       ← NOT CALLED         │
│   Step 5: Assemble T3Cond                       ← NOT CALLED         │
│                                                                      │
│   No audio loading, no resampling, no mel extraction,               │
│   no speaker embedding, no speech tokenization, no CAMPPlus.        │
└──────────────────────────────────────────────────────────────────────┘
```

---

## What Happens Instead: Loading `conds.pt`

At model initialization (`from_pretrained()` or `from_local()`), the model checks for `conds.pt`:

```python
# from_local():
conds = None
if (builtin_voice := ckpt_dir / "conds.pt").exists():
    conds = Conditionals.load(builtin_voice).to(device)
```

This loads a **pre-computed `Conditionals` object** that was created by running `prepare_conditionals()` on a specific reference audio clip during model release. It contains the exact same tensors that would be computed at runtime:

```
conds.pt = Conditionals(
    t3 = T3Cond(
        speaker_emb              = (1, 256) float   ← pre-computed VoiceEncoder output
        cond_prompt_speech_tokens = (1, 150) long    ← pre-computed S3Tokenizer output (6s clip)
        emotion_adv              = (1, 1, 1) float   ← default 0.5
        clap_emb                 = None
        cond_prompt_speech_emb   = None               ← computed lazily by prepare_conditioning
    ),
    gen = {
        "prompt_token":     (1, T_ref) long          ← pre-computed S3Tokenizer tokens
        "prompt_token_len": (1,) long
        "prompt_feat":      (1, T_mel, 80) float     ← pre-computed mel spectrogram
        "prompt_feat_len":  None
        "embedding":        (1, 192) float           ← pre-computed CAMPPlus x-vector
    }
)
```

The `conds.pt` file is a frozen snapshot — **the same built-in voice is used for every call** unless you provide your own reference audio.

---

## generate() Code Path

```python
def generate(self, text, language_id, audio_prompt_path=None, ...):
    if audio_prompt_path:
        self.prepare_conditionals(audio_prompt_path, ...)   # ← SKIPPED
    else:
        assert self.conds is not None, "Please prepare_conditionals first or specify audio_prompt_path"
        # ↑ Uses self.conds loaded from conds.pt at init time

    # Update exaggeration if different from cached value
    if float(exaggeration) != float(self.conds.t3.emotion_adv[0, 0, 0].item()):
        self.conds.t3 = T3Cond(
            speaker_emb=self.conds.t3.speaker_emb,                    # (1, 256) — unchanged
            cond_prompt_speech_tokens=self.conds.t3.cond_prompt_speech_tokens,  # (1, 150) — unchanged
            emotion_adv=exaggeration * torch.ones(1, 1, 1),           # (1, 1, 1) — updated
        ).to(device=self.device)

    # ... rest is identical to ref-audio path
```

---

## Step-by-Step: What Actually Runs

### Step 6: MTLTokenizer (IDENTICAL)

```
INPUT:  text="Γεια σου, πώς είσαι;", language_id="el"

  punc_norm → lowercase → NFKD normalize
  prepend "[el]" → replace spaces → tokenizer.encode
  → (1, ~25) long

  CFG duplicate → pad SOT/EOT

OUTPUT: text_tokens: (2, 27) long       ← SAME as ref-audio path
```

No change. Text processing is completely independent of conditioning.

---

### Step 7: T3.prepare_conditioning (IDENTICAL logic, DIFFERENT source)

```
INPUT: t3_cond from conds.pt (not from live audio)
  cond_prompt_speech_tokens: (1, 150) long   ← from conds.pt (always exactly 150)

  speech_emb: Embedding(8194, 1024)
    (1, 150) → (1, 150, 1024)                ← 150 not 125 (conds.pt uses full 6s clip)

  speech_pos_emb: LearnedPositionEmbeddings(4100, 1024)
    arange(0, 150) → (150, 1024) → (1, 150, 1024)

  sum → cond_prompt_speech_emb: (1, 150, 1024)

OUTPUT: t3_cond.cond_prompt_speech_emb = (1, 150, 1024)
```

**KEY DIFFERENCE:** The built-in voice in `conds.pt` has exactly **150 tokens** (from a full 6-second clip), whereas a 5-second user ref gives only **125 tokens**. This affects the Perceiver cross-attention dimensions.

---

### Step 8: T3CondEnc.forward (SLIGHTLY DIFFERENT shapes)

```
INPUT: t3_cond from conds.pt

  spkr_enc: Linear(256→1024)
    speaker_emb (1, 256) → (1, 1, 1024)      ← SAME shape, DIFFERENT values

  cond_clap: (1, 0, 1024)                     ← SAME

  Perceiver:
    INPUT: (1, 150, 1024)                      ← 150 not 125!
    cross-attn: Q=(1,32,1024), K=V=(1,150,1024)
      scores: (1, 4, 32, 150)                  ← 150 not 125
      output: (1, 32, 1024)
    self-attn: Q=K=V=(1,32,1024)
      scores: (1, 4, 32, 32)                   ← SAME
      output: (1, 32, 1024)
    OUTPUT: (1, 32, 1024)                       ← SAME output shape

  emotion_adv_fc: Linear(1→1024, no bias)
    (1, 1, 1) → (1, 1, 1024)                   ← SAME

  cat → cond_emb: (1, 34, 1024)                ← SAME output shape

OUTPUT: cond_emb: (1, 34, 1024)                ← SAME shape, different values
```

Perceiver cross-attention K/V sequence is 150 instead of 125, but the **output is always (1, 32, 1024)** regardless — that's the whole point of the Perceiver Resampler (fixed-length output from variable-length input).

---

### Step 9: T3.prepare_input_embeds (IDENTICAL shapes)

```
  cond_emb: (1, 34, 1024) → expand → (2, 34, 1024)          ← SAME
  text_embeds: (2, 27, 1024) with [1] zeroed                  ← SAME
  bos_embed: (2, 1, 1024)                                     ← SAME

  cat → input_embeds: (2, 62, 1024)                            ← SAME

OUTPUT: (2, 62, 1024)                                           ← SAME
```

---

### Steps 10–12: T3 Generation (IDENTICAL structure)

```
  LlamaModel first pass:  (2, 62, 1024) → (2, 8194) logits    ← SAME shapes
  Autoregressive loop:     up to 1000 steps                     ← SAME process
  Post-process:            drop invalid tokens                   ← SAME

OUTPUT: speech_tokens: (1, N) long                               ← SAME structure
```

The **values** will differ because different conditioning → different hidden states → different token probabilities → different generated speech tokens. But all tensor shapes are identical.

---

### Step 13: S3Gen Flow (DIFFERENT ref_dict values, DIFFERENT shapes)

```
INPUT:
  speech_tokens:   (1, N) long                  ← from T3 (same structure)
  prompt_token:    (1, T_ref) long              ← from conds.pt (different T_ref!)
  prompt_feat:     (1, T_mel, 80) float         ← from conds.pt (different T_mel!)
  embedding:       (1, 192) float               ← from conds.pt (different values)
```

The built-in voice's reference was likely a full 10-second clip, so:

```
WITH ref audio (5s clip):             WITHOUT ref audio (conds.pt, ~10s clip):
  prompt_token:   (1, 125) long         prompt_token:   (1, 250) long
  prompt_feat:    (1, 250, 80)          prompt_feat:    (1, 500, 80)
  embedding:      (1, 192)              embedding:      (1, 192)
```

This changes all downstream shapes in the flow:

```
  Speaker projection:
    (1, 192) → normalize → Linear(192→80) → (1, 80)           ← SAME shape

  Concat tokens (DIFFERENT):
    cat([prompt(250), speech(N)]) → all_tokens: (1, 250+N)     ← was (1, 125+N)

  Token embedding:
    Embedding(6561, 512)
    (1, 250+N) → (1, 250+N, 512)                               ← was (1, 125+N, 512)

  UpsampleConformerEncoder:
    (1, 250+N, 512) → 2x upsample → (1, 2*(250+N), 512)       ← was (1, 2*(125+N), 512)
    e.g. with N=98: (1, 696, 512)                               ← was (1, 446, 512)

  Encoder projection:
    Linear(512→80)
    → mu: (1, 80, 696)                                          ← was (1, 80, 446)

  Build conditioning:
    conds = zeros(1, 80, 696)                                   ← was zeros(1, 80, 446)
    conds[:, :, :500] = prompt_feat.T                           ← was conds[:, :, :250]

  CausalConditionalCFM (10 Euler steps):
    input: (1, 320, 696)                                        ← was (1, 320, 446)
    → velocity: (1, 80, 696) per step                           ← was (1, 80, 446)
    final: (1, 80, 696)                                          ← was (1, 80, 446)

  Slice target:
    feat[:, :, 500:]                                             ← was feat[:, :, 250:]
    → output_mels: (1, 80, 196)                                 ← SAME (N*2, depends only on N)
    (696 - 500 = 196 = 98*2)                                     ← (446 - 250 = 196)

OUTPUT: output_mels: (1, 80, 196)                                ← SAME final shape
```

The target mel output shape is always `(1, 80, N*2)` regardless of reference length — the reference only affects the **prompt portion** that gets sliced off.

---

### Step 14: HiFT Vocoder (IDENTICAL)

```
INPUT: output_mels: (1, 80, 196)                                ← SAME

  F0 → source → upsample chain → wav

OUTPUT: wav: (1, 94080) @ 24kHz                                  ← SAME
```

---

### Step 15: Final Output (IDENTICAL)

```
OUTPUT: (1, 94080) tensor @ 24kHz, watermarked                   ← SAME
```

---

## Complete Comparison Table

| Step | Component | With Ref Audio (5s) | Without Ref Audio (conds.pt) | Changed? |
|------|-----------|--------------------|-----------------------------|----------|
| 1 | Load audio | `(120000,)` @24kHz | **SKIPPED** | YES |
| 2 | VoiceEncoder | `(80000,)` → `(1, 256)` | **SKIPPED** (loaded from file) | YES |
| 3 | S3Tokenizer (T3) | `(1, 80000)` → `(1, 125)` | **SKIPPED** (loaded from file, `(1, 150)`) | YES |
| 4 | S3Gen.embed_ref | 3 sub-models → ref_dict | **SKIPPED** (loaded from file) | YES |
| 5 | Assemble T3Cond | Built from step 2-4 outputs | **SKIPPED** (loaded from file) | YES |
| 6 | MTLTokenizer | `(1, 25)` → `(2, 27)` | `(1, 25)` → `(2, 27)` | NO |
| 7 | prepare_conditioning | `(1, 125)` → `(1, 125, 1024)` | `(1, 150)` → `(1, 150, 1024)` | **shape** |
| 8 | T3CondEnc | Perceiver K/V: `(1, 125, 1024)` | Perceiver K/V: `(1, 150, 1024)` | **shape** |
| 8 | T3CondEnc output | `(1, 34, 1024)` | `(1, 34, 1024)` | values only |
| 9 | input_embeds | `(2, 62, 1024)` | `(2, 62, 1024)` | values only |
| 10 | LlamaModel | `(2, 62, 1024)` → `(2, 8194)` | `(2, 62, 1024)` → `(2, 8194)` | values only |
| 11 | Autoregressive | `(1, 1)` per step | `(1, 1)` per step | values only |
| 12 | Post-process | `(1, N)` | `(1, N')` (likely different N) | values only |
| 13a | Concat tokens | `(1, 125+N)` | `(1, 250+N')` | **shape** |
| 13b | Token embed | `(1, 125+N, 512)` | `(1, 250+N', 512)` | **shape** |
| 13c | Conformer | `(1, 2*(125+N), 512)` | `(1, 2*(250+N'), 512)` | **shape** |
| 13d | mu | `(1, 80, 2*(125+N))` | `(1, 80, 2*(250+N'))` | **shape** |
| 13e | CFM decoder | `(1, 320, 2*(125+N))` | `(1, 320, 2*(250+N'))` | **shape** |
| 13f | Slice target | `(1, 80, N*2)` | `(1, 80, N'*2)` | N may differ |
| 14 | HiFT vocoder | `(1, N*240*4)` | `(1, N'*240*4)` | N may differ |
| 15 | Final output | `(1, N*960)` @24kHz | `(1, N'*960)` @24kHz | N may differ |

---

## Shape Differences Visualized

```
WITH 5-second reference audio:
═══════════════════════════════

T3 Conditioning:
  cond_prompt_speech_tokens:  (1, 125)                    ← 5s × 25 tok/s
  cond_prompt_speech_emb:     (1, 125, 1024)
  Perceiver cross-attn:       Q(1,4,32,256) × K(1,4,125,256) → scores(1,4,32,125)
  cond_emb:                   (1, 34, 1024)               ← always 34

S3Gen Flow:
  all_tokens:  (1, 125+98)    = (1, 223)
  Conformer:   (1, 446, 512)
  mu:          (1, 80, 446)
  CFM input:   (1, 320, 446)
  full output: (1, 80, 446)
  slice at:    250
  target mel:  (1, 80, 196)


WITHOUT reference audio (conds.pt with ~10s built-in clip):
═══════════════════════════════════════════════════════════

T3 Conditioning:
  cond_prompt_speech_tokens:  (1, 150)                    ← 6s × 25 tok/s (full)
  cond_prompt_speech_emb:     (1, 150, 1024)
  Perceiver cross-attn:       Q(1,4,32,256) × K(1,4,150,256) → scores(1,4,32,150)
  cond_emb:                   (1, 34, 1024)               ← always 34

S3Gen Flow:
  all_tokens:  (1, 250+100)   = (1, 350)                   ← much longer!
  Conformer:   (1, 700, 512)                                ← much longer!
  mu:          (1, 80, 700)                                 ← much longer!
  CFM input:   (1, 320, 700)                                ← much longer!
  full output: (1, 80, 700)
  slice at:    500                                           ← different slice point!
  target mel:  (1, 80, 200)                                 ← similar (depends on N)
```

---

## Key Takeaways

1. **No ref audio = use frozen `conds.pt`** — pre-computed VoiceEncoder embedding, S3Tokenizer tokens, and S3Gen reference features from a specific voice clip bundled with the model.

2. **Steps 1–5 completely skipped** — no audio loading, no neural network inference for conditioning extraction. This makes the first call faster.

3. **T3 input shape is always `(2, 34+L_text+2+1, 1024)`** regardless of reference — the Perceiver Resampler always outputs `(1, 32, 1024)` no matter if the input is 125 or 150 tokens. The conditioning sequence is always 34 tokens.

4. **S3Gen internal shapes change significantly** — a longer reference clip means longer token sequences through the Conformer and CFM decoder. But the **target output** (`output_mels[:, :, T_mel:]`) only depends on the number of generated speech tokens N, not the reference length.

5. **Same voice every time** — without ref audio, every call produces the same built-in voice identity (same speaker_emb, same prosody conditioning). With ref audio, you get voice cloning of the provided speaker.

6. **conds.pt is created once** by running `prepare_conditionals()` on a chosen reference clip and calling `conds.save()`. The exact voice in the built-in file depends on what ResembleAI used when releasing the model.
