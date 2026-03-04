# Copyright (c) 2025 Resemble AI
# MIT License
"""
T3 — Token-to-Token TTS Model
==============================
The core autoregressive model that generates speech tokens from text tokens.

Architecture
------------
    Backbone:    LlamaModel (520M config: 30 layers, 16 heads, dim=1024, intermediate=4096)
    Input:       Custom embeddings (NOT Llama's built-in token embeddings)
    Output:      Two projection heads (text logits + speech logits)

Model dimensions (Llama_520M config):
    hidden_size:        1024
    intermediate_size:  4096
    num_hidden_layers:  30
    num_attention_heads: 16
    head_dim:           64
    Total params:       ~536M

Input sequence layout
---------------------
    [cond_tokens | text_tokens | speech_tokens]
     ↑ 34 tokens   ↑ variable    ↑ variable
     (from T3CondEnc)

    Each segment has its own embedding:
        cond_tokens:   computed by T3CondEnc.forward() → (B, 34, 1024)
        text_tokens:   text_emb(tokens) + text_pos_emb → (B, L_text, 1024)
        speech_tokens: speech_emb(tokens) + speech_pos_emb → (B, L_speech, 1024)

    Concatenated: (B, 34 + L_text + L_speech, 1024) → LlamaModel → hidden_states

Output heads
------------
    text_head:   Linear(1024 → 704 or 2454)  — text token prediction (auxiliary loss)
    speech_head: Linear(1024 → 8194)         — speech token prediction (main loss)

Token vocabulary
----------------
    Text (English):      704 tokens   (start=255, stop=0)
    Text (Multilingual): 2454 tokens  (start=255, stop=0)
    Speech:              8194 tokens  (valid speech: 0-6560, start=6561, stop=6562)

Training
--------
    forward() returns text_logits and speech_logits for cross-entropy loss.
    loss() computes masked CCE for both, ignoring padded positions.

Inference
---------
    inference() uses KV-cache autoregressive generation with:
    - Classifier-Free Guidance (CFG): batch=2 (conditional + unconditional)
    - AlignmentStreamAnalyzer (multilingual): monitors attention for repetition/long-tail
    - Repetition penalty, temperature, min_p, top_p sampling
"""
import logging
from typing import Union, Optional, List

logger = logging.getLogger(__name__)

from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from transformers import LlamaModel, LlamaConfig, GPT2Config, GPT2Model
from transformers.generation.logits_process import (
    LogitsProcessorList,
    RepetitionPenaltyLogitsProcessor,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
    MinPLogitsWarper,
)
from .modules.learned_pos_emb import LearnedPositionEmbeddings

from .modules.cond_enc import T3CondEnc, T3Cond
from .modules.t3_config import T3Config
from .llama_configs import LLAMA_CONFIGS
from .inference.t3_hf_backend import T3HuggingfaceBackend
from .inference.alignment_stream_analyzer import AlignmentStreamAnalyzer
from ..utils import AttrDict


logger = logging.getLogger(__name__)


def _ensure_BOT_EOT(text_tokens: Tensor, hp):
    B = text_tokens.size(0)
    assert (text_tokens == hp.start_text_token).int().sum() >= B, "missing start_text_token"
    assert (text_tokens == hp.stop_text_token).int().sum() >= B, "missing stop_text_token"


class T3(nn.Module):
    """Token-To-Token (T3) TTS model — autoregressive text→speech token generator.

    Uses HuggingFace transformer backbones (LlamaModel or GPT2Model) with custom
    input embeddings and output heads. Tokenization (start/stop tokens) is handled
    externally by the caller.

    NOTE: This model uses relative positional encoding (RoPE for Llama, learned for
    speech/text). With absolute PE, position would need to reset at speech token boundary.

    Components
    ----------
    tfmr : LlamaModel (default) or GPT2Model
        Backbone transformer. Config "Llama_520M": 30 layers, 16 heads, dim=1024.

    cond_enc : T3CondEnc
        Conditioning encoder. See cond_enc.py for detailed shape documentation.
        Output: (B, 34, 1024) conditioning tokens.

    text_emb : Embedding(text_vocab_size → 1024)
        text_vocab_size = 704 (English) or 2454 (Multilingual)

    speech_emb : Embedding(8194 → 1024)
        8194 = 6561 valid speech tokens + start(6561) + stop(6562) + padding

    text_pos_emb : LearnedPositionEmbeddings(max_text_tokens+2, 1024)
        Learned positional embeddings for text tokens.

    speech_pos_emb : LearnedPositionEmbeddings(max_speech_tokens+4, 1024)
        Learned positional embeddings for speech tokens.

    text_head : Linear(1024 → text_vocab_size, no bias)
        Predicts next text token (auxiliary loss during training).

    speech_head : Linear(1024 → 8194, bias=True for GPT2 / False for Llama)
        Predicts next speech token (primary task).
    """

    def __init__(self, hp=None):
        if hp is None:
            hp = T3Config.english_only()
        super().__init__()
        self.hp = hp

        config_dict = LLAMA_CONFIGS[hp.llama_config_name]
        self.is_gpt = config_dict.get("model_type") == "gpt2"

        if self.is_gpt:
            self.cfg = GPT2Config(**config_dict)
            self.tfmr = GPT2Model(self.cfg)
        else:
            self.cfg = LlamaConfig(**config_dict)
            self.tfmr = LlamaModel(self.cfg)

        self.dim = self.cfg.hidden_size  # 1024
        self.deepspeed_patch_applied = False

        # Conditioning encoder: T3Cond → (B, 34, 1024) tokens
        self.cond_enc = T3CondEnc(hp)

        # Token embeddings (custom, NOT using Llama's built-in embeddings)
        self.text_emb = nn.Embedding(hp.text_tokens_dict_size, self.dim)    # (704 or 2454, 1024)
        self.speech_emb = nn.Embedding(hp.speech_tokens_dict_size, self.dim) # (8194, 1024)

        # Learned positional embeddings (separate for text and speech)
        self.text_pos_emb = None
        self.speech_pos_emb = None
        if hp.input_pos_emb == "learned":
            max_text_seq_len = hp.max_text_tokens + 2    # 2048 + 2 = 2050
            self.text_pos_emb = LearnedPositionEmbeddings(max_text_seq_len, self.dim)

            max_mel_seq_len = hp.max_speech_tokens + 2 + 2  # 4096 + 4 = 4100
            self.speech_pos_emb = LearnedPositionEmbeddings(max_mel_seq_len, self.dim)

        # Output projection heads
        self.text_head = nn.Linear(self.cfg.hidden_size, hp.text_tokens_dict_size, bias=False)     # (1024 → 704/2454)
        self.speech_head = nn.Linear(self.cfg.hidden_size, hp.speech_tokens_dict_size, bias=self.is_gpt)  # (1024 → 8194)
        self.compiled = False

    @property
    def device(self):
        return self.speech_head.weight.device

    def prepare_conditioning(self, t3_cond: T3Cond):
        """Embed conditioning speech tokens and run through T3CondEnc.

        This method bridges T3Cond (raw tokens) → T3CondEnc (embedded sequences).
        Speech conditioning tokens must be embedded by T3's own speech_emb before
        the Perceiver Resampler can process them.

        Steps:
            1. speech_emb(cond_prompt_speech_tokens):  (B, 150) → (B, 150, 1024)
            2. + speech_pos_emb (for Llama):           (B, 150, 1024)
            3. Store as t3_cond.cond_prompt_speech_emb
            4. cond_enc(t3_cond) → Perceiver(150→32) + concat → (B, 34, 1024)

        Args:
            t3_cond: T3Cond with speaker_emb, cond_prompt_speech_tokens, emotion_adv

        Returns:
            cond_emb: (B, L_cond, 1024) — conditioning sequence (typically 34 tokens)

        Side effects:
            Sets t3_cond.cond_prompt_speech_emb in-place.
        """
        if t3_cond.cond_prompt_speech_tokens is not None and t3_cond.cond_prompt_speech_emb is None:
            # Embed speech tokens: (B, 150) → (B, 150, 1024)
            t3_cond.cond_prompt_speech_emb = self.speech_emb(t3_cond.cond_prompt_speech_tokens)
            if not self.is_gpt:
                # Add learned positional embeddings (Llama models only)
                t3_cond.cond_prompt_speech_emb += self.speech_pos_emb(t3_cond.cond_prompt_speech_tokens)
        return self.cond_enc(t3_cond)  # (B, len_cond, dim)

    def prepare_input_embeds(
        self,
        *,
        t3_cond: T3Cond,
        text_tokens: torch.LongTensor,
        speech_tokens: torch.LongTensor,
        cfg_weight: float = 0.0,
    ):
        """Build the full input embedding sequence for the transformer backbone.

        Constructs: [cond_emb | text_emb | speech_emb] with position embeddings.

        For Classifier-Free Guidance (CFG), text_tokens has batch=2:
            batch[0] = conditional (real text)
            batch[1] = unconditional (zeroed text embeddings)

        Args:
            t3_cond:       T3Cond conditioning dataclass
            text_tokens:   (B, L_text) long — text token IDs (with start=255, stop=0)
            speech_tokens: (B, L_speech) long — speech token IDs
            cfg_weight:    float — if > 0, batch[1] text embeddings are zeroed (CFG)

        Returns:
            embeds:   (B, L_cond + L_text + L_speech, 1024) — full input embeddings
            len_cond: int — number of conditioning tokens (typically 34)

        Example shapes (inference, multilingual, CFG):
            text_tokens:   (2, L_text)  — duplicated for CFG
            speech_tokens: (2, 1)       — just start token
            cond_emb:      (1, 34, 1024) → expanded to (2, 34, 1024)
            text_emb:      (2, L_text, 1024)
            speech_emb:    (2, 1, 1024)
            embeds:        (2, 34 + L_text + 1, 1024)
        """
        # Condition embeddings: T3Cond → (B_cond, L_cond, 1024), typically (1, 34, 1024)
        cond_emb = self.prepare_conditioning(t3_cond)  # (B, len_cond, dim)

        # Text embeddings: (B, L_text) → (B, L_text, 1024)
        text_emb = self.text_emb(text_tokens)  # (B, len_text, dim)
        if cfg_weight > 0.0 and not self.is_gpt:
            text_emb[1].zero_()  # Zero out unconditional batch for CFG

        # Speech embeddings: (B, L_speech) → (B, L_speech, 1024)
        speech_emb = self.speech_emb(speech_tokens)  # (B, len_speech, dim)

        # Add learned positional embeddings
        if self.hp.input_pos_emb == "learned":
            text_emb = text_emb + self.text_pos_emb(text_tokens)       # (B, L_text, 1024)
            speech_emb = speech_emb + self.speech_pos_emb(speech_tokens) # (B, L_speech, 1024)
        len_cond = cond_emb.size(1)  # typically 34

        # Expand conditioning to match batch size (for CFG: 1 → 2)
        if cond_emb.size(0) != text_emb.size(0):
             cond_emb = cond_emb.expand(text_emb.size(0), -1, -1)

        # Concatenate: [cond | text | speech] → (B, L_total, 1024)
        embeds = torch.stack([
            torch.cat((ce, te, se))
            for ce, te, se in zip(cond_emb, text_emb, speech_emb)
        ])  # (B, length, dim)
        return embeds, len_cond

    def forward(
        self,
        *,
        t3_cond: T3Cond,
        text_tokens: torch.LongTensor,
        text_token_lens: torch.LongTensor,
        speech_tokens: torch.LongTensor,
        speech_token_lens: torch.LongTensor,
        training=False,
    ):
        """Forward pass through the T3 model (training mode).

        Builds the full input sequence [cond | text | speech], runs through the
        Llama backbone, then splices out text and speech hidden states for the
        respective output heads.

        Args:
            t3_cond:          T3Cond — conditioning (speaker_emb, cond_prompt_speech_tokens, emotion_adv)
            text_tokens:      (B, L_text) long — text token IDs with start=255, stop=0
            text_token_lens:  (B,) long — actual lengths (excluding padding)
            speech_tokens:    (B, L_speech) long — speech token IDs (S3Tokenizer output)
            speech_token_lens:(B,) long — actual lengths (excluding padding)
            training:         bool — if True, disables KV cache

        Returns:
            AttrDict with:
                text_logits:    (B, L_text, text_vocab_size)    — text next-token predictions
                text_latents:   (B, L_text, 1024)               — text hidden states
                speech_logits:  (B, L_speech, 8194)             — speech next-token predictions
                speech_latents: (B, L_speech, 1024)             — speech hidden states
                hidden_states:  (B, L_total, 1024)              — full transformer output

        Internal sequence layout:
            Position:  [0..33 | 34..34+L_text-1 | 34+L_text..34+L_text+L_speech-1]
            Content:   [cond  | text             | speech                          ]
        """
        _ensure_BOT_EOT(text_tokens, self.hp)

        # Build input embeddings: (B, L_cond + L_text + L_speech, 1024)
        embeds, len_cond = self.prepare_input_embeds(
            t3_cond=t3_cond,
            text_tokens=text_tokens,
            speech_tokens=speech_tokens,
        )

        # Backbone transformer forward (custom input_embeds, NOT input_ids)
        tfmr_out = self.tfmr.forward(
            input_ids=None,
            inputs_embeds=embeds,
            output_hidden_states=True,
            return_dict=True,
            use_cache=(not training),
        )
        hidden_states = tfmr_out.hidden_states[-1]  # Final layer: (B, L_total, 1024)

        # Splice out text and speech hidden states from the full sequence
        # Layout: [cond(34) | text(L_text) | speech(L_speech)]
        len_text = text_tokens.size(1)
        len_speech = speech_tokens.size(1)
        B, _, dim = hidden_states.shape
        device, dtype = hidden_states.device, hidden_states.dtype
        text_latents = torch.zeros(B, len_text, dim, dtype=dtype, device=device)
        speech_latents = torch.zeros(B, len_speech, dim, dtype=dtype, device=device)
        ttl, stl = text_token_lens, speech_token_lens
        for i in range(B):
            text_end = len_cond + ttl[i].item()                    # end of text region
            speech_start = len_cond + text_tokens.size(1)          # start of speech region
            speech_end = speech_start + stl[i].item()              # end of speech region
            text_latents[i, :ttl[i]] = hidden_states[i, len_cond:text_end]
            speech_latents[i, :stl[i]] = hidden_states[i, speech_start:speech_end]

        # Project to logits
        text_logits = self.text_head(text_latents)      # (B, L_text, text_vocab)
        speech_logits = self.speech_head(speech_latents) # (B, L_speech, 8194)

        return AttrDict(
            text_logits=text_logits,
            text_latents=text_latents,
            speech_logits=speech_logits,
            speech_latents=speech_latents,
            hidden_states=hidden_states,
        )

    def loss(
        self,
        *,
        t3_cond: T3Cond,
        text_tokens: torch.LongTensor,
        text_token_lens: torch.LongTensor,
        speech_tokens: torch.LongTensor,
        speech_token_lens: torch.LongTensor,
    ):
        """Compute training loss (text + speech cross-entropy).

        Args:
            t3_cond:           T3Cond conditioning
            text_tokens:       (B, L_text) long — padded to max length in batch
            text_token_lens:   (B,) long — unpadded lengths
            speech_tokens:     (B, L_speech) long — padded to max length in batch
            speech_token_lens: (B,) long — unpadded lengths

        Returns:
            loss_text:   scalar — cross-entropy on text logits (auxiliary)
            loss_speech: scalar — cross-entropy on speech logits (primary)

        NOTE: Padding positions are masked with IGNORE_ID=-100 (PyTorch ignore_index).
        Text logits use F.cross_entropy with shape (B, vocab, L) convention (no transpose needed
        as the original code had a bug — see usage notes).
        """
        len_text = text_tokens.size(1)
        len_speech = speech_tokens.size(1)
        assert len_text == text_token_lens.max()
        assert len_speech == speech_token_lens.max()

        out = self.forward(
            t3_cond=t3_cond,
            text_tokens=text_tokens,
            text_token_lens=text_token_lens,
            speech_tokens=speech_tokens,
            speech_token_lens=speech_token_lens,
            training=True,
        )  # (B, seq, vocab_size)

        # Calc CCE losses
        IGNORE_ID = -100
        device = out.text_logits.device
        mask_text = torch.arange(len_text, device=device)[None] >= text_token_lens[:, None]  # (B, len_text)
        mask_speech = torch.arange(len_speech, device=device)[None] >= speech_token_lens[:, None]  # (B, len_speech)
        masked_text = text_tokens.masked_fill(mask_text, IGNORE_ID)
        masked_speech = speech_tokens.masked_fill(mask_speech, IGNORE_ID)
        loss_text = F.cross_entropy(out.text_logits, masked_text, ignore_index=IGNORE_ID)
        loss_speech = F.cross_entropy(out.speech_logits, masked_speech, ignore_index=IGNORE_ID)

        return loss_text, loss_speech

    @torch.inference_mode()
    def inference(
        self,
        *,
        t3_cond: T3Cond,
        text_tokens: Tensor,
        initial_speech_tokens: Optional[Tensor]=None,

        # misc conditioning
        prepend_prompt_speech_tokens: Optional[Tensor]=None,

        # HF generate args
        num_return_sequences=1,
        max_new_tokens=None,
        stop_on_eos=True,
        do_sample=True,
        temperature=0.8,
        top_p=0.95,
        min_p=0.05,
        length_penalty=1.0,
        repetition_penalty=1.2,
        cfg_weight=0.5,
    ):
        """Autoregressive speech token generation with KV-cache.

        Generates speech tokens one at a time using the Llama backbone with
        Classifier-Free Guidance (CFG) and various sampling strategies.

        Args:
            t3_cond:         T3Cond — conditioning (speaker_emb(1,256), speech_tokens(1,150), emotion)
            text_tokens:     (L_text,) or (B, L_text) long — text tokens with start=255, stop=0.
                             For CFG: (2, L_text) where batch[0]=conditional, batch[1]=copy.
            initial_speech_tokens: Optional (B, L_init) — starting speech tokens (default: [6561] start token)
            max_new_tokens:  int — max generated tokens (default: hp.max_speech_tokens = 4096)
            temperature:     float — sampling temperature (default 0.8)
            cfg_weight:      float — CFG strength (default 0.5). logits = cond + w*(cond - uncond)
            repetition_penalty: float — penalty for repeated tokens (default 1.2, multilingual uses 2.0)
            min_p:           float — minimum probability filter (default 0.05)
            top_p:           float — nucleus sampling threshold (default 0.95)

        Returns:
            predicted_tokens: (1, N) long — generated speech token IDs (NOT including start token).
                              Includes EOS token (6562) at the end if generation stopped normally.
                              N <= max_new_tokens.

        Generation loop:
            1. Build initial embeds: [cond(34) | text(L) | speech_start(1)] → (2, 34+L+1, 1024)
            2. Initial forward pass → logits + KV cache
            3. For each step:
                a. CFG: logits = cond_logits + cfg_weight * (cond_logits - uncond_logits)
                b. AlignmentStreamAnalyzer (multilingual): detect repetition/long-tail → force EOS
                c. Repetition penalty on generated token history
                d. Temperature scaling
                e. min_p + top_p filtering
                f. Sample next token from softmax(logits)
                g. Stop if EOS (6562) generated
                h. Embed new token + position → next KV-cache forward pass

        AlignmentStreamAnalyzer (multilingual only):
            Monitors attention weights at layer 9 to detect:
            - Long-tail silence (attention drifts past text)
            - Alignment repetition (stuck in a loop)
            - Token repetition (same token 2x in a row)
            Forces EOS token when any of these are detected.
        """
        # Validate / sanitize inputs
        assert prepend_prompt_speech_tokens is None, "not implemented"
        _ensure_BOT_EOT(text_tokens, self.hp)
        text_tokens = torch.atleast_2d(text_tokens).to(dtype=torch.long, device=self.device)

        # Default initial speech to a single start-of-speech token
        if initial_speech_tokens is None:
            initial_speech_tokens = self.hp.start_speech_token * torch.ones_like(text_tokens[:, :1])

        # Prepare custom input embeds
        embeds, len_cond = self.prepare_input_embeds(
            t3_cond=t3_cond,
            text_tokens=text_tokens,
            speech_tokens=initial_speech_tokens,
            cfg_weight=cfg_weight,
        )

        # In order to use the standard HF generate method, we need to extend some methods to inject our custom logic
        # Note the llama-specific logic. Other tfmr types can be added later.

        self.compiled = False

        # TODO? synchronize the expensive compile function
        # with self.compile_lock:
        if not self.compiled:
            # Default to None for English models, only create for multilingual
            alignment_stream_analyzer = None
            if self.hp.is_multilingual:
                alignment_stream_analyzer = AlignmentStreamAnalyzer(
                    self.tfmr,
                    None,
                    text_tokens_slice=(len_cond, len_cond + text_tokens.size(-1)),
                    alignment_layer_idx=9, # TODO: hparam or something?
                    eos_idx=self.hp.stop_speech_token,
                )
                assert alignment_stream_analyzer.eos_idx == self.hp.stop_speech_token

            patched_model = T3HuggingfaceBackend(
                config=self.cfg,
                llama=self.tfmr,
                speech_enc=self.speech_emb,
                speech_head=self.speech_head,
                alignment_stream_analyzer=alignment_stream_analyzer,
            )
            self.patched_model = patched_model
            self.compiled = True

        # # Run normal generate method, which calls our custom extended methods
        # return self.patched_model.generate(
        #     inputs=initial_speech_tokens,
        #     decoder_cond=embeds,
        #     bos_token_id=self.hp.start_speech_token,
        #     eos_token_id=(self.hp.stop_speech_token if stop_on_eos else -1),
        #     pad_token_id=self.hp.stop_speech_token,
        #     max_new_tokens=max_new_tokens or self.hp.max_speech_tokens,
        #     num_return_sequences=num_return_sequences,
        #     temperature=temperature,
        #     min_p=min_p,
        #     length_penalty=length_penalty,
        #     repetition_penalty=repetition_penalty,
        #     do_sample=do_sample,
        #     # cache_implementation=None if not self.compiled else "static",
        # )

        device = embeds.device

        bos_token = torch.tensor([[self.hp.start_speech_token]], dtype=torch.long, device=device)
        bos_embed = self.speech_emb(bos_token)  # shape: (B, 1, embed_dim)
        bos_embed = bos_embed + self.speech_pos_emb.get_fixed_embedding(0)

        # batch_size=2 for CFG
        bos_embed = torch.cat([bos_embed, bos_embed])

        # Combine condition and BOS token for the initial input
        inputs_embeds = torch.cat([embeds, bos_embed], dim=1)

        # Track generated token ids; start with the BOS token.
        generated_ids = bos_token.clone()
        predicted = []  # To store the predicted tokens

        # Instantiate the logits processors.
        top_p_warper = TopPLogitsWarper(top_p=top_p)
        min_p_warper = MinPLogitsWarper(min_p=min_p)
        top_p_warper = TopPLogitsWarper(top_p=top_p)
        repetition_penalty_processor = RepetitionPenaltyLogitsProcessor(penalty=float(repetition_penalty))

        # ---- Initial Forward Pass (no kv_cache yet) ----
        output = self.patched_model(
            inputs_embeds=inputs_embeds,
            past_key_values=None,
            use_cache=True,
            output_attentions=True,
            output_hidden_states=True,
            return_dict=True,
        )
        # Initialize kv_cache with the full context.
        past = output.past_key_values

        # ---- Generation Loop using kv_cache ----
        for i in tqdm(range(max_new_tokens), desc="Sampling", dynamic_ncols=True):
            logits_step = output.logits[:, -1, :]
            # CFG combine  → (1, V)
            cond   = logits_step[0:1, :]
            uncond = logits_step[1:2, :]
            cfg = torch.as_tensor(cfg_weight, device=cond.device, dtype=cond.dtype)
            logits = cond + cfg * (cond - uncond)
            
            # Apply alignment stream analyzer integrity checks
            if self.patched_model.alignment_stream_analyzer is not None:
                if logits.dim() == 1:            # guard in case something upstream squeezed
                    logits = logits.unsqueeze(0) # (1, V)
                # Pass the last generated token for repetition tracking
                last_token = generated_ids[0, -1].item() if len(generated_ids[0]) > 0 else None
                logits = self.patched_model.alignment_stream_analyzer.step(logits, next_token=last_token)  # (1, V)

            # Apply repetition penalty
            ids_for_proc = generated_ids[:1, ...]   # batch = 1
            logits = repetition_penalty_processor(ids_for_proc, logits)  # expects (B,V)
            
            # Apply temperature scaling.
            if temperature != 1.0:
                logits = logits / temperature
                
            # Apply min_p and top_p filtering
            logits = min_p_warper(ids_for_proc, logits)
            logits = top_p_warper(ids_for_proc, logits)

            # Convert logits to probabilities and sample the next token.
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # shape: (B, 1)

            predicted.append(next_token)
            generated_ids = torch.cat([generated_ids, next_token], dim=1)

            # Check for EOS token.
            if next_token.view(-1) == self.hp.stop_speech_token:
                logger.info(f"✅ EOS token detected! Stopping generation at step {i+1}")
                break

            # Get embedding for the new token.
            next_token_embed = self.speech_emb(next_token)
            next_token_embed = next_token_embed + self.speech_pos_emb.get_fixed_embedding(i + 1)

            #  For CFG
            next_token_embed = torch.cat([next_token_embed, next_token_embed])

            # Forward pass with only the new token and the cached past.
            output = self.patched_model(
                inputs_embeds=next_token_embed,
                past_key_values=past,
                output_attentions=True,
                output_hidden_states=True,
                return_dict=True,
            )
            # Update the kv_cache.
            past = output.past_key_values

        # Concatenate all predicted tokens along the sequence dimension.
        predicted_tokens = torch.cat(predicted, dim=1)  # shape: (B, num_tokens)
        return predicted_tokens

    @torch.inference_mode()
    def inference_turbo(self, t3_cond, text_tokens, temperature=0.8, top_k=1000, top_p=0.95, repetition_penalty=1.2,
                        max_gen_len=1000):

        logits_processors = LogitsProcessorList()
        if temperature > 0 and temperature != 1.0:
            logits_processors.append(TemperatureLogitsWarper(temperature))
        if top_k > 0:
            logits_processors.append(TopKLogitsWarper(top_k))
        if top_p < 1.0:
            logits_processors.append(TopPLogitsWarper(top_p))
        if repetition_penalty != 1.0:
            logits_processors.append(RepetitionPenaltyLogitsProcessor(repetition_penalty))


        speech_start_token = self.hp.start_speech_token * torch.ones_like(text_tokens[:, :1])
        embeds, _ = self.prepare_input_embeds(
            t3_cond=t3_cond,
            text_tokens=text_tokens,
            speech_tokens=speech_start_token,
            cfg_weight=0.0,
        )

        generated_speech_tokens = []

        llm_outputs = self.tfmr(
            inputs_embeds=embeds,
            use_cache=True
        )

        hidden_states = llm_outputs[0]
        past_key_values = llm_outputs.past_key_values

        speech_hidden = hidden_states[:, -1:]
        speech_logits = self.speech_head(speech_hidden)

        processed_logits = logits_processors(speech_start_token, speech_logits[:, -1, :])
        probs = F.softmax(processed_logits, dim=-1)
        next_speech_token = torch.multinomial(probs, num_samples=1)

        generated_speech_tokens.append(next_speech_token)
        current_speech_token = next_speech_token

        for _ in tqdm(range(max_gen_len)):
            current_speech_embed = self.speech_emb(current_speech_token)

            llm_outputs = self.tfmr(
                inputs_embeds=current_speech_embed,
                past_key_values=past_key_values,
                use_cache=True
            )

            hidden_states = llm_outputs[0]
            past_key_values = llm_outputs.past_key_values
            speech_logits = self.speech_head(hidden_states)

            input_ids = torch.cat(generated_speech_tokens, dim=1)
            processed_logits = logits_processors(input_ids, speech_logits[:, -1, :])
            if torch.all(processed_logits == -float("inf")):
                print("Warning: All logits are -inf")
                break

            probs = F.softmax(processed_logits, dim=-1)
            next_speech_token = torch.multinomial(probs, num_samples=1)

            generated_speech_tokens.append(next_speech_token)
            current_speech_token = next_speech_token
            if torch.all(next_speech_token == self.hp.stop_speech_token):
                break

        all_tokens = torch.cat(generated_speech_tokens, dim=1)

        # Remove EOS token if present
        if all_tokens.size(1) > 0 and all_tokens[0, -1] == self.hp.stop_speech_token:
            all_tokens = all_tokens[:, :-1]

        return all_tokens
