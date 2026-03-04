"""
T3 Model Configuration.

Defines all hyperparameters for the T3 (Text-to-Token) autoregressive model.
Two variants: English-only (text_tokens_dict_size=704) and multilingual (2454).

Key dimensions
--------------
    n_channels = 1024          — Transformer hidden size (from Llama_520M config)
    speech_tokens_dict_size = 8194  — Speech embedding vocab (6561 valid + SOS + EOS + padding)
    text_tokens_dict_size = 704 (EN) or 2454 (MTL) — Text embedding vocab
    speaker_embed_size = 256   — VoiceEncoder output dimension
    speech_cond_prompt_len = 150 — Reference speech tokens (6s × 25 tok/s)

Token ID assignments
--------------------
    Speech tokens:  0-6560     — Valid S3Tokenizer output (SPEECH_VOCAB_SIZE=6561)
    SOS:            6561       — Start-of-speech token
    EOS:            6562       — End-of-speech token
    Padding:        6563-8193  — Unused padding range

    Text SOT:       255        — Start-of-text delimiter
    Text EOT:       0          — End-of-text delimiter

Backbone
--------
    Llama_520M: 24 layers, 1024 hidden, 16 heads, 64 head_dim, 4096 intermediate
    ~536M total parameters (520M backbone + embeddings + conditioning)
"""
from ..llama_configs import LLAMA_CONFIGS


class T3Config:
    """Configuration for the T3 autoregressive text-to-speech-token model.

    Attributes
    ----------
    start_text_token : int = 255
        SOT (start-of-text) delimiter prepended to text token sequence.
    stop_text_token : int = 0
        EOT (end-of-text) delimiter appended to text token sequence.
    text_tokens_dict_size : int
        Text embedding vocabulary size. 704 for English, 2454 for multilingual.
    max_text_tokens : int = 2048
        Maximum text sequence length (including SOT/EOT).
    start_speech_token : int = 6561
        SOS (start-of-speech) token. Also = SPEECH_VOCAB_SIZE.
    stop_speech_token : int = 6562
        EOS (end-of-speech) token. Signals generation completion.
    speech_tokens_dict_size : int = 8194
        Speech embedding vocabulary size (valid tokens + special + padding).
    max_speech_tokens : int = 4096
        Maximum speech token sequence length for generation.
    llama_config_name : str = "Llama_520M"
        Backbone config key → {hidden_size=1024, num_layers=24, num_heads=16}.
    input_pos_emb : str = "learned"
        Position embedding type. "learned" = LearnedPositionalEmbedding.
    speech_cond_prompt_len : int = 150
        Number of reference speech tokens for conditioning (6s × 25 tok/s).
    encoder_type : str = "voice_encoder"
        Speaker encoder type. Only "voice_encoder" (3-layer LSTM) is implemented.
    speaker_embed_size : int = 256
        VoiceEncoder output dimension.
    use_perceiver_resampler : bool = True
        Whether to use Perceiver to compress 150 speech tokens → 32 conditioning tokens.
    emotion_adv : bool = True
        Whether to use emotion exaggeration conditioning.

    Properties
    ----------
    n_channels : int
        Transformer hidden size from Llama config (1024 for Llama_520M).
    is_multilingual : bool
        True if text_tokens_dict_size == 2454 (multilingual model).
    """
    def __init__(self, text_tokens_dict_size=704):
        self.start_text_token = 255
        self.stop_text_token = 0
        self.text_tokens_dict_size = text_tokens_dict_size
        self.max_text_tokens = 2048

        self.start_speech_token = 6561
        self.stop_speech_token = 6562
        self.speech_tokens_dict_size = 8194
        self.max_speech_tokens = 4096

        self.llama_config_name = "Llama_520M"
        self.input_pos_emb = "learned"
        self.speech_cond_prompt_len = 150

        self.encoder_type = "voice_encoder"
        self.speaker_embed_size = 256
        self.use_perceiver_resampler = True
        self.emotion_adv = True

    @property
    def n_channels(self):
        return LLAMA_CONFIGS[self.llama_config_name]["hidden_size"]
    
    @property
    def is_multilingual(self):
        return self.text_tokens_dict_size == 2454

    @classmethod
    def english_only(cls):
        """Create configuration for English-only TTS model."""
        return cls(text_tokens_dict_size=704)
    
    @classmethod 
    def multilingual(cls):
        """Create configuration for multilingual TTS model."""
        return cls(text_tokens_dict_size=2454)
