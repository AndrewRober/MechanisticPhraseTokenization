from dataclasses import dataclass


@dataclass(frozen=True)
class TokenizerSpec:
    total_vocab_size: int = 16_384
    base_vocab_size: int = 14_336
    phrase_vocab_slots: int = 2_048  # 16,384 - 14,336 = 2,048 (12.5%)
    num_byte_tokens: int = 256
    special_tokens: tuple[str, ...] = ("<|endoftext|>", "<|pad|>")
    min_phrase_words: int = 2
    max_phrase_words: int = 4
    max_phrase_chars: int = 48
    min_doc_freq: int = 25
    reserve_rel_floor: float = 0.20
    reserve_abs_saturation: int = 2_000


@dataclass(frozen=True)
class ModelSpec:
    vocab_size: int = 16_384
    n_layers: int = 7
    d_model: int = 512
    n_heads: int = 8
    d_ff: int = 2_048
    max_seq_len: int = 1_024
    rope_theta: float = 10_000.0
    rms_norm_eps: float = 1e-5
    tie_embeddings: bool = True
    dropout: float = 0.0

    @property
    def head_dim(self) -> int:
        assert self.d_model % self.n_heads == 0
        return self.d_model // self.n_heads


@dataclass(frozen=True)
class TrainSpec:
    peak_lr: float = 6e-4
    min_lr: float = 6e-5
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    warmup_fraction: float = 0.05
    seq_len: int = 1_024
    # Screening & confirmation budgets (defined in B1 reference BPE tokens of raw text)
    pilot_ref_tokens: int = 100_000_000
    stage_b_ref_tokens: int = 120_000_000
    stage_c_ref_tokens: int = 300_000_000
