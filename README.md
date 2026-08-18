# nano-GPT — Transformer from Scratch

A minimal PyTorch implementation of the GPT architecture trained on
Shakespeare character-level text. Every component — multi-head self-attention,
positional embeddings, layer norm, the full training loop — implemented by hand
to build a working understanding of how transformers function at the mechanism level.

**Result: 1.82 cross-entropy loss on Shakespeare dataset**

*Inspired by Andrej Karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT).*

---

## What This Implements

| Component                      | Details                                         |
|--------------------------------|-------------------------------------------------|
| Token + positional embeddings  | Learned, added elementwise                      |
| Multi-head self-attention       | Scaled dot-product with causal mask             |
| Layer normalization             | Pre-norm (before attention and before MLP)      |
| Feed-forward blocks             | 2-layer MLP with GELU activation               |
| Dropout                         | At attention weights and residual connections  |
| Training loop                   | AdamW, gradient clipping, cosine LR decay      |

---

## Architecture

```
Input tokens (character-level)
    │
    ▼
Token Embedding + Positional Embedding
    │
    ▼
N × Transformer Blocks:
    ├── LayerNorm
    ├── Multi-Head Self-Attention (causal mask)
    ├── Residual connection
    ├── LayerNorm
    ├── Feed-Forward MLP (4× hidden dim, GELU)
    └── Residual connection
    │
    ▼
LayerNorm → Linear projection (vocab size)
    │
    ▼
Softmax → Next token prediction
```

---

## Results

| Dataset                   | Loss                    |
|---------------------------|-------------------------|
| Shakespeare (char-level)  | **1.82 cross-entropy**  |

---

## Setup

**Prerequisites:** Python 3.8+, PyTorch 2.0+

```bash
git clone https://github.com/Rahulroy5/nano-GPT.git
cd nano-GPT
pip install torch numpy

# Prepare Shakespeare data
python data/shakespeare_char/prepare.py

# Train (GPU)
python train.py config/train_shakespeare_char.py

# Train (CPU / MacBook)
python train.py config/train_shakespeare_char.py \
  --device=cpu --compile=False --block_size=64 \
  --batch_size=12 --n_layer=4 --n_head=4 --n_embd=128 \
  --max_iters=2000

# Generate text from trained model
python sample.py --out_dir=out-shakespeare-char
```

---

## Project Structure

```
nano-GPT/
├── model.py                           # GPT model definition (~300 lines)
├── train.py                           # Training loop
├── sample.py                          # Text generation from checkpoint
├── bench.py                           # Throughput benchmarking
├── configurator.py                    # Config override system
├── config/
│   └── train_shakespeare_char.py      # Shakespeare training config
├── data/
│   └── shakespeare_char/              # Data prep scripts
├── scaling_laws.ipynb                 # Loss vs model size analysis
└── transformer_sizing.ipynb           # Parameter count exploration
```

---

## Notebooks

- **scaling_laws.ipynb** — Empirical analysis of how loss scales with model
  depth, width, and training steps
- **transformer_sizing.ipynb** — How architecture choices affect total parameter
  count

---

## What I Learned

Implementing from scratch clarified things that using libraries obscures: why causal
masking must be applied before softmax (not after), how positional and token
embeddings interact additively in the residual stream, and why pre-norm trains more
stably than post-norm at scale. The scaling notebooks made empirically concrete
why larger models require proportionally more data — a tradeoff that's easy to
state but hard to internalize without seeing the loss curves yourself.
