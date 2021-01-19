"""Transformer Self-Attention Types

- Example Usage:

```python
import math
import torchtext

log_2_e = math.log2(math.e)
vocab_size = 30_000
max_seq_len = 512

# Load and preprocess the IMDB dataset.
TEXT = torchtext.data.Field(lower=True, include_lengths=True, batch_first=True)
LABEL = torchtext.data.Field(sequential=False)
train, test = torchtext.datasets.IMDB.splits(TEXT, LABEL)
TEXT.build_vocab(train, max_size=vocab_size - 2)
LABEL.build_vocab(train)

# Convert the train and test samples to batch iterables.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_iter, test_iter = torchtext.data.BucketIterator.splits(
    (train, test), batch_size=12, device=device,
)
if max_seq_len is None:
    max_seq_len = max((x.text[0].size(1) for x in train_iter))
    max_seq_len = max_seq_len * 2
    print(f"Setting max_seq_len to: {max_seq_len}")

model = Transformer(128, vocab_size, num_classes=2, max_seq_len=max_seq_len)
...
# TODO: Finish example with tests.
```
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def mask_(m: torch.Tensor, value=0.0, diagonal=False) -> None:
    _, n_h, n_w = m.shape
    offset = 0 if diagonal else 1
    indices = torch.triu_indices(n_h, n_w, offset=offset)
    m[:, indices[0], indices[1]] = value


class WideSelfAttention(nn.Module):
    def __init__(self, emb_dim: int, n_heads: int, do_mask=False):
        super(WideSelfAttention, self).__init__()
        self.emb_dim = emb_dim
        self.n_heads = n_heads
        self.do_mask = do_mask
        self.K = nn.Linear(emb_dim, emb_dim * n_heads, bias=False)
        self.Q = nn.Linear(emb_dim, emb_dim * n_heads, bias=False)
        self.V = nn.Linear(emb_dim, emb_dim * n_heads, bias=False)
        self.union = nn.Linear(n_heads * emb_dim, emb_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n_h = self.n_heads
        bz, t, d = x.shape
        k = self.K(x).view(bz, t, n_h, d)
        q = self.Q(x).view(bz, t, n_h, d)
        v = self.V(x).view(bz, t, n_h, d)
        # Compute scaled dot product self-attention
        # fold heads into the batch size dimension.
        k = k.transpose(1, 2).contiguous().view(bz * n_h, t, d)
        q = q.transpose(1, 2).contiguous().view(bz * n_h, t, d)
        v = v.transpose(1, 2).contiguous().view(bz * n_h, t, d)
        # Intead of dividing the products by sqrt(dim), we scale the
        # <Keys> and <Values> in order to improve memory efficiency.
        q = q / (d ** (1 / 4))
        k = k / (d ** (1 / 4))
        # Obtain the dot product of <Query> + <Keys> and scale.
        dot = torch.bmm(q, k.transpose(1, 2))
        # Mask out the upper half of the dot matrix, excluding the diagonal.
        if self.do_mask:
            mask_(dot, value=float("-inf"), diagonal=False)
        # Obtain the dot row-wise self attention probabilities.
        dot = F.softmax(dot, dim=2)
        # Apply batch self-attention to the <Values>
        y = torch.bmm(dot, v).view(bz, n_h, t, d)
        # Swap number of heads and tensors back in union (unify heads).
        y = y.transpose(1, 2).contiguous().view(bz, t, n_h * d)
        return self.union(y)


class NarrowSelfAttention(nn.Module):
    def __init__(self, emb_dim: int, n_heads: int, do_mask=False):
        super(NarrowSelfAttention, self).__init__()
        self.emb_dim = emb_dim
        self.n_heads = n_heads
        self.do_mask = do_mask
        chunk = emb_dim // n_heads
        self.K = nn.Linear(chunk, chunk, bias=False)
        self.Q = nn.Linear(chunk, chunk, bias=False)
        self.V = nn.Linear(chunk, chunk, bias=False)
        self.union = nn.Linear(n_heads * chunk, emb_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n_h = self.n_heads
        bz, t, d = x.shape
        chunk = d // n_h
        k = self.K(x)
        q = self.Q(x)
        v = self.V(x)
        # Compute scaled dot product self-attention
        # fold heads into the batch size dimension.
        k = k.transpose(1, 2).contiguous().view(bz * n_h, t, chunk)
        q = q.transpose(1, 2).contiguous().view(bz * n_h, t, chunk)
        v = v.transpose(1, 2).contiguous().view(bz * n_h, t, chunk)
        # Instead of dividing the products by sqrt(dim), we scale the
        # <Keys> and <Values> in order to improve memory efficiency.
        q = q / (d ** (1 / 4))
        k = k / (d ** (1 / 4))
        # Obtain the the batch dot product of <Query> + <Keys> and scale.
        dot = torch.bmm(q, k.transpose(1, 2))
        # Mask out the upper half of the dot matrix, excluding the diagonal.
        if self.do_mask:
            mask_(dot, value=float("-inf"), diagonal=False)
        # Obtain the dot row-wise self-attention probabilities.
        dot = F.softmax(dot, dim=2)
        # Apply batch self-attention to the <Values>
        y = torch.bmm(dot, v).view(bz, n_h, t, chunk)
        # Swap the number of heads and inputs (t) back in union (unify heads).
        y = y.transpose(1, 2).contiguous().view(bz, t, chunk * n_h)
        return self.union(y)


class TransformerBlock(nn.Module):
    def __init__(
        self, emb_dim, n_heads, n_hidden, attn_type="wide", do_mask=False, dropout=0.4
    ):
        super(TransformerBlock, self).__init__()
        if attn_type not in ("wide", "narrow"):
            e = "Invalid attention name, expected `wide` or `narrow`, not: {}"
            raise ValueError(e.format(attn_type))
        if attn_type == "wide":
            self.attention = WideSelfAttention(emb_dim, n_heads, do_mask)
        if attn_type == "narrow":
            self.attention = NarrowSelfAttention(emb_dim, n_heads, do_mask)
        self.do_mask = do_mask
        self.l0_norm = nn.LayerNorm(emb_dim)
        self.l1_norm = nn.LayerNorm(emb_dim)
        self.network = nn.Sequential(
            nn.Linear(emb_dim, n_hidden * emb_dim),
            nn.ReLU(),
            nn.Linear(n_hidden * emb_dim, emb_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn = self.attention(x)
        x = self.l0_norm(attn + x)
        x = self.dropout(x)
        w = self.network(x)
        x = self.l1_norm(w + x)
        x = self.dropout(x)
        return x


class Transformer(nn.Module):
    def __init__(
        self,
        num_embed=128,
        vocab_size=50000,
        num_classes=2,
        max_seq_len=512,
        num_blocks=6,
        num_heads=8,
        num_hidden=4,
        attn_type="wide",
    ):
        super(Transformer, self).__init__()
        self.token_embedding = nn.Embedding(vocab_size, num_embed)
        self.position_embedding = nn.Embedding(max_seq_len, num_embed)
        self.transformer_blocks = nn.Sequential(
            *[
                TransformerBlock(num_embed, num_heads, num_hidden, attn_type)
                for block in range(num_blocks)
            ]
        )
        self.classifier = nn.Linear(num_embed, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        token_wx = self.token_embedding(x)
        bz, t, d = token_wx.shape
        pos_wx = self.position_embedding(torch.arange(t))
        pos_wx = pos_wx[None, :, :].expand(bz, t, d)
        inputs = self.transformer_blocks(token_wx + pos_wx)
        output = self.classifier(inputs.mean(dim=1))
        logits = F.log_softmax(output, dim=1)
        return logits
