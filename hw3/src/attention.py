"""
Originally forked from Andrej Karpathy's minGPT,
Modified based on Stanford CS224N 2023-24: Homework 4

John Hewitt <johnhew@stanford.edu>
Ansh Khurana <anshk@stanford.edu>
Soumya Chatterjee <soumyac@stanford.edu>
"""

import math
import logging

import torch
import torch.nn as nn
from torch.nn import functional as F

logger = logging.getLogger(__name__)


def precompute_rotary_emb(dim, max_positions):
    """
    RoPE uses the following sinusoidal functions to encode positions:

    cos(t theta_i) and sin(t theta_i)
        where t is the position and
              theta_i = 1/10000^(-2(i-1)/dim) for i in [1, dim/2]

    Since the maximum length of sequences is known, we can precompute
    these values to speed up training.

    Implement the precompute_rotary_emb function that returns a tensor of
    shape (max_positions, dim/2, 2) where the last dimension contains
    the cos and sin values for each position and each dimension of
    the embedding.
    """

    rope_cache = None
    # TODO: [part g]
    ### YOUR CODE HERE ###
    pass
    ### END YOUR CODE ###
    return rope_cache


def apply_rotary_emb(x, rope_cache):
    """Apply the RoPE to the input tensor x."""
    # TODO: [part g]
    # You might find the following functions useful to convert
    # between real and complex numbers:

    # torch.view_as_real - https://pytorch.org/docs/stable/generated/torch.view_as_real.html
    # torch.view_as_complex - https://pytorch.org/docs/stable/generated/torch.view_as_complex.html

    # Note that during inference, the length of the sequence might be different
    # from the length of the precomputed values. In this case, you should use
    # truncate the precomputed values to match the length of the sequence.

    rotated_x = None
    ### YOUR CODE HERE ###
    pass
    ### END YOUR CODE ###
    return rotated_x

class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    I believe I could have just used torch.nn.MultiheadAttention but their documentation
    is all but absent and code ugly so I don't trust it, rolling my own here.
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads
        self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd)

        self.rope = config.rope
        if self.rope:
            assert (config.n_embd % config.n_head) % 2 == 0

            # TODO: [part g] Precompute the cos and sin values for RoPE and
            # store them in rope_cache.
            # Hint: The maximum sequence length is given by config.block_size.
            rope_cache = None
            ### YOUR CODE HERE ###
            pass
            ### END YOUR CODE ###

            self.register_buffer("rope_cache", rope_cache)

        # regularization
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        # output projection
        self.proj = nn.Linear(config.n_embd, config.n_embd)

        # TODO: causal mask to ensure that attention is only applied to the left in the input sequence
        mask = None
        ### YOUR CODE HERE ###
        mask = torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size)
        ### END YOUR CODE ###

        self.register_buffer("mask", mask)
        self.n_head = config.n_head

    def forward(self, x):
        B, T, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        qkv = self.qkv(x).view(B, T, self.n_head, 3 * (C // self.n_head)).transpose(1, 2) # (B, nh, T, 3*hs)
        q, k, v = qkv.chunk(3, dim=-1)  # each (B, nh, T, hs)

        if self.rope:
            # TODO: [part g] Apply RoPE to the query and key.
            ### YOUR CODE HERE ###
            pass
            ### END YOUR CODE ###

        # TODO: causal self-attention
        # 1. compute attention map (pre-softmax)
        # 2. apply attention mask to the attention map
        # 3. apply softmax to the attention map (hint: masked parts should not be included in the softmax)
        # 4. apply attention dropout to the attention map
        # 5. compute the output by applying the attention map to the value
        # 6. re-assemble all head outputs side by side
        y = None
        ### YOUR CODE HERE ###
        att = (q @ k.transpose(-2, -1)) / (1. / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        ### END YOUR CODE ###

        # output projection
        y = self.resid_drop(self.proj(y))
        return y
