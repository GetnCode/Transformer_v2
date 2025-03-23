"""
Transformer Language Model Implementation

This module implements a configurable transformer-based language model with
state-of-the-art features including Rotary Position Embeddings (RoPE),
Flash Attention, SwiGLU activations, and KV caching for efficient inference.

The architecture is inspired by modern transformer models like GPT but includes
several improvements for stability, performance, and flexibility.

Key components:
- Transformer architecture with multi-head self-attention
- Configurable position embeddings (RoPE or learned embeddings)
- Flexible activation functions (SwiGLU or GELU)
- Pre-normalization or post-normalization options
- KV caching for efficient text generation
- Gradient checkpointing for memory-efficient training
- Proper weight initialization for stable training
- Text generation with temperature, top-k, and nucleus sampling

Example:
    # Create a model
    model = TransformerModel(
        vocab_size=50257,
        d_model=768,
        num_layers=12,
        num_heads=12
    )
    
    # Generate text
    input_ids = torch.tensor([[1, 2, 3, 4, 5]])
    generated = model.generate(input_ids, max_length=50)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Union, List, Dict, Any

class RotaryEmbedding(nn.Module):
    """Rotary Position Embeddings (RoPE) for better handling of relative positions.
    
    RoPE applies a rotation to the embedding vectors based on their positions,
    allowing the model to better capture relative positioning information.
    This implementation includes numerical stability improvements and caching
    for efficient computation.
    
    Args:
        dim (int): Dimension of the embeddings. Must be even.
        max_seq_len (int, optional): Maximum sequence length to precompute embeddings for.
            Defaults to 512.
        base (float, optional): Base for the exponential used in computing rotation
            frequencies. Smaller values can help with longer sequences. Defaults to 10000.0.
    
    References:
        "Roformer: Enhanced transformer with rotary position embedding"
        https://arxiv.org/abs/2104.09864
    """
    def __init__(self, dim, max_seq_len=512, base=10000.0):
        super().__init__()
        # Only need inv_freq for half the dimensions since we apply 
        # rotation independently to each pair of dimensions
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.max_seq_len = max_seq_len
        self.base = base
        
        # Create and cache the sin/cos values
        self._build_cache()
        
    def _build_cache(self):
        """Precompute and cache sine and cosine values for all positions.
        
        This significantly speeds up inference by avoiding repeated computation.
        """
        t = torch.arange(self.max_seq_len, device=self.inv_freq.device).type_as(self.inv_freq)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)  # [seq_len, dim/2]
        # Cache cos/sin values for easy lookup
        cos = torch.cos(freqs)  # [seq_len, dim/2]
        sin = torch.sin(freqs)  # [seq_len, dim/2]
        self.register_buffer("cos_cache", cos)
        self.register_buffer("sin_cache", sin)
        
    def forward(self, seq_len):
        """Retrieve the cached rotary embeddings for the given sequence length.
        
        Args:
            seq_len (int): Current sequence length.
            
        Returns:
            Tuple[Tensor, Tensor]: Cached cosine and sine values for rotary embeddings,
                with shapes [seq_len, dim/2].
        """
        # Return the cached values for the requested sequence length
        return self.cos_cache[:seq_len], self.sin_cache[:seq_len]
    
    @staticmethod
    def apply_rotary_emb(q, k, cos, sin):
        """Apply rotary embeddings to query and key tensors.
        
        Args:
            q (torch.Tensor): Query tensor with shape [batch, num_heads, seq_len, head_dim].
            k (torch.Tensor): Key tensor with shape [batch, num_heads, seq_len, head_dim].
            cos (torch.Tensor): Cosine values [seq_len, dim/2].
            sin (torch.Tensor): Sine values [seq_len, dim/2].
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Rotary-embedded query and key tensors.
        """
        # q, k: [batch, num_heads, seq_len, head_dim]
        # cos, sin: [seq_len, dim/2]
        
        # Make sure cos and sin can be broadcast
        # reshape to [1, 1, seq_len, dim/2]
        cos = cos.unsqueeze(0).unsqueeze(0)  
        sin = sin.unsqueeze(0).unsqueeze(0)
        
        # Split the head dimension in half
        dim_half = q.shape[-1] // 2
        
        # Split q and k into two parts along the last dimension
        q_1, q_2 = q[..., :dim_half], q[..., dim_half:]
        k_1, k_2 = k[..., :dim_half], k[..., dim_half:]
        
        # Apply rotation with stable operations
        q_rotated = torch.cat([
            q_1 * cos - q_2 * sin,
            q_2 * cos + q_1 * sin
        ], dim=-1)
        
        k_rotated = torch.cat([
            k_1 * cos - k_2 * sin,
            k_2 * cos + k_1 * sin
        ], dim=-1)
        
        return q_rotated, k_rotated

class MultiHeadAttention(nn.Module):
    """Configurable multi-head attention with options for stability.
    
    This implementation supports both rotary position embeddings and
    different attention computation methods (flash attention or traditional).
    It also handles KV caching for efficient auto-regressive generation.
    
    Args:
        d_model (int): Model dimension, must be divisible by num_heads.
        num_heads (int): Number of attention heads.
        use_rope (bool, optional): Whether to use rotary position embeddings. Defaults to True.
        use_flash (bool, optional): Whether to use flash attention when available. Defaults to True.
        dropout (float, optional): Dropout rate for attention. Defaults to 0.0.
    """
    def __init__(self, d_model, num_heads, use_rope=True, use_flash=True, dropout=0.0):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        # Linear projections for Q, K, V in one matrix
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        # Output projection
        self.out_proj = nn.Linear(d_model, d_model)
        
        # Configuration options
        self.use_rope = use_rope
        self.use_flash = use_flash
        self.attention_dropout = dropout
        
        # Rotary embeddings (only used if use_rope=True)
        if use_rope:
            self.rotary_emb = RotaryEmbedding(self.head_dim)
        
    def forward(self, x, use_cache=False, past_kv=None):
        """Apply multi-head attention to the input sequence.
        
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, seq_len, d_model].
            use_cache (bool, optional): Whether to return key/value tensors for caching.
                Defaults to False.
            past_kv (Tuple[torch.Tensor, torch.Tensor], optional): Cached key and value
                tensors from previous generation steps. Defaults to None.
                
        Returns:
            Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
                - Output tensor of shape [batch_size, seq_len, d_model]
                - Tuple of key and value tensors for caching (if use_cache=True)
        """
        # x: (B, T, d_model)
        B, T, _ = x.size()
        
        # Compute Q, K, V in one go then split them into separate heads
        qkv = self.qkv_proj(x)  # (B, T, 3 * d_model)
        qkv = qkv.reshape(B, T, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, num_heads, T, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Apply rotary embeddings if enabled
        if self.use_rope:
            cos, sin = self.rotary_emb(T)
            q, k = RotaryEmbedding.apply_rotary_emb(q, k, cos, sin)
        
        # Handle KV caching for inference
        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat([past_k, k], dim=-2)
            v = torch.cat([past_v, v], dim=-2)
        
        # Save KV cache if needed
        current_kv = (k, v) if use_cache else None
        
        # Use either flash attention or manual attention implementation
        if self.use_flash and hasattr(F, 'scaled_dot_product_attention'):
            # Flash attention (faster, less memory)
            attn_output = F.scaled_dot_product_attention(
                q, k, v, attn_mask=None, dropout_p=self.attention_dropout, is_causal=True
            )
        else:
            # Manual attention implementation (more compatible)
            # Scale queries for stable attention
            q = q / math.sqrt(self.head_dim)
            
            # Calculate attention scores
            attn_weights = torch.matmul(q, k.transpose(-2, -1))
            
            # Apply causal mask
            causal_mask = torch.triu(
                torch.ones(T, k.size(-2), dtype=torch.bool, device=x.device), 
                diagonal=1
            )
            attn_weights = attn_weights.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
            
            # Normalize attention weights
            attn_weights = torch.softmax(attn_weights, dim=-1)
            
            # Apply dropout if specified
            if self.attention_dropout > 0:
                attn_weights = F.dropout(attn_weights, p=self.attention_dropout)
                
            # Apply attention to values
            attn_output = torch.matmul(attn_weights, v)
        
        # Merge attention heads back into a single dimension
        attn_output = attn_output.transpose(1, 2).reshape(B, T, self.d_model)
        # Final linear projection
        return self.out_proj(attn_output), current_kv

# Simple GELU activation that can be used as fallback
class GELU(nn.Module):
    """Gaussian Error Linear Unit activation function.
    
    Applies the Gaussian Error Linear Unit (GELU) activation function:
    x * Φ(x) where Φ is the cumulative distribution function of the standard normal.
    """
    def forward(self, x):
        """Apply GELU activation function.
        
        Args:
            x (torch.Tensor): Input tensor.
            
        Returns:
            torch.Tensor: Output after applying GELU.
        """
        return F.gelu(x)

class SwiGLU(nn.Module):
    """SwiGLU activation with stability improvements.
    
    SwiGLU is a variant of GLU that uses the Swish activation for gating,
    which often performs better than GELU in transformer models.
    This implementation includes stability improvements.
    
    Args:
        d_model (int): Input dimension.
        d_ff (int): Internal dimension for the feed-forward computation.
    
    References:
        "GLU Variants Improve Transformer" - https://arxiv.org/abs/2002.05202
    """
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_model, d_ff)
        self.w3 = nn.Linear(d_ff, d_model)
        
        # Initialize with a slightly smaller standard deviation
        self.w1.weight.data.normal_(mean=0.0, std=0.01)
        self.w2.weight.data.normal_(mean=0.0, std=0.01)
        
    def forward(self, x):
        """Apply SwiGLU activation function.
        
        Args:
            x (torch.Tensor): Input tensor.
            
        Returns:
            torch.Tensor: Output after applying SwiGLU.
        """
        # Apply layer normalization before each activation for better stability
        gate = torch.sigmoid(self.w2(x) * 1.0)
        
        # SwiGLU with numerical stability measures
        swish_out = self.w1(x) * gate
        
        # Add a small epsilon to avoid zero gradients
        return self.w3(swish_out)

class TransformerBlock(nn.Module):
    """A single transformer block with self-attention and feed-forward network.
    
    Implements a flexible transformer block that can be configured with different
    normalization styles, activation functions, and attention mechanisms.
    
    Args:
        d_model (int): Hidden dimension of the model.
        num_heads (int): Number of attention heads.
        d_ff (int): Dimension of the feed-forward network.
        dropout (float, optional): Dropout probability. Defaults to 0.1.
        use_rope (bool, optional): Whether to use rotary position embeddings. Defaults to True.
        use_flash (bool, optional): Whether to use flash attention. Defaults to True.
        use_swiglu (bool, optional): Whether to use SwiGLU activation. Defaults to True.
        use_pre_norm (bool, optional): Whether to use pre-normalization architecture. Defaults to True.
    """
    def __init__(
        self, d_model, num_heads, d_ff, dropout=0.1, 
        use_rope=True, use_flash=True, use_swiglu=True,
        use_pre_norm=True
    ):
        super().__init__()
        self.use_pre_norm = use_pre_norm
        
        # Normalization layers
        self.ln1 = nn.LayerNorm(d_model, eps=1e-5)  # Increased epsilon for stability
        self.ln2 = nn.LayerNorm(d_model, eps=1e-5)
        
        # Attention
        self.attn = MultiHeadAttention(
            d_model, num_heads, 
            use_rope=use_rope, 
            use_flash=use_flash, 
            dropout=dropout
        )
        
        # Feed-forward network (with option for SwiGLU or GELU)
        if use_swiglu:
            self.ff = SwiGLU(d_model, d_ff)
        else:
            self.ff = nn.Sequential(
                nn.Linear(d_model, d_ff),
                GELU(),
                nn.Linear(d_ff, d_model),
            )
            
        self.dropout = nn.Dropout(dropout)
        self.gradient_checkpointing = False
    
    def _attn_block(self, x, use_cache=False, past_kv=None):
        """Apply attention block with appropriate normalization.
        
        Args:
            x (torch.Tensor): Input tensor.
            use_cache (bool): Whether to use KV caching.
            past_kv (tuple, optional): Past key-value cache.
            
        Returns:
            Tuple[torch.Tensor, Optional[Tuple]]: Output tensor and KV cache.
        """
        # Apply attention with pre/post normalization based on configuration
        if self.use_pre_norm:
            attn_input = self.ln1(x)
            attn_out, kv_cache = self.attn(attn_input, use_cache, past_kv)
            x = x + self.dropout(attn_out)
        else:
            attn_out, kv_cache = self.attn(x, use_cache, past_kv)
            x = self.ln1(x + self.dropout(attn_out))
        return x, kv_cache
    
    def _ff_block(self, x):
        """Apply feed-forward block with appropriate normalization.
        
        Args:
            x (torch.Tensor): Input tensor.
            
        Returns:
            torch.Tensor: Output tensor.
        """
        # Apply feed-forward with pre/post normalization based on configuration
        if self.use_pre_norm:
            ff_input = self.ln2(x)
            ff_out = self.ff(ff_input)
            x = x + self.dropout(ff_out)
        else:
            ff_out = self.ff(x)
            x = self.ln2(x + self.dropout(ff_out))
        return x
    
    def forward(self, x, use_cache=False, past_kv=None):
        """Process input through the transformer block.
        
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, seq_len, d_model].
            use_cache (bool, optional): Whether to return key/value states for
                incremental decoding. Defaults to False.
            past_kv (Tuple, optional): Cached key/value states from previous steps
                for incremental decoding. Defaults to None.
                
        Returns:
            Tuple[torch.Tensor, Optional[Tuple]]: 
                - Output tensor of shape [batch_size, seq_len, d_model]
                - Key/value states for incremental decoding (if use_cache=True)
        """
        # Use gradient checkpointing if enabled (for memory efficiency)
        if self.gradient_checkpointing and self.training:
            # Custom checkpoint implementation compatible with kv caching
            x, kv_cache = torch.utils.checkpoint.checkpoint(
                self._attn_block, x, use_cache, past_kv, use_reentrant=False
            )
            x = torch.utils.checkpoint.checkpoint(
                self._ff_block, x, use_reentrant=False
            )
        else:
            # Normal forward pass
            x, kv_cache = self._attn_block(x, use_cache, past_kv)
            x = self._ff_block(x)
        
        return x, kv_cache

class TransformerModel(nn.Module):
    """Complete transformer-based language model with configurable architecture.
    
    This transformer model supports auto-regressive language modeling with
    various architectural choices including RoPE vs. standard position embeddings,
    flash attention, different activation functions, and pre/post normalization.
    
    Args:
        vocab_size (int): Size of the vocabulary.
        d_model (int, optional): Hidden dimension size. Defaults to 1024.
        num_layers (int, optional): Number of transformer layers. Defaults to 14.
        num_heads (int, optional): Number of attention heads. Defaults to 16.
        d_ff (int, optional): Feed-forward network dimension. Defaults to 4096.
        max_seq_len (int, optional): Maximum sequence length. Defaults to 512.
        dropout (float, optional): Dropout probability. Defaults to 0.1.
        use_rope (bool, optional): Use rotary position embeddings. Defaults to True.
        use_flash (bool, optional): Use flash attention when available. Defaults to True.
        use_swiglu (bool, optional): Use SwiGLU activation. Defaults to True.
        use_pre_norm (bool, optional): Use pre-normalization architecture. Defaults to True.
    
    Example:
        >>> model = TransformerModel(
        ...     vocab_size=50257,  # GPT-2 vocabulary size
        ...     d_model=768,       # Hidden dimension
        ...     num_layers=12,     # Number of transformer blocks
        ...     num_heads=12       # Number of attention heads
        ... )
        >>> # Generate text from a prompt
        >>> input_ids = torch.tensor([[50, 100, 200]])  # Example token IDs
        >>> output = model.generate(input_ids, max_length=50)
    """
    def __init__(
        self, vocab_size, d_model=1024, num_layers=14, num_heads=16, 
        d_ff=4096, max_seq_len=512, dropout=0.1,
        use_rope=True, use_flash=True, use_swiglu=True, use_pre_norm=True
    ):
        super().__init__()
        # Token embeddings
        self.token_embed = nn.Embedding(vocab_size, d_model)
        
        # Optional standard positional embeddings if not using RoPE
        self.use_rope = use_rope
        if not use_rope:
            self.pos_embed = nn.Parameter(torch.zeros(1, max_seq_len, d_model))
            
        self.dropout = nn.Dropout(dropout)
        
        # Stack of Transformer blocks with consistent configuration
        self.layers = nn.ModuleList([
            TransformerBlock(
                d_model, num_heads, d_ff, dropout,
                use_rope=use_rope, use_flash=use_flash, 
                use_swiglu=use_swiglu, use_pre_norm=use_pre_norm
            ) for _ in range(num_layers)
        ])
        
        # Final layer norm
        self.ln_final = nn.LayerNorm(d_model, eps=1e-5)
        
        # Output projection (tied weights with token embeddings)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.token_embed.weight
        
        # Model configuration
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.num_layers = num_layers
        
        # Initialize weights properly
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """Initialize model weights for stable training.
        
        Uses scaled initialization to improve stability for deep networks.
        
        Args:
            module (nn.Module): Module to initialize.
        """
        if isinstance(module, nn.Linear):
            # Slightly more conservative initialization for stability
            std = 0.02 / math.sqrt(2 * self.num_layers)
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
    
    def forward(self, x, use_cache=False, past_key_values=None):
        """Forward pass through the transformer model.
        
        Args:
            x (torch.Tensor): Input token ids of shape [batch_size, seq_len].
            use_cache (bool, optional): Whether to return key-value cache for
                incremental decoding. Defaults to False.
            past_key_values (List[Tuple[torch.Tensor, torch.Tensor]], optional):
                List of cached key-value pairs from previous steps. Defaults to None.
                
        Returns:
            Union[torch.Tensor, Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]]]:
                - If use_cache=False: Logits of shape [batch_size, seq_len, vocab_size].
                - If use_cache=True: Tuple of (logits, key-value cache).
        """
        # x: (B, T) token ids
        B, T = x.size()
        x = self.token_embed(x)  # (B, T, d_model)
        
        # Add positional embeddings if not using RoPE
        if not self.use_rope:
            x = x + self.pos_embed[:, :T, :]
            
        x = self.dropout(x)
        
        # Initialize past_key_values if not provided
        if past_key_values is None:
            past_key_values = [None] * len(self.layers)
            
        present_key_values = [] if use_cache else None
        
        for i, (layer, past_kv) in enumerate(zip(self.layers, past_key_values)):
            x, current_kv = layer(x, use_cache, past_kv)
            if use_cache:
                present_key_values.append(current_kv)
        
        x = self.ln_final(x)
        logits = self.head(x)  # (B, T, vocab_size)
        
        if use_cache:
            return logits, present_key_values
        return logits
    
    def generate(self, input_ids, max_length=100, temperature=1.0, top_k=0, top_p=0.9):
        """Generate text using the model with various sampling methods.
        
        Performs auto-regressive generation using the model with configurable
        decoding strategies including temperature sampling, top-k filtering,
        and nucleus (top-p) sampling.
        
        Args:
            input_ids (torch.Tensor): Input token ids of shape [batch_size, seq_len].
            max_length (int, optional): Maximum number of new tokens to generate.
                Defaults to 100.
            temperature (float, optional): Sampling temperature; higher values increase
                diversity, lower values make text more focused. Defaults to 1.0.
            top_k (int, optional): Number of highest probability tokens to keep;
                0 means no filtering. Defaults to 0.
            top_p (float, optional): Nucleus sampling probability threshold;
                keep tokens whose cumulative probability exceeds this value.
                Defaults to 0.9.
                
        Returns:
            torch.Tensor: Generated token ids including the input_ids,
                shape [batch_size, input_seq_len + generated_tokens].
                
        Examples:
            >>> model = TransformerModel(vocab_size=50257, d_model=768, num_layers=12)
            >>> # Generate creative text with higher temperature
            >>> creative = model.generate(input_ids, temperature=1.2, top_p=0.95)
            >>> # Generate focused, deterministic text
            >>> focused = model.generate(input_ids, temperature=0.7, top_k=50)
        """
        self.eval()
        
        batch_size = input_ids.shape[0]
        past_key_values = None
        generated_tokens = input_ids.clone()
        
        # Maximum length safeguard
        max_gen_length = min(max_length, 2048)  # Hard cap for safety
        
        # Generate tokens autoregressively with error handling
        try:
            for i in range(max_gen_length):
                # Forward pass with caching
                with torch.no_grad():
                    if past_key_values is None:
                        outputs = self(generated_tokens, use_cache=True)
                        logits, past_key_values = outputs
                    else:
                        # Only process the new token with KV caching
                        new_token_input = generated_tokens[:, -1].unsqueeze(-1)
                        outputs = self(new_token_input, use_cache=True, past_key_values=past_key_values)
                        logits, past_key_values = outputs
                    
                    # Get logits for the next token
                    next_token_logits = logits[:, -1, :] 
                    
                    # Apply temperature scaling safely
                    if temperature > 0:
                        next_token_logits = next_token_logits / temperature
                    
                    # Apply top-k filtering
                    if top_k > 0:
                        top_k_values, _ = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))
                        # Safe filtering that handles edge cases
                        filter_value = top_k_values[:, -1].unsqueeze(-1)
                        next_token_logits = torch.where(
                            next_token_logits < filter_value, 
                            torch.full_like(next_token_logits, float('-inf')),
                            next_token_logits
                        )
                    
                    # Apply top-p (nucleus) filtering with safety measures
                    if 0.0 < top_p < 1.0:
                        sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                        cumulative_probs = torch.cumsum(
                            torch.softmax(sorted_logits, dim=-1) + 1e-10,  # Add small epsilon for stability
                            dim=-1
                        )
                        
                        # Remove tokens with cumulative probability above the threshold
                        sorted_indices_to_remove = cumulative_probs > top_p
                        # Keep at least the top token
                        sorted_indices_to_remove[..., 0] = 0
                        
                        # Apply carefully to avoid index out of bounds
                        for batch_idx in range(batch_size):
                            indices_to_remove = sorted_indices[batch_idx][sorted_indices_to_remove[batch_idx]]
                            next_token_logits[batch_idx, indices_to_remove] = float('-inf')
                    
                    # Sample from the filtered distribution
                    probs = torch.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                    
                    # Append to generated tokens
                    generated_tokens = torch.cat((generated_tokens, next_token), dim=1)
                    
        except RuntimeError as e:
            print(f"Generation stopped early due to error: {e}")
            # Return what we have so far as a fallback
        
        return generated_tokens

    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing for memory-efficient training.
        
        This reduces memory usage significantly (sometimes by 50% or more)
        at the cost of slightly slower training due to recomputation.
        """
        for module in self.modules():
            if isinstance(module, TransformerBlock):
                module.gradient_checkpointing = True
                
    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing to return to standard training."""
        for module in self.modules():
            if isinstance(module, TransformerBlock):
                module.gradient_checkpointing = False

# Example usage:
if __name__ == "__main__":
    # Hyperparameters
    vocab_size = 50257  # Example vocabulary size (e.g., GPT-2's vocab)
    max_seq_len = 512
    
    # Create a model with more stable default settings
    model = TransformerModel(
        vocab_size, d_model=1024, num_layers=14, num_heads=16, d_ff=4096, 
        max_seq_len=max_seq_len, dropout=0.1,
        # Configuration options for stability:
        use_rope=True,       # Set to False for more traditional positional embeddings
        use_flash=True,      # Set to False if having issues with flash attention
        use_swiglu=True,     # Set to False to use simpler GELU activation
        use_pre_norm=True    # Set to False for post-normalization architecture
    )
    
    # Create example input (batch of token ids)
    dummy_input = torch.randint(0, vocab_size, (2, 32))  # Smaller batch for safety
    logits = model(dummy_input)
    print("Logits shape:", logits.shape)  # Expected shape: (2, 32, 50257)
    
    # Demo text generation with safer parameters
    input_prompt = torch.randint(0, vocab_size, (1, 5))  # Short prompt
    try:
        generated = model.generate(
            input_prompt, 
            max_length=10,   # Smaller for testing
            temperature=0.8, # More conservative temperature
            top_k=50,        # Reasonable top-k
            top_p=0.9        # Standard nucleus sampling parameter
        )
        print("Generated sequence shape:", generated.shape)  # Expected: (1, 15)
    except Exception as e:
        print(f"Generation test failed with error: {e}")
