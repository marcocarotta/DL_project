import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary
import logging

# Set up logging configuration
logging.basicConfig(level=logging.INFO) 

class CausalSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        assert embed_dim % num_heads == 0, "Embedding size must be divisible by the number of heads"
        
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5  # Scaling factor for attention scores
        
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)  # Query, Key, Value projection
        self.out_proj = nn.Linear(embed_dim, embed_dim)  # Output projection
        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, N, E = x.shape  # Batch size, Sequence length, Embedding size
        qkv = self.qkv(x)  # Shape: (B, N, 3*E)
        q, k, v = qkv.chunk(3, dim=-1)  # Split into query, key, and value tensors
        
        # Reshape for multi-head attention
        q = q.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # (B, num_heads, N, head_dim)
        k = k.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute scaled dot-product attention
        attn_scores = (q @ k.transpose(-2, -1)) * self.scale  # (B, num_heads, N, N)

        # Create causal mask, where the values above the diagonal are set to -inf so softmax will be equal to zero
        mask = torch.tril(torch.ones(N, N, device=x.device)).unsqueeze(0).unsqueeze(0)  # (1, 1, N, N)
        attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
        
        attn_probs = F.softmax(attn_scores, dim=-1)  # Attention probabilities
        attn_probs = self.attn_dropout(attn_probs)
        
        attn_output = (attn_probs @ v).transpose(1, 2).contiguous()  # Combine values
        attn_output = attn_output.view(B, N, E)  # Concatenate heads
        
        return self.proj_dropout(self.out_proj(attn_output))  # Apply final linear layer

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_hid_dim, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.attn = CausalSelfAttention(embed_dim, num_heads, dropout)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, ff_hid_dim),
            nn.GELU(),
            nn.Linear(ff_hid_dim, embed_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        # Causal self-attention with residual connection
        x = x + self.attn(self.ln1(x))
        # Feed-forward network with residual connection
        x = x + self.mlp(self.ln2(x))
        return x

class CharTransformer(nn.Module):
    def __init__(self, vocab_size, block_size, embed_dim, num_heads, num_layers, ff_hid_dim = 3072, dropout=0.1, multitask=False):
        super().__init__()
        # Model hyperparameters
        self.block_size = block_size # Needed for model summary

        # Token embeddings
        self.tok_emb = nn.Embedding(vocab_size, embed_dim)
        # Positional embeddings
        self.pos_emb = nn.Parameter(torch.zeros(1, block_size, embed_dim))
        # Dropout
        self.dropout = nn.Dropout(dropout)
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ff_hid_dim, dropout)
            for _ in range(num_layers)
        ])
        # Final layer normalization
        self.ln_f = nn.LayerNorm(embed_dim)
        # Output projection to vocabulary size (for main task)
        self.lm_head = nn.Linear(embed_dim, vocab_size)
        
        # Auxiliary head for multitask learning (for MLM)
        self.multitask = multitask
        if multitask:
            self.aux_head = nn.Linear(embed_dim, vocab_size)  # Auxiliary head for masked positions

        # logging
                # Logger setup
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.INFO)  # Default level; users can change as needed.
        self.logger.info("Model initialized")
    
    def forward(self, idx, masked_positions=None):
        """
        Forward pass.
        Args:
            idx: Input indices (B, N)
            masked_positions: Binary mask (B, N) indicating positions for auxiliary task (if multitask=True)
        Returns:
            logits: Main task logits (B, N, vocab_size)
            aux_logits: Auxiliary task logits (masked positions only) (optional)
        """
        B, N = idx.shape

        # Token and positional embeddings
        tok_emb = self.tok_emb(idx)  # (B, N, embed_dim)
        pos_emb = self.pos_emb[:, :N, :]  # Slice positional embeddings
        x = self.dropout(tok_emb + pos_emb)

        # Transformer blocks
        for block in self.blocks:
            x = block(x)

        # Final layer normalization
        x = self.ln_f(x)
        logits = self.lm_head(x)  # Main task logits (B, N, vocab_size)

        # Auxiliary logits (only for masked positions)
        if self.multitask and masked_positions is not None:
            aux_logits = self.aux_head(x[masked_positions])  # Predict only for masked positions
            return logits, aux_logits
        else:
            return logits
    
    def summary(self): 
        """
        Print model summary using torchinfo.summary
        A wrapper class is needed to provide a forward pass with single tensor output
        """

        wrapper = CharTransformerSummaryWrapper(self)
        summary_string = summary(
            wrapper,
            input_size=(32,wrapper.original_model.block_size),
            dtypes=[torch.long],
            device='cpu',
            verbose=0 # Suppress print output as it will be redirected to logger
        )
        self.logger.info("### Model summary###\n" + str(summary_string))

class CharTransformerSummaryWrapper(nn.Module):
    """
    Wrapper class to provide a summary method for the CharTransformer model
    It is necessary to wrap the model in a new class to provide a summary method as the torchinfo.summary function requires a forward pass with single tensor output
    """
    def __init__(self, original_model):
        super(CharTransformerSummaryWrapper, self).__init__()
        self.original_model = original_model

    def forward(self, idx):
        if self.original_model.multitask: 
            logits, _ = self.original_model(idx) # Unpack and return only logits
        else:
            logits = self.original_model(idx) # No unpacking needed
        return logits
        

if __name__ == "__main__":
    # Example instantiation
    vocab_size = 65  # Example vocabulary size for the Shakespeare dataset
    block_size = 128  # Sequence length
    embed_dim = 768  # Embedding dimensions
    num_heads = 8  # Number of attention heads
    num_layers = 12  # Number of transformer blocks
    ff_hid_dim = 3072  # Feed-forward hidden dimensions
    dropout = 0.1  # Dropout rate

    # Instantiate the model
    model = CharTransformer(vocab_size, block_size, embed_dim, num_heads, num_layers, ff_hid_dim, dropout, multitask=False)

    # Example forward pass
    idx = torch.randint(0, vocab_size, (32, block_size))  # Random input (batch_size=32)
    logits, _ = model(idx)  # Logits shape: (32, 128, vocab_size)

    # Print model summary 
    model.summary()
