"""
Small Language Model (SLM) baseado em Transformer
Arquitetura simples mas eficiente para aprendizado local
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadAttention(nn.Module):
    """Mecanismo de Multi-Head Attention"""

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model deve ser divisível por num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        """Calcula a atenção escalonada"""
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)

        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)

        output = torch.matmul(attn_probs, V)
        return output

    def split_heads(self, x):
        """Divide a última dimensão em (num_heads, d_k)"""
        batch_size, seq_length, d_model = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)

    def combine_heads(self, x):
        """Combina as cabeças de atenção"""
        batch_size, _, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)

    def forward(self, Q, K, V, mask=None):
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))

        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
        output = self.W_o(self.combine_heads(attn_output))
        output = self.dropout(output)
        return output


class FeedForward(nn.Module):
    """Rede Feed-Forward"""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.fc2(self.dropout(F.relu(self.fc1(x))))


class TransformerBlock(nn.Module):
    """Bloco Transformer"""

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Self-attention com conexão residual
        attn_output = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))

        # Feed-forward com conexão residual
        ff_output = self.ff(x)
        x = self.norm2(x + self.dropout(ff_output))

        return x


class SmallLanguageModel(nn.Module):
    """Small Language Model (SLM)"""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        num_heads: int = 4,
        num_layers: int = 4,
        d_ff: int = 512,
        max_seq_length: int = 100,
        dropout: float = 0.1
    ):
        """
        Inicializa a SLM

        Args:
            vocab_size: Tamanho do vocabulário
            d_model: Dimensão do modelo
            num_heads: Número de cabeças de atenção
            num_layers: Número de camadas transformer
            d_ff: Dimensão da rede feed-forward
            max_seq_length: Comprimento máximo da sequência
            dropout: Taxa de dropout
        """
        super().__init__()

        self.d_model = d_model
        self.vocab_size = vocab_size

        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_length, d_model)

        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        # Camada de saída
        self.fc_out = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)

        # Inicialização dos pesos
        self._init_weights()

    def _init_weights(self):
        """Inicializa os pesos do modelo"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def create_causal_mask(self, seq_length, device):
        """Cria máscara causal para impedir atenção a tokens futuros"""
        mask = torch.tril(torch.ones(seq_length, seq_length, device=device))
        return mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_length, seq_length)

    def forward(self, x):
        """
        Forward pass

        Args:
            x: Tensor de entrada (batch_size, seq_length)

        Returns:
            Logits (batch_size, seq_length, vocab_size)
        """
        batch_size, seq_length = x.shape
        device = x.device

        # Embeddings
        positions = torch.arange(0, seq_length, device=device).unsqueeze(0).expand(batch_size, seq_length)
        token_emb = self.token_embedding(x)
        pos_emb = self.position_embedding(positions)

        x = self.dropout(token_emb + pos_emb)

        # Máscara causal
        mask = self.create_causal_mask(seq_length, device)

        # Transformer blocks
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x, mask)

        # Saída
        logits = self.fc_out(x)

        return logits

    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Gera novos tokens

        Args:
            idx: Tensor de entrada (batch_size, seq_length)
            max_new_tokens: Número máximo de tokens a gerar
            temperature: Temperatura para sampling
            top_k: Número de tokens mais prováveis a considerar

        Returns:
            Tensor com tokens gerados
        """
        self.eval()
        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Limita o contexto ao tamanho máximo
                idx_cond = idx if idx.size(1) <= self.max_seq_length else idx[:, -self.max_seq_length:]

                # Forward pass
                logits = self(idx_cond)

                # Pega os logits do último token
                logits = logits[:, -1, :] / temperature

                # Top-k sampling
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float('Inf')

                # Aplica softmax
                probs = F.softmax(logits, dim=-1)

                # Amostra o próximo token
                idx_next = torch.multinomial(probs, num_samples=1)

                # Concatena com a sequência
                idx = torch.cat((idx, idx_next), dim=1)

        return idx


if __name__ == "__main__":
    # Teste do modelo
    vocab_size = 100
    batch_size = 2
    seq_length = 50

    model = SmallLanguageModel(
        vocab_size=vocab_size,
        d_model=128,
        num_heads=4,
        num_layers=3,
        d_ff=256
    )

    # Entrada aleatória
    x = torch.randint(0, vocab_size, (batch_size, seq_length))

    # Forward pass
    logits = model(x)
    print(f"Shape de entrada: {x.shape}")
    print(f"Shape de saída: {logits.shape}")

    # Teste de geração
    generated = model.generate(x[:1, :10], max_new_tokens=20)
    print(f"Shape do texto gerado: {generated.shape}")

    # Contagem de parâmetros
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal de parâmetros: {total_params:,}")
