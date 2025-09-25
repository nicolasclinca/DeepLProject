# --- FILE: model_from_scratch.py ---

import torch
import torch.nn as nn
import math


# ----------------------------------------------------------------------
# 1. BLOCCHI DI COSTRUZIONE FONDAMENTALI
# ----------------------------------------------------------------------

class PositionalEncoding(nn.Module):
    """
    Implementazione del Positional Encoding sinusoidale.
    Aggiunge informazioni sulla posizione dei token ai loro embedding.
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1) # aggiunge una dimensione
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))

        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe.transpose(0, 1))  # Shape: (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: Tensor, shape [batch_size, seq_len, d_model]
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    """
    Implementazione della Multi-Head Attention.
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        # TODO: devono essere per forza quadrate?

        self.dropout = nn.Dropout(dropout)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        """
        Esegue il calcolo della attenzione multi-testa
        """
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            # Applica la maschera impostando a -inf i valori che non vogliamo considerare
            scores = scores.masked_fill(mask == 0, -1e9)

        attention_weights = torch.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        output = torch.matmul(attention_weights, V)
        return output, attention_weights

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: torch.Tensor = None) -> torch.Tensor:
        batch_size = query.size(0)

        # 1. Proiezioni lineari
        Q = self.W_q(query)
        K = self.W_k(key)
        V = self.W_v(value)

        # 2. Suddivisione in 'num_heads'
        # Shape: (batch_size, num_heads, seq_len, d_k)
        Q = Q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # 3. Scaled Dot-Product Attention
        x, _ = self.scaled_dot_product_attention(Q, K, V, mask)

        # 4. Concatenazione degli head e proiezione finale
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        x = self.W_o(x)

        return x


class PositionwiseFeedForward(nn.Module):
    """
    Implementazione del Feed-Forward Network.
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(self.relu(self.linear1(x))))


# ----------------------------------------------------------------------
# 2. BLOCCHI DELL'ENCODER E DEL DECODER
# ----------------------------------------------------------------------

class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        # Self-Attention sub-layer
        attn_output = self.self_attn(src, src, src, src_mask)
        src = self.norm1(src + self.dropout1(attn_output))  # Connessione residuale + LayerNorm

        # Feed-Forward sub-layer
        ff_output = self.feed_forward(src)
        src = self.norm2(src + self.dropout2(ff_output))  # Connessione residuale + LayerNorm

        return src


class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, tgt: torch.Tensor, encoder_output: torch.Tensor, tgt_mask: torch.Tensor,
                src_mask: torch.Tensor) -> torch.Tensor:
        # Masked Self-Attention sub-layer
        attn_output = self.self_attn(tgt, tgt, tgt, tgt_mask)
        tgt = self.norm1(tgt + self.dropout1(attn_output))

        # Cross-Attention sub-layer (Q dal decoder, K e V dall'encoder)
        cross_attn_output = self.cross_attn(query=tgt, key=encoder_output, value=encoder_output, mask=src_mask)
        tgt = self.norm2(tgt + self.dropout2(cross_attn_output))

        # Feed-Forward sub-layer
        ff_output = self.feed_forward(tgt)
        tgt = self.norm3(tgt + self.dropout3(ff_output))

        return tgt


# ----------------------------------------------------------------------
# 3. ASSEMBLAGGIO FINALE DEL MODELLO
# ----------------------------------------------------------------------

class Encoder(nn.Module):
    def __init__(self, layer, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([layer for _ in range(num_layers)])

    def forward(self, src, src_mask):
        for layer in self.layers:
            src = layer(src, src_mask)
        return src


class Decoder(nn.Module):
    def __init__(self, layer, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([layer for _ in range(num_layers)])

    def forward(self, tgt, encoder_output, tgt_mask, src_mask):
        for layer in self.layers:
            tgt = layer(tgt, encoder_output, tgt_mask, src_mask)
        return tgt


class EncoderDecoderTransformer(nn.Module):
    """
    Il modello NanoSocrates completo, costruito da zero.
    """

    def __init__(self, vocab_size: int, d_model: int, num_layers: int, num_heads: int, d_ff: int, dropout: float,
                 pad_idx: int):
        super().__init__()

        self.pad_idx = pad_idx

        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, dropout)

        # Encoder e Decoder
        encoder_layer = EncoderLayer(d_model, num_heads, d_ff, dropout)
        self.encoder = Encoder(encoder_layer, num_layers)

        decoder_layer = DecoderLayer(d_model, num_heads, d_ff, dropout)
        self.decoder = Decoder(decoder_layer, num_layers)

        # Output finale
        self.generator = nn.Linear(d_model, vocab_size)

    def _create_padding_mask(self, sequence):
        # Shape: (batch_size, 1, 1, seq_len)
        return (sequence != self.pad_idx).unsqueeze(1).unsqueeze(2)

    def _generate_square_subsequent_mask(self, sz):
        # Genera una maschera triangolare superiore
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def encode(self, src: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        src_emb = self.positional_encoding(self.token_embedding(src))
        return self.encoder(src_emb, src_mask)

    def decode(self, tgt: torch.Tensor, encoder_output: torch.Tensor, tgt_mask: torch.Tensor,
               src_mask: torch.Tensor) -> torch.Tensor:
        tgt_emb = self.positional_encoding(self.token_embedding(tgt))
        return self.decoder(tgt_emb, encoder_output, tgt_mask, src_mask)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor):
        # Creazione delle maschere
        src_padding_mask = self._create_padding_mask(src)
        tgt_padding_mask = self._create_padding_mask(tgt)

        tgt_len = tgt.shape[1]
        tgt_subsequent_mask = self._generate_square_subsequent_mask(tgt_len).to(src.device)

        # La maschera del decoder combina la maschera per il padding e quella per i token futuri
        tgt_mask = tgt_padding_mask & tgt_subsequent_mask

        # Flusso Encoder -> Decoder
        encoder_output = self.encode(src, src_padding_mask)
        decoder_output = self.decode(tgt, encoder_output, tgt_mask, src_padding_mask)

        # Proiezione finale al vocabolario
        output = self.generator(decoder_output)
        return output
