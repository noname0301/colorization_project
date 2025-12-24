import torch.nn as nn
import torch.nn.functional as F

class SelfAttentionLayer(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.0):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, x, pos):
        return x if pos is None else x + pos

    def forward(self, x, query_pos=None, attn_mask=None, key_padding_mask=None):
        # Pre norm
        x2 = self.norm(x)

        q = k = self.with_pos_embed(x, query_pos)
        x2 = self.attention(q, k, x2, attn_mask=attn_mask, key_padding_mask=key_padding_mask)[0]

        # Residual connection
        x = x + self.dropout(x2)
        
        return x
    

class CrossAttentionLayer(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.0):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, x, pos):
        return x if pos is None else x + pos

    def forward(self, x, y, query_pos=None, pos=None, attn_mask=None, key_padding_mask=None):
        # Pre norm
        x2 = self.norm(x)

        x2 = self.attention(query=self.with_pos_embed(x2, query_pos), 
                               key=self.with_pos_embed(y, pos),
                               value=y, attn_mask=attn_mask, key_padding_mask=key_padding_mask)[0]

        # Residual connection
        x = x + self.dropout(x2)
        
        return x
    

class FFNLayer(nn.Module):
    def __init__(self, d_model, dim_feedforward, dropout=0.0):
        super().__init__()
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self._reset_parameters()


    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, x):
        # Pre norm
        x2 = self.norm(x)

        x2 = self.linear2(self.dropout(F.relu(self.linear1(x2))))

        # Residual connection
        x = x + self.dropout(x2)
        
        return x
    

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2):
        super().__init__()
        self.activation = nn.ReLU()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = self.activation(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x