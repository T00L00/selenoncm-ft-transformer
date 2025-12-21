import torch
import torch.nn as nn

class FeatureTokenizer(nn.Module):
    """
    Turns morphological features into tokens.

    token_j = feature_id_embed[j] + (x_j * w_j + b_j)

    Implemented efficiently as:
      - id embedding: nn.Embedding(D, d_model)
      - value projection: w, b as nn.Parameter of shape [D, d_model]
    """
    def __init__(self, num_features: int, d_model: int):
        super().__init__()
        self.num_features = num_features
        self.d_model = d_model

        self.feature_embed = nn.Embedding(num_features, d_model)
        self.w = nn.Parameter(torch.randn(num_features, d_model) * 0.02)
        self.b = nn.Parameter(torch.zeros(num_features, d_model))

        # precompute indices [0..D-1] so we don't have to recreate every call
        self.register_buffer("feat_idx", torch.arange(num_features, dtype=torch.long))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, D] float
        returns: [B, D, d_model]
        """
        B, D = x.shape
        assert D == self.num_features

        # feature identity embedding
        # [D, d_model] -> [1, D, d_model] -> [B, D, d_model]
        e = self.feature_embed(self.feat_idx).unsqueeze(0).expand(B, -1, -1)

        # value embedding: x_j * w_j + b_j
        # x: [B, D] -> [B, D, 1]
        x_ = x.unsqueeze(-1)
        val = x_ * self.w.unsqueeze(0) + self.b.unsqueeze(0)  # [B, D, d_model]

        return e + val
    
class TabTransformerClassifier(nn.Module):
    def __init__(
        self,
        num_features: int,
        d_model: int = 128,
        n_heads: int = 8,
        n_layers: int = 3,
        dropout: float = 0.2,
        token_dropout: float = 0.1, # randomly mask features at train time
        mlp_hidden: int = 128,
    ):
        super().__init__()
        self.num_features = num_features
        self.d_model = d_model
        self.token_dropout = token_dropout

        self.tokenizer = FeatureTokenizer(num_features, d_model)

        # CLS token
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.cls, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4*d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True, # important: expects [B, T, d]
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.norm = nn.LayerNorm(d_model)

        self.head = nn.Sequential(
            nn.Linear(d_model, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, 1), # logits for binary classification
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, D]
        returns logits: [B]
        """
        B, D = x.shape

        # Feature/token dropout: mask some feature values to 0 (after scaling)
        if self.training and self.token_dropout > 0:
            mask = (torch.rand(B, D, device=x.device) > self.token_dropout).float()
            x = x * mask

        tok = self.tokenizer(x) # [B, D, d_model]

        cls = self.cls.expand(B, -1, -1) # [B, 1, d_model]
        seq = torch.cat([cls, tok], dim=1) # [B, 1+D, d_model]

        h = self.encoder(seq) # [B, 1+D, d_model]
        h_cls = self.norm(h[:, 0, :]) # [B, d_model]

        logits = self.head(h_cls).squeeze(-1) # [B]
        return logits