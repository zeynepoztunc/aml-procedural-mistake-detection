import torch
import torch.nn as nn
from core.models.blocks import fetch_input_dim


class LSTM(nn.Module):
    def __init__(self, config):
        super(LSTM, self).__init__()
        self.config = config
        input_dim = fetch_input_dim(config)
        hidden_dim = 512
        num_layers = 2
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.5 if num_layers > 1 else 0
        )
        
        self.fc = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        # x shape: (batch, seq_len, input_dim) or (seq_len, input_dim)
        
        # Check for NaNs in input and replace them with zero (Robustness)
        x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Ensure input is 3D (Batch, Seq, Dim)
        # If 2D (Seq, Dim), treat as Batch=1
        is_unbatched = x.dim() == 2
        if is_unbatched:
            x = x.unsqueeze(0)
            
        # LSTM forward
        # out shape: (batch, seq_len, hidden_dim)
        out, (h_n, c_n) = self.lstm(x)
        
        # Many-to-Many Classification
        # We want predictions for EVERY time step to match the target shape (Seq, 1)
        # out: (1, Seq, Hidden) -> (1, Seq, 1)
        logits = self.fc(out)
        
        if is_unbatched:
            # Remove batch dim: (1, Seq, 1) -> (Seq, 1)
            logits = logits.squeeze(0)
        
        return logits
