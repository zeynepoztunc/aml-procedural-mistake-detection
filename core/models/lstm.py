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
        # x shape: (batch, seq_len, input_dim)
        
        # Check for NaNs in input and replace them with zero (Robustness)
        x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # LSTM forward
        # out shape: (batch, seq_len, hidden_dim) if batch_first=True
        # OR (seq_len, hidden_dim) if unbatched input
        out, (h_n, c_n) = self.lstm(x)
        
        # Handle both batched (3D) and unbatched (2D) output
        if out.dim() == 3:
            # (Batch, Seq, Hidden) -> Take last time step for each batch
            last_output = out[:, -1, :]
        elif out.dim() == 2:
            # (Seq, Hidden) -> Take last time step
            last_output = out[-1, :].unsqueeze(0) # Add batch dim back: (1, Hidden)
        else:
            raise ValueError(f"Unexpected LSTM output shape: {out.shape}")
        
        # Classification head
        logits = self.fc(last_output)
        
        return logits
