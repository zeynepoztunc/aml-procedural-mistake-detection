import torch
from torch import nn

from core.models.blocks import EncoderLayer, Encoder, MLP, fetch_input_dim


class ErFormer(nn.Module):

    def __init__(self, config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config
        input_dimension = fetch_input_dim(config)

        # Initialize the transformer encoder
        step_encoder_layer = EncoderLayer(d_model=input_dimension, dim_feedforward=2048, nhead=8, batch_first=True)
        self.step_encoder = Encoder(step_encoder_layer, num_layers=1)
        
        # Decoder input dimension matches encoder output for single-modality
        decoder_input_dimension = input_dimension  # Use same dimension as encoder
        
        # Initialize the MLP decoder
        self.decoder = MLP(decoder_input_dimension, 512, 1)
        self.apply(init_weights)  # Apply weight initialization

    def forward(self, input_data):
        # Check for NaNs in input and replace them with zero
        input_data = torch.nan_to_num(input_data, nan=0.0, posinf=1.0, neginf=-1.0)

        # Encode the input
        encoded_output = self.step_encoder(input_data)
        _, dim = encoded_output.shape

        # For backbones with different dimensions (EgoVLP=256, Omnivore=1024, etc.)
        # Simply use the full encoded output for single-modality case
        # Multi-modality handling is only needed for ImageBind with multiple inputs
        
        if dim <= 1024:
            # Single modality (EgoVLP: 256, SlowFast/X3D: 400, Omnivore: 1024)
            final_encoded = encoded_output
        else:
            # Multi-modality case (ImageBind with multiple modalities)
            # Split and combine based on 1024-dim chunks
            video_output = encoded_output[:, :1024]
            if dim // 1024 == 2:
                audio_output = encoded_output[:, 1024:2048]
                final_encoded = 0.65 * video_output + 0.35 * audio_output
            elif dim // 1024 == 3:
                audio_output = encoded_output[:, 1024:2048]
                text_output = encoded_output[:, 2048:3072]
                final_encoded = 0.4 * video_output + 0.3 * audio_output + 0.3 * text_output
            elif dim // 1024 >= 4:
                audio_output = encoded_output[:, 1024:2048]
                text_output = encoded_output[:, 2048:3072]
                depth_output = encoded_output[:, 3072:]
                final_encoded = 0.25 * video_output + 0.25 * audio_output + 0.25 * text_output + 0.25 * depth_output
            else:
                final_encoded = video_output

        # Decode the output
        final_output = self.decoder(final_encoded)

        # Check for NaNs in output and replace them with zero
        # final_output = torch.nan_to_num(final_output, nan=0.0, posinf=1.0, neginf=-1.0)

        return final_output

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        torch.nn.init.constant_(m.bias, 0)
        torch.nn.init.constant_(m.weight, 1.0)