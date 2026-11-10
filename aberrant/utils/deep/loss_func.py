"""Loss function types for deep learning models."""

from torch import nn

# Reconstruction losses an autoencoder can be trained with
AutoencoderLoss = nn.MSELoss | nn.L1Loss | nn.SmoothL1Loss | nn.HuberLoss
