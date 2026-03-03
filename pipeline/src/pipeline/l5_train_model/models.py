import torch
import torch.nn as nn
from torch.utils.data import Dataset

class DecoderDataset(Dataset):
    def __init__(self, inputs: torch.Tensor, targets: torch.Tensor):
        self.inputs = inputs
        self.targets = targets

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]

class Decoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        dropout_p: float = 0.1,
    ):
        super().__init__()
        mid_dim = (input_dim + output_dim) // 2
        self.threshold = 0.5
        
        # Using Sequential organizes layers and makes forward() cleaner
        self.layers = nn.Sequential(
            nn.Linear(input_dim, mid_dim),
            nn.LayerNorm(mid_dim),
            nn.GELU(),
            nn.Dropout(dropout_p),
            
            nn.Linear(mid_dim, mid_dim),
            nn.LayerNorm(mid_dim),
            nn.GELU(),
            
            nn.Linear(mid_dim, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (bsz, input_dim)
        out = self.layers(x) # (bsz, output_dim)
        return torch.nn.functional.normalize(out, p=2.0, dim=1)
    

class DecoderWrapper(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_classes: int,
    ):
        super().__init__()
        # Create vectornode and finetune
        self.vectors = nn.Parameter(torch.rand(num_classes, output_dim))
        self.decoder = Decoder(input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.decoder(x) # (bsz, output_dim)
        out = torch.matmul(out, self.vectors.T) # (bsz, num_classes)
        return out
