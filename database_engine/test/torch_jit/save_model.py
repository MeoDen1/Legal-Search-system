import torch
import torch.nn as nn

class TestModel(nn.Module):
    def __init__(self, input_dim: int = 128, output_dim: int = 128):
        super().__init__()

        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)

model = TestModel()
traced_model = torch.jit.script(model, torch.randn(1, 128))
traced_model.save("model.jit")
