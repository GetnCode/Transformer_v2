# Advanced Transformer Language Model

A flexible and efficient transformer-based language model implementation with state-of-the-art features including Rotary Position Embeddings (RoPE), Flash Attention, and more.

## Features

- **Modern Architecture**: Implements a decoder-only transformer similar to GPT models
- **Configurable Design**: Easily adjust model size, activation functions, and more
- **Advanced Mechanisms**:
  - Rotary Position Embeddings (RoPE) for better sequence handling
  - Flash Attention for faster training and inference
  - SwiGLU activation functions for better performance
  - KV caching for efficient text generation
  - Gradient checkpointing for memory-efficient training
- **Text Generation**: Built-in methods for sampling with temperature, top-k, and nucleus (top-p) sampling
- **Stability Improvements**: Robust initialization and numerical safeguards

## Installation

### Requirements

- Python 3.8+
- PyTorch 2.0+ (for Flash Attention support)

### Steps

1.  Clone the repository:

    ```bash
    git clone <repository_url>
    cd <repository_directory>
    ```

2.  Install the required packages:

    ```bash
    pip install torch torchvision torchaudio
    pip install -r requirements.txt
    ```

## Usage

### Model Configuration

You can configure the model by adjusting the following hyperparameters in the `TransformerModel` class:

- `vocab_size`: Size of the vocabulary.
- `d_model`: Hidden dimension size (e.g., 512, 768, 1024).
- `num_layers`: Number of transformer layers (e.g., 6, 12, 14).
- `num_heads`: Number of attention heads (e.g., 8, 12, 16).
- `d_ff`: Feed-forward network dimension (e.g., 2048, 4096).
- `dropout`: Dropout probability.
- `use_rope`: Use Rotary Position Embeddings.
- `use_flash`: Use Flash Attention.
- `use_swiglu`: Use SwiGLU activation.
- `use_pre_norm`: Use pre-normalization.

Example:

```python
from model import TransformerModel
import torch

# Create a small model
model = TransformerModel(
    vocab_size=50257,
    d_model=512,
    num_layers=6,
    num_heads=8
)

# Example input
input_ids = torch.randint(0, 50257, (1, 32))
output = model(input_ids)
print(output.shape)
```

### Text Generation

Use the `generate` method to generate text:

```python
from model import TransformerModel
import torch

# Create a model
model = TransformerModel(
    vocab_size=50257,
    d_model=768,
    num_layers=12,
    num_heads=12
)

# Example input
input_ids = torch.tensor([[1, 2, 3, 4, 5]])

# Generate text
generated = model.generate(input_ids, max_length=50, temperature=0.7, top_k=50)
print(generated)
```

### Hyperparameter Tuning for Model Sizes

- **Small Model**:
  ```python
  model = TransformerModel(
      vocab_size=50257,
      d_model=512,
      num_layers=6,
      num_heads=8,
      d_ff=2048
  )
  ```
- **Medium Model**:
  ```python
  model = TransformerModel(
      vocab_size=50257,
      d_model=768,
      num_layers=12,
      num_heads=12,
      d_ff=3072
  )
  ```
- **Large Model**:
  ```python
  model = TransformerModel(
      vocab_size=50257,
      d_model=1024,
      num_layers=14,
      num_heads=16,
      d_ff=4096
  )
  ```

Adjusting `d_model`, `num_layers`, `num_heads`, and `d_ff` will significantly impact the model size and performance.

## Training Tutorial

### Data Preparation

1.  **Prepare your dataset**: Ensure your data is tokenized and ready for training.
2.  **Create data loaders**: Use `torch.utils.data.DataLoader` to efficiently load data in batches.

### Training Script

Here’s a basic training loop example:

```python
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from model import TransformerModel

# Sample dataset (replace with your actual dataset)
class SimpleDataset(Dataset):
    def __init__(self, data, vocab_size):
        self.data = data
        self.vocab_size = vocab_size

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx]), torch.tensor(self.data[idx][1:] + [0])  # Input and target

# Generate synthetic data
vocab_size = 50257
sample_data = [[torch.randint(1, vocab_size, (10,)).item() for _ in range(10)] for _ in range(100)]

dataset = SimpleDataset(sample_data, vocab_size)
data_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Model and optimizer
model = TransformerModel(vocab_size=vocab_size, d_model=256, num_layers=4, num_heads=4)
optimizer = AdamW(model.parameters(), lr=1e-4)

# Training loop
num_epochs = 3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch, (inputs, targets) in enumerate(data_loader):
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)

        # Compute loss
        loss = torch.nn.functional.cross_entropy(outputs.view(-1, vocab_size), targets.view(-1), ignore_index=0)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if batch % 10 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch}, Loss: {loss.item()}")

    print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {total_loss/len(data_loader)}")

print("Training finished!")
```

### Running the Training Script

1.  Save the above script as `train.py`.
2.  Run the script:

    ```bash
    python train.py
    ```

This tutorial provides a basic example. For real-world scenarios, consider using more sophisticated data handling, validation, and checkpointing.
