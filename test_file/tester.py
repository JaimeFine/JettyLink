import sys
import struct
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

device = "cuda"

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 5 * 5, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.conv(x)
        x = torch.flatten(x, 1)
        return self.fc(x)
    
model = Net().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()

def read_exact(n):
    return sys.stdin.buffer.read(n)

while True:
    header = read_exact(32)
    if not header:
        break

    magic, batch_idx, dtype, data_len, label_len = struct.unpack("<4sIIQQ", header)

    if magic != b"BTS0":
        print("Bad header!", flush=True)
        break

    data_raw = read_exact(data_len)
    label_raw = read_exact(label_len)

    X = np.frombuffer(data_raw, dtype=np.float32).reshape(2, 1, 28, 28)
    y = np.frombuffer(label_raw, dtype=np.uint8)

    x = torch.tensor(X, device=device)
    y = torch.tensor(y, dtype=torch.long, device=device)

    optimizer.zero_grad()
    output = model(X)
    loss = loss_fn(output, y)
    loss.backward()
    optimizer.step()

    acc = (output.argmax(dim=1) == y).float().mean().item()

    print(f"batch {batch_idx} | loss={loss.item():.4f} | acc={acc*100:.1f}%", flush=True)