import sys
import struct
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gc
import traceback

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
    
try:
    model = Net().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()
except Exception as e:
    print(f"Initialization error: {str(e)}", flush=True)
    sys.exit(1)

while True:
    header = sys.stdin.buffer.read(32)
    if len(header) == 0:
        break

    if header[:4] != b"BTS0":
        print("bad magic", flush=True)
        break

    batch_idx = int.from_bytes(header[4:8], "little")
    data_len = int.from_bytes(header[16:24], "little")
    label_len = int.from_bytes(header[24:32], "little")

    data_raw = sys.stdin.buffer.read(data_len)
    label_raw = sys.stdin.buffer.read(label_len)

    try:
        X = np.frombuffer(data_raw, dtype=np.float32).reshape(-1, 1, 28, 28)
        y = np.frombuffer(label_raw, dtype=np.uint8)

        x = torch.tensor(X, device=device)
        y = torch.tensor(y, dtype=torch.long, device=device)

        optimizer.zero_grad()
        out = model(x)
        loss = loss_fn(out, y)
        loss.backward()
        optimizer.step()

        acc = (out.argmax(dim=1) == y).float().mean().item()
        print(f"batch {batch_idx:4d} | loss={loss.item():.4f} | acc={acc*100:5.1f}%", flush=True)
    except Exception as e:
        print(f"Error processing batch {batch_idx}: {str(e)}\n{traceback.format_exc()}", flush=True)
    finally:
        try:
            del x, y, out, loss, X
            if 'cuda' in device:
                torch.cuda.empty_cache()
            gc.collect()
        except:
            pass