# JettyLink

> **NOTE**: *This is the initial version of the project, more feature and newer version please check main branch!*

This repository demonstrates an example of Jetson connection with PC: **custom TCP-based batch sender and trainer pipeline** for PyTorch, using Go as the client and server, and Python for model training on a Jetson device. It is useful for **streaming data to an embedded device for on-device deep learning training** and can serve as a reference for similar setups.

---

## Project Structure

* `batch_sender.go`: Runs on your PC. Reads MNIST dataset, packages batches, and sends them over TCP to the Jetson.
* `batch_interchanger.go`: Runs on the Jetson. Receives batches from the PC, forwards them to the Python trainer, and returns training feedback.
* `tester.py`: Runs on the Jetson. Implements a PyTorch CNN for MNIST and performs training on the received batches.

---

## Features

* **Batch streaming:** Sends MNIST images in configurable batch sizes over TCP.
* **On-device training:** Python code on Jetson performs CNN training with CUDA acceleration.
* **Real-time feedback:** Training loss and accuracy per batch are sent back to the PC.
* **Safe memory handling:** NumPy arrays copied to avoid PyTorch warnings about non-writable arrays.
* **Lightweight Go servers:** Minimal dependencies, easy to adapt for other datasets or models.

---

## Requirements

* **PC (sender)**

  * Go 1.20+
  * MNIST dataset files: `train-images.idx3-ubyte`, `train-labels.idx1-ubyte`

* **Jetson (receiver & trainer)**

  * Go 1.20+
  * Python 3.10+
  * PyTorch with CUDA
  * Numpy
  * > Mine is Jetson Orin Nano Super (4GB RAM, extremely limited resource)

---

## Setup

### 1. On the PC

1. Place MNIST data files in the same folder as `batch_sender.go`.
2. Update the IP address of the Jetson in `batch_sender.go`:

```go
conn, err := net.Dial("tcp", "JETSON_IP:9000")
```

3. Run the batch sender:

```bash
go run batch_sender.go
```
s
### 2. On the Jetson

1. Ensure Python environment has required packages:
**Be aware of the compatibility, this is the trickiest part! Do not trust AI on that, you'll thank me later!**
```bash
python3 -m venv venv
source venv/bin/activate
pip install torch numpy
```

2. Start the Go interchanger server:

```bash
go run batch_interchanger.go
```

3. The server will automatically start `tester.py` in the background to receive batches and train the model.

---

## How It Works

1. `batch_sender.go` reads MNIST images and labels, normalizes images, and converts them to little-endian float32 bytes.
2. Each batch is sent with a 32-byte **header** describing batch index, data length, and label length.
3. `batch_interchanger.go` receives batches, forwards them to `tester.py`, and relays training output back to the PC.
4. `tester.py` performs forward/backward passes using PyTorch on GPU, reports loss and accuracy, and returns `"OK"` when a batch is successfully processed.
5. Cleanup ensures GPU memory is freed between batches to avoid CUDA allocation errors.

---

## Notes

* The system is optimized for **Jetson embedded devices** but can be adapted to any Linux system with CUDA.
* Warnings about **non-writable NumPy arrays** are suppressed after copying arrays (`.copy()`) before converting to tensors.
* You can adjust `BatchSize` in `batch_sender.go` to control memory usage and throughput.

---

## Reference Use

* Use this setup as a template for **remote batch training**, **embedded AI experiments**, or **custom TCP-based ML pipelines**.
* Modify the model in `tester.py` or dataset loading in `batch_sender.go` for your own tasks.
* The code demonstrates **interoperability between Go and Python** for high-performance ML workflows.

---

> **Note:**
> There are also three tiny program called infer.py, server.go, and client.go, which I used for checking the connectivity before everything.
> You can ignore it...
