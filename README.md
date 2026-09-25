# MNIST Classifier

A small PyTorch project that trains a fully connected neural network to
recognize handwritten digits from the [MNIST](https://yann.lecun.com/exdb/mnist/)
dataset.

## Requirements

- Python 3.14 or newer
- PyTorch
- torchvision
- NumPy
- Matplotlib

The dependencies are listed in `pyproject.toml`.

## Setup

Create and activate a virtual environment, then install the project:

```bash
python3.14 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

## Train the classifier

After installation, start the project's training process. MNIST is downloaded
into `data/MNIST` if it is not already present. Training runs for 100 epochs and
prints training loss, test loss, and test accuracy after each epoch.

The model is a multilayer perceptron with this shape:

```text
784 input features -> 128 ReLU units -> 10 output classes
```

The output logits are trained with cross-entropy loss and the Adam optimizer.

## Project contents

| File | Purpose |
| --- | --- |
| `pyproject.toml` | Project metadata and dependencies |
| `data/MNIST/` | Downloaded MNIST dataset files |

## Notes

- Set the random seed when you need repeatable experiments.
- Dataset files and generated model weights are ignored by Git.
- This project is intended as a small learning example, not a production
	inference service.
