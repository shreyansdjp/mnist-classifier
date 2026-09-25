"""
MNIST handwritten digit classifier
A minimal but complete PyTorch training pipeline.

Setup:
    pip install torch torchvision
    python mnist_classifier.py

Expect ~97-98% test accuracy after 5 epochs.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# ---------------------------------------------------------------------------
# Config -- every knob worth turning lives here, not buried in the code below.
# ---------------------------------------------------------------------------
BATCH_SIZE = 64
EPOCHS = 5
LR = 1e-3
SEED = 0

torch.manual_seed(SEED)  # makes runs reproducible so you can compare changes


# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------
def get_device():
    """CUDA (NVIDIA) > MPS (Apple Silicon) > CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class MLP(nn.Module):
    """784 -> 128 -> 10 multilayer perceptron."""

    def __init__(self, in_features=28 * 28, hidden=128, n_classes=10):
        super().__init__()  # required: registers the module machinery
        self.net = nn.Sequential(
            nn.Flatten(),  # [B, 1, 28, 28] -> [B, 784]
            nn.Linear(in_features, hidden),
            nn.ReLU(),  # the nonlinearity that makes depth matter
            nn.Linear(hidden, n_classes),  # raw logits -- NO softmax here
        )

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------
def grad_norm(model):
    """L2 norm of all gradients, concatenated.

    Read it like a pulse. Steadily shrinking is healthy. Collapsing to ~0 early
    means the model stopped learning; exploding to hundreds means your LR is
    too high. You will lean on this hard in the advanced project.
    """
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += p.grad.detach().norm(2).item() ** 2
    return total**0.5


# ---------------------------------------------------------------------------
# Train / eval loops
# ---------------------------------------------------------------------------
def train_one_epoch(model, loader, loss_fn, optimizer, device):
    model.train()  # enables dropout/batchnorm training behavior (unused here, but be in the habit)
    running_loss, seen = 0.0, 0

    for batch_idx, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)

        # --- forward ---
        logits = model(images)  # [B, 10]
        loss = loss_fn(logits, labels)  # scalar

        # --- backward: the three-line ritual ---
        optimizer.zero_grad()  # clear last step's grads (PyTorch accumulates by default)
        loss.backward()  # autograd fills p.grad for every parameter
        optimizer.step()  # apply the update rule

        running_loss += loss.item() * labels.size(0)
        seen += labels.size(0)

        if batch_idx % 200 == 0:
            print(
                f"    batch {batch_idx:4d} | loss {loss.item():.4f} "
                f"| grad norm {grad_norm(model):.4f}"
            )

    return running_loss / seen


@torch.no_grad()  # no gradients needed at eval time -> less memory, faster
def evaluate(model, loader, loss_fn, device):
    model.eval()
    total_loss, correct, seen = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        total_loss += loss_fn(logits, labels).item() * labels.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        seen += labels.size(0)

    return total_loss / seen, correct / seen


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    device = get_device()
    print(f"Using device: {device}\n")

    # ToTensor scales pixels to [0, 1]; Normalize centers them using MNIST's
    # own mean/std. Centered inputs keep early gradients well-scaled.
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )

    train_set = datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )
    test_set = datasets.MNIST(
        root="./data", train=False, download=True, transform=transform
    )

    # shuffle=True on train only -- shuffling test data changes nothing but costs time.
    train_loader = DataLoader(
        train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=2
    )
    test_loader = DataLoader(test_set, batch_size=256, shuffle=False, num_workers=2)

    model = MLP().to(device)
    loss_fn = nn.CrossEntropyLoss()  # expects logits + integer class labels
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model has {n_params:,} parameters\n")

    for epoch in range(1, EPOCHS + 1):
        print(f"Epoch {epoch}/{EPOCHS}")
        train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
        test_loss, test_acc = evaluate(model, test_loader, loss_fn, device)
        print(
            f"  train loss {train_loss:.4f} | test loss {test_loss:.4f} "
            f"| test acc {test_acc * 100:.2f}%\n"
        )

    torch.save(model.state_dict(), "mnist_mlp.pt")
    print("Saved weights to mnist_mlp.pt")


if __name__ == "__main__":
    # This guard is required for num_workers > 0 on Windows and macOS spawn.
    main()
