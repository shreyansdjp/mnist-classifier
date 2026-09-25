import torchvision
import torch, torch.nn as nn, torch.optim as optim

torch.manual_seed(0)

DATA_ROOT = "./data"

transform = torchvision.transforms.Compose(
    [
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.1307,), (0.3081,)),
    ]
)

training_dataset = torchvision.datasets.MNIST(
    root=DATA_ROOT, train=True, download=True, transform=transform
)

testing_dataset = torchvision.datasets.MNIST(
    root=DATA_ROOT, train=False, download=True, transform=transform
)

training_images = training_dataset.data
training_labels = training_dataset.targets

testing_images = testing_dataset.data
testing_labels = testing_dataset.targets


class NeuralNet(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.layer1 = nn.Linear(in_features, 128)
        self.activation = nn.ReLU()
        self.out_layer = nn.Linear(128, out_features)

    def forward(self, input):
        input = self.layer1(input)
        input = self.activation(input)
        input = self.out_layer(input)
        return input


class VerySimpleNeuralNet(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.layer = nn.Linear(in_features, out_features)

    def forward(self, input):
        return self.layer(input)


EPOCHS = 100
BATCH_SIZE = 64
LR = 1e-4

# model = NeuralNet(28 * 28, 10)
model = VerySimpleNeuralNet(28 * 28, 10)
loss_fn = nn.CrossEntropyLoss()
# optimizer = optim.SGD(model.parameters(), lr=LR)
optimizer = optim.Adam(model.parameters(), lr=LR)

# go through each epoch
for epoch in range(EPOCHS):
    # Training
    # set model in train mode
    model.train()
    # batch the data
    running_train_loss = 0
    for i in range(0, len(training_images) - 32, BATCH_SIZE):
        batch_images = training_images[i : i + BATCH_SIZE].to(torch.float)
        batch_labels = training_labels[i : i + BATCH_SIZE].to(torch.long)
        # feed the input
        batch_images = batch_images.reshape(64, 28 * 28)
        logits = model(batch_images)
        # calulate loss from logits
        loss = loss_fn(logits, batch_labels)
        # optimizer finds the blame
        loss.backward()
        # step to next batch item to train
        optimizer.step()
        optimizer.zero_grad()
        running_train_loss += loss.item() * batch_images.size(0)
        # break
    avg_train_loss = running_train_loss / len(training_images)

    # Evaluation (pro tip, use validation data and use testing separate for better checking)
    # set the model in eval mode with no grad
    correct_predictions = 0
    running_test_loss = 0.0
    model.eval()
    with torch.no_grad():
        for i in range(0, len(testing_images) - 32, BATCH_SIZE):
            batch_images = testing_images[i : i + BATCH_SIZE].to(torch.float)
            batch_labels = testing_labels[i : i + BATCH_SIZE].to(torch.long)
            batch_images = batch_images.reshape(64, 28 * 28)
            logits = model(batch_images)
            loss = loss_fn(logits, batch_labels)
            running_test_loss += loss.item() * batch_images.size(0)
            predictions = torch.argmax(logits, dim=1)
            correct_predictions += (predictions == batch_labels).sum().item()
    avg_test_loss = running_test_loss / len(testing_images)
    test_accuracy = (correct_predictions / len(testing_images)) * 100
    # print(avg_test_loss, test_accuracy)
    # send test data to model to see testing loss
    print(
        f"Epoch {epoch+1:02d}/{EPOCHS} | "
        f"Train Loss: {avg_train_loss:.4f} | "
        f"Test Loss: {avg_test_loss:.4f} | "
        f"Test Acc: {test_accuracy:.2f}%"
    )
    # break

# once test loss goes up the last epoch was the one with best model
