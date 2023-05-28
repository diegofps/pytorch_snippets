import torch
import os

from torch import nn
from torch.utils.data import DataLoader
from utils import train, test, select_device, load_dataset


# Get cpu, gpu or mps device for training.
print("Selecting device")
device = select_device()
print(f"Using {device} device")



# Download training data from open datasets.
print("Creating dataset objects")
training_data, test_data = load_dataset('mnist')


# Create data loaders.
print("Creating data loaders")
batch_size = 64
train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

for X, y in test_dataloader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break


print("Defining the model")
class NeuralNetwork2(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv_stack = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            nn.MaxPool2d(kernel_size=2),
        )

        self.flatten = nn.Flatten()

        self.dense_stack = nn.Sequential(
            nn.Linear(in_features=1600, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=10)
        )

    def forward(self, x):
        x = self.conv_stack(x)
        x = self.flatten(x)
        x = self.dense_stack(x)
        return x


model = NeuralNetwork2().to(device)
print(model)


print("Creating CrossEntropyLoss")
loss_fn = nn.CrossEntropyLoss()


print("Training with Adam")
optimizer = torch.optim.Adam(model.parameters())
epochs = 3
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer, device)
    test(test_dataloader, model, loss_fn, device)
print("Done!")

optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
print("Training with SGD")
epochs = 5
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer, device)
    test(test_dataloader, model, loss_fn, device)
print("Done!")


os.makedirs('models', exist_ok=True)
model_filepath = os.path.join('models', "conv2d.pth")


print("Exporting to file")
torch.save(model.state_dict(), model_filepath)
print(f"Saved PyTorch Model State to {model_filepath}")


print("Loading from exported file")
model = NeuralNetwork2().to(device)
model.load_state_dict(torch.load(model_filepath))


print("Running single instance")
classes = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

model.eval()
x, y = test_data[0][0], test_data[0][1]
with torch.no_grad():
    x = torch.unsqueeze(x, dim=0)
    x = x.to(device)
    pred = model(x)
    predicted, actual = classes[pred[0].argmax(0)], classes[y]
    print(f'Predicted: "{predicted}", Actual: "{actual}"')