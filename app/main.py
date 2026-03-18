import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

#Device (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Transform (convert image to Tensor)
transform = transforms.Compose([
    transforms.ToTensor()
])

#load dataset
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

#Define simple neural network
class MNISTModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.model(x)
        return x
    
model = MNISTModel().to(device)

#Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# training loop
epochs = 3

for epoch in range(epochs):
    total_loss = 0

    for images, labels in train_loader:
        imgaes, labels = images.to(device), labels.to(device)

        outputs = model(imgaes)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

    # Save model
torch.save(model.state_dict(), "../model/mnist_model.pth")

print("Model saved as mnist_model.pth")