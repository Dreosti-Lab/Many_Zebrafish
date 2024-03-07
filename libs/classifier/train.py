import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
from torchsummary import summary

#----------------------------------------------------------
# Load environment file and variables
import os
from dotenv import load_dotenv
load_dotenv()
libs_path = os.getenv('LIBS_PATH')
base_path = os.getenv('BASE_PATH')

# Set library paths
import sys
sys.path.append(libs_path)
#----------------------------------------------------------

# Locals libs
import classifier.model as model
import classifier.dataset as dataset

# Reimport
import importlib
importlib.reload(dataset)
importlib.reload(model)

# Get user name
import os
username = os.getlogin()

# Specify paths
experiment_folder = base_path + '/PPI'
dataset_folder = experiment_folder + '/_dataset'
model_path = experiment_folder + '/_model'

# Prepare datasets
train_data, test_data = dataset.prepare(dataset_folder, 0.8)

# Create datasets
train_dataset = dataset.custom(data_paths=train_data, augment=True)
test_dataset = dataset.custom(data_paths=test_data, augment=True)

# Create data loaders
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=True)

# Inspect dataset?
inspect = False
if inspect:
    train_features, train_targets = next(iter(train_dataloader))
    for i in range(9):
        plt.subplot(3,3,i+1)
        feature = train_features[i]
        target = train_targets[i]
        feature = (feature + 2.0) / 4.0
        image = np.transpose(feature, (1,2,0))
        plt.imshow(image)
        plt.plot(target[0] * 224, target[1] * 224, 'g+', markersize=15,)
    plt.show()

# Reimport
importlib.reload(dataset)
importlib.reload(model)

# Instantiate model
importlib.reload(model)
custom_model = model.custom()

# Set loss function
loss_fn = torch.nn.BCELoss()
optimizer = torch.optim.Adam(custom_model.parameters(), lr=0.001)

# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using {device} device")

# Move model to device
custom_model.to(device)
summary(custom_model, (20, 48, 48))

# Define training
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = custom_model(X)
        loss = loss_fn(pred, y.unsqueeze(1))

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 2 == 0:
            loss, current = loss.item(), batch * len(X)
            pixel_loss = np.sqrt(loss) * 48.0
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}], pixel_loss: {pixel_loss:>5f}")

# Define testing
def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y.unsqueeze(1)).item()
    test_loss /= num_batches
    pixel_loss = np.sqrt(test_loss) * 48.0
    print(f"Test Error: \n Avg loss: {test_loss:>8f}, pixel_loss: {pixel_loss:>5f}\n")

# TRAIN
epochs = 250
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, custom_model, loss_fn, optimizer)
    test(test_dataloader, custom_model, loss_fn)
print("Done!")




# Display image and label.
train_features, train_targets = next(iter(test_dataloader))
print(f"Feature batch shape: {train_features.size()}")
print(f"Targets batch shape: {train_targets.size()}")

# Let's run it
train_features_gpu = train_features.to(device)
outputs = custom_model(train_features_gpu)
outputs = outputs.cpu().detach().numpy()

# Examine predictions
for i in range(9):
    plt.subplot(3,3,i+1)
    feature = train_features[i]
    target = train_targets[i]
    output = outputs[i]
    feature = (feature + 2.0) / 4.0
    image = np.transpose(feature, (1,2,0))
    plt.imshow(image)
    plt.plot(output[0] * 224, output[1] * 224, 'yo', markersize=15, fillstyle='full')
    plt.plot(target[0] * 224, target[1] * 224, 'g+', markersize=15,)
plt.show()






# Save model
torch.save(custom_model.state_dict(), output_path + '/custom.pt')


# FIN