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
model_path = experiment_folder + '/classification_model.pt'

# Prepare datasets
train_data, test_data = dataset.prepare(dataset_folder, 0.8)

# Create datasets
train_dataset = dataset.custom(data_paths=train_data, augment=True)
test_dataset = dataset.custom(data_paths=test_data, augment=True)

# Create data loaders
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=True)

# Inspect dataset?
inspect = True
if inspect:
    train_features, train_targets = next(iter(train_dataloader))
    target = train_targets
    for i in range(9):
        plt.subplot(3,3,i+1)
        if target[i] == 0:
            plt.title("No Response")
        else:
            plt.title("Response")
        feature = train_features[i]
        feature = np.uint8((feature + 1.0) * 127.0)
        feature = np.swapaxes(feature, 2, 0)
        plt.imshow(feature)
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
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

# Define testing
def test(dataloader, model, loss_fn):
    num_correct = 0
    num_wrong = 0
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y.unsqueeze(1)).item()
            for i in range(len(pred)):
                target = y[i]
                output = pred[i]
                answer = (target > 0.5)
                prediction = (output > 0.5)[0]
                if answer == prediction:
                    num_correct += 1
                else:
                    num_wrong += 1
    accuracy = 100.0 * num_correct/(num_correct+num_wrong)
    test_loss /= num_batches
    print(f"Test Error: \n Avg loss: {test_loss:>8f}, Test Accuracy: {accuracy}%\n")

# TRAIN
epochs = 250
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, custom_model, loss_fn, optimizer)
    test(test_dataloader, custom_model, loss_fn)
print("Done!")


# Evaluate

# Display data and classification
features, targets = next(iter(test_dataloader))

# Let's run it
features = features.to(device)
outputs = custom_model(features)
outputs = outputs.cpu().detach().numpy()

# Evaluate performance
num_correct = 0
num_wrong = 0

for i in range(len(outputs)):
    target = train_targets[i]
    output = outputs[i]
    answer = (target > 0.5)
    prediction = (output > 0.5)[0]
    if answer == prediction:
        num_correct += 1
    else:
        num_wrong += 1
print(f'Performance ({num_correct} vs {num_wrong}): {100.0 * num_correct/(num_correct+num_wrong)}%')

## Examine predictions
#for i in range(9):
#    plt.subplot(3,3,i+1)
#    feature = train_features[i]
#    target = train_targets[i]
#    output = outputs[i]
#    feature = np.uint8((feature + 1.0) * 127.0)
#    feature = np.swapaxes(feature, 2, 0)
#    plt.imshow(feature)
#    answer = (target > 0.5)
#    prediction = (output > 0.5)[0]
#    if answer == prediction:
#        plt.title("Correct")
#    else:
#        plt.title("Wrong")
#plt.show()






# Save model
torch.save(custom_model.state_dict(), model_path + '/custom.pt')


# FIN