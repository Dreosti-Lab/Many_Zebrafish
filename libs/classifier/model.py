import torch


# Define model (which extends the NN module)
class custom(torch.nn.Module):

    # Initialize
    def __init__(self):
        super(custom, self).__init__()

        self.conv1 = torch.nn.Conv2d(20, 32, 5)
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.relu1 = torch.nn.ReLU(inplace=True)
        self.conv2 = torch.nn.Conv2d(32, 64, 5)
        self.relu2 = torch.nn.ReLU(inplace=True)
        self.flatten = torch.nn.Flatten()
        self.linear1 = torch.nn.Linear(64 * 9 * 9, 120)
        self.linear2 = torch.nn.Linear(120, 84)
        self.linear3 = torch.nn.Linear(84, 16)
        self.linear4 = torch.nn.Linear(16,1)
        self.sigmoid = torch.nn.Sigmoid()

    # Forward
    def forward(self, x):
        x = self.conv1(x)
        #print(x.size())
        x = self.relu1(x)
        x = self.pool(x)
        #print(x.size())
        x = self.conv2(x)
        #print(x.size())
        x = self.relu2(x)
        x = self.pool(x)
        #print(x.size())
        x = self.flatten(x)
        #print(x.size())
        x = self.linear1(x)
        #print(x.size())
        x = self.linear2(x)
        x = self.linear3(x)
        x = self.linear4(x)
        x = self.sigmoid(x)
        return x
