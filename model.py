import torch
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
import torch.utils.data as data_utils
import numpy as np

class ResNetBlock(torch.nn.Module):
    def __init__(self,neurons_in, neurons_out):
        super().__init__()
        layers = [
            torch.nn.Mish(),
            torch.nn.Linear(neurons_in,neurons_out),
            torch.nn.Mish(),
            torch.nn.Linear(neurons_out, neurons_out),
        ]
        self.res_block = torch.nn.Sequential(*layers)
        self.identity = torch.nn.Identity()
        if neurons_in != neurons_out:
            self.identity = torch.nn.Linear(neurons_in, neurons_out)

    def forward(self,x):
        return self.identity(x) + self.res_block(x)

class nn(torch.nn.Module):
    def __init__(self,neurons_in, neurons_out):
        super().__init__()
        layers = [
            ResNetBlock(neurons_in,16),
            ResNetBlock(16,32),
            ResNetBlock(32,64),
            ResNetBlock(64, 32),
            torch.nn.Linear(32,neurons_out),
            torch.nn.Sigmoid()
        ]
        self.network = torch.nn.Sequential(*layers)

    def forward(self,x):
        x = self.network(x)[0]
        return x

data = pd.read_csv("hotel.csv")
data_y = data["is_canceled"]
data = data[["adults", "children", "babies", "is_repeated_guest", "previous_cancellations", "deposit_type"]]
ordinal_encoder = OrdinalEncoder()
data = ordinal_encoder.fit_transform(data)
data = pd.DataFrame(data)
train_x = data.iloc[:int(len(data)*.8)]
train_y = data_y.iloc[:int(len(data_y)*.8)]
test_x = data.iloc[int(len(data)*.8):]
test_y = data_y.iloc[int(len(data_y)*.8):]

train = data_utils.TensorDataset(torch.Tensor(np.array(train_x)), torch.Tensor(np.array(train_y)))
train_loader = data_utils.DataLoader(train, batch_size=1, shuffle=True)
test = data_utils.TensorDataset(torch.Tensor(np.array(test_x)), torch.Tensor(np.array(test_y)))
test_loader = data_utils.DataLoader(test, batch_size=1, shuffle=True)
#training-loop
model = nn(6,1)
n_epochs = 1
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
loss = torch.nn.BCEWithLogitsLoss()

for epoch in range(n_epochs):
    print(epoch)
    for input, label in train_loader:
        output = model(input)
        class_loss = loss(output, label)
        optimizer.zero_grad()
        class_loss.backward()
        optimizer.step()

num_corr = 0
num_sam = 0

with torch.no_grad():
    for x,y in test_loader:
        pred = model(x)
        if pred.item() > .5:
            pred = torch.Tensor([1])
        else:
            pred = torch.Tensor([0])
        if pred == y:
            num_corr += 1
        num_sam += 1

print(num_corr/num_sam)


