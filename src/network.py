import torch
import pickle 
from torch.nn import Module, LeakyReLU, Dropout, Conv1d, Flatten, Linear
import matplotlib.pyplot as plt
from torch.nn import CrossEntropyLoss
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from torch.optim import Adam
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
import random
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
seed = 0
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
 
def normalize_input(x):
    m = np.mean(x, axis=1, keepdims=True)
    X = x - m
    return X

class GlassesNet(Module):

    def __init__(self):

        super(GlassesNet, self).__init__()
        self.lrelu = LeakyReLU()
        self.dropout1 = Dropout(0)
        self.dropout2 = Dropout(0)
        self.dropout3 = Dropout(0)
        self.dropout4 = Dropout(0.25)
        self.layer1 = Conv1d(7, 8, 7, 3)
        self.layer2 = Conv1d(8, 16, 5, 2)
        self.layer3 = Conv1d(16, 16, 3, 2)
        self.layer4 = Conv1d(16, 16, 3, 2)
        self.flatten = Flatten()
        self.fc = Linear(112, 3)

    def forward(self, x):
        y = self.dropout1(self.lrelu(self.layer1(x)))
        y = self.dropout2(self.lrelu(self.layer2(y)))
        y = self.dropout3(self.lrelu(self.layer3(y)))
        y = self.dropout4(self.lrelu(self.layer4(y)))
        y = self.flatten(y)
        y = self.fc(y)
        return y

class SimpleDataset(Dataset):

    def __init__(self, x, y):

        super(SimpleDataset, self).__init__()
        self.x = torch.tensor(x).permute(0, 2, 1).float()
        self.y = torch.tensor(y).long()

    def __len__(self):
        return len(self.x)

    def __getitem__(self, ind):
        return self.x[ind], self.y[ind]
         

with open("../dataset/data.pickle", "rb") as f:
    data = pickle.load(f)


for key in data:
    if key != "test":
        data[key] = np.array(data[key])
print(data.keys())
print(data["walk"].shape, data["up"].shape, data["down"].shape)
X = np.concatenate([data["walk"], data["up"], data["down"]], axis=0)
X = normalize_input(X)
y = np.concatenate([
    np.zeros((len(data["walk"]), 1)),
    np.ones((len(data["up"]), 1)),
    np.ones((len(data["down"]), 1)) * 2
], axis=0).astype(np.int64)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle=True, stratify=y, random_state=seed)
y_train = y_train[:, 0]
y_test = y_test[:, 0]

dataset = SimpleDataset(X_train, y_train)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=12)
model = GlassesNet()
model.to(device)

epochs = 30
optimizer = Adam(model.parameters())
loss_fn = CrossEntropyLoss()

train_acc = []
val_acc = []
best_predictions = None
best_loss = np.inf
best_model = None
for epoch in range(epochs):
    model.train()
    train_acc.append(0)
    print("Epoch:", epoch)
    for X_tr, y_tr in dataloader:
        X_tr = X_tr.to(device)
        y_tr = y_tr.to(device)
        y_pred = model(X_tr)
        optimizer.zero_grad()
        loss = loss_fn(y_pred, y_tr)
        loss.backward()
        optimizer.step()
        train_acc[-1] += torch.sum((torch.argmax(y_pred, axis=1) == y_tr).float()).cpu().numpy()
    train_acc[-1] /= len(dataset)
    model.eval()
    y_val = model(torch.tensor(X_test).permute(0, 2, 1).to(device).float())
    accuracy = torch.mean((torch.argmax(y_val, axis=1) == torch.tensor(y_test).to(device)).float()).cpu().numpy()
    loss = loss_fn(y_val.cpu(), torch.tensor(y_test)).detach().numpy()
    print("Accuracy", accuracy, "Loss", loss)
    val_acc.append(accuracy)
    if loss < best_loss:
        best_loss = loss
        best_predictions = torch.argmax(y_val, axis=1).cpu().numpy()
        best_model = model.state_dict()

model.load_state_dict(best_model)
model = model.cpu()
model.eval()
cm = confusion_matrix(y_test, best_predictions, normalize="pred")
ax = sns.heatmap(cm, annot=True, cmap="Blues")
ax.set_xlabel("Predicted")
ax.set_ylabel("True")
plt.show()
plt.plot(train_acc, label="train")
plt.plot(val_acc, label="validation")
plt.legend()
plt.show()

test = data["test"]
for walk in test:
    walk = normalize_input(walk)
    walk = torch.tensor(walk).float().permute(0, 2, 1)
    colors = ["red", "green", "blue"]
    c_test = []
    prediction = model(walk)
    for label in torch.argmax(prediction, axis=1).reshape(-1):
        c_test.append(colors[label])
    plt.scatter(np.arange(len(prediction)), torch.argmax(prediction, axis=1), c = c_test)
    plt.yticks([0, 1, 2], ["walk", "up", "down"])
    plt.show()
    prediction = torch.nn.functional.softmax(prediction, dim=1).T.reshape(1, 3, -1)
    prediction = torch.nn.ReflectionPad1d(2)(prediction)
    prediction = torch.nn.functional.conv1d(prediction, torch.ones((3, 1, 5))/5, padding="valid", groups=3).detach().cpu().numpy()[0].T
    c_test = []
    position = 0
    positions = []
    for label in np.argmax(prediction, axis=1).reshape(-1):
        position += (((label + 1) % 3) - 1)
        positions.append(position)
        c_test.append(colors[label])
    plt.scatter(np.arange(len(prediction)), np.argmax(prediction, axis=1), c = c_test)
    plt.yticks([0, 1, 2], ["walk", "up", "down"])
    plt.figure()
    plt.plot(positions)
    plt.show()

