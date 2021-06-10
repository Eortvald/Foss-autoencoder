import torch
import numpy as np
from sklearn.metrics import accuracy_score
from torch import nn, optim
from data.dataload_collection import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device} device')

traindataloader = Ktrain_loader
testdataloader = Ktest_loader


# ANN Definition

class ANN(nn.Module):

    def __init__(self, n_inputs, hidden_out):
        super(ANN, self).__init__()

        self.hidden1 = nn.Linear(n_inputs, hidden_out[0])
        self.af1 = nn.ReLU()
        self.hidden2 = nn.Linear(hidden_out[0], hidden_out[1])
        self.af2 = nn.ReLU()
        self.hidden3 = nn.Linear(hidden_out[1], hidden_out[2])
        self.af3 = nn.ReLU()
        self.hidden4 = nn.Linear(hidden_out[2], hidden_out[3])
        self.af4 = nn.Softmax()

    def forward(self, X):
        X = self.hidden1(X)
        X = self.af1(X)
        X = self.hidden2(X)
        X = self.af2(X)
        X = self.hidden3(X)
        X = self.af3(X)
        # last layer and output
        X = self.hidden4(X)
        # X = self.af4(X)  #Disabled soft max!
        return X


learningrate = 0.001


# train model
def train_model(traindataloader, model, ENC):
    # Optimizing
    train_loss = 0
    dataset_size = len(traindataloader.dataset)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learningrate, momentum=0.9)

    for i, (inputs, targets) in enumerate(traindataloader):
        # ENCODER HERE
        inputs = inputs.to(device)
        inputs = ENC(inputs).to(device)
        print(inputs)

        optimizer.zero_grad()
        yhat = model(inputs)
        loss = criterion(yhat, targets)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()

    train_loss /= dataset_size
    print(f'Avg train loss: {train_loss}')


# test model

def model_evaluate(testdataloader, model, ENC):
    dataset_size = len(traindataloader.dataset)
    test_loss = 0
    criterion = nn.CrossEntropyLoss()
    correct = 0
    total = 0
    with torch.no_grad():
        for i, (inputs, label) in enumerate(testdataloader):
            # Evaluating model on test set
            # ENCODER HERE
            inputs = inputs.to(device)
            inputs = ENC(inputs).to(device)
            yhat = model(inputs)
            # Purely for print statement
            loss = criterion(yhat, label)
            test_loss += loss.item()

            _, yhat = torch.max(yhat, 1)

            for label, prediction in zip(label, yhat):
                if label == yhat:
                    correct += 1
                total += 1

    ACC = 100 * float(correct) / total

    # Accuracy calculation and print
    test_loss /= dataset_size
    print(f'Avg. test loss {test_loss}  | Accuray: {ACC}')


if __name__ == "__main__":

    model = ANN(30)
    model = model.to(device)
    learningrate = 0.001  # Insert LR
    epochs = 5  # Insert epochs
    hidden_out = [64, 32, 16, 8]
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learningrate, momentum=0.9)

    for epoch in range(epochs):
        print(f'\n\t\t------------------------------Epoch: {epoch + 1}------------------------------')
        train_model(traindataloader, model, ENC=None)

        print('\t\t\t>>>>>>>>>>>>>>>>TEST RESULTS<<<<<<<<<<<<<<<<<')
        model_evaluate(testdataloader, model, ENC=None)
