import torch
import numpy
from sklearn.metrics import accuracy_score
from .nn import Linear
from .nn import ReLU
from .nn import Module
from .optim import SGD
from .nn import CrossEntropyLoss
from .nn import Softmax
from data.dataload_collection import *

traindataloader = Ktrain_loader
testdataloader = Ktest_loader

#Hyperparameters
learningrate = 0.001#Insert LR
epochs = 5 #Insert epochs
hidden_out = [32,24,16,8]

#ANN Definition
class ANN(Module):
    def __init__(self, n_inputs):
        super(ANN, self).__init__()
        self.hidden1 = Linear(n_inputs, hidden_out[0])
        self.af1 = ReLU()
        self.hidden2 = Linear(hidden_out[0],hidden_out[1])
        self.af2 = ReLU()
        self.hidden3 = Linear(hidden_out[1],hidden_out[2])
        self.af3 = ReLU()
        self.hidden4 = Linear(hidden_out[2],hidden_out[3])
        self.af4 = Softmax()

    def forward(self,X):
        X = self.hidden1(X)
        X = self.af1(X)
        X = self.hidden2(X)
        X = self.af2(X)
        X = self.hidden3(X)
        X = self.af3(X)
        #last layer and output
        X = self.hidden4(X)
        X = self.af4(X)
        return X

#train model
def train_model(traindataloader, model):
    #Optimizing
    criterion = CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=learningrate, momentum=0.9)
    #Epochs
    for epoch in range(epochs):
        for i, (inputs, targets) in enumerate(traindataloader):
            optimizer.zero_grad()
            yhat = model(inputs)
            loss = criterion(yhat,targets)
            loss.backward()
            optimizer.step()

def model_evaluate(testdataloader, model):
    predictions, actuals = list(), list()
    for i, (inputs, targets) in enumerate(testdataloader):
        #Evaluating model on test set
        yhat = model(inputs)
        yhat = yhat.detach().numpy()
        actual = targets.numpy()
        #Converting to class labels
        yhat = argmax(yhat,axis=1)
        #Reshaping
        yhat = yhat.reshape((len(yhat),1))
        actual = actual.reshape((len(actual), 1))
        #Lists
        predictions.append(yhat)
        actuals.append(actual)
    predictions, actuals = vstack(predictions), vstack(actuals)
    #Accuracy calculation
    accuracy = accuracy_score(actuals, predictions)
    return accuracy
def predict(row, model):
    row = Tensor([row])
    yhat = model(row)
    yhat = yhat.detach().numpy()
    return yhat

model = ANN(60)
train_model(traindataloader, model)
acc = model_evaluate(testdataloader, model)
print ('Accuracy: %.4f' % acc)

yhat = predict(row,model)
print('Predicted: %.4f (class=%d)' % (yhat, yhat.round()))
