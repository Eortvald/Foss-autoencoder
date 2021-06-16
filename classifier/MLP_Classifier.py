import torch
import numpy as np
from sklearn.metrics import accuracy_score
import matplotlib.pylab as plt
from torch import nn, optim
from data.dataload_collection import *
from autoencoder.CAE_model import *

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

    #        self.hidden2 = nn.Linear(hidden_out[0], hidden_out[1])
    #        self.af2 = nn.ReLU()
    #        self.hidden3 = nn.Linear(hidden_out[1], hidden_out[2])
    #        self.af3 = nn.ReLU()

    def forward(self, X):
        X = self.hidden1(X)
        X = self.af1(X)
        #        X = self.hidden2(X)
        #        X = self.af2(X)
        #        X = self.hidden3(X)
        #        X = self.af3(X)
        return X


learningrate = 0.001


# train model
def train_model(traindataloader, model, ENC):
    # Optimizing
    train_loss = 0
    dataset_size = len(traindataloader.dataset)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learningrate, momentum=0.9)

    for i, (inputs, label) in enumerate(traindataloader):
        # ENCODER HERE
        inputs = inputs.to(device)
        label = label.to(device)
        inputs = ENC(inputs).to(device)
        # print(inputs)

        optimizer.zero_grad()
        yhat = model(inputs)
        # print(f'yhat:{yhat}\n')
        # print(f'label: {label}')
        loss = criterion(yhat, label)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()

    train_loss /= dataset_size
    print(f'Avg train loss: {train_loss}')

    return train_loss


# test model

def model_evaluate(testdataloader, model, ENC):
    dataset_size = len(testdataloader.dataset)
    test_loss = 0
    criterion = nn.CrossEntropyLoss()
    correct = 0
    total = 0

    label_correct = {1: 0,
                     2: 0,
                     3: 0,
                     4: 0,
                     5: 0,
                     6: 0,
                     7: 0,
                     8: 0}

    label_total = {1: 0,
                     2: 0,
                     3: 0,
                     4: 0,
                     5: 0,
                     6: 0,
                     7: 0,
                     8: 0}

    with torch.no_grad():
        for i, (inputs, label) in enumerate(testdataloader):
            # Evaluating model on test set
            # ENCODER HERE
            label = label.to(device)
            inputs = inputs.to(device)
            inputs = ENC(inputs).to(device)
            yhat = model(inputs)
            # Purely for print statement
            loss = criterion(yhat, label)
            test_loss += loss.item()

            _, yhat = torch.max(yhat, 1)

            for la, pre in zip(label.detach().cpu().numpy(), yhat.detach().cpu().numpy()):

                label_total[int(la)] += 1

                if la == pre:
                    label_correct[int(pre)] += 1
                    correct += 1
                total += 1

        ACC = 100 * float(correct) / total

        # Accuracy calculation and print
        test_loss /= dataset_size
        print(f'Avg. test loss {test_loss}  | Accuray: {ACC}')

    return test_loss


if __name__ == "__main__":

    X = torch.ones([1, 8, 200, 89]).to(device)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    aemodel = CAE(z_dim=30).to(device)
    aemodel.load_state_dict(torch.load('./autoencoder/model_dicts/CAE_10Kmodel.pth', map_location=device))
    aemodel.eval()

    ENCO = lambda img: aemodel.encode(img)

    hidden_out = [8, 10, 8]
    ANN_10Kmodel = ANN(30, hidden_out)
    ANN_10Kmodel = ANN_10Kmodel.to(device)
    learningrate = 0.001  # Insert LR
    epochs = 50  # Insert epochs

    train_log = []
    test_log = []

    for epoch in range(epochs):
        print(f'\n\t\t------------------------------Epoch: {epoch + 1}------------------------------')
        tr_loss = train_model(traindataloader, ANN_10Kmodel, ENCO)
        train_log.append(tr_loss)

        print('\t\t\t>>>>>>>>>>>>>>>>TEST RESULTS<<<<<<<<<<<<<<<<<')
        te_loss = model_evaluate(testdataloader, ANN_10Kmodel, ENCO)
        test_log.append(te_loss)

    torch.save(ANN_10Kmodel.state_dict(), 'classifier/model_dicts/ANN_10Kmodel.pth')

    img_name = f"plots/classifier-plots/ANN_10K_Results-{str(datetime.now())[5:-10].replace(' ', '_').replace(':', '-')}.png"
    plt.plot(np.arange(len(train_log)), train_log, label='Train')  # etc.
    plt.plot(np.arange(len(test_log)), test_log, label='Test')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title("Train vs Test loss")
    plt.legend()
    plt.savefig(img_name, transparent=False)
