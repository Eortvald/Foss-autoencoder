import torch
import numpy as np
from sklearn.metrics import accuracy_score
import matplotlib.pylab as plt
from torch import nn, optim
from data.MyDataset import *
from autoencoder.CAE_model import *
from datetime import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device} device')


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

    def forward(self, X):
        X = self.hidden1(X)
        X = self.af1(X)
        X = self.hidden2(X)
        X = self.af2(X)
        X = self.hidden3(X)
        X = self.af3(X)
        return X


learningrate = 0.001


# train model
def train_model(traindataloader, model, ENC):
    # Optimizing
    train_loss = 0
    dataset_size = len(traindataloader.dataset)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learningrate, momentum=0.9)
    # optimizer = optim.Adam(model.parameters(), lr=learningrate)

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
        print(f'{len(inputs)*(i+1)}/{dataset_size}')

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

    confusion = np.zeros((7, 7))

    label_correct = {1.: 0,
                     2.: 0,
                     3.: 0,
                     4.: 0,
                     5.: 0,
                     6.: 0,
                     7.: 0}

    label_total = {1.: 0,
                   2.: 0,
                   3.: 0,
                   4.: 0,
                   5.: 0,
                   6.: 0,
                   7.: 0}

    with torch.no_grad():
        for i, (img, label) in enumerate(testdataloader):
            # Evaluating model on test set
            # ENCODER HERE
            label = label.to(device)
            inputs = img.to(device)
            inputs = ENC(inputs)
            yhat = model(inputs)
            # Purely for print statement
            print(yhat)
            loss = criterion(yhat, label)
            print(loss)
            test_loss += loss.item()


            _, yhat = torch.max(yhat, 1)
            print('~~~~~~~~~~~~~~~~~~~~~~~~~')
            print(yhat)
            for la, pre in zip(label.detach().cpu().numpy(), yhat.detach().cpu().numpy()):
                print(la)
                print(pre)

                label_total[la] += 1
                if pre == la:
                    label_correct[pre] += 1
                    correct += 1
                total += 1

                confusion[int(la - 1)][int(pre - 1)] += 1

        print(confusion)
        ACC = 100 * float(correct) / total

        c_acc = [label_correct[i] / label_total[i] * 100 for i in [1., 2., 3., 4., 5., 6., 7.]]

        per_class = {'Oat': c_acc[1],
                     'Broken': c_acc[2],
                     'Rye': c_acc[3],
                     'Wheat': c_acc[4],
                     'BarleyGreen': c_acc[5],
                     'Cleaved': c_acc[6],
                     'Skinned': c_acc[7]}

        # Accuracy calculation and print
        test_loss /= dataset_size
        print(f'Avg. test loss {test_loss}  | Accuracy: {ACC} \n Class Accuracy {per_class}')
        print(confusion)

    return test_loss


if __name__ == "__main__":

    MEAN = np.load('../MEAN.npy')
    STD = np.load('../STD.npy')

    label_path = '../preprocess/Classifier_labels.csv'
    PATH_dict = {
        '10K_remote': 'M:/R&D/Technology access controlled/Projects access controlled/AIFoss/Data/Foss_student/tenkblobs/',
        '10K_gamer': 'C:/ASB/Projects/EyefossAutoencoder/Fagprojekt-2021/tenkblobs/',
        '224': 'M:/R&D/Technology access controlled/Projects access controlled/AIFoss/Data/Foss_student/tenhblobsA/',
        'validation_grain': 'C:/ASB/Projects/EyefossAutoencoder/Fagprojekt-2021/validation_grain/',
        'validation_blob': 'C:/ASB/Projects/EyefossAutoencoder/Fagprojekt-2021/validation_blob/',
        'grainmix': 'C:/ASB/Projects/EyefossAutoencoder/Fagprojekt-2021/grainmix/'
    }

    DATA_SET = 'validation_blob'
    PATH = PATH_dict[DATA_SET]

    BSIZE = 5
    classes = ['Oat', 'Broken', 'Rye', 'Wheat', 'BarleyGreen', 'Cleaved', 'Skinned']
    hidden_out = [16, 12, 10]
    ANN_10Kmodel = ANN(30, hidden_out)
    ANN_10Kmodel = ANN_10Kmodel.to(device)
    num_epochs = 50
    learning_rate = 1e-3
    w_decay = 1e-5
    PIN = True
    torch.cuda.empty_cache()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    aemodel = CAE(z_dim=30).to(device)
    aemodel.load_state_dict(torch.load('../autoencoder/model_dicts/PTH_Grain/CAE_69.pth', map_location=device))
    aemodel.eval()

    TFORM = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=MEAN, std=STD)])
    ENCO = lambda img: aemodel.encode(img)

    traindata = KornDataset(data_path=PATH + '/train/', transform=TFORM, label_path=label_path)
    trainload = DataLoader(traindata, batch_size=BSIZE, shuffle=True, num_workers=0, pin_memory=PIN)

    testdata = KornDataset(data_path=PATH + '/test/', transform=TFORM, label_path=label_path)
    testload = DataLoader(testdata, batch_size=BSIZE, shuffle=True, num_workers=0, pin_memory=PIN)

    train_log = []
    test_log = []

    start_time = timeit.default_timer()
    for epoch in range(num_epochs):
        print(f'\n\t\t------------------------------Epoch: {epoch + 1}------------------------------')
        tr_loss = train_model(trainload, ANN_10Kmodel, ENCO)
        train_log.append(tr_loss)

        print('\t\t\t>>>>>>>>>>>>>>>>TEST RESULTS<<<<<<<<<<<<<<<<<')

        te_loss = model_evaluate(testload, ANN_10Kmodel, ENCO)
        test_log.append(te_loss)

        torch.save(ANN_10Kmodel.state_dict(), f'classifier/model_dicts/ANN_{num_epochs}.pth')

    end_time = timeit.default_timer() - start_time

    print(end_time)

    torch.save(ANN_10Kmodel.state_dict(), 'classifier/model_dicts/ANN_Big.pth')

    SESSION = str(datetime.now())[5:-10].replace(' ', '_').replace(':', '-')
    np.savez(f'model_dicts/session_results-{SESSION}', train_log, test_log)

    figR, ax = plt.subplots()
    ax.plot(train_log, label='Train')
    ax.plot(test_log, label='Test')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title(f"Train vs Test \t - {DATA_SET} \nBatch size:{BSIZE} | Dataset size:{len(trainload.dataset)}")
    ax.legend()
    figR.savefig(f"../plots/autoencoder-plots/session_results-{SESSION}.png")

    # https://towardsdatascience.com/pytorch-tabular-multiclass-classification-9f8211a123ab

"""
#DONT REMOVE IS NEEDED LATER
from sklearn import metrics

# Constants
C="Cat"
F="Fish"
H="Hen"

# True values
y_true = [C,C,C,C,C,C, F,F,F,F,F,F,F,F,F,F, H,H,H,H,H,H,H,H,H]
# Predicted values
y_pred = [C,C,C,C,H,F, C,C,C,C,C,C,H,H,F,F, C,C,C,H,H,H,H,H,H]

# Print the confusion matrix
print(metrics.confusion_matrix(y_true, y_pred))

# Print the precision and recall, among other metrics
print(metrics.classification_report(y_true, y_pred, digits=3))

"""
