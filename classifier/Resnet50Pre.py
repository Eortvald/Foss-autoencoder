# MyResNet50
from __future__ import print_function
from __future__ import division
from data.dataload_collection import Ktest_loader, Ktrain_loader
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import matplotlib.pylab as plt
import numpy as np
import time
import copy
from datetime import datetime
from torchsummary import summary

# Number of classes in the dataset
num_classes = 7
# Batch size for training (change depending on how much memory you have)
batch_size = 300

# Number of epochs to train for
num_epochs = 100

# Flag for feature extracting. When False, we finetune the whole model,
#   when True we only update the reshaped layer params
feature_extract = True

def buildResNet50Model(numClasses):
    # get the stock PyTorch ResNet50 model w/ pretrained set to True
    model = torchvision.models.resnet34(pretrained=True)

    # freeze all model parameters so we don’t backprop through them during training (except the FC layer that will be replaced)
    for param in model.parameters():
        param.requires_grad = False
    # end for

    model.conv1 = nn.Conv2d(8, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

    return model
# end function
def train_model(model, dataloaders, criterion, optimizer, num_epochs):
    since = time.time()

    train_acc_history = []
    val_acc_history = []
    val_loss_history = []
    train_loss_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0


    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds[1] == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects / len(dataloaders[phase].dataset)
            if phase == 'train':
                train_loss_history.append(epoch_loss)
                train_acc_history.append(epoch_acc)
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_epoch = epoch
                best_model_wts = copy.deepcopy(model.state_dict())
                best_loss = epoch_loss
            if phase == 'val':
                val_acc_history.append(epoch_acc)
                val_loss_history.append(epoch_loss)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    print('Best val loss: {:4f}'.format(best_loss))
    print('Best val Epoch {:0f}'.format(best_epoch))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history, val_loss_history, train_loss_history, train_acc_history

def initialize_model(num_classes):
    # Initialize these variables
    model_ft = None
    input_size = 0
    model_ft = buildResNet50Model(num_classes)
    #set_parameter_requires_grad(model_ft, feature_extract)
    num_ftrs = model_ft.fc.in_features#[0]#.in_features
    model_ft.fc = nn.Linear(num_ftrs, num_classes)
    input_size = 224
    return model_ft, input_size

#initialize model
model_ft, input_size = initialize_model(num_classes)
print(summary(model_ft.cuda(),(8,224,224)))
print("Initializing Datasets and Dataloaders...")
dataloaders_dict = {'train': Ktrain_loader,
                    'val': Ktest_loader
}
# Device configuration
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
print(device)

model_ft = model_ft.to(device)

params_to_update = model_ft.parameters()
print("Params to learn:")
if feature_extract:
    params_to_update = []
    for name,param in model_ft.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
            print("\t",name)
else:
    for name,param in model_ft.named_parameters():
        if param.requires_grad == True:
            print("\t",name)

# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(params_to_update, lr=0.001, momentum=0.9)

criterion = nn.CrossEntropyLoss()
model_ft, val_acc_history, val_loss_history, train_loss_history, train_acc_history= train_model(model_ft, dataloaders_dict, criterion, optimizer_ft, num_epochs=num_epochs)

#torch.save(model_ft.state_dict(), './model_dicts/Res18.pth')
#val_loss_history1 = val_loss_history
#train_loss_history1 = train_loss_history
#val_acc_history1 = val_acc_history
#train_acc_history1 = train_acc_history

np.save('./res34test', torch.tensor(val_loss_history).cpu().numpy())
np.save('./res34train', torch.tensor(train_loss_history).cpu().numpy())
np.save('./res34acctest', torch.tensor(val_acc_history).cpu().numpy())
np.save('./res34acctrain',torch.tensor(train_acc_history).cpu().numpy())

img_name = f"../plots/classifier-plots/ResNet_Results-{str(datetime.now())[5:-10].replace(' ', '_').replace(':', '-')}.png"
plt.plot(np.arange(len(train_loss_history)), train_loss_history, label='Train')  # etc.
plt.plot(np.arange(len(val_loss_history)), val_loss_history, label='Test')
minpos = val_loss_history.index(min(val_loss_history))
plt.axvline(minpos, linestyle="--", color='r', label ='Minimum loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title("ResNet34: Train vs Test loss")
plt.legend()
plt.savefig(img_name, transparent=False)