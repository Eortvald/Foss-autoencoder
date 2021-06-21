import numpy as np
import matplotlib.pylab as plt


testaccuracyres18 = np.load('res18acctest.npy')
trainaccuracyres18 = np.load('res18acctrain.npy')
testlossres18 = np.load('res18test.npy')
trainlossres18 = np.load('res18train.npy')
testaccuracyres34 = np.load('res34acctest.npy')
trainaccuracyres34 = np.load('res34acctrain.npy')
testlossres34 = np.load('res34test.npy')
trainlossres34 = np.load('res34train.npy')
testaccuracyres50 = np.load('res50acctest.npy')
trainaccuracyres50 = np.load('res50acctrain.npy')
testlossres50 = np.load('res50test.npy')
trainlossres50 = np.load('res50train.npy')

#Accuracy plots
fig, (ax1,ax2, ax3) = plt.subplots(1,3, sharey=True, constrained_layout=True)
#plt.supxlabel('Epochs')
plt.suptitle("ResNet accuracies")
ax1.plot(np.arange(len(testaccuracyres18)), testaccuracyres18, label='Resnet 18 Test', color='tab:blue')
ax1.plot(np.arange(len(trainaccuracyres18)), trainaccuracyres18, label='Resnet 18 Train',color='tab:blue', linestyle=':')
ax1.set_title("Resnet 18")
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Accuracy')
ax1.legend(loc='lower right')
ax2.plot(np.arange(len(testaccuracyres34)), testaccuracyres34, label='Resnet 34 Test', color='tab:orange')
ax2.plot(np.arange(len(trainaccuracyres34)), trainaccuracyres34, label='Resnet 34 Train', color='tab:orange', linestyle=':')
ax2.set_title("Resnet 34")
ax2.legend(loc='lower right')
ax2.set_xlabel('Epochs')
ax3.plot(np.arange(len(testaccuracyres50)), testaccuracyres50, label='Resnet 50 Test', color='tab:green')
ax3.plot(np.arange(len(trainaccuracyres50)), trainaccuracyres50, label='Resnet 50 Train', color='tab:green', linestyle=':')
ax3.set_title("Resnet 50")
ax3.legend(loc='lower right')
ax3.set_xlabel('Epochs')

img_name = f"../plots/classifier-plots/ResNetAccuracies{str(datetime.now())[5:-10].replace(' ', '_').replace(':', '-')}.png"
plt.savefig(img_name, transparent=False)

#Loss plots
fig, (ax4,ax5,ax6) = plt.subplots(3,1, sharex=True, constrained_layout=True)
#plt.supxlabel('Epochs')
plt.suptitle("ResNet losses")
ax4.plot(np.arange(len(testlossres18)), testlossres18, label='Resnet 18 Test', color='tab:blue')
ax4.plot(np.arange(len(trainlossres18)), trainlossres18, label='Resnet 18 Train',color='tab:blue', linestyle=':')
ax4.set_title("Resnet 18")
ax4.set_xlabel('Epochs')
ax4.set_ylabel('Loss')
ax4.legend(loc='upper right')
ax5.plot(np.arange(len(testlossres34)), testlossres34, label='Resnet 34 Test', color='tab:orange')
ax5.plot(np.arange(len(trainlossres34)), trainlossres34, label='Resnet 34 Train', color='tab:orange', linestyle=':')
ax5.set_title("Resnet 34")
ax5.legend(loc='upper right')
ax5.set_xlabel('Epochs')
ax6.plot(np.arange(len(testlossres50)), testlossres50, label='Resnet 50 Test', color='tab:green')
ax6.plot(np.arange(len(trainlossres50)), trainlossres50, label='Resnet 50 Train', color='tab:green', linestyle=':')
ax6.set_title("Resnet 50")
ax6.legend(loc='upper right')
ax6.set_xlabel('Epochs')

img_name = f"../plots/classifier-plots/ResNetLosses{str(datetime.now())[5:-10].replace(' ', '_').replace(':', '-')}.png"
plt.savefig(img_name, transparent=False)