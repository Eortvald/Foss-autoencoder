import numpy as np
import matplotlib as plt


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