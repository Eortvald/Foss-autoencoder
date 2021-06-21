import numpy as np
import matplotlib.pyplot as plt


train_loss = np.array([])


test_loss = np.array([])





figA, ax = plt.subplots()
ax.plot(train_loss, label='Train')
ax.plot(test_loss, label='Test')
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.set_title(f"Train vs Test \t - Grain \nBatch size:3000 | z_dim:30 | Dataset size: 120K")
ax.legend()
figA.savefig(f"../plots/autoencoder-plots/Final_AEsession_loss.png")
plt.show()

