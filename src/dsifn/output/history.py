import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

history = pd.read_csv('src/dsifn/output/log.csv')

plt.subplot(1, 2, 1)
plt.plot(history['epoch'], history['Train_loss'])
plt.plot(history['epoch'], history['Test_loss'])
plt.legend(['Train_loss', 'Test_loss'])
plt.xlabel('Epochs')
plt.ylabel('Cross-Entropy Loss')
plt.title('DSIFN Loss vs Epochs')

plt.subplot(1, 2, 2)
plt.plot(history['epoch'], history['Train_f1_score'])
plt.plot(history['epoch'], history['Test_f1_score'])
plt.legend(['Train_f1_score', 'Test_f1_score'])
plt.xlabel('Epochs')
plt.ylabel('F1 Score')
plt.title('DSIFN F1 Score vs Epochs')

plt.show()