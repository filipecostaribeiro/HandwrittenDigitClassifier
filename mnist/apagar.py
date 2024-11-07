#%%
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
import struct
import pickle
#%%

import numpy as np
import struct

def load_mnist_images(filename):
    """Lê o arquivo de imagens MNIST e retorna um array NumPy com as imagens."""
    with open(filename, 'rb') as f:
        # Ler o cabeçalho do arquivo: primeiro 16 bytes
        magic, num_images, rows, cols = struct.unpack('>IIII', f.read(16))
        # Ler o restante dos dados e converter para numpy
        images = np.frombuffer(f.read(), dtype=np.uint8)
        # Reshape para (n_imagens, altura, largura)
        images = images.reshape(num_images, rows, cols)
    return images

def load_mnist_labels(filename):
    """Lê o arquivo de rótulos MNIST e retorna um array NumPy com os labels."""
    with open(filename, 'rb') as f:
        # Ler o cabeçalho do arquivo: primeiro 8 bytes
        magic, num_labels = struct.unpack('>II', f.read(8))
        # Ler os rótulos restantes e converter para numpy
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return labels

# Exemplo de uso:
train_images = load_mnist_images('train-images.idx3-ubyte')
train_labels = load_mnist_labels('train-labels.idx1-ubyte')

print(f"Imagens de treino: {train_images.shape}")  # Ex.: (60000, 28, 28)
print(f"Rótulos de treino: {train_labels.shape}")  # Ex.: (60000,)

#%%
indices_0 = np.where(train_labels == 0)[0]

# Índices dos elementos iguais a 1
indices_1 = np.where(train_labels == 1)[0]

X0 = train_images[indices_0]
X0 = X0.reshape(-1,28*28)
X1 = train_images[indices_1]
X1 = X1.reshape(-1,28*28)

X = np.vstack([X0,X1])

a = train_labels[indices_0]
b = train_labels[indices_1]
Y = np.concatenate([a,b])
Y= Y.reshape(-1,1)
print(X.shape)
print(Y.shape)
#%%
