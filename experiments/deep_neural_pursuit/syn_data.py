from numpy.random import uniform
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from numpy.random import normal
from tensorflow import device, convert_to_tensor
from functools import partial
import numpy as np


def init_(shape, dtype=None, m=2):
    out = normal(0, 0.5, size=shape)
    out[m:] = 0.0
    return convert_to_tensor(out, dtype=dtype)

def syn(m):
    X = uniform(-1, 1, size=(8500, 10000))
    init = partial(init_, m=m)
    with device("/CPU"):
        model = Sequential([
            Dense(50, input_shape=(10000,), activation="relu",  kernel_initializer=init),
            Dense(30, activation="relu"),
            Dense(15, activation="relu"),
            Dense(10, activation="relu"),
            Dense(1, activation="sigmoid")
        ])

        model.compile()

        y = 1*(model(X).numpy() > 0.5)

    flip = np.random.choice(range(8500), size=int(8500*0.05))
    
    y[flip, :] = 1-y[flip, :]
    train_size=1000
    return X[:train_size, :], X[train_size:, :], y[:train_size], y[train_size:]