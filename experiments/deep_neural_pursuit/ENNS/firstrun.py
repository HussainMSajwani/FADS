from DNP import DeepNet
import numpy as np
from torch import tensor

dnp = DeepNet(10)

X_ = np.load("/home/shussain/experiments/deep_neural_pursuit/X_train.npy")
y_ = np.load("/home/shussain/experiments/deep_neural_pursuit/y_train.npy")


X = dnp.add_bias(tensor(X_))
y = tensor(y_)

ret = dnp.train(X, y, return_select=True)

print("")