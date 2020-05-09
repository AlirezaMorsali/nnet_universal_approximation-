import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset


def func_sum(x, y):
    # x : m by n numpy matrix
    # use numpy as np
    # m : input dimention
    # n : input samples

    return x + y


def func_prod(x, y):
    # x : m by n numpy matrix
    # use numpy as np
    # m : input dimention
    # n : input samples

    return np.multiply(x, y)


def func_divide(x, y):
    # x : m by n numpy matrix
    # use numpy as np
    # m : input dimention must be equal to 2
    # n : input samples

    return np.divide(x, y)


def func_weight(x, y):
    # x : m by n numpy matrix
    # use numpy as np
    # m : input dimention must be equal to 2
    # n : input samples

    return np.divide(x, np.power(y, 2))


def partition_dataset(x, z, valid_ratio=0.1, shuffle=True, seed=1234):

    if shuffle:
        np.random.seed(seed)  # Set the random seed of numpy.
        indices = np.random.permutation(x.shape[0])
    else:
        indices = np.arange(x.shape[0])

    train_idx, valid_idx = np.split(indices, [int((1.0 - valid_ratio) * len(indices))])
    train_data, valid_data = x[train_idx], x[valid_idx]
    tgt = np.array(z)
    train_labels, valid_labels = tgt[train_idx].tolist(), tgt[valid_idx].tolist()

    return train_data, train_labels, valid_data, valid_labels


def create_dataset(x, z, n):
    """
  Slice the first n point/labels and create a torch.utils.data.DataLoader.

  Args:
     images: numpy array of images.
     labels: list of labels associated with the images.
     n: the number of images/labels to slice.

  Return:
     A torch.utils.data.TensorDataset to be used with a torch.utils.data.DataLoader.

  """
    data = torch.tensor(x[:n], dtype=torch.float)
    labels = torch.tensor(z[:n], dtype=torch.float)
    dataset = TensorDataset(data, labels)

    return dataset


class Simple_NN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        current_dim = input_dim
        self.layers = nn.ModuleList()

        for hdim in hidden_dim:
            self.layers.append(nn.Linear(current_dim, hdim))
            current_dim = hdim
        self.layers.append(nn.Linear(current_dim, output_dim))

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
        out = self.layers[-1](x)

        return out
