import imageio
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import cm
from matplotlib.ticker import LinearLocator
from mpl_toolkits.mplot3d import Axes3D
from torch.utils.data import DataLoader, Dataset, TensorDataset

import utils as u

matplotlib.use("TkAgg")


#%% # Make data.
nf = 0
X = np.arange(1, 10, 0.25)
xlen = len(X)
Y = np.arange(1, 10, 0.25)
ylen = len(Y)
X, Y = np.meshgrid(X, Y)

if nf == 0:
    Z = u.func_sum(X, Y)
elif nf == 1:
    Z = u.func_prod(X, Y)
elif nf == 2:
    Z = u.func_divide(X, Y)
elif nf == 3:
    Z = u.func_weight(X, Y)
x = np.stack((X.reshape(-1), Y.reshape(-1)), axis=1)
# z = Z.reshape(-1)
z = X.reshape(-1) + Y.reshape(-1)

#%%
n = 1000
batch_size = 64
EPOCH = 100
train_data, train_labels, valid_data, valid_labels = u.partition_dataset(
    x, z
)  # To complete.
train_dataset = u.create_dataset(train_data, train_labels, n)
train_dataloader = DataLoader(
    train_dataset, batch_size=batch_size, drop_last=False, shuffle=True
)
hidden_dim = [2]

net = u.Simple_NN(2, 1, hidden_dim)
net1 = u.Simple_NN(2, 1, hidden_dim)
# with torch.no_grad():
#     net.layers[0].weight.copy_(torch.eye(2))
#     net.layers[1].weight.copy_(torch.ones(1, 2))
#     net.layers[0].bias.copy_(torch.zeros(2))
#     net.layers[1].bias.copy_(torch.zeros(1))

#%%
optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
loss_func = torch.nn.MSELoss()  # this is for regression mean squared loss
fig1 = plt.figure()
axo = fig1.add_subplot(111, projection="3d")
my_images = []
# start training

for epoch in range(EPOCH):
    print(epoch)

    for step, (batch_x, batch_y) in enumerate(
        train_dataloader
    ):  # for each training step

        # b_x = Variable(batch_x)
        # b_y = Variable(batch_y)
        # net.layers[0]._parameters['weight'] = torch.eye(2)
        # net.layers[0]._parameters['bias'] = torch.zeros(2, 2)

        prediction = net(batch_x)  # input x and predict based on x

        loss = loss_func(
            prediction, batch_y.view(-1, 1)
        )  # must be (1. nn output, 2. target)

        optimizer.zero_grad()  # clear gradients for next train
        loss.backward()  # backpropagation, compute gradients
        optimizer.step()  # apply gradients

        if step == 1:
            # plot and show learning process

            plt.cla()
            axo.set_title("UAT - 2D Function", fontsize=10)
            axo.set_xlabel("Independent variable 1", fontsize=10)
            axo.set_ylabel("Independent variable 2", fontsize=10)
            axo.set_zlabel("Dependent variable", fontsize=0)
            axo.set_xlim(0, 10.0)
            axo.set_ylim(0, 10.0)
            axo.set_zlim(0, 20)
            axo.scatter(
                batch_x.data[:, 0].numpy(),
                batch_x.data[:, 1].numpy(),
                prediction.data.numpy(),
                color="green",
                alpha=0.8,
            )
            axo.plot_wireframe(
                X, Y, u.func_sum(X, Y), color="red", rstride=10, cstride=10
            )

            # Used to return the plot as an image array
            # (https://ndres.me/post/matplotlib-animated-gifs-easily/)
            fig1.canvas.draw()  # draw the canvas, cache the renderer
            image = np.frombuffer(fig1.canvas.tostring_rgb(), dtype="uint8")
            image = image.reshape(fig1.canvas.get_width_height()[::-1] + (3,))

            my_images.append(image)

# save images as a gif
imageio.mimsave("Final Result.gif", my_images, fps=12)
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
plt.cla()
ax.set_title("Regression Analysis - model 3, Batches", fontsize=10)
ax.set_xlabel("Independent variable 1", fontsize=10)
ax.set_ylabel("Independent variable 2", fontsize=10)
ax.set_zlabel("Dependent variable", fontsize=0)
ax.set_xlim(0, 10.0)
ax.set_ylim(0, 10.0)
ax.set_zlim(0, 20)
ax.plot_wireframe(X, Y, u.func_sum(X, Y), color="red", rstride=10, cstride=10)
prediction = net(torch.tensor(x, dtype=torch.float))  # input x and predict based on x
ax.scatter(
    X.reshape(-1), Y.reshape(-1), prediction.data.numpy(), color="green", alpha=0.2
)
plt.show()
plt.savefig("./final result.png")
#
