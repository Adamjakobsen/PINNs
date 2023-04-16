import numpy as np
import scipy.io
import torch
import torch.optim as optim
import torch.nn as nn
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

torch.manual_seed(42)
np.random.seed(42)


class PINN1D(nn.Module):
    def __init__(self, num_layers, num_neurons, device):
        """
        Initializes the PINN.

        args:
            num_layers(int):    Number of hidden layers in the PINN.
            num_neurons(int):   Number of neurons in each hidden layer.
            device(str):        Device to run the PINN on.
        """
        super(PINN1D, self).__init__()
        self.num_layers = num_layers
        self.num_neurons = num_neurons

        self.device = device

        layers = [nn.Linear(3, num_neurons), nn.Tanh()]
        for _ in range(num_layers - 2):
            layers += [nn.Linear(num_neurons, num_neurons), nn.Tanh()]
        layers += [nn.Linear(num_neurons, 2)]

        self.model = nn.Sequential(*layers).to(device)

    def forward(self, x):
        return self.model(x)

    def predict(self, x):
        x = torch.tensor(x, dtype=torch.float32)
        return self.model(x).to('cpu').detach().numpy()

    def predict_torch(self, x):
        x = torch.tensor(x, dtype=torch.float32)
        return self.model(x)


def model_loss_2D(pinn, x, y_true, a, k, mu1, mu2, eps, b, h, D):
    """
    Function to compute the PDE loss

    args:
        pinn (torch.nn.Module): instance of the NN class
        x (torch.tensor): input data (t,x,y)
        y (torch.tensor): target data (V,W)
        Y_true (torch.tensor): target data (V,W)
        a (float): 
        k (float): 
        mu1 (float): 
        mu2 (float): 
        eps (float): 
        b (float): 
        h (float): 
        D (float):
        x_left (float): left boundary of the domain
        x_right (float): right boundary of the domain
        T_ic (float): initial time

    returns:
        PDE_loss (torch.tensor): PDE_loss

    """

    t_space = x[:, 0:1].clone().requires_grad_(True)
    x_space = x[:, 1:2].clone().requires_grad_(True)
    y_space = x[:, 2:3].clone().requires_grad_(True)

    x_min = torch.min(x_space)
    x_max = torch.max(x_space)
    y_min = torch.min(y_space)
    y_max = torch.max(y_space)
    T_ic = torch.min(t_space)

    y_pred = pinn(torch.cat([t_space, x_space, y_space], dim=1))
    V, W = y_pred[:, 0:1], y_pred[:, 1:2]
    V_true, W_true = y_true[:, 0:1], y_true[:, 1:2]

    dVdt = torch.autograd.grad(V, t_space, grad_outputs=torch.ones_like(
        V), create_graph=True, allow_unused=True)[0]
    dWdt = torch.autograd.grad(W, t_space, grad_outputs=torch.ones_like(
        W), create_graph=True, allow_unused=True)[0]

    dVdx = torch.autograd.grad(V, x_space, grad_outputs=torch.ones_like(
        V), create_graph=True, allow_unused=True)[0]

    if dVdx is not None:
        dVdxx = torch.autograd.grad(dVdx, x_space, grad_outputs=torch.ones_like(
            dVdx), create_graph=True, allow_unused=True)[0]

    else:
        raise ValueError("Gradients dVdx are not computed correctly.")

    dVdy = torch.autograd.grad(V, y_space, grad_outputs=torch.ones_like(
        V), create_graph=True, allow_unused=True)[0]

    if dVdy is not None:
        dVdyy = torch.autograd.grad(dVdy, y_space, grad_outputs=torch.ones_like(
            dVdy), create_graph=True, allow_unused=True)[0]

    else:
        raise ValueError("Gradients dVdy are not computed correctly.")

    eq_V = dVdt - D*(dVdxx + dVdyy) + k*V*(V-a)*(V-1) + W*V
    eq_W = dWdt - (eps + (mu1 * W) / (mu2 + V)) * (-W - k * V * (V - b - 1))

    loss_V = torch.mean(torch.square(eq_V))
    loss_W = torch.mean(torch.square(eq_W))
    loss_data = torch.mean(torch.square(V - V_true)) + \
        torch.mean(torch.square(W - W_true))

    loss_BC = boundary_loss(x, dVdx, dVdy, x_min, x_max, y_min, y_max)
    loss_IC = initial_condition_loss(x, y_pred, y_true, T_ic)

    loss = loss_V + loss_W + loss_data + loss_BC + loss_IC

    return loss


def boundary_loss(x, dV_dx, dV_dy, x_min, x_max, y_min, y_max):
    on_boundary_x = (x[:, 1:2] == x_min) | (x[:, 1:2] == x_max)
    on_boundary_y = (x[:, 2:3] == y_min) | (x[:, 2:3] == y_max)

    loss_x = torch.mean(torch.square(dV_dx[on_boundary_x]))
    loss_y = torch.mean(torch.square(dV_dy[on_boundary_y]))

    return loss_x + loss_y


def initial_condition_loss(x, y_pred, y_true, T_ic):
    """
    Function to compute the initial condition loss

    args:
        x (torch.tensor): input data (t,x,y)
        y (torch.tensor): target data (V,W)

    returns:
        initial_condition_loss (torch.tensor): initial_condition_loss

    """

    T = x[:, 0:1].reshape(-1, 1)
    T_ic = torch.tensor(T_ic, dtype=torch.float32)

    idx_init = torch.where(torch.isclose(T, T_ic))[0]

    true = y_true[idx_init]
    pred = y_pred[idx_init]

    return torch.mean(torch.square(true - pred))


def train(model, optimizer, data_loader, a, k, mu1, mu2, eps, b, h, D, device, pinn):
    model.train()
    for x, y_true in data_loader:
        x = x.to(device)
        y_true = y_true.to(device)
        x.requires_grad = True  # Set requires_grad to True

        optimizer.zero_grad()  # Clear gradients
        loss = model_loss_2D(pinn, x, y_true, a, k, mu1, mu2, eps, b, h, D)

        loss.backward()  # Compute gradients with backpropagation

        optimizer.step()  # Update parameters

        # Ensure all tensors are on the same device
        for param in model.parameters():
            param.data = param.data.to(device)
            if param.grad is not None:
                param.grad = param.grad.to(device)

    return loss.item()


def generate_data():
    mat = scipy.io.loadmat('./data/data_2d_vertical_planar.mat')
    V = mat['Vsav']
    W = mat['Wsav']
    t = mat['t']
    x = mat['x']
    y = mat['y']

    T, X, Y = np.meshgrid(t, x, y)

    X = X.reshape(-1, 1)
    Y = Y.reshape(-1, 1)
    T = T.reshape(-1, 1)

    V = V.reshape(-1, 1)
    W = W.reshape(-1, 1)

    return np.hstack((T, X, Y)), np.hstack((V, W))


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device: ", device)

    num_neurons = 60
    num_layers = 5
    n_epochs = 2000
    batch_size = 64
    test_size = 0.9

    # Create PINN object
    pinn = PINN1D(num_layers, num_neurons, device).to(device)

    X, Y = generate_data()

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, random_state=True, test_size=test_size)

    # Add noise to training data
    Y_train = Y_train + 0.01*np.random.randn(*Y_train.shape)

    dataset_train = torch.utils.data.TensorDataset(torch.tensor(
        X_train, dtype=torch.float32).to(device), torch.tensor(Y_train, dtype=torch.float32).to(device))

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, batch_size=batch_size, shuffle=True)

    # Parameters for the Aliev-Panfilov model
    a = 0.01
    k = 8.0
    mu1 = 0.2
    mu2 = 0.3
    eps = 0.002
    b = 0.15
    h = 0.1
    D = 0.1

    loss_history = []
    optimizer = optim.Adam(pinn.parameters(), lr=0.005)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2000, gamma=0.3)

    for epoch in range(n_epochs):
        loss = train(pinn, optimizer, data_loader_train,
                     a, k, mu1, mu2, eps, b, h, D, device, pinn)
        scheduler.step()
        loss_history.append(loss)
        if epoch % 1000 == 0:
            print(f"Epoch: {epoch}, Loss: {loss}")

    # Save the model
    save_model_path = "./models"
    torch.save(pinn.state_dict(), save_model_path +
               f"/2D_pinn_aliev_panfilov_{num_layers}x{num_neurons}_{n_epochs}_{test_size}.pt")

    # Plot the loss history
    plt.plot(loss_history)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()


if __name__ == "__main__":
    main()
