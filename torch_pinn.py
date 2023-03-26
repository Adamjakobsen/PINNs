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
        super(PINN1D, self).__init__()
        self.num_layers = num_layers
        self.num_neurons = num_neurons

        self.device = device

        layers = [nn.Linear(2, num_neurons), nn.Tanh()]
        for _ in range(num_layers - 2):
            layers += [nn.Linear(num_neurons, num_neurons), nn.Tanh()]
        layers += [nn.Linear(num_neurons, 2)]

        self.model = nn.Sequential(*layers).to(device)

    def forward(self, x):
        return self.model(x)

    def predict(self, x):
        x = torch.tensor(x, dtype=torch.float32)
        return self.model(x).to('cpu').detach().numpy()


def model_loss(pinn, x, y_true, a, k, mu1, mu2, eps, b, h, D, x_left, x_right, T_ic):
    x_space, t_space = x[:, 0:1].clone().requires_grad_(
        True), x[:, 1:2].clone().requires_grad_(True)
    y_pred = pinn(torch.cat([x_space, t_space], dim=1))

    V, W = y_pred[:, 0:1], y_pred[:, 1:2]
    V_true, W_true = y_true[:, 0:1], y_true[:, 1:2]

    dVdt = torch.autograd.grad(V, t_space, grad_outputs=torch.ones_like(
        V), create_graph=True, allow_unused=True)[0]
    dWdt = torch.autograd.grad(W, t_space, grad_outputs=torch.ones_like(
        W), create_graph=True, allow_unused=True)[0]

    dVdx = torch.autograd.grad(V, x_space, grad_outputs=torch.ones_like(
        V), create_graph=True, allow_unused=True)[0]
    dWdx = torch.autograd.grad(W, x_space, grad_outputs=torch.ones_like(
        W), create_graph=True, allow_unused=True)[0]

    if dVdx is not None and dWdx is not None:
        dVdxx = torch.autograd.grad(dVdx, x_space, grad_outputs=torch.ones_like(
            dVdx), create_graph=True, allow_unused=True)[0]
        dWdxx = torch.autograd.grad(dWdx, x_space, grad_outputs=torch.ones_like(
            dWdx), create_graph=True, allow_unused=True)[0]
    else:
        raise ValueError("Gradients dVdx or dWdx are not computed correctly.")

    eq_V = dVdt - D * dVdxx + k * V * (V - a) * (V - 1) + W * V
    eq_W = dWdt - (eps + (mu1 * W) / (mu2 + V)) * (-W - k * V * (V - b - 1))

    loss_V = torch.mean(torch.square(eq_V))
    loss_W = torch.mean(torch.square(eq_W))
    loss_data = torch.mean(torch.square(V - V_true)) + \
        torch.mean(torch.square(W - W_true))
    
    loss_BC = boundary_loss(x,y_pred,x_left, x_right,pinn)
    loss_IC = initial_condition_loss(x,y_pred,V_true,W_true,T_ic,pinn)

    

    return loss_V + loss_W + loss_data + loss_BC + loss_IC

def boundary_loss(x, y,x_left, x_right, pinn):
    v_x = torch.autograd.grad(y[:, 0], x, create_graph=True, grad_outputs=torch.ones_like(y[:, 0]))[0][:, 0:1]
    on_boundary = (x[:, 0] == x_left) | (x[:, 0] == x_right)
    v_x_boundary = v_x[on_boundary]
    loss = torch.mean(v_x_boundary**2)
    return loss

def initial_condition_loss(x, y, observe_train, v_train,T_ic,pinn):
    T = x[:, -1].reshape(-1, 1)
    T_ic = torch.tensor(T_ic, dtype=torch.float32)
    idx_init = torch.where(torch.isclose(T, T_ic))[0]
    y_init = y[idx_init]
    v_init = v_train[idx_init]
    loss = torch.mean((y_init - v_init)**2)
    return loss


def train(model, optimizer, data_loader, a, k, mu1, mu2, eps, b, h, D, device, x_left, x_right, T_ic):
    model.train()
    for x, y_true in data_loader:
        x = x.to(device)
        y_true = y_true.to(device)
        x.requires_grad = True  # Set requires_grad to True

        optimizer.zero_grad()
        loss = model_loss(model, x, y_true, a, k, mu1, mu2,
                          eps, b, h, D, x_left, x_right, T_ic)
        loss.backward()
        optimizer.step()

        # Ensure all tensors are on the same device
        for param in model.parameters():
            param.data = param.data.to(device)
            if param.grad is not None:
                param.grad = param.grad.to(device)

    return loss.item()


def generate_data():
    mat = scipy.io.loadmat('./data/data_1d_left.mat')
    V = mat['Vsav']
    W = mat['Wsav']
    t = mat['t']
    x = mat['x']

    X, T = np.meshgrid(x, t)
    X = X.reshape(-1, 1)
    
    T = T.reshape(-1, 1)

    V = V.reshape(-1, 1)
    W = W.reshape(-1, 1)

    


    x_left = np.min(x)
    x_right = np.max(x)

    T_ic = np.min(t)

    return np.hstack((X, T)), np.hstack((V, W)), x_left, x_right, T_ic


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device: ", device)

    num_neurons = 32
    num_layers = 3
    n_epochs = 20000
    batch_size = 512

    # Create PINN object
    pinn = PINN1D(num_layers, num_neurons, device).to(device)

    # Generate data for training and testing
    X, Y, x_left, x_right, T_ic = generate_data()
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y,random_state=True, test_size=0.9)

    #Add noise to training data
    Y_train=Y_train + 0.01*np.random.randn(*Y_train.shape)

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
    """ 
    optimizer = optim.Adam(pinn.parameters(), lr=0.005)
    loss = train(pinn, optimizer, data_loader_train,
                 a, k, mu1, mu2, eps, b, h, D, device, x_left, x_right, T_ic)
    while loss > 0.1:
        loss = train(pinn, optimizer, data_loader_train,
                     a, k, mu1, mu2, eps, b, h, D, device, x_left, x_right, T_ic)
        print(f" Loss: {loss}")
    """
    loss_history = []
    optimizer = optim.Adam(pinn.parameters(), lr=0.005)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2000, gamma=0.5)
    for epoch in range(n_epochs):
        loss = train(pinn, optimizer, data_loader_train,
                     a, k, mu1, mu2, eps, b, h, D, device, x_left, x_right, T_ic)
        scheduler.step()
        loss_history.append(loss)
        if epoch % 1000 == 0:
            print(f"Epoch: {epoch}, Loss: {loss}")

    # Save the model
    torch.save(pinn.state_dict(), "pinn_aliev_panfilov_20000_noise_rand.pt")

    # Test the model
    pinn.eval()
    Y_pred = pinn.predict(torch.tensor(X, dtype=torch.float32).to(device))

    mat = scipy.io.loadmat('./data/data_1d_left.mat')
    V = mat['Vsav']
    x = mat['x']
    # print("Mean squared error:", np.mean((Y_pred - Y_test) ** 2))
    V_pred = Y_pred[:, 0]
    V_pred = np.reshape(V_pred, (V.shape[0], V.shape[1]))
    print(f"Shape V_pred:{V_pred.shape}")
    print(f'Shape x: {x.shape}')
    print(f'shape V_GT: {V.shape}')
    plt.plot(x[0, :], V_pred[15, :], label="Predicted")
    plt.plot(x[0, :], V[15, :], label="True")
    plt.xlabel("x")
    plt.ylabel("V")
    plt.legend()
    plt.show()

    #plot loss
    plt.plot(np.linspace(0,20000,len(loss_history)),loss_history)
    plt.show()


if __name__ == "__main__":
    main()
