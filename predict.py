import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import scipy.io
from torch_pinn import PINN1D,generate_data


def load_model(model_path, num_layers, num_neurons, device):
    model = PINN1D(num_layers, num_neurons, device).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def predict(model, x, device):
    x_tensor = torch.tensor(x, dtype=torch.float32).to(device)
    return model.predict(x_tensor)

def update_plot(frame_number, v_predict, V, line1, line2, text):
    line1.set_data(
        np.arange(v_predict.shape[0]), v_predict[:, frame_number])
    line2.set_data(np.arange(V.shape[0]), V[:, frame_number])

    rmse = np.sqrt(
        np.mean((v_predict[:, frame_number] - V[:, frame_number]) ** 2))
    text.set_text(f"RMSE: {rmse:.4e}")
    return line1, line2, text

def animate(v_predict, V):
    fig, ax = plt.subplots()
    ax.set_ylim([0, max(v_predict.max(), V.max()) * 1.1])
    ax.set_xlim([0, v_predict.shape[0]])
    ax.set_xlabel("Cell")
    ax.set_ylabel("V")
    line1, = ax.plot([], [], label='Predicted')
    line2, = ax.plot([], [], linestyle='dashed', label='GT')
    ax.legend(loc='upper left')
    text = ax.text(0.95, 0.95, "", transform=ax.transAxes,
                    ha="right", va="top")
    print(v_predict.shape, V.shape)

    return FuncAnimation(fig, update_plot, frames=v_predict.shape[1],
                            fargs=(v_predict, V, line1, line2, text), interval=40, blit=True)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device: ", device)

    model_path = "pinn_aliev_panfilov_20000_noise_rand.pt"
    num_layers = 3
    num_neurons = 32

    model = load_model(model_path, num_layers, num_neurons, device)

    # Load test data or create input data for predictions
    # For example, you can load the data from the previous script
    mat = scipy.io.loadmat('./data/data_1d_left.mat')
    V = mat['Vsav']
    x = mat['x']
    t = mat['t']
    X, Y, x_left, x_right, T_ic = generate_data()
    

    Y_pred = predict(model, X, device)
    # print("Mean squared error:", np.mean((Y_pred - Y_test) ** 2))
    V_pred = Y_pred[:, 0]
    V_pred = np.reshape(V_pred, (V.shape[0], V.shape[1]))
    print(f"Shape V_pred:{V_pred.shape}")
    print(f'Shape x: {x.shape}')
    print(f'shape V_GT: {V.shape}')
    idx_t_pred = 30
    #Compute RMSE
    rmse = np.sqrt(np.mean((V_pred - V) ** 2))
    print(f"RMSE: {rmse:.4e}")

    anim = animate(V_pred, V)
    plt.show()


    #Testing with higher resolution prediction
    x_hr=np.linspace(np.min(x),np.max(x),1000)
    t_hr=np.linspace(np.min(t),np.max(t),70)

    X_hr,T_hr=np.meshgrid(x_hr,t_hr)
    X_hr = X_hr.reshape(-1,1)
    T_hr = T_hr.reshape(-1,1)
    X_test_hr=np.hstack((X_hr,T_hr))
    print(np.shape(t))
    idx_hr=np.where(np.isclose(t_hr,11))

    V_pred_hr=predict(model,X_test_hr,device)[:,0]
    V_pred_hr = np.reshape(V_pred_hr, (70, 1000))


    
    plt.plot(x[0, :], V[idx_t_pred, :], label="True")
    plt.plot(x_hr, V_pred_hr[idx_t_pred, :],linestyle='dashed', label="Predicted")
    plt.xlabel("x")
    plt.ylabel("V")
    plt.legend()
    plt.show()




if __name__ == "__main__":
    main()